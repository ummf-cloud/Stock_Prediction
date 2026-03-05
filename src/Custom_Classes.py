"""
src/Custom_Classes.py
=====================
Custom sklearn-compatible transformers used across the Bitcoin signal pipeline.

Classes
-------
FeatureEngineer      — Generates MACD, BB_Width, ROC, MOM from a price series
AutoPowerTransformer — Selectively applies Yeo-Johnson to skewed numeric columns
FeatureSelector      — Drops high-missing, high-cardinality, and low-correlation cols
PairFeatureEngineer  — Rolling OLS spread + z-score for pairs-trading features
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew


# ══════════════════════════════════════════════════════════════════════════════
#  FeatureEngineer
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Generates 4 technical-indicator features per window from a single-column
    price DataFrame (column must be named 'Close' or be the first column).

    Features per window (total = 4 × len(windows)):
        MACD_{w}      — EMA(w) - EMA(2w)
        BB_Width_{w}  — 4 × rolling std (= upper BB - lower BB)
        ROC_{w}       — Rate-of-change (%)
        MOM_{w}       — Raw price difference over w periods

    Returns a numpy array of shape (n_samples, 4 × n_windows).
    The Close column is NOT included in the output to prevent data leakage.

    Parameters
    ----------
    windows : list of int, default [10]
        Look-back periods for each set of indicators.
    """

    def __init__(self, windows=None):
        self.windows = windows if windows is not None else [10]

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        return self  # stateless

    # ------------------------------------------------------------------
    def transform(self, X, y=None):
        # ── Normalise input to a DataFrame with a 'Close' column ──────
        if isinstance(X, np.ndarray):
            X_ = pd.DataFrame(X, columns=['Close'])
        else:
            X_ = X.copy()
            if 'Close' not in X_.columns:
                X_.columns = ['Close']

        price    = X_['Close']
        new_cols = []

        for w in self.windows:
            # 1. MACD — trend signal
            ema_fast = price.ewm(span=w, adjust=False).mean()
            ema_slow = price.ewm(span=w * 2, adjust=False).mean()
            X_[f'MACD_{w}'] = ema_fast - ema_slow
            new_cols.append(f'MACD_{w}')

            # 2. Bollinger Band Width — volatility
            std = price.rolling(window=w).std()
            X_[f'BB_Width_{w}'] = 4 * std       # = (sma+2σ) - (sma-2σ)
            new_cols.append(f'BB_Width_{w}')

            # 3. Rate of Change — momentum %
            X_[f'ROC_{w}'] = price.pct_change(periods=w) * 100
            new_cols.append(f'ROC_{w}')

            # 4. Momentum — raw difference
            X_[f'MOM_{w}'] = price.diff(periods=w)
            new_cols.append(f'MOM_{w}')

        # Return ONLY engineered columns (no Close leakage)
        return X_[new_cols].values

    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        names = []
        for w in self.windows:
            names += [f'MACD_{w}', f'BB_Width_{w}', f'ROC_{w}', f'MOM_{w}']
        return np.array(names)


# ══════════════════════════════════════════════════════════════════════════════
#  AutoPowerTransformer
# ══════════════════════════════════════════════════════════════════════════════

class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Yeo-Johnson power transform only to numeric columns whose absolute
    skewness exceeds `threshold`.
    """

    def __init__(self, threshold=0.75):
        self.threshold   = threshold
        self.skewed_cols = []
        self.pt          = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return self

        skewness          = numeric_df.apply(lambda x: skew(x.dropna()))
        self.skewed_cols  = skewness[abs(skewness) > self.threshold].index.tolist()

        if self.skewed_cols:
            self.pt.fit(X[self.skewed_cols])
        return self

    def transform(self, X):
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.skewed_cols:
            X_copy[self.skewed_cols] = self.pt.transform(X_copy[self.skewed_cols])
        return X_copy


# ══════════════════════════════════════════════════════════════════════════════
#  FeatureSelector
# ══════════════════════════════════════════════════════════════════════════════

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Drops columns that are:
      - mostly missing  (> missing_threshold)
      - high-cardinality categoricals  (> cardinality_threshold unique ratio)
      - weakly correlated with target  (abs corr < corr_threshold)
    """

    def __init__(self, missing_threshold=0.3,
                 corr_threshold=0.03,
                 cardinality_threshold=0.9):
        self.missing_threshold     = missing_threshold
        self.corr_threshold        = corr_threshold
        self.cardinality_threshold = cardinality_threshold
        self.features_to_keep      = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. Missing values filter
        null_ratios  = X.isnull().mean()
        cols_ok      = null_ratios[null_ratios <= self.missing_threshold].index.tolist()
        X_filtered   = X[cols_ok]

        # 2. High-cardinality filter (categoricals only)
        cat_cols     = X_filtered.select_dtypes(exclude='number').columns
        high_card    = [c for c in cat_cols
                        if X_filtered[c].nunique() / len(X_filtered) > self.cardinality_threshold]
        remaining_cats = [c for c in cat_cols if c not in high_card]

        # 3. Correlation filter (numerics only)
        numeric_X    = X_filtered.select_dtypes(include='number')
        if y is not None and not numeric_X.empty:
            temp_df      = numeric_X.copy()
            temp_df['_target'] = y
            correlations = temp_df.corr()['_target'].abs().drop('_target')
            numeric_keep = correlations[correlations >= self.corr_threshold].index.tolist()
        else:
            numeric_keep = numeric_X.columns.tolist()

        self.features_to_keep = numeric_keep + remaining_cats
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.features_to_keep]


# ══════════════════════════════════════════════════════════════════════════════
#  PairFeatureEngineer
# ══════════════════════════════════════════════════════════════════════════════

class PairFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes rolling-OLS spread, beta, z-score, spread std, and beta stability
    for a pair of price series — intended for pairs-trading models.

    Parameters
    ----------
    window : int, default 60
        Look-back window for rolling OLS regression.
    """

    def __init__(self, window=60):
        self.window      = window
        self.last_beta_  = None
        self.last_alpha_ = None
        self.is_fitted_  = False

    def fit(self, X, y=None):
        if len(X) < self.window:
            raise ValueError(
                f"Data length {len(X)} is less than window size {self.window}"
            )
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Must call fit() before transform().")

        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=['price_a', 'price_b'])
        else:
            df = X.copy()
            df.columns = ['price_a', 'price_b']

        df[['spread', 'beta']] = self._rolling_regression(df)
        df['z_score']          = self._z_score(df['spread'])
        df['spread_std']       = df['spread'].rolling(self.window).std()
        df['beta_stability']   = df['beta'].rolling(self.window).std()

        return df

    # ------------------------------------------------------------------
    def _rolling_regression(self, df):
        spreads = np.full(len(df), np.nan)
        betas   = np.full(len(df), np.nan)
        a_vals  = df['price_a'].values
        b_vals  = df['price_b'].values

        for i in range(self.window, len(df)):
            y      = a_vals[i - self.window: i]
            x      = b_vals[i - self.window: i]
            model  = sm.OLS(y, sm.add_constant(x)).fit()
            alpha, beta = model.params[0], model.params[1]
            betas[i]   = beta
            spreads[i] = a_vals[i] - (beta * b_vals[i] + alpha)
            self.last_alpha_, self.last_beta_ = alpha, beta

        return pd.DataFrame({'spread': spreads, 'beta': betas}, index=df.index)

    def _z_score(self, spread):
        mean = spread.rolling(self.window).mean()
        std  = spread.rolling(self.window).std()
        return (spread - mean) / std
