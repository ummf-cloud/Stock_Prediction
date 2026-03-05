import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew

class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.skewed_cols = []
        self.pt = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return self
        skewness = numeric_df.apply(lambda x: skew(x.dropna()))
        self.skewed_cols = skewness[abs(skewness) > self.threshold].index.tolist()
        if self.skewed_cols:
            self.pt.fit(X[self.skewed_cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
            X_copy = pd.DataFrame(X_copy)
        if self.skewed_cols:
            X_copy[self.skewed_cols] = self.pt.transform(X_copy[self.skewed_cols])
        return X_copy

class PairFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, window=60):
        self.window = window
        self.last_beta_ = None
        self.last_alpha_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        if len(X) < self.window:
            # For inference, we may have less data, but for training we need the window
            pass
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=['price_a', 'price_b'])
        else:
            df = X.copy()
            df.columns = ['price_a', 'price_b']
        
        # 1. Compute Rolling Spread and Beta
        df[['spread', 'beta']] = self._compute_rolling_regression(df)

        # 2. Derive Statistics-based Features
        df['z_score'] = self._calculate_z_score(df['spread'])
        df['spread_std'] = df['spread'].rolling(self.window).std()
        df['beta_stability'] = df['beta'].rolling(self.window).std()
        
        # Add a lagged return feature to match your notebook training
        df['return_lag'] = df['price_a'].pct_change().shift(1)
        
        return df

    def _compute_rolling_regression(self, df):
        spreads = np.full(len(df), np.nan)
        betas = np.full(len(df), np.nan)
        a_vals = df['price_a'].values
        b_vals = df['price_b'].values

        # Ensure we have enough data for at least one regression
        start_idx = self.window if len(df) > self.window else 1
        
        for i in range(start_idx, len(df) + 1):
            window_start = max(0, i - self.window)
            y = a_vals[window_start:i]
            x = b_vals[window_start:i]
            
            if len(y) < 2: continue # Need at least 2 points
            
            x_with_const = sm.add_constant(x)
            model = sm.OLS(y, x_with_const).fit()
            
            alpha, beta = model.params[0], model.params[1]
            if i < len(df):
                betas[i] = beta
                spreads[i] = a_vals[i] - (beta * b_vals[i] + alpha)
            else:
                # Handle the last element for live prediction
                self.last_alpha_, self.last_beta_ = alpha, beta
            
        return pd.DataFrame({'spread': spreads, 'beta': betas}, index=df.index)

    def _calculate_z_score(self, spread_series):
        rolling_mean = spread_series.rolling(self.window, min_periods=1).mean()
        rolling_std = spread_series.rolling(self.window, min_periods=1).std()
        return (spread_series - rolling_mean) / rolling_std
