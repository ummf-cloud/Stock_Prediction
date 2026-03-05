"""
src/feature_utils.py
====================
Utility functions for downloading and preparing financial data.

Functions
---------
extract_features()               — Downloads stock/FX/index data via yfinance + FRED
get_bitcoin_historical_prices()  — Downloads BTC/USD daily prices from CoinGecko
"""

import datetime
import os
import sys

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pandas_datareader.data as web


# ══════════════════════════════════════════════════════════════════════════════
#  extract_features  (stock / FX / macro features — used by the stock model)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features():
    """
    Downloads 1 year of MSFT/IBM/GOOGL prices, JPY/USD, GBP/USD, SP500, DJIA,
    and VIX from Yahoo Finance + FRED, then computes 5-day log returns.

    Returns
    -------
    pd.DataFrame
        Feature matrix (no target column) with one row per 5-day period.
    """
    return_period = 5

    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE   = datetime.date.today().strftime("%Y-%m-%d")

    stk_tickers = ['MSFT', 'IBM', 'GOOGL']
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    Y = np.log(stk_data.loc[:, ('Adj Close', 'MSFT')]).diff(return_period).shift(-return_period)
    Y.name = Y.name[-1] + '_Future'

    X1 = np.log(stk_data.loc[:, ('Adj Close', ['GOOGL', 'IBM'])]).diff(return_period)
    X1.columns = X1.columns.droplevel()
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    X = pd.concat([X1, X2, X3], axis=1)

    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    dataset.index.name = 'Date'

    features = dataset.sort_index().reset_index(drop=True).iloc[:, 1:]
    return features


# ══════════════════════════════════════════════════════════════════════════════
#  get_bitcoin_historical_prices  (used by the Bitcoin model & Streamlit app)
# ══════════════════════════════════════════════════════════════════════════════

def get_bitcoin_historical_prices(days: int = 60) -> pd.DataFrame:
    """
    Fetches daily BTC/USD closing prices from the CoinGecko public API.

    Parameters
    ----------
    days : int, default 60
        Number of historical days to retrieve.

    Returns
    -------
    pd.DataFrame
        Index : datetime (UTC date, time-normalised)
        Column: 'Close Price (USD)'
    """
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily',
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data   = response.json()
    prices = data['prices']

    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')

    # Drop duplicate dates (CoinGecko sometimes returns an extra "today" row)
    df = df[~df.index.duplicated(keep='last')]

    return df
