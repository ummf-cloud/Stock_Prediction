import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
#from datetime import datetime, timedelta
import os
import sys

import os
import sys


# ... continue with your script ...

def extract_features():
    return_period = 5
    
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    
    # AAPL updated here
    stk_tickers = ['MPWR', 'AAPL'] #['AAPL', 'IBM', 'GOOGL']
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']
    
    #stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    #ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    # AAPL future returns
    #Y = np.log(stk_data.loc[:, ('Adj Close', 'AAPL')]).diff(return_period).shift(-return_period)
    Y = stk_data.loc[:, ('Adj Close', 'AAPL')]
    Y.name = 'AAPL'

    X = stk_data.loc[:, ('Adj Close', 'MPWR')]
    Y.name = 'MPWR'


    
    #X1 = np.log(stk_data.loc[:, ('Adj Close', ('GOOGL', 'IBM'))]).diff(return_period)
    #X1.columns = X1.columns.droplevel()
    #X2 = np.log(ccy_data).diff(return_period)
    #X3 = np.log(idx_data).diff(return_period)

    #X = pd.concat([X1, X2, X3], axis=1)
    
    # The 4 NEW features
    X['AAPL_SMA_14'] = stk_data.loc[:, ('Adj Close', 'AAPL')].rolling(window=14).mean()
    X['AAPL_Volatility'] = stk_data.loc[:, ('High', 'AAPL')] - stk_data.loc[:, ('Low', 'AAPL')]
    X['AAPL_Momentum_14'] = stk_data.loc[:, ('Adj Close', 'AAPL')].pct_change(14)
    X['Is_Quarter_End'] = X.index.is_quarter_end.astype(int)
    
    dataset = pd.concat([Y, X], axis=1).dropna()#.iloc[::return_period, :]
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.name]
    dataset.index.name = 'Date'
    
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    #features = features.iloc[:,1:]
    
    return features

def get_bitcoin_historical_prices(days = 60):
    
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily' # Ensure we get daily granularity
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df






