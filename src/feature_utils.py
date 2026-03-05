import numpy as np
import pandas as pd
import datetime
import yfinance as yf

def extract_features_pair(ticker_a='JPM', ticker_b='BAC'):
    """
    Fetches the last year of data for the cointegrated pair.
    Used by Streamlit to provide data to the SageMaker endpoint.
    """
    # Pull extra days to ensure the 'rolling window' in Custom_Classes has enough data
    START_DATE = (datetime.date.today() - datetime.timedelta(days=450)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    
    stk_tickers = [ticker_a, ticker_b]
    
    print(f"Downloading data for {stk_tickers}...")
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)

    # Use Adj Close for both
    df_a = stk_data.loc[:, ('Adj Close', ticker_a)]
    df_b = stk_data.loc[:, ('Adj Close', ticker_b)]

    dataset = pd.concat([df_a, df_b], axis=1).dropna()
    dataset.columns = [ticker_a, ticker_b]
    dataset.index.name = 'Date'
    
    return dataset.sort_index()

def get_bitcoin_historical_prices(days=60):
    """Placeholder function required if your Streamlit app expects it."""
    try:
        data = yf.download("BTC-USD", period=f"{days}d", interval="1d")
        return data[['Close']]
    except Exception as e:
        print(f"Error fetching BTC: {e}")
        return pd.DataFrame()
