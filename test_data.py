import yfinance as yf
import pandas as pd

def test_fetch(ticker):
    print(f"Fetching {ticker}...")
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    print(f"Columns: {data.columns}")
    print(f"Shape: {data.shape}")
    print(data.head())
    print("-" * 20)
    
    # Check for multi-index
    if isinstance(data.columns, pd.MultiIndex):
        print("MultiIndex detected, dropping level...")
        data.columns = data.columns.droplevel(1)
        print(f"New Columns: {data.columns}")
        print(data.head())

test_fetch("GC=F")
test_fetch("QQQ")
