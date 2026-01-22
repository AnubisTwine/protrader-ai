import yfinance as yf
import pandas as pd

try:
    ticker = "AAPL"
    print(f"--- Fetching News for {ticker} ---")
    t = yf.Ticker(ticker)
    news = t.news
    if news:
        print(f"News count: {len(news)}")
        print(f"First item keys: {news[0].keys()}")
        print(f"First item raw: {news[0]}")
    else:
        print("No news found.")

    print("\n--- Fetching 5m Data ---")
    df = yf.download(ticker, period="1d", interval="5m", progress=False)
    print(df.head())
    print("\nIndex type:", type(df.index))
    if len(df) > 1:
        diff = df.index.to_series().diff().dropna()
        print("\nTime diffs head:", diff.head())
        print("Max gap:", diff.max())

except Exception as e:
    print(f"Error: {e}")
