import yfinance as yf
import pandas as pd

class DataLoader:
    @staticmethod
    def get_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        # Ensure dates are strings
        # yfinance interval limits: 1m (7 days), 2m-15m (60 days), 30m-1h (730 days)
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False)
        except Exception as e:
            print(f"Error downloading data: {e}")
            return pd.DataFrame()
        
        # Handle MultiIndex columns (yfinance v0.2+)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        # Ensure we have data
        if data.empty:
            return data

        # Fix column names if needed (sometimes they come in lowercase or differ)
        data.columns = [c.capitalize() for c in data.columns]
        
        # Drop rows with missing values (especially for indicators)
        data.dropna(inplace=True)

        if hasattr(data.index, 'tz_localize'):
             # Remove timezone info to avoid comparison issues
             data.index = data.index.tz_localize(None)
             
        return data
