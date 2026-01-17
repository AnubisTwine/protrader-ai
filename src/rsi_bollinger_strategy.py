from .strategy import Strategy
from .indicators import Indicators
import pandas as pd

class RsiBollingerStrategy(Strategy):
    def __init__(self, data, rsi_period=14, rsi_lower=30, rsi_upper=70, bb_period=20, bb_std=2):
        super().__init__(data)
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.bb_period = bb_period
        self.bb_std = bb_std

    def generate_signals(self) -> pd.DataFrame:
        df = self.data.copy()
        
        # Calculate Indicators
        df['RSI'] = Indicators.rsi(df['Close'], self.rsi_period)
        df['BB_Upper'], df['BB_Lower'] = Indicators.bollinger_bands(df['Close'], self.bb_period, self.bb_std)
        
        df['signal'] = 0
        
        # Mean Reversion Logic
        # Buy when Oversold (RSI < Lower) AND Price touches Lower Band (potential bounce)
        buy_condition = (df['RSI'] < self.rsi_lower) & (df['Close'] < df['BB_Lower'])
        
        # Sell when Overbought (RSI > Upper) AND Price touches Upper Band (potential reversal)
        sell_condition = (df['RSI'] > self.rsi_upper) & (df['Close'] > df['BB_Upper'])
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Note: This is a mean reversion strategy.
        # It buys dips and sells rips.
        
        return df
