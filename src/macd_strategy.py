from .strategy import Strategy
from .indicators import Indicators
import pandas as pd

class MacdStrategy(Strategy):
    def __init__(self, data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(data)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self) -> pd.DataFrame:
        df = self.data.copy()
        
        # Calculate MACD
        df['MACD'], df['Signal_Line'], df['Hist'] = Indicators.macd(
            df['Close'], 
            fast=self.fast_period, 
            slow=self.slow_period, 
            signal=self.signal_period
        )
        
        df['signal'] = 0
        
        # Buy: MACD crosses above Signal Line
        # We can detect this by checking if MACD > Signal Line AND Previous MACD < Previous Signal Line
        # But vectorised way:
        df.loc[df['MACD'] > df['Signal_Line'], 'signal'] = 1
        df.loc[df['MACD'] < df['Signal_Line'], 'signal'] = -1
        
        # State-based signal (1 for hold Long, -1 for hold Short/Flat)
        # The engine generally enters on '1' if flat.
        
        return df
