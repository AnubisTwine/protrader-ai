from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.position = 0
        self.trades = []
        self.equity = []

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Implement logic to generate signals.
        Should return the dataframe with a 'signal' column (1: Buy, -1: Sell, 0: Hold)
        """
        pass
