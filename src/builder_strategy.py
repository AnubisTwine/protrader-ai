from .strategy import Strategy
from .indicators import Indicators
import pandas as pd
import numpy as np

class BuilderStrategy(Strategy):
    def __init__(self, data: pd.DataFrame, entry_rules: list, exit_rules: list):
        """
        entry_rules: List of dicts describing conditions.
            e.g. {
                'ind1': 'RSI', 'params1': {'period': 14}, 
                'op': '<', 
                'ind2': 'Value', 'params2': {'value': 30}
            }
        """
        super().__init__(data)
        self.entry_rules = entry_rules
        self.exit_rules = exit_rules

    def _get_indicator_series(self, df: pd.DataFrame, name: str, params: dict) -> pd.Series:
        name = name.upper()
        
        if name == 'CLOSE':
            return df['Close']
        elif name == 'OPEN':
            return df['Open']
        elif name == 'HIGH':
            return df['High']
        elif name == 'LOW':
            return df['Low']
        elif name == 'VOLUME':
            return df['Volume']
        elif name == 'VALUE':
            return pd.Series(params.get('value', 0), index=df.index)
        
        # Indicator library calls
        if name == 'SMA':
            return Indicators.sma(df['Close'], period=params.get('period', 20))
        elif name == 'EMA':
            return Indicators.ema(df['Close'], period=params.get('period', 20))
        elif name == 'RSI':
            return Indicators.rsi(df['Close'], period=params.get('period', 14))
        elif name == 'MACD' or name == 'MACD LINE':
             m, s, h = Indicators.macd(df['Close'], fast=params.get('fast', 12), slow=params.get('slow', 26), signal=params.get('signal', 9))
             return m
        elif name == 'MACD SIGNAL':
             m, s, h = Indicators.macd(df['Close'], fast=params.get('fast', 12), slow=params.get('slow', 26), signal=params.get('signal', 9))
             return s
        elif name == 'MACD HIST':
             m, s, h = Indicators.macd(df['Close'], fast=params.get('fast', 12), slow=params.get('slow', 26), signal=params.get('signal', 9))
             return h
             
        elif name == 'BOLLINGER':
             # Upper or Lower?
             u, l = Indicators.bollinger_bands(df['Close'], period=params.get('period', 20), std_dev=params.get('std', 2))
             band = params.get('band', 'upper')
             return u if band == 'upper' else l
             
        return pd.Series(0, index=df.index)

    def _evaluate_rule(self, df: pd.DataFrame, rule: dict) -> pd.Series:
        # Get series and ensure we name/store them for visualization if they are indicators
        s1 = self._get_indicator_series(df, rule['ind1'], rule['params1'])
        s2 = self._get_indicator_series(df, rule['ind2'], rule['params2'])
        
        # Helper to generate a unique readable name
        def get_name(ind, p):
             if ind == 'Value': return None # Don't plot constants usually
             if ind in ['Close', 'Open', 'High', 'Low', 'Volume']: return None
             
             # Format: RSI_14, SMA_50, MACD_12_26_9
             if ind in ['RSI', 'SMA', 'EMA']:
                 return f"{ind}_{p.get('period')}"
             if 'MACD' in ind:
                 return f"{ind}" # Simplified, usually standard
             if 'BOLLINGER' in ind:
                 return f"BB_{p.get('period')}_{p.get('band')}"
             return ind

        name1 = get_name(rule['ind1'], rule['params1'])
        name2 = get_name(rule['ind2'], rule['params2'])
        
        # Save to dataframe if valid name and not already there
        if name1 and name1 not in df.columns:
            df[name1] = s1
        if name2 and name2 not in df.columns:
            df[name2] = s2
            
        op = rule['op']
        
        if op == '>':
            return s1 > s2
        elif op == '<':
            return s1 < s2
        elif op == '>=':
            return s1 >= s2
        elif op == '<=':
            return s1 <= s2
        elif op == '==':
            return s1 == s2
        elif op == 'Crosses Above':
            # (Current S1 > Current S2) AND (Prev S1 < Prev S2)
            prev_s1 = s1.shift(1)
            prev_s2 = s2.shift(1)
            return (s1 > s2) & (prev_s1 < prev_s2)
        elif op == 'Crosses Below':
             prev_s1 = s1.shift(1)
             prev_s2 = s2.shift(1)
             return (s1 < s2) & (prev_s1 > prev_s2)
             
        return pd.Series(False, index=df.index)

    def generate_signals(self) -> pd.DataFrame:
        df = self.data.copy()
        
        # Initialize condition masks
        buy_condition = pd.Series(True, index=df.index)
        sell_condition = pd.Series(True, index=df.index)
        
        # If no rules, do nothing
        if not self.entry_rules:
            buy_condition = pd.Series(False, index=df.index)
            
        for rule in self.entry_rules:
            result = self._evaluate_rule(df, rule)
            buy_condition = buy_condition & result
            
        # Exit rules (if empty, maybe use inverse of buy? or simple manual exit)
        # For now, if exit rules exist, use them. If not, maybe use inverse of Entry?
        # Standard: Entry is Long, Exit is Flat/Short.
        if not self.exit_rules:
            # Default behavior: If not buy, are we selling? 
            # Or just Hold? Let's assume explicit exit needed, or user wants flip.
            sell_condition = pd.Series(False, index=df.index)
        else:
             for rule in self.exit_rules:
                result = self._evaluate_rule(df, rule)
                sell_condition = sell_condition & result
        
        df['signal'] = 0
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
