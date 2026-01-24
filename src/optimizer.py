import itertools
import pandas as pd
from .engine import BacktestEngine
from .simple_strategy import SmaCrossStrategy
from .rsi_bollinger_strategy import RsiBollingerStrategy
from .macd_strategy import MacdStrategy

class StrategyOptimizer:
    def __init__(self, data, strategy_type, fixed_params=None):
        self.data = data
        self.strategy_type = strategy_type
        self.fixed_params = fixed_params or {} 
        
    def run_optimization(self, param_ranges, metric='return_pct'):
        """
        param_ranges: dict of arg_name -> list of values
        e.g. {'short': [10, 20], 'long': [50, 100]}
        """
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        combinations = list(itertools.product(*values))
        
        results = []
        
        # Limit iterations for safety if too large
        if len(combinations) > 50:
            print(f"Warning: {len(combinations)} combinations. This might take a while.")
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            # Validation: Ensure logic (e.g. short < long)
            if self.strategy_type == "SMA Crossover":
                if params.get('short', 10) >= params.get('long', 50):
                    continue

            # Create Strategy Instance
            strategy = None
            if self.strategy_type == "SMA Crossover":
                 strategy = SmaCrossStrategy(self.data, short_window=params.get('short', 10), long_window=params.get('long', 50))
            elif self.strategy_type == "RSI + Bollinger":
                 strategy = RsiBollingerStrategy(self.data, 
                                                 rsi_period=params.get('rsi_p', 14), 
                                                 rsi_lower=params.get('rsi_l', 30), 
                                                 rsi_upper=params.get('rsi_u', 70), 
                                                 bb_period=params.get('bb_p', 20), 
                                                 bb_std=params.get('bb_s', 2))
            elif self.strategy_type == "MACD":
                 strategy = MacdStrategy(self.data, 
                                         fast_period=params.get('fast', 12), 
                                         slow_period=params.get('slow', 26), 
                                         signal_period=params.get('signal', 9))
            
            if strategy:
                engine = BacktestEngine(initial_capital=self.fixed_params.get('initial_capital', 10000))
                res = engine.run(strategy, 
                                 multiplier=self.fixed_params.get('multiplier', 1), 
                                 leverage=self.fixed_params.get('leverage', 1), 
                                 fixed_size=self.fixed_params.get('fixed_size', 0),
                                 stop_loss_pct=self.fixed_params.get('stop_loss_pct', 0),
                                 take_profit_pct=self.fixed_params.get('take_profit_pct', 0))
                
                # Store result
                res_metric = res['metrics'].get(metric, 0)
                if metric == 'return_pct': res_metric = res['return_pct']
                
                results.append({
                    'params': params,
                    'metric': res_metric,
                    'win_rate': res['metrics']['win_rate'],
                    'trades': res['metrics']['total_trades'],
                    'sharpe': res['metrics']['sharpe_ratio'],
                    'drawdown': res['metrics']['max_drawdown']
                })
                
        # Sort by metric
        results.sort(key=lambda x: x['metric'], reverse=True)
        return results
