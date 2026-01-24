import pandas as pd
from .strategy import Strategy

class BacktestEngine:
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trade_log = []
        self.equity_curve = []

    def run(self, strategy: Strategy, multiplier: float = 1.0, leverage: float = 1.0, fixed_size: int = 0, pyramiding_limit: int = 1, stop_loss_pct: float = 0.0, take_profit_pct: float = 0.0):
        data = strategy.generate_signals()
        
        self.entry_count = 0
        self.avg_entry_price = 0.0
        
        for index, row in data.iterrows():
            price = row['Close']
            # Fallback to Close if High/Low missing
            high = row.get('High', price)
            low = row.get('Low', price)
            signal = row.get('signal', 0)
            
            # --- CHECK STOPS (Before Signal) ---
            exit_signal = False
            if self.position > 0:
                 sl_price = self.avg_entry_price * (1 - stop_loss_pct)
                 tp_price = self.avg_entry_price * (1 + take_profit_pct)
                 
                 exit_price = None
                 exit_type = None
                 
                 # Check Low for SL
                 if stop_loss_pct > 0 and low <= sl_price:
                      exit_price = sl_price
                      exit_type = "STOP_LOSS"
                 # Check High for TP
                 elif take_profit_pct > 0 and high >= tp_price:
                      exit_price = tp_price
                      exit_type = "TAKE_PROFIT"
                      
                 if exit_price:
                      pnl = (exit_price - self.avg_entry_price) * self.position * multiplier
                      self.capital += pnl
                      self.trade_log.append({
                            'Date': index,
                            'Type': exit_type,
                            'Price': exit_price,
                            'Size': self.position, 
                            'Value': pnl
                      })
                      self.position = 0
                      self.entry_count = 0
                      self.avg_entry_price = 0.0
                      exit_signal = True # Position closed

            # Mark to Market Equity
            # Equity = Cash + Unrealized PnL
            unrealized_pnl = 0
            if self.position > 0:
                unrealized_pnl = (price - self.avg_entry_price) * self.position * multiplier
            
            current_equity = self.capital + unrealized_pnl
            self.equity_curve.append({'Date': index, 'Equity': current_equity})

            # Don't process entry signals if we just stopped out same bar
            if exit_signal:
                continue

            # BUY
            if signal == 1 and (self.position == 0 or (self.position > 0 and self.entry_count < pyramiding_limit)):
                
                # Determine how many to buy
                buy_price = price
                
                if fixed_size > 0:
                    quantity = fixed_size
                else:
                    # Auto size: Allocate remaining margin?
                    # Margin Used = Position * CurrentPrice * Multiplier / Leverage
                    margin_used = (self.position * price * multiplier) / leverage
                    available_equity = current_equity - margin_used
                    
                    if available_equity > 0:
                        # buying_power = available_equity * leverage
                        # But simpler: How many contracts can we hold max?
                        # Max Position Value = CurrentEquity * Leverage
                        # Max Contracts = Max Pos Value / (Price * Multiplier)
                        # Qty = Max Contracts - Current Position
                        
                        max_val = current_equity * leverage
                        max_contracts = int(max_val // (price * multiplier))
                        quantity = max_contracts - self.position
                        if quantity < 0: quantity = 0
                    else:
                        quantity = 0

                if quantity > 0:
                     # Check if we have enough margin to open
                     if fixed_size > 0:
                         # User override for Futures: Trust user has margin, check only bankruptcy
                         margin_checks_out = current_equity > 0
                     else:
                        margin_needed = (quantity * price * multiplier) / leverage
                        margin_checks_out = current_equity >= margin_needed
                     
                     if margin_checks_out:
                         # EXECUTE BUY
                         cost_basis = quantity * price * multiplier
                         
                         # Update Weighted Average Entry Price
                         self.avg_entry_price = ((self.avg_entry_price * self.position) + (price * quantity)) / (self.position + quantity)
                         
                         self.position += quantity
                         self.entry_count += 1
                         
                         self.trade_log.append({
                            'Date': index,
                            'Type': 'BUY',
                            'Price': price,
                            'Size': quantity, # Changed from Shares
                            'Value': cost_basis
                        })

            # SELL (Exit)
            elif signal == -1 and self.position > 0:
                # Close Position
                sell_price = price
                pnl = (sell_price - self.avg_entry_price) * self.position * multiplier
                
                self.capital += pnl
                self.position = 0
                self.entry_count = 0
                self.avg_entry_price = 0
                
                self.trade_log.append({
                    'Date': index,
                    'Type': 'SELL',
                    'Price': price,
                    'Size': self.position, # Just for log, actually 0 now
                    'Value': pnl # Realized PnL
                })

        # Final Close
        if self.position > 0:
            final_price = data.iloc[-1]['Close']
            pnl = (final_price - self.avg_entry_price) * self.position * multiplier
            self.capital += pnl
            self.position = 0

            
        # Calculate Performance Metrics
        equity_df = pd.DataFrame(self.equity_curve).set_index('Date')
        
        # Returns
        equity_df['Returns'] = equity_df['Equity'].pct_change()
        
        # Max Drawdown
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
        max_drawdown = equity_df['Drawdown'].min() * 100 # In percentage
        
        # Sharpe Ratio (assuming 252 trading days, 0% risk free for simplicity)
        if equity_df['Returns'].std() > 0:
            sharpe_ratio = (equity_df['Returns'].mean() / equity_df['Returns'].std()) * (252 ** 0.5)
        else:
            sharpe_ratio = 0.0

        # Detailed Trade Stats
        trades_df = pd.DataFrame(self.trade_log)
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0
        total_trades_count = 0
        total_wins = 0
        total_losses = 0

        if not trades_df.empty and 'Type' in trades_df.columns:
            # Filter solely for realized PnL explicitly
            # In this engine, SELL/Sl/TP entries have 'Value' as PnL? 
            # Looking at code: 
            # SELL: 'Value': pnl
            # STOP_LOSS: 'Value': pnl
            # TAKE_PROFIT: 'Value': pnl
            # BUY: 'Value': cost_basis (positive)
            
            # So we sum 'Value' where Type != 'BUY'
            exits = trades_df[trades_df['Type'].isin(['SELL', 'STOP_LOSS', 'TAKE_PROFIT'])]
            total_trades_count = len(exits)
            
            if not exits.empty:
                wins = exits[exits['Value'] > 0]
                losses = exits[exits['Value'] <= 0]
                
                total_wins = len(wins)
                total_losses = len(losses)
                
                win_rate = (total_wins / total_trades_count) * 100
                gross_profit = wins['Value'].sum()
                gross_loss = abs(losses['Value'].sum())
                
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.99
                avg_trade = exits['Value'].mean()

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'return_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades_df,
            'equity_curve': equity_df,
            'metrics': {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade': avg_trade,
                'total_trades': total_trades_count,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_wins': total_wins,
                'total_losses': total_losses
            }
        }
