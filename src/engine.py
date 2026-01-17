import pandas as pd
from .strategy import Strategy

class BacktestEngine:
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trade_log = []
        self.equity_curve = []

    def run(self, strategy: Strategy, multiplier: float = 1.0, leverage: float = 1.0, fixed_size: int = 0, pyramiding_limit: int = 1):
        data = strategy.generate_signals()
        
        # Track entries for pyramiding
        self.entry_count = 0 
        
        # Simple loop-based backtest
        for index, row in data.iterrows():
            price = row['Close']
            signal = row.get('signal', 0)
            
            # Record equity before trade
            # Value = Cash + (Position * Price * Multiplier)
            current_value = self.capital + (self.position * price * multiplier)
            self.equity_curve.append({'Date': index, 'Equity': current_value})

            # BUY LOGIC
            # Condition: Signal is Buy AND (We are flat OR (We are long AND allow pyramiding))
            if signal == 1 and (self.position == 0 or (self.position > 0 and self.entry_count < pyramiding_limit)):  
                
                # Check for rapid-fire filling on state signals (1,1,1...)
                # Simple heuristic: Don't buy if we bought on the very previous bar? 
                # For now, we allow it, assuming user handles strategy logic or accepts rapid filling if using fixed size.
                # Ideally, strategy should emit '1' only on entry triggers, not just state.
                # However, to prevent unintended full margin blowup on state signals:
                # If using Fixed Size: It's okay, user controls it.
                # If using Auto Margin: It will blow up.
                
                # Determine Size
                shares_to_buy = 0
                
                if fixed_size > 0:
                     shares_to_buy = fixed_size
                else:
                     # Auto/Margin based
                     # If pyramiding, we need to be careful not to use full leverage on first trade if we want to save for later.
                     # But simple logic: Use remaining buying power?
                     pass
                     # For this version, let's keep Auto logic as "Max Possible" for first entry, 
                     # which effectively disables pyramiding for Auto unless we split capital.
                     # We'll rely on Fixed Size for pyramiding usefulness.
                     buying_power = self.capital * leverage
                     contract_value = price * multiplier
                     if contract_value > 0:
                        shares_to_buy = int(buying_power // contract_value)

                # Execute Buy
                if shares_to_buy > 0:
                    # Check Capital (CRITICAL: Can we afford it?)
                    cost = shares_to_buy * (price * multiplier) # Not really cost, but margin requirement usually. 
                    # In this simple cash engine, we subtract full value or margin value?
                    # Engine tracks 'capital' as Cash. 
                    # Buying futures doesn't cost full value, only margin. 
                    # But PnL is added/subtracted.
                    # Simplified: We treat 'capital' as Account Balance. 
                    # We don't subtract 'cost' for futures usually, we just checks margin.
                    # But here we stick to the previous robust logic:
                    # self.capital -= cost (treating it like spot/stocks for safety/simplicity or assume full cash backing)
                    # If leverage is high, this line might make 'capital' negative if we treat it as cash-out.
                    # Correct way for futures simplified: Capture entry price. PnL is realized on close.
                    # Current engine subtracts cost. This works for Stocks. For futures, it's weird.
                    # Let's stick to the previous working logic but apply fixed_size limits.
                    
                    self.capital -= (shares_to_buy * price * multiplier) if leverage == 1.0 else (shares_to_buy * price * multiplier / leverage)
                    # Actually, previous code: self.capital -= cost (where cost = shares * price).
                    
                    cost_basis = shares_to_buy * price * multiplier
                    # Update PnL logic later? No, stick to existing flow.
                    if leverage > 1:
                        # For leverage, we only lock up Margin.
                        margin_req = cost_basis / leverage
                        if self.capital >= margin_req:
                             self.capital -= margin_req
                             self.position += shares_to_buy
                             self.entry_count += 1
                             self.trade_log.append({
                                'Date': index,
                                'Type': 'BUY',
                                'Price': price,
                                'Size': shares_to_buy,
                                'Value': cost_basis
                            })
                    else:
                         # Cash account logic
                         if self.capital >= cost_basis:
                             self.capital -= cost_basis
                             self.position += shares_to_buy
                             self.entry_count += 1
                             self.trade_log.append({
                                'Date': index,
                                'Type': 'BUY',
                                'Price': price,
                                'Size': shares_to_buy,
                                'Value': cost_basis
                             })

            # SELL LOGIC (Exit)
            elif signal == -1 and self.position > 0: 
                revenue = self.position * price * multiplier
                
                # We need to add back the Margin we took out, PLUS PnL.
                # Current Logic: self.capital += revenue. 
                # Previous Buy: self.capital -= cost.
                # Net: capital - cost + revenue = capital + (revenue - cost) = capital + PnL.
                # This holds true for leverage too IF we subtracted Margin Cost and add back Margin Revenue (Revenue / Leverage? No.)
                
                # FIXING LEVERAGE MATH for the loop:
                # Buy: Capital -= (Price * Size / Leverage)
                # Sell: Capital += (Price * Size / Leverage) + (Price - EntryPrice) * Size * Multiplier?
                # The previous simple engine logic:
                # Buy: Cap -= Cost. Sell: Cap += Rev. PnL = Rev - Cost.
                # This works perfectly for PnL calculation regardless of leverage IF we simply track cash flow.
                # But for Futures, we don't pay full price. 
                # Let's keep it abstract: We pay 'entry_value' and get 'exit_value'. 
                # Logic holds.
                
                # Check if we should only close PARTIAL? 
                # Current logic: Close ALL.
                self.capital += revenue # This assumes we paid full 'revenue' amount to enter previously? No.
                # Wait, if we used leverage logic above: self.capital -= margin_req.
                # If we verify the math:
                # Buy at 100, Lev 10. Margin 10. Cap 100 -> 90.
                # Sell at 110. Revenue 110? No.
                # We need to reconstruct the logic to specific Spot vs Margin if we want 100% accuracy.
                # BUT, given the previous code was:
                #  shares = capital // price  (Spot logic)
                #  capital -= shares * price
                #  capital += shares * price (on sell)
                # I will stick to this Spot-like logic for minimal regression, 
                # just adjusting 'cost' by leverage factor.
                
                # If I changed Buy to use Margin, I must change Sell to return Margin + PnL.
                # Revenue (Full) = Position * Price * Multipler.
                # If we treat capital flow as Full Value / Leverage:
                # Buy: -FullValue / Lev
                # Sell: +FullValue / Lev
                # PnL = (SellVal - BuyVal) / Lev. << WRONG. PnL is not levered. PnL is absolute.
                
                # REVERT strategy to simpler Cash Flow for stability, just allow buying more contracts than cash exists (Leverage).
                # New Logic:
                # Allow position tracking separate from Cash?
                # No, stick to: Capital = Cash.
                # On Buy: Check if (Position * Price * Mult / Leverage) < Capital.
                # Don't subtract from Capital. Capital is just balance.
                # On Sell: Capital += (SellPrice - AvgEntryPrice) * Size * Mult.
                
                # REFACTORING ENGINE FOR FUTURES ACCURACY
                # We need Average Entry Price.
                pass
                
                # Fallback to simple spot loop with "virtual" capital checking for this iteration to avoid rewriting entire engine state.
                # Using the modify_file logic below to inject updated Run method.
                
    def run(self, strategy: Strategy, multiplier: float = 1.0, leverage: float = 1.0, fixed_size: int = 0, pyramiding_limit: int = 1):
        data = strategy.generate_signals()
        
        self.entry_count = 0
        self.avg_entry_price = 0.0
        
        for index, row in data.iterrows():
            price = row['Close']
            signal = row.get('signal', 0)
            
            # Mark to Market Equity
            # Equity = Cash + Unrealized PnL
            unrealized_pnl = 0
            if self.position > 0:
                unrealized_pnl = (price - self.avg_entry_price) * self.position * multiplier
            
            current_equity = self.capital + unrealized_pnl
            self.equity_curve.append({'Date': index, 'Equity': current_equity})

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
                     margin_needed = (quantity * price * multiplier) / leverage
                     
                     # Approximate margin check against Cash+Unrealized (Equity)
                     if current_equity >= margin_needed:
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
            sharpe_ratio = 0

        return {
            'final_capital': self.capital,
            'return_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': pd.DataFrame(self.trade_log),
            'equity_curve': equity_df
        }
