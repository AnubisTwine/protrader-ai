from src.data_loader import DataLoader
from src.simple_strategy import SmaCrossStrategy
from src.engine import BacktestEngine
import datetime

def main():
    ticker = 'AAPL'
    start = '2020-01-01'
    end = '2023-01-01'
    
    print(f"Fetching data for {ticker}...")
    data = DataLoader.get_data(ticker, start, end)
    
    if data.empty:
        print("No data fetched.")
        return

    print("Running Backtest...")
    strategy = SmaCrossStrategy(data, short_window=20, long_window=50)
    engine = BacktestEngine(initial_capital=10000)
    results = engine.run(strategy)
    
    print("\nResults:")
    print(f"Final Capital: ${results['final_capital']:.2f}")
    print(f"Return: {results['return_pct']:.2f}%")
    print(f"Total Trades: {len(results['trades'])}")
    
    if not results['trades'].empty:
        print("\nLast 5 Trades:")
        print(results['trades'].tail())

if __name__ == "__main__":
    main()
