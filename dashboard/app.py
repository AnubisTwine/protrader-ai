import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import importlib
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="ProTrader AI", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- MODULE IMPORT & RELOAD ---
import src.data_loader
import src.simple_strategy
import src.rsi_bollinger_strategy
import src.engine
import src.indicators

importlib.reload(src.data_loader)
importlib.reload(src.simple_strategy)
importlib.reload(src.rsi_bollinger_strategy)
importlib.reload(src.engine)
importlib.reload(src.indicators)

from src.data_loader import DataLoader
from src.simple_strategy import SmaCrossStrategy
from src.rsi_bollinger_strategy import RsiBollingerStrategy
from src.engine import BacktestEngine

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1f2833;
        border-right: 1px solid #45a29e;
    }
    
    /* Headings */
    h1, h2, h3, h4, .big-font {
        color: #66fcf1;
        font-weight: 700;
        text-shadow: 0px 0px 10px rgba(102, 252, 241, 0.3);
    }
    
    /* KPI Cards */
    .metric-card {
        background: linear-gradient(145deg, #1f2833, #0b0c10);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #45a29e;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 252, 241, 0.2);
    }
    .metric-title {
        color: #c5c6c7;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #fff;
        font-size: 2rem;
        font-weight: bold;
        margin-top: 5px;
    }
    .metric-positive { color: #66fcf1; }
    .metric-negative { color: #ff6b6b; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #45a29e 0%, #66fcf1 100%);
        color: #0b0c10;
        border: none;
        padding: 12px 25px;
        border-radius: 50px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(102, 252, 241, 0.4);
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(102, 252, 241, 0.6);
        color: #000;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f2833;
        border-radius: 5px;
        color: #c5c6c7;
        border: 1px solid transparent;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #45a29e;
        color: #0b0c10;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'bt_results' not in st.session_state:
    st.session_state.bt_results = None
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = "GC=F"

# --- SIDEBAR NAV ---
st.sidebar.image("https://img.icons8.com/nolan/96/bullish.png", width=60)
st.sidebar.markdown("## ProTrader AI")
nav = st.sidebar.radio("Navigation", ["Dashboard", "Strategy Lab", "Trade Journal", "Settings"])

st.sidebar.markdown("---")

# --- GLOBAL SETTINGS (Always Visible) ---
st.sidebar.header("Global Asset Config")
asset_class = st.sidebar.selectbox("Active Asset", ["Gold (GC=F)", "Nasdaq (QQQ)", "S&P 500 (SPY)", "Bitcoin (BTC-USD)", "Custom"], index=0)

if asset_class == "Custom":
    ticker = st.sidebar.text_input("Ticker", value="AAPL")
elif "Gold" in asset_class: ticker = "GC=F"
elif "Nasdaq" in asset_class: ticker = "QQQ"
elif "S&P" in asset_class: ticker = "SPY"
elif "Bitcoin" in asset_class: ticker = "BTC-USD"

st.sidebar.markdown("---")

# --- HELPER FUNCTIONS ---
def get_kpi_card(title, value, is_positive=True, prefix="", suffix=""):
    color_class = "metric-positive" if is_positive else "metric-negative"
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value {color_class}">{prefix}{value}{suffix}</div>
    </div>
    """

def run_backtest_logic(start_date, end_date, interval, initial_capital, strategy_type, params, multiplier, leverage, fixed_size, pyramiding_limit):
    try:
        # Load Data
        fetch_start = start_date
        # Auto-adjust logic for intraday
        if interval in ['5m', '15m', '30m', '60m', '1h']:
             limit = 59 if 'm' in interval else 729
             min_date = datetime.date.today() - datetime.timedelta(days=limit)
             if fetch_start < min_date: fetch_start = min_date

        data = DataLoader.get_data(ticker, str(fetch_start), str(end_date), interval=interval)
        
        if data.empty:
            return None, "No Data Found"

        # Init Strategy
        if strategy_type == "SMA Crossover":
            strategy = SmaCrossStrategy(data, short_window=params['short'], long_window=params['long'])
        else:
            strategy = RsiBollingerStrategy(data, rsi_period=params['rsi_p'], rsi_lower=params['rsi_l'], rsi_upper=params['rsi_u'], bb_period=params['bb_p'], bb_std=params['bb_s'])
        
        # Run Engine
        engine = BacktestEngine(initial_capital=initial_capital)
        results = engine.run(strategy, multiplier=multiplier, leverage=leverage, fixed_size=fixed_size, pyramiding_limit=pyramiding_limit)
        
        # Calculate Extra Stats for Journal/Dash
        trades = results['trades']
        win_rate = 0
        profit_factor = 0
        avg_trade = 0
        total_trades_count = 0
        
        if not trades.empty and 'Type' in trades.columns:
            # Filter solely for EXITS (Sells) which have the realized PnL in 'Value'
            exits = trades[trades['Type'] == 'SELL']
            total_trades_count = len(exits)

            if not exits.empty:
                wins = exits[exits['Value'] > 0]
                losses = exits[exits['Value'] <= 0]
                
                win_rate = (len(wins) / len(exits)) * 100
                total_profit = wins['Value'].sum()
                total_loss = abs(losses['Value'].sum())
                
                profit_factor = total_profit / total_loss if total_loss > 0 else 99.99
                avg_trade = exits['Value'].mean()
        
        results['metrics'] = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'total_trades': total_trades_count
        }
        
        return results, None
        
    except Exception as e:
        return None, str(e)


# --- VIEW: DASHBOARD ---
if nav == "Dashboard":
    st.markdown(f"# 📊 Command Center: {ticker}")
    
    # If no results, show "Ready" state
    if st.session_state.bt_results is None:
        st.info("👋 Welcome to ProTrader AI. The system is ready. Go to the 'Strategy Lab' to generate your first backtest.")
        
        # Placeholder / Market Check
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Market Snapshot (Live)")
            with st.spinner("Fetching live market data..."):
                try:
                    now = datetime.date.today()
                    start_snap = now - datetime.timedelta(days=100)
                    df_snap = DataLoader.get_data(ticker, str(start_snap), str(now), interval='1d')
                    if not df_snap.empty:
                        fig = go.Figure(data=[go.Candlestick(x=df_snap.index, open=df_snap['Open'], high=df_snap['High'], low=df_snap['Low'], close=df_snap['Close'])])
                        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        last_close = df_snap['Close'].iloc[-1]
                        prev_close = df_snap['Close'].iloc[-2]
                        chg_pct = ((last_close - prev_close)/prev_close)*100
                        col_text = "#66fcf1" if chg_pct >= 0 else "#ff6b6b"
                        
                        st.markdown(f"""
                        <div style="font-size: 3rem; font-weight: bold; color: {col_text}">
                            ${last_close:.2f} <span style="font-size: 1.5rem">({chg_pct:+.2f}%)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("No live data available for this ticker.")
                except Exception as e:
                    st.warning(f"Could not fetch live data: {e}")
        
        with col2:
            st.markdown("### Quick Actions")
            st.markdown("Jump straight into analysis:")
            if st.button("🚀 Launch Strategy Lab"):
                st.write("Please select 'Strategy Lab' from the sidebar!") 

    else:
        # DISPLAY RESULTS DASHBOARD
        res = st.session_state.bt_results
        metrics = res['metrics']
        
        # 1. TOP ROW CARDS
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(get_kpi_card("Net Profit", f"{res['final_capital'] - 10000:.2f}", (res['final_capital'] >= 10000), "$"), unsafe_allow_html=True)
        with c2: st.markdown(get_kpi_card("Win Rate", f"{metrics['win_rate']:.1f}", (metrics['win_rate'] >= 50), suffix="%"), unsafe_allow_html=True)
        with c3: st.markdown(get_kpi_card("Profit Factor", f"{metrics['profit_factor']:.2f}", (metrics['profit_factor'] >= 1.5)), unsafe_allow_html=True)
        with c4: st.markdown(get_kpi_card("Sharpe Ratio", f"{res['sharpe_ratio']:.2f}", (res['sharpe_ratio'] > 1.0)), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. VISUALS
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            st.markdown("### 📈 Equity Growth")
            fig_eq = px.area(res['equity_curve'], x=res['equity_curve'].index, y="Equity", template="plotly_dark")
            fig_eq.update_traces(line_color='#66fcf1', fillcolor='rgba(102, 252, 241, 0.1)')
            fig_eq.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_eq, use_container_width=True)
        
        with col_side:
            st.markdown("### 🎯 Trade Distribution")
            # Create a Pie generic
            trades = res['trades']
            exits = trades[trades['Type'] == 'SELL']
            if not exits.empty:
                wins = len(exits[exits['Value'] > 0])
                losses = len(exits[exits['Value'] <= 0])
                fig_pie = px.pie(values=[wins, losses], names=['Wins', 'Losses'], color_discrete_sequence=['#66fcf1', '#ff6b6b'], hole=0.6)
                fig_pie.update_layout(template="plotly_dark", height=350, showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)

# --- VIEW: STRATEGY LAB ---
elif nav == "Strategy Lab":
    st.markdown("# 🧪 Strategy Lab")
    
    col_input, col_run = st.columns([1, 4])
    
    with col_input:
        with st.form("backtest_config"):
            st.markdown("### Configuration")
            # Dates
            start_date = st.date_input("Start", value=datetime.date.today() - datetime.timedelta(days=365))
            end_date = st.date_input("End", value=datetime.date.today())
            interval = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m"])
            
            # Capital & Risk
            st.markdown("### Risk Management")
            initial_cap = st.number_input("Capital", value=10000)
            
            # Asset Defaults
            def_mult = 1.0
            def_lev = 1.0
            if "Gold" in asset_class: 
                def_mult = 5.0
                def_lev = 10.0
            elif "Nasdaq" in asset_class:
                def_mult = 10.0
                def_lev = 10.0
                
            multiplier = st.number_input("Multiplier", value=def_mult, help="Value of 1 point movement per contract.")
            leverage = st.number_input("Leverage", 1.0, 100.0, def_lev, help="10x means 10% margin required.")
            
            mode = st.selectbox("Sizing", ["Fixed Contracts", "Use Full Margin"])
            fixed_size = 0
            
            if mode == "Fixed Contracts":
                fixed_size = st.number_input("Contracts", 1, 100, 1)
            else:
                st.caption("Strategy will use 100% of capital * leverage for each trade.")
                
            pyramiding = st.number_input("Max Pyramiding", 1, 5, 1)
            
            # Strategy
            st.markdown("### Strategy Logic")
            strat = st.selectbox("Model", ["SMA Crossover", "RSI + Bollinger"])
            params = {}
            if strat == "SMA Crossover":
                params['short'] = st.number_input("Short Window", 5, 50, 20)
                params['long'] = st.number_input("Long Window", 20, 200, 50)
            else:
                params['rsi_p'] = st.number_input("RSI Per", 5, 30, 14)
                params['rsi_l'] = st.number_input("RSI Low", 10, 40, 30)
                params['rsi_u'] = st.number_input("RSI High", 60, 90, 70)
                params['bb_p'] = st.number_input("BB Per", 10, 50, 20)
                params['bb_s'] = st.number_input("BB Std", 1.0, 3.0, 2.0)
                
            run_btn = st.form_submit_button("RUN BACKTEST")
            
    with col_run:
        if run_btn:
            with st.spinner(f"Simulating {strat} on {ticker}..."):
                res, err = run_backtest_logic(start_date, end_date, interval, initial_cap, strat, params, multiplier, leverage, fixed_size, pyramiding)
                
                if res:
                    st.session_state.bt_results = res
                    st.success("Backtest Complete! Check the Dashboard for insights.")
                    
                    # Immediate Feedback
                    m = res['metrics']
                    row1 = st.columns(4)
                    row1[0].metric("PnL", f"${res['final_capital'] - initial_cap:.2f}")
                    row1[1].metric("Win Rate", f"{m['win_rate']:.1f}%")
                    row1[2].metric("Sharpe", f"{res['sharpe_ratio']:.2f}")
                    row1[3].metric("Trades", m['total_trades'])
                    
                    st.markdown("### 🔍 Price Action Analysis")
                    # Visualizing the trade entries
                    trades = res['trades']
                    fig = go.Figure()
                    
                    st.line_chart(res['equity_curve']['Equity'])

                else:
                    st.error(err)
        else:
            if st.session_state.bt_results:
                 st.info("Last Run Results Available")
                 res = st.session_state.bt_results
                 st.dataframe(res['trades'].tail())
            else:
                 st.markdown("""
                 <div style="display: flex; justify-content: center; align-items: center; height: 400px; color: #45a29e; border: 2px dashed #333; border-radius: 20px;">
                    <h3>Ready to Initialize Simulation</h3>
                 </div>
                 """, unsafe_allow_html=True)

# --- VIEW: JOURNAL ---
elif nav == "Trade Journal":
    st.markdown("# 📓 Trade Journal")
    
    if st.session_state.bt_results:
        trades = st.session_state.bt_results['trades']
        exits = trades[trades['Type'] == 'SELL'].copy()
        
        if not exits.empty:
            # Stats Header
            col1, col2, col3 = st.columns(3)
            col1.markdown(get_kpi_card("Total Trades", len(exits), True), unsafe_allow_html=True)
            col2.markdown(get_kpi_card("Avg Win", f"${exits[exits['Value'] > 0]['Value'].mean():.2f}", True), unsafe_allow_html=True)
            col3.markdown(get_kpi_card("Avg Loss", f"${exits[exits['Value'] <= 0]['Value'].mean():.2f}", False), unsafe_allow_html=True)
            
            st.markdown("### Transaction Ledger")
            
            # Format nicely
            exits['Date'] = exits['Date'].dt.strftime('%Y-%m-%d %H:%M')
            exits = exits.rename(columns={'Value': 'PnL (Realized)', 'Size': 'Contracts'})
            
            def color_pnl(val):
                color = '#66fcf1' if val > 0 else '#ff6b6b'
                return f'color: {color}'

            st.dataframe(
                exits[['Date', 'Price', 'Contracts', 'PnL (Realized)']].style.map(color_pnl, subset=['PnL (Realized)']),
                use_container_width=True,
                height=600
            )
        else:
            st.info("No closed trades to display.")
    else:
        st.warning("No backtest data available. Run a simulation in the Strategy Lab first.")
