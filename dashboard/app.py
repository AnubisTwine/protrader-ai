import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
TIMEFRAMES = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

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
        
        # Add history for visualization
        results['history'] = data
        
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
        st.markdown("### Configuration")
        
        # --- Date Configuration ---
        if 'bt_start' not in st.session_state:
            st.session_state.bt_start = datetime.date.today() - datetime.timedelta(days=365)
        if 'bt_end' not in st.session_state:
            st.session_state.bt_end = datetime.date.today()
        
        # Store multi-timeframe results
        if 'multi_tf_results' not in st.session_state:
            st.session_state.multi_tf_results = {}

        def apply_date_preset():
            p = st.session_state.period_preset
            today = datetime.date.today()
            if p == "1 Month": 
                st.session_state.bt_start = today - datetime.timedelta(days=30)
            elif p == "3 Months":
                st.session_state.bt_start = today - datetime.timedelta(days=90)
            elif p == "6 Months": 
                st.session_state.bt_start = today - datetime.timedelta(days=180)
            elif p == "12 Months": 
                st.session_state.bt_start = today - datetime.timedelta(days=365)
            st.session_state.bt_end = today

        def set_custom_date():
            st.session_state.period_preset = "Custom"

        st.selectbox(
            "Quick Range", 
            ["Custom", "1 Month", "3 Months", "6 Months", "12 Months"], 
            index=4, 
            key="period_preset", 
            on_change=apply_date_preset
        )
        
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            start_date = st.date_input("Start", key="bt_start", format="DD/MM/YYYY", on_change=set_custom_date)
        with c_d2:
            end_date = st.date_input("End", key="bt_end", format="DD/MM/YYYY", on_change=set_custom_date)

        with st.form("backtest_config"):
            interval = st.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("1d") if "1d" in TIMEFRAMES else 0)
            
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
            # 1. Determine relevant timeframes (Selected + All Below)
            try:
                main_tf_index = TIMEFRAMES.index(interval)
                # Get all timeframes up to and including the selected one, reversed so main is first? 
                # Or user wants "below". Below usually means "lower timeframe" (shorter duration).
                # Our list is sorted smallest to largest (mostly).
                # "1m", "2m", ... "1d", ...
                # If "1d" is selected, we want "1d" and everything before it.
                # However, restrict "1m" and "2m" if range is too large to avoid freezing/errors?
                # For now, let's just take them all.
                
                # We want the main one first for immediate display.
                # Then the rest in descending order (largest to smallest) or just list them.
                
                # Let's get the list of timeframes to test:
                test_timeframes = [interval] + [tf for tf in TIMEFRAMES[:main_tf_index] if tf != interval]
                # Reverse the lower ones so we go from close-to-main down to 1m
                test_timeframes = [interval] + sorted(TIMEFRAMES[:main_tf_index], key=lambda x: TIMEFRAMES.index(x), reverse=True)
                
            except ValueError:
                test_timeframes = [interval]

            st.session_state.multi_tf_results = {} # Reset
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, tf in enumerate(test_timeframes):
                status_text.text(f"Simulating {strat} on {tf} ({i+1}/{len(test_timeframes)})...")
                
                # Adjust start date for lower timeframes if needed to prevent yfinance errors?
                # yfinance handles it gracefully usually, or returns empty.
                # But to save time, maybe cap history for < 15m?
                # User simply asked to test. Let's let logic handle it.
                
                res, err = run_backtest_logic(start_date, end_date, tf, initial_cap, strat, params, multiplier, leverage, fixed_size, pyramiding)
                
                if res:
                    st.session_state.multi_tf_results[tf] = res
                else:
                    # If it fails (e.g. no data for 1h in 1990), we just skip
                    pass
                
                progress_bar.progress((i + 1) / len(test_timeframes))
                
            status_text.empty()
            progress_bar.empty()
            
            # Set the main one as the current active result
            if interval in st.session_state.multi_tf_results:
                st.session_state.bt_results = st.session_state.multi_tf_results[interval]
                st.success(f"Multi-Timeframe Analysis Complete! Main View: {interval}")
            elif len(st.session_state.multi_tf_results) > 0:
                 # If main failed but others worked
                 first_key = list(st.session_state.multi_tf_results.keys())[0]
                 st.session_state.bt_results = st.session_state.multi_tf_results[first_key]
                 st.warning(f"Main timeframe {interval} had no data. Showing {first_key} instead.")
            else:
                 st.session_state.bt_results = None
                 st.error("No data found for any timeframe.")

        # --- DISPLAY RESULTS / MULTI-TF SELECTOR ---
        
        if st.session_state.bt_results:
            
            # --- MULTI-TIMEFRAME SELECTOR ---
            if len(st.session_state.multi_tf_results) > 1:
                st.markdown("### 🕰️ Multi-Timeframe Analysis")
                
                # Create a summary DataFrame
                summary_data = []
                for tf, res in st.session_state.multi_tf_results.items():
                    m = res['metrics']
                    summary_data.append({
                        "Timeframe": tf,
                        "PnL": res['final_capital'] - initial_cap,
                        "Win Rate %": m['win_rate'],
                        "Profit Factor": m['profit_factor'],
                        "Trades": m['total_trades']
                    })
                
                df_summary = pd.DataFrame(summary_data)
                
                # Display Summary Table with formatting
                st.dataframe(
                    df_summary.style.format({
                        "PnL": "${:.2f}", 
                        "Win Rate %": "{:.1f}", 
                        "Profit Factor": "{:.2f}"
                    }), 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Selector to switch view
                selected_tf = st.selectbox(
                    "🔍 Select Timeframe for Deeper Insight:", 
                    list(st.session_state.multi_tf_results.keys()),
                    index=0
                )
                
                # Update the main session state result based on selection
                if selected_tf != st.session_state.bt_results.get('timeframe_id', ''): # Simple check logic
                     st.session_state.bt_results = st.session_state.multi_tf_results[selected_tf]
                     # Mark it so we know which one
                     st.session_state.bt_results['timeframe_id'] = selected_tf
                     st.rerun()

            # --- MAIN RESULTS VIEW (Existing Code) ---
            res = st.session_state.bt_results
            # Use stored initial capital if available, else fallback to widget value
            initial_cap_val = res.get('initial_capital', initial_cap)
            
            m = res['metrics']
            net_profit = res['final_capital'] - initial_cap_val
            data_hist = res.get('history', pd.DataFrame())
            
            # --- OVERVIEW METRICS ---
            st.markdown("### 📊 Performance Summary")
            
            # Additional Metrics
            max_dd = res.get('max_drawdown', 0.0)
            bh_return = 0.0
            if not data_hist.empty:
                bh_return = ((data_hist['Close'].iloc[-1] - data_hist['Close'].iloc[0]) / data_hist['Close'].iloc[0]) * 100
                
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Net Profit", f"${net_profit:,.2f}", f"{(net_profit/initial_cap_val)*100:.2f}%")
            col_m2.metric("Total Trades", m['total_trades'])
            col_m3.metric("Profit Factor", f"{m['profit_factor']:.2f}")
            col_m4.metric("Max Drawdown", f"{max_dd:.2f}%")
            col_m5.metric("Buy & Hold", f"{bh_return:.2f}%")

            st.markdown("---")
            
            # --- CHART VISUALIZATION ---
            st.markdown("### 📈 Trade Visualization")
            
            if not data_hist.empty:
                # Create Subplots: Row 1 = Candle, Row 2 = Equity
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3],
                                    subplot_titles=(f"Price Action & Trades ({ticker})", "Equity Curve"))
                
                # 1. Candlestick
                fig.add_trace(go.Candlestick(
                    x=data_hist.index,
                    open=data_hist['Open'],
                    high=data_hist['High'],
                    low=data_hist['Low'],
                    close=data_hist['Close'],
                    name="Price"
                ), row=1, col=1)
                
                # 2. Indicators (Optional - e.g. SMA)
                # We could pull indicators from data if we saved them... mostly they are in 'data' if strategy calculated them?
                # The 'data' returned by strategy.generate_signals() has indicators.
                # Let's check columns for common indicators like 'SMA_20', 'upper', etc.
                for col in data_hist.columns:
                    if col.startswith('SMA') or col in ['upper', 'lower', 'basis']:
                        fig.add_trace(go.Scatter(x=data_hist.index, y=data_hist[col], name=col,
                                                 line=dict(width=1), opacity=0.7), row=1, col=1)
                
                # 3. Trade Markers
                trades = res['trades']
                if not trades.empty:
                    # Buys
                    buys = trades[trades['Type'] == 'BUY']
                    if not buys.empty:
                        fig.add_trace(go.Scatter(
                            x=buys['Date'], y=buys['Price'],
                            mode='markers',
                            name='Buy',
                            marker=dict(symbol='triangle-up', size=12, color='#00cc96', line=dict(width=1, color='black'))
                        ), row=1, col=1)
                    
                    # Sells
                    sells = trades[trades['Type'] == 'SELL']
                    if not sells.empty:
                        fig.add_trace(go.Scatter(
                            x=sells['Date'], y=sells['Price'],
                            mode='markers',
                            name='Sell',
                            marker=dict(symbol='triangle-down', size=12, color='#ef553b', line=dict(width=1, color='black'))
                        ), row=1, col=1)

                # 4. Equity Curve
                eq_curve = res['equity_curve']
                fig.add_trace(go.Scatter(
                    x=eq_curve.index, y=eq_curve['Equity'],
                    name="Equity",
                    line=dict(color='#66fcf1', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(102, 252, 241, 0.1)'
                ), row=2, col=1)
                
                fig.update_layout(
                    template="plotly_dark",
                    height=700,
                    hovermode='x unified',
                    showlegend=True,
                    xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)

            # --- LIST OF TRADES ---
            st.markdown("### 📝 List of Trades")
            trades = res['trades']
            if not trades.empty:
                # Filter for Exits (Closed Trades)
                closed_trades = trades[trades['Type'] == 'SELL'].copy()
                
                if not closed_trades.empty:
                    # Calculate Stats for Table
                    # Reconstruct Entry Price for display: Entry = Exit - (PnL / (Size * Mult))
                    # Note: This implies Avg Entry Price if multiple entries.
                    # PnL = (Exit - Entry) * Size * Mult
                    # PnL / Size / Mult = Exit - Entry
                    # Entry = Exit - (PnL / (Size * Mult))
                    
                    # Ensure numeric
                    closed_trades['Value'] = pd.to_numeric(closed_trades['Value'])
                    closed_trades['Size'] = pd.to_numeric(closed_trades['Size'])
                    
                    # Avoid division by zero
                    def calc_entry(row):
                         s = row['Size'] if row['Size'] != 0 else 1
                         m = multiplier if multiplier != 0 else 1
                         return row['Price'] - (row['Value'] / (s * m))

                    closed_trades['Entry Price'] = closed_trades.apply(calc_entry, axis=1)
                    
                    # Format for Display
                    display_df = pd.DataFrame()
                    display_df['Exit Date'] = closed_trades['Date']
                    display_df['Type'] = "Long" # We only have Longs in this simple engine
                    display_df['Entry Price'] = closed_trades['Entry Price']
                    display_df['Exit Price'] = closed_trades['Price']
                    display_df['Contracts'] = closed_trades['Size']
                    display_df['PnL'] = closed_trades['Value']
                    display_df['PnL %'] = (display_df['PnL'] / (display_df['Entry Price'] * display_df['Contracts'] * multiplier)) * 100
                    
                    # Formatting
                    st.dataframe(
                        display_df.style.format({
                            "Entry Price": "{:.2f}",
                            "Exit Price": "{:.2f}",
                            "PnL": "${:.2f}",
                            "PnL %": "{:.2f}%",
                            "Contracts": "{:.0f}"
                        }).map(lambda x: f"color: {'#66fcf1' if x>0 else '#ff6b6b'}", subset=['PnL', 'PnL %']),
                        use_container_width=True
                    )
                else:
                    st.info("No trades closed yet.")
            else:
                 st.info("No trades generated.")
        else:
             if not run_btn: # Only show ready state if not just ran (which would settle into results or error)
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
