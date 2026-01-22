import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import json # Added for AI response parsing
import textwrap
import importlib
import datetime
import yfinance as yf
from textblob import TextBlob
import openai
from openai import OpenAI


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
import src.macd_strategy
import src.builder_strategy

importlib.reload(src.data_loader)
importlib.reload(src.simple_strategy)
importlib.reload(src.rsi_bollinger_strategy)
importlib.reload(src.engine)
importlib.reload(src.indicators)
importlib.reload(src.macd_strategy)
importlib.reload(src.builder_strategy)

from src.data_loader import DataLoader
from src.simple_strategy import SmaCrossStrategy
from src.rsi_bollinger_strategy import RsiBollingerStrategy
from src.macd_strategy import MacdStrategy
from src.builder_strategy import BuilderStrategy
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
    /* Card Style */
    .card-container {
        background-color: #1f2833;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #c5c6c7;
    }
    .card-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #66fcf1;
        margin-bottom: 15px;
        border-bottom: 1px solid #45a29e;
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'bt_results' not in st.session_state:
    st.session_state.bt_results = None
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = "GC=F"
if 'saved_strategies' not in st.session_state:
    st.session_state.saved_strategies = {}
if 'builder_entry_rules' not in st.session_state: st.session_state.builder_entry_rules = []
if 'builder_exit_rules' not in st.session_state: st.session_state.builder_exit_rules = []
if 'navigation' not in st.session_state: st.session_state.navigation = "AI Trading Suite"
if 'github_token' not in st.session_state: st.session_state.github_token = ""

# --- SIDEBAR NAV ---
st.sidebar.image("https://img.icons8.com/nolan/96/bullish.png", width=60)
st.sidebar.markdown("## ProTrader AI")
nav = st.sidebar.radio("Navigation", ["AI Trading Suite", "Strategy Lab", "Strategy Builder", "Trade Journal", "Settings"], key="navigation")

if not st.session_state.github_token:
    st.sidebar.warning("⚠️ No GitHub Token found. AI features will be limited. Go to Settings to configure.")

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
TIMEFRAMES = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"]

def get_kpi_card(title, value, is_positive=True, prefix="", suffix=""):
    color_class = "metric-positive" if is_positive else "metric-negative"
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value {color_class}">{prefix}{value}{suffix}</div>
    </div>
    """

def run_backtest_logic(start_date, end_date, interval, initial_capital, strategy_type, params, multiplier, leverage, fixed_size, pyramiding_limit, stop_loss_pct=0.0, take_profit_pct=0.0):
    try:
        # Load Data
        fetch_start = start_date
        # Auto-adjust logic for intraday
        if interval in ['5m', '15m', '30m', '60m', '90m', '1h']:
             if interval in ['60m', '90m', '1h']:
                 limit = 729
             else:
                 limit = 59
             
             min_date = datetime.date.today() - datetime.timedelta(days=limit)
             if fetch_start < min_date: fetch_start = min_date

        data = DataLoader.get_data(ticker, str(fetch_start), str(end_date), interval=interval)
        
        if data.empty:
            return None, "No Data Found"

        # Init Strategy
        if strategy_type == "SMA Crossover":
            strategy = SmaCrossStrategy(data, short_window=params['short'], long_window=params['long'])
        elif strategy_type == "RSI + Bollinger":
            strategy = RsiBollingerStrategy(data, rsi_period=params['rsi_p'], rsi_lower=params['rsi_l'], rsi_upper=params['rsi_u'], bb_period=params['bb_p'], bb_std=params['bb_s'])
        elif strategy_type == "MACD":
            strategy = MacdStrategy(data, fast_period=params['fast'], slow_period=params['slow'], signal_period=params['signal'])
        else:
            # Assume any other strategy type found here with entry_rules/exit_rules is a Custom/Builder one
            # matches "Active Builder State" or any Saved Strategy Name
            if 'entry_rules' in params and 'exit_rules' in params:
                 strategy = BuilderStrategy(data, entry_rules=params['entry_rules'], exit_rules=params['exit_rules'])
            else:
                 return None, f"Unknown Strategy: {strategy_type}"
        
        # Run Engine
        engine = BacktestEngine(initial_capital=initial_capital)
        results = engine.run(strategy, multiplier=multiplier, leverage=leverage, fixed_size=fixed_size, pyramiding_limit=pyramiding_limit, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
        
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


if nav == "AI Trading Suite":
    st.markdown(f"# 🤖 AI Trading Suite {ticker}")
    
    # AI Analysis Request State
    if 'ai_analysis_text' not in st.session_state:
        st.session_state.ai_analysis_text = ""
    if 'ai_analysis_json' not in st.session_state:
        st.session_state.ai_analysis_json = None
    if 'active_indicators' not in st.session_state:
        st.session_state.active_indicators = ["SMA (20)", "RSI (14)"] # Defaults

    # --- TOP CONTROLS ---
    # Custom stylized timeframe selector using tabs or columns to look like buttons
    # Since st.pills is new, we use radio with horizontal for stability or standard pills if version allows. 
    # For now, let's use a nice horizontal radio which is reliable.
    
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1, 1, 2], gap="medium")
    
    with c_ctrl1:
        st.write("⏱️ **Timeframe**")
        selected_tf = st.selectbox("Select Timeframe", ["5m", "15m", "30m", "1h", "4h", "1d", "1wk"], index=5, label_visibility="collapsed")

    with c_ctrl2:
        st.write("📊 **Indicators**")
        with st.popover("Add Indicators"):
            st.markdown("### Chart Overlays")
            ind_sma20 = st.checkbox("SMA (20)", value="SMA (20)" in st.session_state.active_indicators)
            ind_sma50 = st.checkbox("SMA (50)", value="SMA (50)" in st.session_state.active_indicators)
            ind_sma200 = st.checkbox("SMA (200)", value="SMA (200)" in st.session_state.active_indicators)
            ind_bb = st.checkbox("Bollinger Bands", value="Bollinger Bands" in st.session_state.active_indicators)
            
            st.markdown("### Oscillators")
            ind_rsi = st.checkbox("RSI (14)", value="RSI (14)" in st.session_state.active_indicators)
            ind_macd = st.checkbox("MACD", value="MACD" in st.session_state.active_indicators)
            
            # Update State
            current_inds = []
            if ind_sma20: current_inds.append("SMA (20)")
            if ind_sma50: current_inds.append("SMA (50)")
            if ind_sma200: current_inds.append("SMA (200)")
            if ind_bb: current_inds.append("Bollinger Bands")
            if ind_rsi: current_inds.append("RSI (14)")
            if ind_macd: current_inds.append("MACD")
            st.session_state.active_indicators = current_inds

    with c_ctrl3:
        st.write("🧠 **AI Analyst**")
        if st.button("Generate AI Market Report", type="primary", use_container_width=True):
             st.session_state.run_ai_analysis = True
        else:
             st.session_state.run_ai_analysis = False


    # 2. Fetch Data
    with st.spinner(f"Fetching {selected_tf} market data for {ticker}..."):
        try:
            now = datetime.datetime.now()
            
            # Determine start date based on timeframe to optimize fetch and meet yfinance limits
            if selected_tf == "5m":
                start_date = now - datetime.timedelta(days=5) # Max ~60 days
                interval_yf = "5m"
            elif selected_tf == "15m":
                start_date = now - datetime.timedelta(days=10)
                interval_yf = "15m"
            elif selected_tf == "30m":
                start_date = now - datetime.timedelta(days=20)
                interval_yf = "30m"
            elif selected_tf == "1h":
                start_date = now - datetime.timedelta(days=45) # Max 730 days
                interval_yf = "1h"
            elif selected_tf == "4h":
                start_date = now - datetime.timedelta(days=90)
                interval_yf = "1h" # Use 1h data
            elif selected_tf == "1wk":
                start_date = now - datetime.timedelta(days=730)
                interval_yf = "1wk"
            else: # 1d
                start_date = now - datetime.timedelta(days=365)
                interval_yf = "1d"
            
            # Use DataLoader
            df = DataLoader.get_data(ticker, str(start_date.date()), str(now.date() + datetime.timedelta(days=1)), interval=interval_yf)
            
            if not df.empty:
                active_inds = st.session_state.active_indicators
                
                # Base Logic - CALCULATIONS
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                
                if "SMA (50)" in active_inds:
                    df['SMA_50'] = df['Close'].rolling(window=50).mean()
                
                if "SMA (200)" in active_inds:
                    df['SMA_200'] = df['Close'].rolling(window=200).mean()
                
                if "Bollinger Bands" in active_inds:
                    df['BB_Upper'] = df['Close'].rolling(window=20).mean() + (df['Close'].rolling(window=20).std() * 2)
                    df['BB_Lower'] = df['Close'].rolling(window=20).mean() - (df['Close'].rolling(window=20).std() * 2)
                
                if "RSI (14)" in active_inds:
                    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).mean() / df['Close'].diff().apply(lambda x: abs(x) if x < 0 else 0).rolling(14).mean()))
                
                if "MACD" in active_inds:
                    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = exp1 - exp2
                    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
                
                # --- CARD 1: CHART ---
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-header">📉 Market Overview: {ticker} ({selected_tf})</div>', unsafe_allow_html=True)
                
                # Setup Subplots (Price on Top, RSI/MACD on Bottom)
                # Determine chart structure based on indicators
                row_specs = [0.7] # Main chart
                subplot_titles = ["Price Action"]
                
                has_rsi = "RSI (14)" in active_inds
                has_macd = "MACD" in active_inds
                
                rows = 1
                if has_rsi: 
                    rows += 1
                    row_specs.append(0.2)
                    subplot_titles.append("RSI")
                if has_macd:
                    rows += 1
                    row_specs.append(0.2)
                    subplot_titles.append("MACD")
                
                # Normalize row heights logic is a bit manual in plotly make_subplots
                # Simplifying: Fixed 2 rows if any oscillator, else 1
                # Actually, make_subplots supports auto sizing but let's be explicit
                
                if rows == 1:
                    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
                elif rows == 2:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                else: # 3 rows (Price, RSI, MACD)
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05)


                # 1. Candlestick (Row 1)
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"), row=1, col=1)
                
                # 2. Overlays (Row 1)
                if "SMA (20)" in active_inds:
                     fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA (20)', line=dict(color='orange', width=1)), row=1, col=1)
                if "SMA (50)" in active_inds:
                     fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA (50)', line=dict(color='blue', width=1)), row=1, col=1)
                if "SMA (200)" in active_inds:
                     fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA (200)', line=dict(color='white', width=2)), row=1, col=1)
                if "Bollinger Bands" in active_inds:
                     fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
                     fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dot'), fill='tonexty'), row=1, col=1)

                # 3. Oscillators
                current_row = 2
                if has_rsi:
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI (14)', line=dict(color='#66fcf1', width=1)), row=current_row, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
                    current_row += 1
                
                if has_macd:
                    fig.add_trace(go.Bar(x=df.index, y=df['MACD']-df['Signal_Line'], name='MACD Hist'), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD Line', line=dict(color='#66fcf1')), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='orange')), row=current_row, col=1)

                # Check if intraday to fix gaps
                if selected_tf in ["5m", "15m", "30m", "1h", "4h"]:
                    for r in range(1, rows + 1):
                        fig.update_xaxes(
                            type='category', 
                            nticks=10,
                            tickmode='auto',
                            showticklabels=False, # Removed dates completely
                            row=r, col=1
                        )
                else:
                     for r in range(1, rows + 1):
                        fig.update_xaxes(showticklabels=False, row=r, col=1) # Removed dates completely
                
                # Initialize AI Annotations if available
                if 'ai_analysis_json' in st.session_state and st.session_state.ai_analysis_json:
                     try:
                         data = st.session_state.ai_analysis_json
                         # Plot Support Levels
                         for level in data.get('support_levels', []):
                             fig.add_hline(y=level, line_dash="dot", line_color="#00cc96", annotation_text=f"Sup {level}", annotation_position="bottom right", row=1, col=1)
                         # Plot Resistance Levels
                         for level in data.get('resistance_levels', []):
                             fig.add_hline(y=level, line_dash="dot", line_color="#ef553b", annotation_text=f"Res {level}", annotation_position="top right", row=1, col=1)
                     except:
                         pass

                fig.update_layout(
                    template="plotly_dark", 
                    height=500 + (rows - 1) * 150, 
                    margin=dict(l=0,r=0,t=0,b=0), 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True) # End Card



                # --- CALCULATE SENTIMENT VARIABLES ---
                last_close = df['Close'].iloc[-1]
                last_sma = df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else last_close
                
                if has_rsi:
                    last_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
                else:
                    last_rsi = 50
                
                sentiment_score = 0
                if last_close > last_sma: sentiment_score += 0.5
                else: sentiment_score -= 0.5
                
                if len(df) > 5:
                    change_last_5 = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
                    if change_last_5 > 0: sentiment_score += 0.5
                    else: sentiment_score -= 0.5
                
                if sentiment_score >= 0.5:
                    sent_text = "BULLISH 🐂"
                    sent_color = "#00cc96"
                    bar_val = 0.85
                elif sentiment_score <= -0.5:
                    sent_text = "BEARISH 🐻"
                    sent_color = "#ef553b"
                    bar_val = 0.15
                else:
                    sent_text = "NEUTRAL ⚖️"
                    sent_color = "#b8b8b8"
                    bar_val = 0.5

                # --- CARD 2: SENTIMENT ---
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.markdown('<div class="card-header">🌡️ Technical Sentiment</div>', unsafe_allow_html=True)
                
                c_s1, c_s2 = st.columns([1, 3])
                with c_s1:
                     st.markdown(f"<div style='text-align:center; font-size: 2rem; font-weight:bold; color:{sent_color}; margin-top: 10px;'>{sent_text}</div>", unsafe_allow_html=True)
                with c_s2:
                     st.progress(bar_val)
                     st.caption(f"Based on Trend (SMA20) & Momentum (RSI: {last_rsi:.1f})")
                
                st.markdown('</div>', unsafe_allow_html=True) # End Card

                # --- CARD 3: AI ANALYSIS (GitHub Models) ---
                if st.session_state.run_ai_analysis:
                     with st.spinner("🤖 Communicating with GitHub Models (Copilot)..."):
                         if not st.session_state.github_token:
                             st.error("Please configure your GitHub Token in 'Settings' first.")
                         else:
                             try:
                                 # Configure OpenAI Client for GitHub Models
                                 # Endpoint: https://models.inference.ai.azure.com
                                 client = OpenAI(
                                     base_url="https://models.inference.ai.azure.com",
                                     api_key=st.session_state.github_token,
                                 )
                                 
                                 # Prepare Context
                                 recent_data = df.tail(15).to_string() # Increased context
                                 
                                 # Fetch Concurrent Data for Multi-Timeframe Prompt
                                 mtf_context = ""
                                 for m_tf in ["15m", "1h", "1d"]:
                                     try:
                                         if m_tf == "15m": dx_start = now - datetime.timedelta(days=5)
                                         elif m_tf == "1h": dx_start = now - datetime.timedelta(days=20)
                                         else: dx_start = now - datetime.timedelta(days=365)
                                         
                                         dx = DataLoader.get_data(ticker, str(dx_start.date()), str(now.date() + datetime.timedelta(days=1)), interval=m_tf)
                                         if not dx.empty:
                                            # Simple technicals for the prompt
                                            curr = dx['Close'].iloc[-1]
                                            sma_ref = dx['Close'].rolling(20).mean().iloc[-1] if len(dx) > 20 else curr
                                            mtf_context += f"\nTimeframe {m_tf}: Price={curr:.2f}, SMA20={sma_ref:.2f}, Last5Candles={dx.tail(5)['Close'].tolist()}"
                                     except:
                                         pass

                                 # Construct Header with Indicators
                                 ind_summary = f"Indicators Active: {', '.join(active_inds)}\n"
                                 if "SMA (50)" in active_inds: ind_summary += f"SMA(50): {df['SMA_50'].iloc[-1]:.2f}\n"
                                 if "SMA (200)" in active_inds: ind_summary += f"SMA(200): {df['SMA_200'].iloc[-1]:.2f}\n"
                                 if "RSI (14)" in active_inds: ind_summary += f"RSI(14): {df['RSI'].iloc[-1]:.2f}\n"
                                 if "MACD" in active_inds: ind_summary += f"MACD: {df['MACD'].iloc[-1]:.2f}\n"

                                 prompt_content = f"""
                                 You are an expert financial trading analyst. Analyze the following market data for {ticker} across multiple timeframes.
                                 
                                 A. SELECTED TIMEFRAME ({selected_tf}) CONTEXT:
                                 - Current Price: {last_close:.2f}
                                 - SMA(20): {last_sma:.2f}
                                 - Technical Sentiment: {sent_text}
                                 {ind_summary}
                                 
                                 Recent OHLCV Data ({selected_tf}):
                                 {recent_data}
                                 
                                 B. CROSS-TIMEFRAME CONTEXT:
                                 {mtf_context}
                                 
                                 Task:
                                 Analyze the market and provide a detailed JSON response that covers specific analysis for 15m, 1h, and 1d timeframes.
                                 
                                 Required JSON Structure:
                                 {{
                                    "analysis_summary": "Concise market analysis (max 150 words). Use markdown.",
                                    "trend": "Bullish/Bearish/Neutral",
                                    "support_levels": [price_float_1, price_float_2],
                                    "resistance_levels": [price_float_1, price_float_2],
                                    "key_signal": "BUY / SELL / WAIT",
                                    "timeframes": {{
                                        "15m": {{
                                            "trend": "BULLISH/BEARISH/NEUTRAL",
                                            "analysis": "Short sentence analysis.",
                                            "support": [level1],
                                            "resistance": [level1],
                                            "rating": 1-5 (int)
                                        }},
                                        "1h": {{
                                            "trend": "BULLISH/BEARISH/NEUTRAL",
                                            "analysis": "Short sentence analysis.",
                                            "support": [level1],
                                            "resistance": [level1],
                                            "rating": 1-5 (int)
                                        }},
                                        "1d": {{
                                            "trend": "BULLISH/BEARISH/NEUTRAL",
                                            "analysis": "Short sentence analysis.",
                                            "support": [level1],
                                            "resistance": [level1],
                                            "rating": 1-5 (int)
                                        }}
                                    }}
                                 }}
                                 
                                 Ensure the price levels are derived from the recent data provided.
                                 """
                                 
                                 response = client.chat.completions.create(
                                     messages=[
                                         {"role": "system", "content": "You are a helpful and professional financial analyst that outputs strict JSON."},
                                         {"role": "user", "content": prompt_content}
                                     ],
                                     model="gpt-4o", # Using GPT-4o via GitHub Models
                                     temperature=0.5,
                                     response_format={ "type": "json_object" } # Request JSON mode if supported, otherwise prompt handles it
                                 )
                                 
                                 content = response.choices[0].message.content
                                 
                                 # Parse JSON
                                 try:
                                     data = json.loads(content)
                                     st.session_state.ai_analysis_json = data
                                     st.session_state.ai_analysis_text = data.get("analysis_summary", "No analysis provided.")
                                 except Exception as e:
                                     # Fallback if JSON fails
                                     st.session_state.ai_analysis_text = content
                                     st.session_state.ai_analysis_json = None
                                     st.warning(f"Failed to parse AI JSON: {e}")

                                 # Force Rerun to update chart annotations
                                 st.rerun()
                                 
                             except Exception as e:
                                 st.error(f"GitHub Inference API Error: {e}")
                
                 # Display Analysis (Analysis Text)
                if st.session_state.ai_analysis_text:
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    st.markdown('<div class="card-header">🧠 GitHub Copilot AI Analysis</div>', unsafe_allow_html=True)
                    
                    # Display Signal Badge
                    if st.session_state.ai_analysis_json:
                        sig = st.session_state.ai_analysis_json.get("key_signal", "WAIT")
                        color = "#b8b8b8"
                        if "BUY" in sig.upper(): color = "#00cc96"
                        elif "SELL" in sig.upper(): color = "#ef553b"
                        st.markdown(f"<span style='background-color:{color}; padding:5px 10px; border-radius:5px; font-weight:bold; color:black;'>SIGNAL: {sig}</span>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown(st.session_state.ai_analysis_text)
                    
                    # Display Levels Text
                    if st.session_state.ai_analysis_json:
                        s_levels = st.session_state.ai_analysis_json.get("support_levels", [])
                        r_levels = st.session_state.ai_analysis_json.get("resistance_levels", [])
                        st.caption(f"**Supports:** {s_levels} | **Resistances:** {r_levels}")

                    st.markdown('</div>', unsafe_allow_html=True)
                
                # --- CARD: MULTI-TIMEFRAME ANALYSIS ---
                if 'ai_analysis_json' in st.session_state and st.session_state.ai_analysis_json:
                    ai_data = st.session_state.ai_analysis_json
                    ai_mtf_data = ai_data.get("timeframes", {})
                    
                    if ai_mtf_data:
                        st.markdown('<div class="card-container">', unsafe_allow_html=True)
                        st.markdown('<div class="card-header">🕒 Multi-Timeframe Matrix</div>', unsafe_allow_html=True)
                        
                        mtf_cols = st.columns(3)
                        mtf_tfs = ["15m", "1h", "1d"]
                        
                        for i, tf in enumerate(mtf_tfs):
                            if tf in ai_mtf_data:
                                d = ai_mtf_data[tf]
                                trend = d.get("trend", "NEUTRAL")
                                rating = d.get("rating", 3)
                                analysis = d.get("analysis", "No Data")
                                sup = d.get("support", [])
                                res = d.get("resistance", [])
                                
                                # Styling based on trend
                                border_color = "#45a29e"
                                if "BULL" in trend.upper(): border_color = "#00cc96"
                                elif "BEAR" in trend.upper(): border_color = "#ef553b"
                                
                                
                                with mtf_cols[i]:
                                    html_content = f"""
                                    <div style="background-color: #1f2833; padding: 15px; border-radius: 10px; border: 1px solid {border_color}; height: 100%;">
                                        <div style="text-align: center; margin-bottom: 10px;">
                                            <span style="font-size: 0.8em; color: #888; font-weight: bold; letter-spacing: 2px;">INTERVAL</span><br>
                                            <span style="font-size: 1.5em; font-style: italic; font-weight: 900; color: #66fcf1;">{tf.upper()}</span>
                                        </div>
                                        
                                        <div style="margin-bottom: 15px;">
                                            <div style="font-size: 0.7em; color: #888; text-transform: uppercase; font-weight: bold;">Trend Vector</div>
                                            <div style="color: {border_color}; font-weight: bold;">{trend}</div>
                                            <div style="color: #ffd700;">{'⭐' * rating}</div>
                                        </div>
                                        
                                        <div style="margin-bottom: 15px; font-size: 0.9em; line-height: 1.4; color: #ccc;">
                                            {analysis}
                                        </div>
                                        
                                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.8em;">
                                            <div style="background: rgba(0, 204, 150, 0.1); padding: 5px; border-radius: 5px; text-align: center;">
                                                <div style="color: #00cc96; font-weight: bold;">DEMAND</div>
                                                <div>{', '.join(map(str, sup)) if sup else '-'}</div>
                                            </div>
                                            <div style="background: rgba(239, 85, 59, 0.1); padding: 5px; border-radius: 5px; text-align: center;">
                                                <div style="color: #ef553b; font-weight: bold;">SUPPLY</div>
                                                <div>{', '.join(map(str, res)) if res else '-'}</div>
                                            </div>
                                        </div>
                                    </div>
                                    """
                                    st.markdown(textwrap.dedent(html_content), unsafe_allow_html=True)
                                    
                        st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.warning(f"No data available for {ticker} on {selected_tf} timeframe.")

        except Exception as e:
            st.error(f"Error fetching chart data: {e}")

    # --- CARD 4: NEWS ---
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📰 Global News Intelligence</div>', unsafe_allow_html=True)
    
    with st.spinner("Analyzing global news wires..."):
        try:
            t_obj = yf.Ticker(ticker)
            news_list = t_obj.news
            
            if news_list:
                # Fetch more news (up to 10)
                for item in news_list[:10]: 
                    # Support new YF news structure (v0.2+)
                    if 'content' in item:
                        c = item['content']
                        title = c.get('title', 'No Title')
                        link = c.get('clickThroughUrl', {}).get('url', '#')
                        publisher = c.get('provider', {}).get('displayName', 'Unknown')
                    else:
                        title = item.get('title', 'No Title')
                        link = item.get('link', '#')
                        publisher = item.get('publisher', 'Unknown')
                    
                    blob = TextBlob(title)
                    polarity = blob.sentiment.polarity
                    
                    if polarity > 0.05:
                        impact = "POSITIVE"
                        imp_color = "#00cc96"
                    elif polarity < -0.05:
                        impact = "NEGATIVE"
                        imp_color = "#ef553b"
                    else:
                        impact = "NEUTRAL"
                        imp_color = "#b8b8b8"
                        
                    with st.container():
                        st.markdown(
                            f'''
                            <div style="background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid {imp_color}; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                                <div style="font-size: 1.1em; font-weight: bold;"><a href="{link}" target="_blank" style="text-decoration: none; color: white;">{title}</a></div>
                                <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.85em; color: #ccc;">
                                    <span>{publisher}</span>
                                    <span style="color: {imp_color}; font-weight: bold; background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;">{impact}</span>
                                </div>
                            </div>
                            ''', 
                            unsafe_allow_html=True
                        )
            else:
                st.info("No recent news found for this asset.")
                
        except Exception as e:
            st.warning(f"Could not load news: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True) # End Card

    st.markdown("---")


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

        # Strategy Selection (Outside Form for interactivity)
        st.markdown("### Strategy Logic")
        
        # Combine standard + saved custom strategies
        standard_strats = ["SMA Crossover", "RSI + Bollinger", "MACD"]
        custom_strats = list(st.session_state.saved_strategies.keys())
        
        # We handle "Builder" separately on its own page now, but we can allow "Unsaved Builder" or similar 
        # provided the session state is populated.
        # But user requested "Use them to backtest". 
        # Let's show: Standard..., Saved Strat 1, Saved Strat 2...
        
        all_models = standard_strats + ["Active Builder State"] + custom_strats
        
        strat = st.selectbox("Model", all_models, help="Standard: Built-in strategies.\nActive Builder State: Uses the current unsaved usage in the 'Strategy Builder' page.\nSaved: Your custom named strategies.")

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
            
            # --- NEW RISK INPUTS ---
            c_sl, c_tp = st.columns(2)
            with c_sl:
                stop_loss = st.number_input("Stop Loss %", 0.0, 20.0, 0.0, step=0.5, help="0.0 to disable") / 100
            with c_tp:
                take_profit = st.number_input("Take Profit %", 0.0, 50.0, 0.0, step=0.5, help="0.0 to disable") / 100

            mode = st.selectbox("Sizing", ["Fixed Contracts", "Use Full Margin"])
            fixed_size = 0
            
            if mode == "Fixed Contracts":
                fixed_size = st.number_input("Contracts", 1, 100, 1)
            else:
                st.caption("Strategy will use 100% of capital * leverage for each trade.")
            
            pyramiding = st.number_input("Max Pyramiding", 1, 10, 1, help="Max open trades allowed at once.")

            # Strategy Params
            params = {}
            if strat == "SMA Crossover":
                params['short'] = st.number_input("Short Window", 5, 50, 20)
                params['long'] = st.number_input("Long Window", 20, 200, 50)
            elif strat == "RSI + Bollinger":
                params['rsi_p'] = st.number_input("RSI Per", 5, 30, 14)
                params['rsi_l'] = st.number_input("RSI Low", 10, 40, 30)
                params['rsi_u'] = st.number_input("RSI High", 60, 90, 70)
                params['bb_p'] = st.number_input("BB Per", 10, 50, 20)
                params['bb_s'] = st.number_input("BB Std", 1.0, 3.0, 2.0)
            elif strat == "MACD":
                params['fast'] = st.number_input("Fast Period", 2, 20, 12)
                params['slow'] = st.number_input("Slow Period", 10, 50, 26)
                params['signal'] = st.number_input("Signal Period", 2, 20, 9)
            elif strat == "Active Builder State":
                st.info("Using rules currently defined in Strategy Builder.")
                params['entry_rules'] = st.session_state.builder_entry_rules
                params['exit_rules'] = st.session_state.builder_exit_rules
                st.caption(f"Entry Rules: {len(params['entry_rules'])}, Exit Rules: {len(params['exit_rules'])}")
            elif strat in st.session_state.saved_strategies:
                st.success(f"Loaded '{strat}'")
                rules = st.session_state.saved_strategies[strat]
                params['entry_rules'] = rules['entry']
                params['exit_rules'] = rules['exit']
                st.caption(f"Entry Rules: {len(params['entry_rules'])}, Exit Rules: {len(params['exit_rules'])}")
                
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
                
                res, err = run_backtest_logic(start_date, end_date, tf, initial_cap, strat, params, multiplier, leverage, fixed_size, pyramiding, stop_loss, take_profit)
                
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
                    width="stretch",
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
                
                # 2. Indicators
                # Automatically plot columns that look like overlays (SMA, EMA, BB)
                overlay_indicators = ['SMA', 'EMA', 'upper', 'lower', 'basis']
                for col in data_hist.columns:
                    if any(x in col for x in overlay_indicators):
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
                
                st.plotly_chart(fig, width="stretch")

                # --- NEW VISUALS: DRAWDOWN & HEATMAP ---
                col_viz1, col_viz2 = st.columns(2)
                
                # 1. Drawdown Chart (Underwater Plot)
                with col_viz1:
                    st.markdown("#### 🌊 Underwater Plot (Drawdown)")
                    equity_df = res['equity_curve']
                    fig_dd = px.area(equity_df, x=equity_df.index, y='Drawdown', 
                                    title='Drawdown %', color_discrete_sequence=['#ff6b6b'])
                    fig_dd.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=40,b=0),
                                        yaxis_tickformat='.1%')
                    st.plotly_chart(fig_dd, use_container_width=True)

                # 2. Monthly Returns Heatmap
                with col_viz2:
                    st.markdown("#### 📅 Monthly Performance")
                    try:
                        # Resample to Monthly Returns
                        # Need to handle potential empty index or non-datetime
                        # 'ME' is Month End in newer pandas, 'M' in older. 'M' is deprecated in newest. 
                        # Using 'M' for safety as it works in most recent versions with warning, but 'ME' fails in old.
                        monthly = equity_df['Equity'].resample('M').last().pct_change()
                        monthly_df = monthly.to_frame(name='Return')
                        monthly_df['Year'] = monthly_df.index.year
                        monthly_df['Month'] = monthly_df.index.month_name()
                        
                        # Pivot for Heatmap
                        heatmap_data = monthly_df.pivot(index='Year', columns='Month', values='Return')
                        
                        # Sort months correctly
                        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                                     'July', 'August', 'September', 'October', 'November', 'December']
                        # Ensure only months that exist in the data columns are in the order list to avoid error if columns missing?
                        # No, reindex works fine with missing cols (inserts NaNs)
                        heatmap_data = heatmap_data.reindex(columns=month_order)
                        
                        fig_hm = px.imshow(heatmap_data, 
                                          labels=dict(x="Month", y="Year", color="Return"),
                                          x=heatmap_data.columns,
                                          y=heatmap_data.index,
                                          color_continuous_scale="RdBu",
                                          text_auto='.1%')
                        fig_hm.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=40,b=0))
                        st.plotly_chart(fig_hm, use_container_width=True)
                    except Exception as e:
                        st.caption(f"Not enough data for heatmap or error: {e}")

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
                        width="stretch"
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

# --- SETTINGS VIEW ---
elif nav == "Settings":
    st.markdown("# ⚙️ Settings")
    st.markdown("### 🔑 API Keys")
    
    gh_token = st.text_input("GitHub Token (for AI Models)", type="password", value=st.session_state.github_token, help="Enter a GitHub Personal Access Token (PAT) to access GitHub Models.")
    if st.button("Save Token"):
        st.session_state.github_token = gh_token
        st.success("GitHub Token saved!")

# --- VIEW: STRATEGY BUILDER ---
elif nav == "Strategy Builder":
    st.markdown("# 🛠️ Custom Strategy Builder")
    
    st.markdown("Build your strategy by adding rules below. Once configured, **save** it to use in the Strategy Lab.")
    
    # --- SAVE STRATEGY SECTION ---
    with st.expander("💾 Save Strategy config", expanded=True):
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            strat_name_input = st.text_input("Strategy Name", placeholder="My New Strategy")
        with col_s2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Save Strategy", use_container_width=True):
                if strat_name_input:
                    # Save current entry/exit rules to session state
                    st.session_state.saved_strategies[strat_name_input] = {
                        'entry': st.session_state.builder_entry_rules.copy(),
                        'exit': st.session_state.builder_exit_rules.copy()
                    }
                    st.success(f"Strategy '{strat_name_input}' saved successfully!")
                else:
                    st.error("Please enter a strategy name.")

    # --- BUILDER UI ---
    b_col1, b_col2 = st.columns(2)
             
    # Helper to render rule creator
    def render_rule_creator(key_prefix, target_list_key):
         with st.container():
             # Standard Indicator List
             IND_LIST_1 = ["Close", "RSI", "SMA", "EMA", "MACD LINE", "MACD SIGNAL", "MACD HIST"]
             IND_LIST_2 = ["Value", "Close", "SMA", "EMA", "MACD LINE", "MACD SIGNAL", "MACD HIST"]
             
             c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
             with c1:
                 ind1 = st.selectbox("Indicator 1", IND_LIST_1, key=f"{key_prefix}_i1")
                 p1 = {}
                 if ind1 in ["RSI", "SMA", "EMA"]:
                    p1['period'] = st.number_input("Period", 2, 200, 14, key=f"{key_prefix}_p1", label_visibility="collapsed")
                 elif "MACD" in ind1: 
                    pass
             with c2:
                 op = st.selectbox("Op", [">", "<", "Crosses Above", "Crosses Below"], key=f"{key_prefix}_op")
             with c3:
                 ind2 = st.selectbox("Indicator 2 / Value", IND_LIST_2, key=f"{key_prefix}_i2")
                 p2 = {}
                 if ind2 == "Value":
                     p2['value'] = st.number_input("Value", 0.0, 50000.0, 50.0, key=f"{key_prefix}_v2", label_visibility="collapsed")
                 elif ind2 in ["SMA", "EMA"]:
                     p2['period'] = st.number_input("Period", 2, 200, 50, key=f"{key_prefix}_p2", label_visibility="collapsed")
             with c4:
                 if st.button("Add", key=f"{key_prefix}_add"):
                     rule = {
                         'ind1': ind1, 'params1': p1,
                         'op': op,
                         'ind2': ind2, 'params2': p2
                     }
                     st.session_state[target_list_key].append(rule)
                     st.rerun()

    # Entry Rules Section
    with b_col1:
         st.markdown("#### Entry Rules (Long)")
         # List existing
         for i, rule in enumerate(st.session_state.builder_entry_rules):
             p1_s = f"({rule['params1'].get('period','')})" if 'period' in rule['params1'] else ""
             p2_val = rule['params2'].get('value', '')
             p2_s = f"({rule['params2'].get('period','')})" if 'period' in rule['params2'] else f" {p2_val}"
             
             st.text(f"{i+1}. {rule['ind1']}{p1_s} {rule['op']} {rule['ind2']}{p2_s}")
             if st.button(f"🗑️", key=f"del_entry_{i}"):
                 st.session_state.builder_entry_rules.pop(i)
                 st.rerun()

         st.markdown("---")
         st.markdown("**Add New Entry Rule**")
         render_rule_creator("entry", "builder_entry_rules")

    # Exit Rules Section
    with b_col2:
         st.markdown("#### Exit Rules (Long)")
         for i, rule in enumerate(st.session_state.builder_exit_rules):
             p1_s = f"({rule['params1'].get('period','')})" if 'period' in rule['params1'] else ""
             p2_val = rule['params2'].get('value', '')
             p2_s = f"({rule['params2'].get('period','')})" if 'period' in rule['params2'] else f" {p2_val}"
             
             st.text(f"{i+1}. {rule['ind1']}{p1_s} {rule['op']} {rule['ind2']}{p2_s}")
             if st.button(f"🗑️", key=f"del_exit_{i}"):
                 st.session_state.builder_exit_rules.pop(i)
                 st.rerun()
                 
         st.markdown("---")
         st.markdown("**Add New Exit Rule**")
         render_rule_creator("exit", "builder_exit_rules")
    
    st.markdown("---")



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
                width="stretch",
                height=600
            )
        else:
            st.info("No closed trades to display.")
    else:
        st.warning("No backtest data available. Run a simulation in the Strategy Lab first.")
