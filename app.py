import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from analytics.cointegration import CointegrationAnalyzer


def valid_df(df, min_rows=2):
    return df is not None and not df.empty and len(df) >= min_rows


# Import modules
from config import *
from utils.logger import get_logger
from data.storage import storage
from data.ingestion import BinanceWSCollector
from data.resampler import DataResampler
from analytics.statistics import StatisticsCalculator
from analytics.cointegration import CointegrationAnalyzer
from analytics.microstructure import MicrostructureAnalyzer
from analytics.signals import SignalGenerator
from alerts.alert_engine import alert_engine, AlertRule

logger = get_logger()

# Page configuration
st.set_page_config(
    page_title="Quantitative Trading Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stApp {background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);}
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #00d4ff;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2329 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #00d4ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 10px 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #0080ff 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 212, 255, 0.4);
    }
    
    /* Success/Warning badges */
    .badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .badge-success {background: #10b981; color: white;}
    .badge-warning {background: #f59e0b; color: white;}
    .badge-danger {background: #ef4444; color: white;}
    .badge-info {background: #3b82f6; color: white;}
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2329;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0080ff 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'collector' not in st.session_state:
    st.session_state.collector = None
if 'symbols' not in st.session_state:
    st.session_state.symbols = DEFAULT_SYMBOLS
if 'collecting' not in st.session_state:
    st.session_state.collecting = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'selected_symbols_pair' not in st.session_state:
    st.session_state.selected_symbols_pair = DEFAULT_SYMBOLS[:2] if len(DEFAULT_SYMBOLS) >= 2 else []
if 'alert_rules' not in st.session_state:
    st.session_state.alert_rules = []

# Header
col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='background: linear-gradient(90deg, #00d4ff 0%, #0080ff 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 48px; font-weight: 800; margin: 0;'>
            ⚡ Quantitative Trading Analytics
        </h1>
        <p style='color: #64748b; font-size: 16px; margin: 10px 0;'>
            Real-Time Market Microstructure & Statistical Arbitrage Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar - Data Collection Controls
with st.sidebar:
    st.markdown("### 🎯 Data Collection")
    
    # Symbol input
    symbols_input = st.text_input(
        "Trading Pairs (comma-separated)",
        value=",".join(st.session_state.symbols),
        help="Enter symbol pairs like: btcusdt,ethusdt"
    )
    
    # Start/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True, type="primary"):
            if not st.session_state.collecting:
                symbols = [s.strip().lower() for s in symbols_input.split(',') if s.strip()]
                st.session_state.symbols = symbols
                
                collector = BinanceWSCollector(symbols)
                collector.start()
                st.session_state.collector = collector
                st.session_state.collecting = True
                st.success(f"✅ Collecting: {', '.join(symbols)}")
                logger.info(f"Started collection for: {symbols}")
                
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            if st.session_state.collecting and st.session_state.collector:
                st.session_state.collector.stop()
                st.session_state.collecting = False
                st.info("🛑 Collection stopped")
                logger.info("Collection stopped")
    
    # Status
    if st.session_state.collecting:
        st.markdown('<div class="badge badge-success">🟢 Live</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge badge-danger">🔴 Stopped</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data stats
    st.markdown("### 📊 Data Statistics")
    available_symbols = storage.get_all_symbols()
    
    if available_symbols:
        for symbol in available_symbols:
            count = storage.get_tick_count(symbol)
            st.markdown(f"""
            <div style='background: #1e2329; padding: 10px; border-radius: 8px; margin: 5px 0;'>
                <strong style='color: #00d4ff;'>{symbol.upper()}</strong><br/>
                <span style='color: #64748b;'>{count:,} ticks</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No data collected yet")
    
    st.markdown("---")
    
    # Analysis parameters
    st.markdown("### ⚙️ Analysis Settings")
    
    analysis_timeframe = st.selectbox(
        "Timeframe",
        options=list(TIMEFRAMES.keys()),
        index=1
    )
    
    rolling_window = st.slider(
        "Rolling Window",
        min_value=20,
        max_value=200,
        value=50,
        step=10
    )
    
    regression_method = st.selectbox(
        "Regression Method",
        options=["ols", "huber"],
        index=0
    )
    
    # Export button
    st.markdown("---")
    st.markdown("### 💾 Export Data")
    
    if available_symbols and st.button("📥 Export All to CSV", use_container_width=True):
        for symbol in available_symbols:
            filename = f"data_store/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            storage.export_to_csv(symbol, filename)
        st.success(f"Exported {len(available_symbols)} files")

# Main content area
# Get available symbols for analysis
available_symbols = storage.get_all_symbols()

if len(available_symbols) < 1:
    st.warning("""
    ### ⚠️ No Data Available
    
    Please start data collection from the sidebar to begin analysis.
    
    **Steps:**
    1. Enter trading symbols (e.g., btcusdt,ethusdt)
    2. Click "▶️ Start"
    3. Wait for data to accumulate (30+ seconds recommended)
    """)
    st.stop()

# Symbol pair selection for pairs trading analysis
st.markdown("### 🔀 Select Symbol Pair for Analysis")
col1, col2 = st.columns(2)

with col1:
    symbol1 = st.selectbox(
        "Primary Symbol",
        options=available_symbols,
        index=0 if available_symbols else None
    )
    
with col2:
    symbol2_options = [s for s in available_symbols if s != symbol1]
    symbol2 = st.selectbox(
        "Secondary Symbol",
        options=symbol2_options if symbol2_options else available_symbols,
        index=0 if symbol2_options else None
    )

# Auto-refresh
if st.session_state.collecting:
    if st.button("🔄 Refresh", use_container_width=False):
        st.rerun()
    
    # Show update time
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview & Prices",
    "🔄 Spread & Cointegration",
    "🔬 Microstructure",
    "🎯 Trading Signals",
    "🚨 Alerts & Monitoring",
    "📊 Advanced Analytics"
])

# TAB 1: Overview & Prices
with tab1:
    st.markdown("## 💹 Real-Time Price Action")
    
    # Fetch data
    df1 = storage.get_recent_ticks(symbol1, n=5000)
    df2 = storage.get_recent_ticks(symbol2, n=5000) if symbol2 else pd.DataFrame()
    
    if df1.empty:
        st.warning(f"⚠️ No data for {symbol1}. Please wait for data collection...")
    else:
        # Resample to OHLCV
        freq_str = {'1s': '1S', '1m': '1T', '5m': '5T'}[analysis_timeframe]
        ohlcv1 = DataResampler.ticks_to_ohlcv(df1, freq_str)
        ohlcv1 = DataResampler.add_technical_indicators(ohlcv1)
        
        if not df2.empty:
            ohlcv2 = DataResampler.ticks_to_ohlcv(df2, freq_str)
            ohlcv2 = DataResampler.add_technical_indicators(ohlcv2)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price1 = df1['price'].iloc[-1] if not df1.empty else 0
            price_change1 = ((current_price1 - df1['price'].iloc[0]) / df1['price'].iloc[0] * 100) if len(df1) > 1 else 0
            st.metric(
                f"{symbol1.upper()} Price",
                f"${current_price1:,.2f}",
                f"{price_change1:+.2f}%"
            )
            
        with col2:
            if not df2.empty:
                current_price2 = df2['price'].iloc[-1]
                price_change2 = ((current_price2 - df2['price'].iloc[0]) / df2['price'].iloc[0] * 100) if len(df2) > 1 else 0
                st.metric(
                    f"{symbol2.upper()} Price",
                    f"${current_price2:,.2f}",
                    f"{price_change2:+.2f}%"
                )
                
        with col3:
            vol1 = df1['size'].sum()
            st.metric("Total Volume", f"{vol1:,.2f}", f"{symbol1.upper()}")
            
        with col4:
            tick_count1 = len(df1)
            st.metric("Tick Count", f"{tick_count1:,}", "trades")
        
        # Candlestick chart with dual Y-axis
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{symbol1.upper()} & {symbol2.upper() if symbol2 else ""} Price Action', 
                          'Volume', 'RSI')
        )
        
        # Candlestick for symbol1
        if not ohlcv1.empty:
            fig.add_trace(
                go.Candlestick(
                    x=ohlcv1['timestamp'],
                    open=ohlcv1['open'],
                    high=ohlcv1['high'],
                    low=ohlcv1['low'],
                    close=ohlcv1['close'],
                    name=symbol1.upper(),
                    increasing_line_color='#00d4ff',
                    decreasing_line_color='#ff4444'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if 'sma_20' in ohlcv1.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ohlcv1['timestamp'],
                        y=ohlcv1['sma_20'],
                        name='SMA 20',
                        line=dict(color='#ffa500', width=1)
                    ),
                    row=1, col=1
                )
                
            # Volume bars
            colors = ['#00d4ff' if row['close'] >= row['open'] else '#ff4444' 
                     for _, row in ohlcv1.iterrows()]
            fig.add_trace(
                go.Bar(
                    x=ohlcv1['timestamp'],
                    y=ohlcv1['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # RSI
            if 'rsi' in ohlcv1.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ohlcv1['timestamp'],
                        y=ohlcv1['rsi'],
                        name='RSI',
                        line=dict(color='#00d4ff', width=2)
                    ),
                    row=3, col=1
                )
                # Overbought/Oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        
        # Add symbol2 on secondary y-axis
        ohlcv2 = None
        if ohlcv2 is not None and not ohlcv2.empty:
            fig.add_trace(
                go.Scatter(
                    x=ohlcv2['timestamp'],
                    y=ohlcv2['close'],
                    name=symbol2.upper(),
                    line=dict(color='#ff6b9d', width=2),
                    yaxis='y2'
                ),
                row=1, col=1
            )
        
        fig.update_layout(
            height=800,
            template='plotly_dark',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            yaxis2=dict(
                overlaying='y',
                side='right',
                showgrid=False
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table
        st.markdown("### 📊 Statistical Summary")
        
        stats_calc = StatisticsCalculator()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {symbol1.upper()} Statistics")
            if not ohlcv1.empty and len(ohlcv1) >= 2:
                basic_stats1 = stats_calc.basic_stats(ohlcv1, 'close')
                returns_stats1 = stats_calc.returns_analysis(ohlcv1, 'close')
                
                stats_df1 = pd.DataFrame({
                    'Metric': ['Mean Price', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis',
                              'Mean Return %', 'Volatility (Annual)', 'Sharpe Approx'],
                    'Value': [
                        f"${basic_stats1.get('mean', 0):.2f}",
                        f"${basic_stats1.get('std', 0):.2f}",
                        f"${basic_stats1.get('min', 0):.2f}",
                        f"${basic_stats1.get('max', 0):.2f}",
                        f"{basic_stats1.get('skew', 0):.4f}",
                        f"{basic_stats1.get('kurtosis', 0):.4f}",
                        f"{returns_stats1.get('mean_return', 0)*100:.4f}%",
                        f"{returns_stats1.get('volatility_annual', 0)*100:.2f}%",
                        f"{returns_stats1.get('sharpe_approx', 0):.4f}"
                    ]
                })
                st.dataframe(stats_df1, use_container_width=True, hide_index=True)
                
        with col2:
            if ohlcv2 is not None and not ohlcv2.empty and len(ohlcv2) >= 2:
                st.markdown(f"#### {symbol2.upper()} Statistics")
                basic_stats2 = stats_calc.basic_stats(ohlcv2, 'close')
                returns_stats2 = stats_calc.returns_analysis(ohlcv2, 'close')
                
                stats_df2 = pd.DataFrame({
                    'Metric': ['Mean Price', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis',
                              'Mean Return %', 'Volatility (Annual)', 'Sharpe Approx'],
                    'Value': [
                        f"${basic_stats2.get('mean', 0):.2f}",
                        f"${basic_stats2.get('std', 0):.2f}",
                        f"${basic_stats2.get('min', 0):.2f}",
                        f"${basic_stats2.get('max', 0):.2f}",
                        f"{basic_stats2.get('skew', 0):.4f}",
                        f"{basic_stats2.get('kurtosis', 0):.4f}",
                        f"{returns_stats2.get('mean_return', 0)*100:.4f}%",
                        f"{returns_stats2.get('volatility_annual', 0)*100:.2f}%",
                        f"{returns_stats2.get('sharpe_approx', 0):.4f}"
                    ]
                })
                st.dataframe(stats_df2, use_container_width=True, hide_index=True)

# TAB 2: Spread & Cointegration
with tab2:
    st.markdown("## 🔄 Pairs Trading & Cointegration Analysis")
    
    if df1.empty or df2.empty:
        st.warning("⚠️ Need data from both symbols for cointegration analysis")
    elif len(df1) < MIN_DATA_POINTS or len(df2) < MIN_DATA_POINTS:
        st.warning(f"⚠️ Need at least {MIN_DATA_POINTS} data points. Currently: {len(df1)}, {len(df2)}")
    else:
        coint_analyzer = CointegrationAnalyzer()
        
        # Calculate hedge ratio
        hedge_ratio = np.nan
        hr_stats = {}

        if valid_df(ohlcv1) and valid_df(ohlcv2):
            hedge_ratio, hr_stats = coint_analyzer.calculate_hedge_ratio(
                ohlcv1,
                ohlcv2,
                price_col='close',
                method=regression_method
            )
        else:
            st.info("Waiting for sufficient OHLC data for hedge ratio calculation")

        
        # Display hedge ratio metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
        with col2:
            st.metric("R²", f"{hr_stats.get('r_squared', 0):.4f}")
        with col3:
            st.metric("Data Points", f"{hr_stats.get('data_points', 0):,}")
        with col4:
            st.metric("Method", regression_method.upper())
        
        # Calculate spread
        spread_df = None  # always define first
        if valid_df(ohlcv1) and valid_df(ohlcv2) and not np.isnan(hedge_ratio):
            spread_df = coint_analyzer.calculate_spread(
                ohlcv1,
                ohlcv2,
                hedge_ratio,
                price_col='close'
            )
        else:
            st.info("Waiting for sufficient OHLC data to compute spread")

        if valid_df(spread_df):
            spread_df = coint_analyzer.calculate_zscore(spread_df)
        
        if valid_df(spread_df):
            # Calculate z-score
            spread_df = coint_analyzer.calculate_zscore(spread_df, rolling_window)
            
            # Current z-score
            current_zscore = spread_df['zscore'].iloc[-1] if 'zscore' in spread_df.columns else np.nan
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Z-Score", f"{current_zscore:.3f}")
            with col2:
                spread_mean = spread_df['spread'].mean() if 'spread' in spread_df.columns else 0
                st.metric("Spread Mean", f"{spread_mean:.4f}")
            with col3:
                spread_std = spread_df['spread'].std() if 'spread' in spread_df.columns else 0
                st.metric("Spread Std Dev", f"{spread_std:.4f}")
            
            # Spread and Z-score chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price Spread', 'Z-Score')
            )
            
            # Spread
            fig.add_trace(
                go.Scatter(
                    x=spread_df['timestamp'],
                    y=spread_df['spread'],
                    name='Spread',
                    line=dict(color='#00d4ff', width=2)
                ),
                row=1, col=1
            )
            
            if 'spread_mean' in spread_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=spread_df['timestamp'],
                        y=spread_df['spread_mean'],
                        name='Mean',
                        line=dict(color='#ffa500', dash='dash', width=1)
                    ),
                    row=1, col=1
                )
            
            # Z-score
            if 'zscore' in spread_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=spread_df['timestamp'],
                        y=spread_df['zscore'],
                        name='Z-Score',
                        line=dict(color='#00d4ff', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 212, 255, 0.2)'
                    ),
                    row=2, col=1
                )
                
                # Entry/exit thresholds
                fig.add_hline(y=2, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
                fig.add_hline(y=-2, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
                fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
            
            fig.update_layout(
                height=600,
                template='plotly_dark',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cointegration tests
            st.markdown("### 🧪 Statistical Tests")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ADF Test (Stationarity)")
                if len(spread_df) >= MIN_COINTEGRATION_POINTS:
                    adf_result = coint_analyzer.adf_test(spread_df['spread'])
                    
                    is_stationary = adf_result.get('is_stationary', False)
                    badge_class = "badge-success" if is_stationary else "badge-warning"
                    status = "✅ Stationary" if is_stationary else "⚠️ Non-Stationary"
                    
                    st.markdown(f'<div class="badge {badge_class}">{status}</div>', unsafe_allow_html=True)
                    
                    st.write(f"**Test Statistic:** {adf_result.get('test_statistic', 0):.4f}")
                    st.write(f"**P-Value:** {adf_result.get('p_value', 0):.4f}")
                    st.write(f"**Lags Used:** {adf_result.get('lags_used', 0)}")
                else:
                    st.info(f"Need {MIN_COINTEGRATION_POINTS}+ points for ADF test")
                    
            with col2:
                st.markdown("#### Cointegration Test")
                if len(spread_df) >= MIN_COINTEGRATION_POINTS:
                    coint_result = coint_analyzer.cointegration_test(ohlcv1, ohlcv2, 'close')
                    
                    is_coint = coint_result.get('cointegrated', False)
                    badge_class = "badge-success" if is_coint else "badge-warning"
                    status = "✅ Cointegrated" if is_coint else "⚠️ Not Cointegrated"
                    
                    st.markdown(f'<div class="badge {badge_class}">{status}</div>', unsafe_allow_html=True)
                    
                    st.write(f"**Test Statistic:** {coint_result.get('test_statistic', 0):.4f}")
                    st.write(f"**P-Value:** {coint_result.get('p_value', 0):.4f}")
                else:
                    st.info(f"Need {MIN_COINTEGRATION_POINTS}+ points for cointegration test")
            
            # Half-life
            if len(spread_df) >= 30:
                half_life = coint_analyzer.half_life(spread_df['spread'])
                st.metric("Mean Reversion Half-Life", f"{half_life:.1f} periods" if not np.isinf(half_life) else "∞ (No MR)")

with tab3:
    st.markdown("## 🔬 Market Microstructure Analytics")
    st.caption("Advanced tick-level analysis - order flow, liquidity, volatility regimes")
    
    if df1.empty:
        st.warning("⚠️ No data available for microstructure analysis")
    else:
        micro_analyzer = MicrostructureAnalyzer()
        
        # Tick velocity
        st.markdown("### ⚡ Tick Velocity & Activity")
        tick_velocity_df = micro_analyzer.tick_velocity(df1, window_seconds=60)
        
        if not tick_velocity_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                current_velocity = tick_velocity_df['tick_velocity'].iloc[-1]
                st.metric("Current Tick Velocity", f"{current_velocity:.2f} ticks/sec")
            with col2:
                avg_velocity = tick_velocity_df['tick_velocity'].mean()
                st.metric("Avg Tick Velocity", f"{avg_velocity:.2f} ticks/sec")
            with col3:
                max_velocity = tick_velocity_df['tick_velocity'].max()
                st.metric("Peak Velocity", f"{max_velocity:.2f} ticks/sec")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tick_velocity_df['timestamp'],
                y=tick_velocity_df['tick_velocity'],
                name='Tick Velocity',
                line=dict(color='#00d4ff', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 255, 0.2)'
            ))
            
            fig.update_layout(
                title='Tick Arrival Rate Over Time',
                xaxis_title='Time',
                yaxis_title='Ticks per Second',
                height=400,
                template='plotly_dark',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Order Flow Imbalance
        st.markdown("### 💹 Order Flow Imbalance")
        ofi_df = micro_analyzer.order_flow_imbalance(df1, window=100)
        
        if not ofi_df.empty:
            current_ofi = ofi_df['ofi_normalized'].iloc[-1]
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if current_ofi > 0.1:
                    st.markdown('<div class="badge badge-success">🟢 Buying Pressure</div>', unsafe_allow_html=True)
                elif current_ofi < -0.1:
                    st.markdown('<div class="badge badge-danger">🔴 Selling Pressure</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge badge-info">⚪ Neutral</div>', unsafe_allow_html=True)
                    
                st.metric("OFI", f"{current_ofi:.4f}")
                
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ofi_df['timestamp'],
                    y=ofi_df['ofi_normalized'],
                    name='Order Flow Imbalance',
                    line=dict(color='#00d4ff', width=2),
                    fill='tozeroy'
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_hline(y=0.2, line_dash="dot", line_color="green", opacity=0.3)
                fig.add_hline(y=-0.2, line_dash="dot", line_color="red", opacity=0.3)
                
                fig.update_layout(
                    title='Normalized Order Flow Imbalance',
                    height=300,
                    template='plotly_dark',
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#0e1117',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Volatility Regime Detection
        st.markdown("### 📊 Volatility Regime Analysis")
        vol_regime = micro_analyzer.volatility_regime(df1, 'price', windows=[20, 50, 100])
        
        col1, col2, col3, col4 = st.columns(4)
        
        regime = vol_regime.get('regime', 'unknown')
        confidence = vol_regime.get('confidence', 0)
        
        with col1:
            regime_colors = {'low': '#10b981', 'medium': '#f59e0b', 'high': '#ef4444'}
            regime_color = regime_colors.get(regime, '#64748b')
            st.markdown(f"""
            <div style='background: {regime_color}; padding: 20px; border-radius: 12px; text-align: center;'>
                <h3 style='margin: 0; color: white;'>{regime.upper()}</h3>
                <p style='margin: 5px 0 0 0; color: white; opacity: 0.9;'>Volatility Regime</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        with col3:
            vol_ratio = vol_regime.get('vol_ratio', 1.0)
            st.metric("Vol Ratio", f"{vol_ratio:.2f}x")
        with col4:
            current_vol = vol_regime.get('current_vol', 0)
            st.metric("Current Vol", f"{current_vol*100:.2f}%")
        
        # Liquidity Score
        st.markdown("### 💧 Liquidity Analysis")
        liq_df = micro_analyzer.liquidity_score(df1, window=100)
        
        if not liq_df.empty:
            current_liq = liq_df['liquidity_score_normalized'].iloc[-1]
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if current_liq > 70:
                    liq_status = "🟢 High Liquidity"
                    badge_class = "badge-success"
                elif current_liq > 40:
                    liq_status = "🟡 Medium Liquidity"
                    badge_class = "badge-warning"
                else:
                    liq_status = "🔴 Low Liquidity"
                    badge_class = "badge-danger"
                    
                st.markdown(f'<div class="badge {badge_class}">{liq_status}</div>', unsafe_allow_html=True)
                st.metric("Liquidity Score", f"{current_liq:.1f}/100")
                
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=liq_df['timestamp'],
                    y=liq_df['liquidity_score_normalized'],
                    name='Liquidity Score',
                    line=dict(color='#00d4ff', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 212, 255, 0.2)'
                ))
                
                fig.update_layout(
                    title='Liquidity Score Over Time',
                    yaxis=dict(range=[0, 100]),
                    height=300,
                    template='plotly_dark',
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#0e1117',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Momentum Quality
        st.markdown("### 🎯 Momentum Quality Assessment")
        momentum_quality = micro_analyzer.momentum_quality(df1, 'price', window=50)
        
        if momentum_quality:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                quality = momentum_quality.get('quality', 'unknown')
                quality_colors = {'high': '#10b981', 'medium': '#f59e0b', 'low': '#ef4444'}
                color = quality_colors.get(quality, '#64748b')
                st.markdown(f"""
                <div style='background: {color}; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4 style='margin: 0; color: white;'>{quality.upper()}</h4>
                    <p style='margin: 0; color: white; opacity: 0.9; font-size: 12px;'>Quality</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                direction = momentum_quality.get('direction', 'neutral')
                dir_color = '#10b981' if direction == 'bullish' else '#ef4444'
                dir_icon = '📈' if direction == 'bullish' else '📉'
                st.markdown(f"""
                <div style='background: {dir_color}; padding: 15px; border-radius: 8px; text-align: center;'>
                    <h4 style='margin: 0; color: white;'>{dir_icon} {direction.upper()}</h4>
                    <p style='margin: 0; color: white; opacity: 0.9; font-size: 12px;'>Direction</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                trend_strength = momentum_quality.get('trend_strength', 0)
                st.metric("Trend Strength", f"{trend_strength:.3f}")
                
            with col4:
                momentum_score = momentum_quality.get('momentum_score', 0)
                st.metric("Momentum Score", f"{momentum_score:.3f}")

# TAB 4: Trading Signals
with tab4:
    st.markdown("## 🎯 Trading Signals & Opportunity Scoring")

    spread_df = None

    if df1 is not None and df2 is not None and not df1.empty and not df2.empty:
        hedge_ratio, hr_stats = CointegrationAnalyzer.calculate_hedge_ratio(
            df1,
            df2,
            price_col="price"   # 🔑 IMPORTANT

        )

    # Step 2: only compute spread if hedge ratio is valid
        if not np.isnan(hedge_ratio):
            spread_df = CointegrationAnalyzer.calculate_spread(
                df1,
                df2,
                hedge_ratio,
                price_col="price"   # 🔑 IMPORTANT

            )

            if valid_df(spread_df):
                spread_df = CointegrationAnalyzer.calculate_zscore(spread_df)

    
    if (
        df1 is None or df1.empty or
        df2 is None or df2.empty or
        spread_df is None or spread_df.empty
    ):
        st.warning("⚠️ Need spread data for signal generation")
    else:
        signal_gen = SignalGenerator()
        
        # Get current z-score
        current_zscore = (
            spread_df['zscore'].iloc[-1] 
            if 'zscore' in spread_df.columns and not spread_df['zscore'].empty 
            else np.nan
            )
        
        # Generate mean reversion signal
        mr_signal = signal_gen.mean_reversion_signal(
            current_zscore, 
            entry_threshold=2.0, 
            exit_threshold=0.5
            )
        
        # Generate momentum signal
        momentum_signal = signal_gen.momentum_signal(ohlcv1, fast_window=12, slow_window=26)
        
        # Opportunity score
        liq_score = liq_df['liquidity_score_normalized'].iloc[-1] if not liq_df.empty else 50
        mom_quality = momentum_quality.get('momentum_score', 0.5) if momentum_quality else 0.5
        vol_regime_str = vol_regime.get('regime', 'medium')
        
        opp_score = signal_gen.opportunity_score(
            ohlcv1,
            zscore=current_zscore,
            vol_regime=vol_regime_str,
            momentum_quality=mom_quality,
            liquidity=liq_score
        )
        
        # Display opportunity score
        st.markdown("### 🎲 Overall Trading Opportunity")
        
        total_score = opp_score.get('total_score', 0)
        quality = opp_score.get('quality', 'Unknown')
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Gauge chart for opportunity score
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=total_score,
                title={'text': "Opportunity Score"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00d4ff"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ef4444"},
                        {'range': [40, 60], 'color': "#f59e0b"},
                        {'range': [60, 75], 'color': "#10b981"},
                        {'range': [75, 100], 'color': "#00d4ff"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                paper_bgcolor='#0e1117',
                font={'color': "white", 'family': "Arial"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e2329 0%, #2d3748 100%); 
                        padding: 20px; border-radius: 12px; text-align: center;'>
                <h2 style='margin: 0; color: #00d4ff;'>{quality}</h2>
                <p style='margin: 5px 0 0 0; color: #64748b;'>Quality Rating</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            components = opp_score.get('components', {})
            st.markdown("**Score Breakdown:**")
            for comp, value in components.items():
                st.progress(value/100, text=f"{comp.title()}: {value:.0f}")
        
        # Signal cards
        st.markdown("### 📡 Active Signals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Mean Reversion Signal")
            signal = mr_signal.get('signal', 'neutral')
            strength = mr_signal.get('strength', 0)
            
            signal_colors = {'long': '#10b981', 'short': '#ef4444', 'exit': '#f59e0b', 'neutral': '#64748b'}
            signal_icons = {'long': '🟢 LONG', 'short': '🔴 SHORT', 'exit': '⚪ EXIT', 'neutral': '⚫ NEUTRAL'}
            
            st.markdown(f"""
            <div style='background: {signal_colors.get(signal, '#64748b')}; 
                        padding: 20px; border-radius: 12px; text-align: center;'>
                <h2 style='margin: 0; color: white;'>{signal_icons.get(signal, signal.upper())}</h2>
                <p style='margin: 10px 0 0 0; color: white; opacity: 0.9;'>
                    Strength: {strength*100:.0f}% | Z-Score: {current_zscore:.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("#### Momentum Signal")
            mom_sig = momentum_signal.get('signal', 'neutral')
            mom_strength = momentum_signal.get('strength', 0)
            
            st.markdown(f"""
            <div style='background: {signal_colors.get(mom_sig, '#64748b')}; 
                        padding: 20px; border-radius: 12px; text-align: center;'>
                <h2 style='margin: 0; color: white;'>{signal_icons.get(mom_sig, mom_sig.upper())}</h2>
                <p style='margin: 10px 0 0 0; color: white; opacity: 0.9;'>
                    Strength: {mom_strength*100:.0f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Simple backtest
        st.markdown("### 📈 Strategy Backtest (Z-Score Mean Reversion)")
        
        backtest_results = signal_gen.backtest_simple_strategy(
            spread_df,
            entry_z=2.0,
            exit_z=0.0
        )
        
        if backtest_results and 'total_trades' in backtest_results:
            if backtest_results['total_trades'] > 0:
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Trades", backtest_results['total_trades'])
                with col2:
                    win_rate = backtest_results['win_rate'] * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                with col3:
                    st.metric("Total P&L", f"${backtest_results['total_pnl']:.2f}")
                with col4:
                    st.metric("Avg P&L", f"${backtest_results['avg_pnl']:.2f}")
                with col5:
                    st.metric("Return", f"{backtest_results['return_pct']:.2f}%")
                    
                st.info("💡 **Note:** This is a simplified backtest without transaction costs, slippage, or position sizing.")
            else:
                st.info("No trades executed in backtest period. Try adjusting entry/exit thresholds.")

# TAB 5: Alerts & Monitoring
with tab5:
    st.markdown("## 🚨 Alert System & Real-Time Monitoring")
    
    # Alert rule creation
    st.markdown("### ➕ Create New Alert Rule")
    
    with st.form("create_alert"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alert_name = st.text_input("Alert Name", placeholder="e.g., High Z-Score")
        with col2:
            alert_symbol = st.selectbox("Symbol", options=available_symbols)
        with col3:
            alert_metric = st.selectbox("Metric", options=["zscore", "price", "volume", "rsi", "ofi"])
        
        col1, col2 = st.columns(2)
        with col1:
            alert_operator = st.selectbox("Operator", options=[">", "<", ">=", "<=", "=="])
        with col2:
            alert_threshold = st.number_input("Threshold", value=2.0, step=0.1)
        
        submitted = st.form_submit_button("➕ Add Alert", use_container_width=True)
        
        if submitted and alert_name:
            new_rule = AlertRule(
                rule_id=f"rule_{len(st.session_state.alert_rules)}",
                name=alert_name,
                symbol=alert_symbol,
                metric=alert_metric,
                operator=alert_operator,
                threshold=alert_threshold
            )
            alert_engine.add_rule(new_rule)
            st.success(f"✅ Alert '{alert_name}' created!")
            st.rerun()
    
    # Display active rules
    st.markdown("### 📋 Active Alert Rules")
    
    rules = alert_engine.get_rules()
    
    if rules:
        for rule in rules:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                enabled_badge = "badge-success" if rule['enabled'] else "badge-danger"
                enabled_text = "🟢 Enabled" if rule['enabled'] else "🔴 Disabled"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{rule['name']}</strong> 
                    <span class="badge {enabled_badge}">{enabled_text}</span><br/>
                    <small style='color: #64748b;'>
                        {rule['symbol'].upper()} | {rule['metric']} {rule['operator']} {rule['threshold']}
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.write(f"Triggers: {rule['trigger_count']}")
            with col3:
                last_triggered = rule['last_triggered']
                if last_triggered:
                    st.write(f"Last: {last_triggered[:19]}")
                else:
                    st.write("Never triggered")
            with col4:
                if st.button("🗑️", key=f"del_{rule['rule_id']}"):
                    alert_engine.remove_rule(rule['rule_id'])
                    st.rerun()
    else:
        st.info("No alert rules configured. Create one above!")
    
    # Recent alerts
    st.markdown("### 📢 Recent Alerts")
    
    recent_alerts = alert_engine.get_alerts(n=20)
    
    if recent_alerts:
        alerts_df = pd.DataFrame(recent_alerts)
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts triggered yet.")
    
    # Statistics
    alert_stats = alert_engine.get_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rules", alert_stats['total_rules'])
    with col2:
        st.metric("Enabled Rules", alert_stats['enabled_rules'])
    with col3:
        st.metric("Total Alerts", alert_stats['total_alerts_triggered'])

with tab6:
    st.markdown("## 📊 Advanced Analytics & Visualizations")
    
    if df1.empty:
        st.warning("⚠️ No data available")
    else:
        # 3D Correlation Surface (UNIQUE FEATURE)
        st.markdown("### 🌐 3D Rolling Correlation Surface")
        
        if not df2.empty and len(df1) >= 100:
            # Calculate rolling correlation with multiple windows
            windows = [20, 50, 100]
            corr_data = []
            
            for window in windows:
                corr_df = None  # always define first
                if valid_df(ohlcv1) and valid_df(ohlcv2):
                    corr_df = coint_analyzer.rolling_correlation(
                        ohlcv1,
                        ohlcv2,
                        window=window,
                        price_col='close'
                    )
                else:
                    st.info("Waiting for sufficient data to compute rolling correlation")

                if valid_df(corr_df) and 'rolling_corr' in corr_df.columns:
                    for idx, row in corr_df.iterrows():
                        corr_data.append({
                            'window': window,
                            'timestamp': row['timestamp'],
                            'correlation': row['rolling_corr']
                        })
            
            if corr_data:
                corr_3d_df = pd.DataFrame(corr_data)
                
                # Create 3D surface plot
                fig = go.Figure(data=[go.Surface(
                    z=corr_3d_df.pivot(index='timestamp', columns='window', values='correlation').values,
                    x=windows,
                    y=corr_3d_df['timestamp'].unique(),
                    colorscale='Viridis',
                    showscale=True
                )])
                
                fig.update_layout(
                    title='3D Rolling Correlation Across Time & Window Sizes',
                    scene=dict(
                        xaxis_title='Window Size',
                        yaxis_title='Time',
                        zaxis_title='Correlation',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                    ),
                    height=600,
                    paper_bgcolor='#0e1117',
                    font={'color': "white"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need two symbols with 100+ data points for 3D correlation")
        
        # Correlation heatmap
        st.markdown("### 🔥 Cross-Asset Correlation Heatmap")
        
        if len(available_symbols) >= 2:
            # Get data for all symbols
            symbol_data = {}
            for sym in available_symbols[:5]:  # Limit to 5 symbols
                sym_df = storage.get_recent_ticks(sym, n=1000)
                if not sym_df.empty:
                    freq_str = {'1s': '1S', '1m': '1T', '5m': '5T'}[analysis_timeframe]
                    ohlcv = DataResampler.ticks_to_ohlcv(sym_df, freq_str)
                    if not ohlcv.empty:
                        symbol_data[sym] = ohlcv.set_index('timestamp')['close']
            
            if len(symbol_data) >= 2:
                # Create correlation matrix
                combined_df = pd.DataFrame(symbol_data)
                corr_matrix = combined_df.corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=[s.upper() for s in corr_matrix.columns],
                    y=[s.upper() for s in corr_matrix.index],
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 12},
                    colorbar=dict(title="Correlation")
                ))
                
                fig.update_layout(
                    title='Asset Correlation Matrix',
                    height=500,
                    template='plotly_dark',
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#0e1117'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Time series decomposition
        st.markdown("### 📐 Price Decomposition & Trend Analysis")
        
        if len(ohlcv1) >= 50:
            from scipy.signal import savgol_filter
            
            # Smooth trend using Savitzky-Golay filter
            window_length = min(51, len(ohlcv1) if len(ohlcv1) % 2 == 1 else len(ohlcv1) - 1)
            trend = savgol_filter(ohlcv1['close'], window_length, 3)
            
            # Detrend to get cyclic component
            cyclic = ohlcv1['close'] - trend
            
            # Residual
            residual = cyclic - cyclic.rolling(10).mean()
            
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Original Price', 'Trend', 'Cyclic Component', 'Residual')
            )
            
            fig.add_trace(go.Scatter(x=ohlcv1['timestamp'], y=ohlcv1['close'], 
                                    name='Price', line=dict(color='#00d4ff')), row=1, col=1)
            fig.add_trace(go.Scatter(x=ohlcv1['timestamp'], y=trend, 
                                    name='Trend', line=dict(color='#ffa500')), row=2, col=1)
            fig.add_trace(go.Scatter(x=ohlcv1['timestamp'], y=cyclic, 
                                    name='Cyclic', line=dict(color='#10b981')), row=3, col=1)
            fig.add_trace(go.Scatter(x=ohlcv1['timestamp'], y=residual, 
                                    name='Residual', line=dict(color='#ef4444')), row=4, col=1)
            
            fig.update_layout(
                height=800,
                template='plotly_dark',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution analysis
        st.markdown("### 📊 Returns Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            returns = ohlcv1['close'].pct_change().dropna()
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                marker_color='#00d4ff',
                opacity=0.7
            ))
            
            # Add normal distribution overlay
            from scipy.stats import norm
            x_range = np.linspace(returns.min(), returns.max(), 100)
            y_norm = norm.pdf(x_range, returns.mean(), returns.std()) * len(returns) * (x_range[1] - x_range[0])
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_norm,
                name='Normal Distribution',
                line=dict(color='#ffa500', width=2)
            ))
            
            fig.update_layout(
                title='Returns Histogram vs Normal Distribution',
                xaxis_title='Returns',
                yaxis_title='Frequency',
                height=400,
                template='plotly_dark',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Q-Q plot
            from scipy import stats as sp_stats
            
            (osm, osr), (slope, intercept, r) = sp_stats.probplot(returns, dist="norm")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=osm,
                y=osr,
                mode='markers',
                name='Returns',
                marker=dict(color='#00d4ff', size=6)
            ))
            
            # Add reference line
            fig.add_trace(go.Scatter(
                x=osm,
                y=slope * osm + intercept,
                mode='lines',
                name='Normal',
                line=dict(color='#ffa500', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='Q-Q Plot (Normal Distribution)',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                height=400,
                template='plotly_dark',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Time-series table with export
        st.markdown("### 📋 Time-Series Data Table")
        
        display_df = ohlcv1[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(100)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Export button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{symbol1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #64748b;'>
    <p>⚡ Quantitative Trading Analytics | Built with Streamlit, Plotly & Real-Time WebSockets</p>
    <p style='font-size: 12px;'>Designed for high-frequency trading & statistical arbitrage research</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh mechanism
if st.session_state.collecting:
    time.sleep(UPDATE_INTERVAL / 1000)
    st.rerun()

