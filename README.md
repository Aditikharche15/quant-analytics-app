<<<<<<< HEAD
QuanFusion Trading Analytics Platform
A comprehensive real-time trading analytics application featuring market microstructure analysis, statistical arbitrage, and advanced visualization capabilities.
🎯 Project Overview
This application provides institutional-grade analytics for cryptocurrency futures trading, designed specifically for MFT firms engaged in statistical arbitrage, market-making, and quantitative research.
Unique Features

Real-Time Microstructure Analytics

Tick velocity measurement (trades per second)
Order flow imbalance detection
Liquidity scoring algorithm
Volatility regime classification


3D Correlation Visualization

Multi-timeframe correlation surface
Interactive 3D plots for cross-asset analysis


Smart Trading Signals

Mean reversion opportunity scoring
Momentum quality assessment
Composite opportunity scoring (0-100)


Advanced Statistical Tests

ADF test for stationarity
Cointegration testing
Half-life calculation for mean reversion


Configurable Alert System

Custom alert rules (z-score, price, volume, RSI)
Real-time monitoring
Alert history tracking



🚀 Quick Start
Prerequisites

Python 3.13.3
Windows OS (configured for)
Internet connection for WebSocket data

Installation
bash# Clone or download the project
cd trading_analytics

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m streamlit run app.py
The application will open in your browser at http://localhost:8501
📁 Project Structure
trading_analytics/
├── app.py                     # Main Streamlit application
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/
│   ├── ingestion.py          # WebSocket data collection
│   ├── storage.py            # SQLite + in-memory storage
│   └── resampler.py          # OHLCV aggregation
│
├── analytics/
│   ├── statistics.py         # Basic statistical analysis
│   ├── cointegration.py      # Spread & hedge ratio calculation
│   ├── microstructure.py     # Tick-level analytics
│   └── signals.py            # Trading signal generation
│
├── alerts/
│   └── alert_engine.py       # Alert management system
│
├── utils/
│   ├── logger.py             # Logging configuration
│   └── helpers.py            # Utility functions
Dashboard Tabs

Overview & Prices

Real-time candlestick charts
Technical indicators (SMA, RSI, MACD)
Volume analysis
Statistical summaries


Spread & Cointegration

Hedge ratio calculation (OLS/Huber regression)
Spread visualization with z-scores
ADF test for stationarity
Cointegration testing
Mean reversion half-life


Microstructure (UNIQUE)

Tick velocity tracking
Order flow imbalance
Volatility regime detection
Liquidity scoring
Momentum quality assessment


Trading Signals

Mean reversion signals
Momentum crossover signals
Composite opportunity score (0-100)
Simple backtest results


Alerts & Monitoring

Create custom alert rules
Monitor triggered alerts
Enable/disable rules
Alert statistics


Advanced Analytics

3D correlation surface visualization
Cross-asset correlation heatmap
Time-series decomposition
Returns distribution analysis
Q-Q plots for normality testing
Data export functionality



🔧 Configuration
Edit config.py to customize:
pythonDEFAULT_SYMBOLS = ["btcusdt", "ethusdt"]  # Trading pairs
TIMEFRAMES = {"1s": 1, "1m": 60, "5m": 300}  # Sampling intervals
ROLLING_WINDOWS = [20, 50, 100, 200]  # Window sizes
MIN_DATA_POINTS = 30  # Minimum for analytics
MIN_COINTEGRATION_POINTS = 100  # For ADF/cointegration tests
📊 Analytics Methodology
1. Data Collection

Source: Binance Futures WebSocket API
Format: Real-time tick data (timestamp, symbol, price, size)
Storage: Hybrid (in-memory + SQLite for persistence)

2. Data Resampling

Aggregation to OHLCV bars (1s, 1m, 5m)
Volume-weighted average price (VWAP)
Technical indicators (SMA, EMA, RSI, Bollinger Bands, ATR)

3. Cointegration Analysis

Hedge Ratio: Calculated via OLS or Huber regression
Spread: S = Y - β * X
Z-Score: (Spread - μ) / σ
Tests:

Augmented Dickey-Fuller (stationarity)
Engle-Granger (cointegration)



4. Microstructure Metrics
Tick Velocity
Velocity = Ticks_in_window / Time_window
Measures market activity and interest.
Order Flow Imbalance
OFI = (Buy_Volume - Sell_Volume) / Total_Volume
Indicates buying/selling pressure using tick rule.
Liquidity Score
Liquidity = (Volume_Ratio) / (1 + Time_Between_Trades_Ratio)
Normalized to 0-100 scale.
Volatility Regime
Regime = Current_Vol / Historical_Vol

Low: < 0.7
Medium: 0.7 - 1.3
High: > 1.3

5. Signal Generation
Mean Reversion Signal

Entry: |z-score| > 2.0
Exit: |z-score| < 0.5
Direction: Long (z < -2), Short (z > 2)

Opportunity Score (0-100)
Score = 0.30 * Z-Score_Component 
      + 0.25 * Volatility_Component
      + 0.25 * Momentum_Component
      + 0.20 * Liquidity_Component
🏗️ Architecture
Design Philosophy
The system follows microservices-inspired design principles:

Separation of Concerns

Data ingestion layer (WebSocket)
Storage layer (SQLite + in-memory)
Analytics layer (modular calculators)
Presentation layer (Streamlit)


Scalability Considerations

Thread-safe storage with locks
Batch insertion for efficiency
Deque for bounded memory usage
Plugin architecture for new analytics


Extensibility

Easy to add new symbols
Modular analytics (add new calculators)
Pluggable alert rules
Flexible timeframe configuration



Data Flow
WebSocket → Tick Buffer → [Batch Insert] → SQLite + Memory
                ↓
         Analytics Engine
                ↓
    ┌───────────┴───────────┐
    ↓                       ↓
Statistics          Microstructure
Cointegration       Signals
    ↓                       ↓
         Alert Engine
                ↓
         Streamlit UI
🔒 Production Considerations
Current Limitations

Single-machine deployment
No authentication/authorization
Limited to 100K ticks in memory per symbol
No distributed processing

Future Enhancements

Scalability

Redis for distributed caching
Kafka for event streaming
TimescaleDB for time-series data
Microservices architecture


Performance

Cython for bottleneck functions
Async I/O for WebSocket
Batch processing with Dask/Ray
GPU acceleration for ML models


Reliability

Health checks and monitoring
Graceful degradation
Circuit breakers
Backup/recovery mechanisms


Security

JWT authentication
Role-based access control
API rate limiting
Encrypted communications



🧪 Testing
Manual Testing Checklist

 WebSocket connection stability
 Data persistence across restarts
 Alert triggering accuracy
 Chart rendering performance
 CSV export functionality
 Multi-symbol handling

Known Issues

Initial data collection requires 30-60 seconds
Some analytics need 100+ data points
Auto-refresh can be CPU-intensive

💡 ChatGPT/AI Usage
How AI Was Used

Algorithm Implementation

ADF test integration with statsmodels
Cointegration testing logic
Statistical calculations verification


UI/UX Design

Streamlit layout suggestions


Documentation

README structure
Code comments
API documentation



Example Prompts Used
"Create a WebSocket collector class for Binance Futures that handles reconnection"

"Implement ADF test for time series stationarity using statsmodels"

"Design a modern dark-themed Streamlit UI with gradient backgrounds and glassmorphism"

Overall system architecture design
Unique microstructure analytics algorithms
Integration between modules
Performance optimization strategies
Business logic for trading signals

📈 Performance Metrics

Data Throughput: ~100 ticks/second per symbol
Memory Usage: ~50MB for 100K ticks (2 symbols)
Storage: ~1MB per 10K ticks (SQLite)
UI Refresh Rate: 500ms (configurable)
WebSocket Latency: <100ms average

📝 License & Disclaimer
This project is for educational purposes only.
⚠️ IMPORTANT:

Not financial advice
No warranty or guarantee
Use at your own risk
Test thoroughly before any real trading

👨‍💻 Author
Created as a student evaluation assignment demonstrating:

Full-stack development (backend + frontend)
Real-time data processing
Advanced analytics implementation
Modern UI/UX design
System architecture planning

🙏 Acknowledgments

Binance for WebSocket API
Streamlit for rapid UI development
Plotly for interactive visualizations
Statsmodels for statistical tests


