import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_store"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database
SQLITE_DB = DATA_DIR / "ticks.db"

# WebSocket settings
BINANCE_WS_URL = "wss://fstream.binance.com/ws"
DEFAULT_SYMBOLS = ["btcusdt", "ethusdt"]
RECONNECT_DELAY = 5  # seconds

# Sampling intervals (seconds)
TIMEFRAMES = {
    "1s": 1,
    "1m": 60,
    "5m": 300,
}

# Analytics parameters
ROLLING_WINDOWS = [20, 50, 100, 200]
DEFAULT_WINDOW = 50
MIN_DATA_POINTS = 30  # Minimum for basic analytics
MIN_COINTEGRATION_POINTS = 100  # For ADF test

# Alert settings
ALERT_CHECK_INTERVAL = 1  # seconds
MAX_ALERTS_STORED = 1000

# UI settings
CHART_HEIGHT = 400
UPDATE_INTERVAL = 500  # milliseconds for frontend refresh

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"