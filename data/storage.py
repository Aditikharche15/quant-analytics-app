import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from threading import Lock
from typing import List, Dict, Optional
from config import SQLITE_DB
from utils.logger import get_logger

logger = get_logger()

class TickStorage:
    """Hybrid storage: in-memory for real-time, SQLite for persistence."""
    
    def __init__(self, max_memory_ticks=100000):
        self.max_memory_ticks = max_memory_ticks
        self.memory_buffer = {}  # symbol -> deque of ticks
        self.lock = Lock()
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(SQLITE_DB))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_ts ON ticks(symbol, timestamp)")
        conn.commit()
        conn.close()
        logger.info("Database initialized")
        
    def insert_tick(self, tick: Dict):
        """Insert single tick into both memory and DB."""
        with self.lock:
            symbol = tick['symbol']
            
            # Memory buffer
            if symbol not in self.memory_buffer:
                self.memory_buffer[symbol] = deque(maxlen=self.max_memory_ticks)
            self.memory_buffer[symbol].append(tick)
            
        # Persist to DB (non-blocking in production would use queue)
        try:
            conn = sqlite3.connect(str(SQLITE_DB))
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO ticks (timestamp, symbol, price, size) VALUES (?, ?, ?, ?)",
                (tick['timestamp'], tick['symbol'], tick['price'], tick['size'])
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"DB insert error: {e}")
            
    def insert_batch(self, ticks: List[Dict]):
        """Batch insert for efficiency."""
        if not ticks:
            return
            
        with self.lock:
            for tick in ticks:
                symbol = tick['symbol']
                if symbol not in self.memory_buffer:
                    self.memory_buffer[symbol] = deque(maxlen=self.max_memory_ticks)
                self.memory_buffer[symbol].append(tick)
        
        try:
            conn = sqlite3.connect(str(SQLITE_DB))
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO ticks (timestamp, symbol, price, size) VALUES (?, ?, ?, ?)",
                [(t['timestamp'], t['symbol'], t['price'], t['size']) for t in ticks]
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Batch insert error: {e}")
            
    def get_recent_ticks(self, symbol: str, n: int = 1000) -> pd.DataFrame:
        """Get recent ticks from memory."""
        with self.lock:
            if symbol not in self.memory_buffer or len(self.memory_buffer[symbol]) == 0:
                return pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'size'])
            
            data = list(self.memory_buffer[symbol])[-n:]
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(
                df['timestamp'],
                format='mixed',
                errors='coerce'
            )
            df = df.dropna(subset=['timestamp'])

            return df
            
    def get_ticks_range(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Query ticks from DB by time range."""
        try:
            conn = sqlite3.connect(str(SQLITE_DB))
            query = """
                SELECT timestamp, symbol, price, size 
                FROM ticks 
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(symbol, start.isoformat(), end.isoformat())
            )
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            logger.error(f"Range query error: {e}")
            return pd.DataFrame(columns=['timestamp', 'symbol', 'price', 'size'])
            
    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols with data."""
        with self.lock:
            return list(self.memory_buffer.keys())
            
    def get_tick_count(self, symbol: str) -> int:
        """Get count of ticks in memory for a symbol."""
        with self.lock:
            return len(self.memory_buffer.get(symbol, []))
            
    def clear_symbol(self, symbol: str):
        """Clear memory buffer for a symbol."""
        with self.lock:
            if symbol in self.memory_buffer:
                self.memory_buffer[symbol].clear()
                
    def export_to_csv(self, symbol: str, filepath: str):
        """Export all data for a symbol to CSV."""
        conn = sqlite3.connect(str(SQLITE_DB))
        query = "SELECT * FROM ticks WHERE symbol = ? ORDER BY timestamp"
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} ticks to {filepath}")

# Global storage instance
storage = TickStorage()