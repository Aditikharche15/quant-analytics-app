import pandas as pd
import numpy as np
from typing import Dict
from utils.logger import get_logger

logger = get_logger()

class DataResampler:
    """Resample tick data to OHLCV bars."""
    
    @staticmethod
    def ticks_to_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Convert tick data to OHLCV format.
        
        Args:
            df: DataFrame with columns [timestamp, price, size]
            freq: Pandas frequency string ('1S', '1T', '5T', etc.)
            
        Returns:
            OHLCV DataFrame
        """
        if df.empty or len(df) < 2:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades'])
            
        df = df.copy()
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        # Resample
        ohlcv = pd.DataFrame()
        ohlcv['open'] = df['price'].resample(freq).first()
        ohlcv['high'] = df['price'].resample(freq).max()
        ohlcv['low'] = df['price'].resample(freq).min()
        ohlcv['close'] = df['price'].resample(freq).last()
        ohlcv['volume'] = df['size'].resample(freq).sum()
        ohlcv['trades'] = df['price'].resample(freq).count()
        
        # Remove incomplete bars and NaN
        ohlcv = ohlcv.dropna()
        ohlcv.reset_index(inplace=True)
        
        return ohlcv
        
    @staticmethod
    def calculate_vwap(df: pd.DataFrame, freq: str = '1T') -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        if df.empty:
            return pd.Series(dtype=float)
            
        df = df.copy()
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        df['pv'] = df['price'] * df['size']
        
        pv_sum = df['pv'].resample(freq).sum()
        vol_sum = df['size'].resample(freq).sum()
        
        vwap = pv_sum / vol_sum
        return vwap.dropna()
        
    @staticmethod
    def calculate_tick_stats(df: pd.DataFrame, freq: str = '1T') -> pd.DataFrame:
        """Calculate tick-level statistics."""
        if df.empty:
            return pd.DataFrame()
            
        df = df.copy()
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        stats = pd.DataFrame()
        stats['tick_count'] = df['price'].resample(freq).count()
        stats['avg_trade_size'] = df['size'].resample(freq).mean()
        stats['total_volume'] = df['size'].resample(freq).sum()
        stats['price_std'] = df['price'].resample(freq).std()
        
        # Tick direction (buy/sell pressure approximation)
        df['price_change'] = df['price'].diff()
        stats['buy_ticks'] = (df['price_change'] > 0).resample(freq).sum()
        stats['sell_ticks'] = (df['price_change'] < 0).resample(freq).sum()
        stats['order_imbalance'] = (stats['buy_ticks'] - stats['sell_ticks']) / stats['tick_count']
        
        return stats.dropna()
        
    @staticmethod
    def add_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to OHLCV data."""
        if ohlcv.empty or len(ohlcv) < 20:
            return ohlcv
            
        df = ohlcv.copy()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean() if len(df) >= 50 else np.nan
        
        # Exponential moving average
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        
        # ATR (Average True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        return df