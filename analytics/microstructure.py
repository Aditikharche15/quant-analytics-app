import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats
from utils.logger import get_logger

logger = get_logger()

class MicrostructureAnalyzer:
    """Analyze market microstructure from tick data."""
    
    @staticmethod
    def tick_velocity(df: pd.DataFrame, window_seconds: int = 60) -> pd.DataFrame:
        """
        Calculate tick arrival rate (ticks per second).
        High velocity = high activity/interest.
        """
        if df.empty or 'timestamp' not in df.columns:
            return pd.DataFrame()
            
        df = df.copy()
        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        
        # Count ticks in rolling window
        result = pd.DataFrame()
        result['tick_count'] = df['price'].rolling(f'{window_seconds}S').count()
        result['tick_velocity'] = result['tick_count'] / window_seconds  # ticks/second
        
        result.reset_index(inplace=True)
        return result
        
    @staticmethod
    def order_flow_imbalance(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """
        Calculate order flow imbalance (buy vs sell pressure).
        OFI > 0: buying pressure, OFI < 0: selling pressure
        """
        if df.empty or len(df) < 2:
            return pd.DataFrame()
            
        df = df.copy()
        df = df.sort_values('timestamp')
        
        # Approximate trade direction using tick rule
        df['price_change'] = df['price'].diff()
        df['trade_direction'] = np.sign(df['price_change'])
        df.loc[df['trade_direction'] == 0, 'trade_direction'] = np.nan
        df['trade_direction'].fillna(method='ffill', inplace=True)
        
        # Signed volume
        df['signed_volume'] = df['size'] * df['trade_direction']
        
        # Rolling OFI
        df['ofi'] = df['signed_volume'].rolling(window).sum()
        df['ofi_normalized'] = df['ofi'] / df['size'].rolling(window).sum()
        
        return df[['timestamp', 'ofi', 'ofi_normalized', 'trade_direction']]
        
    @staticmethod
    def volume_weighted_spread(df: pd.DataFrame, window: int = 50) -> pd.Series:
        """
        Estimate effective spread using volume and price changes.
        Proxy for transaction costs.
        """
        if df.empty or len(df) < window:
            return pd.Series(dtype=float)
            
        df = df.copy()
        df = df.sort_values('timestamp')
        
        # Price changes weighted by volume
        df['price_change'] = df['price'].diff().abs()
        df['vol_weighted_change'] = df['price_change'] * df['size']
        
        spread_proxy = df['vol_weighted_change'].rolling(window).sum() / df['size'].rolling(window).sum()
        
        return spread_proxy
        
    @staticmethod
    def trade_intensity_profile(df: pd.DataFrame, n_bins: int = 24) -> Dict:
        """
        Analyze trading intensity across time of day.
        Returns distribution of trades by hour.
        """
        if df.empty or 'timestamp' not in df.columns:
            return {}
            
        df = df.copy()
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        hourly_counts = df.groupby('hour').size()
        hourly_volume = df.groupby('hour')['size'].sum()
        
        return {
            'hourly_trades': hourly_counts.to_dict(),
            'hourly_volume': hourly_volume.to_dict(),
            'peak_hour': int(hourly_counts.idxmax()) if not hourly_counts.empty else None
        }
        
    @staticmethod
    def volatility_regime(df: pd.DataFrame, price_col: str = 'price', 
                         windows: list = [20, 50, 100]) -> Dict:
        """
        Detect volatility regime using multiple timeframes.
        Returns current regime: 'low', 'medium', 'high'
        """
        if df.empty or price_col not in df.columns:
            return {'regime': 'unknown', 'confidence': 0.0}
            
        returns = df[price_col].pct_change().dropna()
        
        if len(returns) < max(windows):
            return {'regime': 'insufficient_data', 'confidence': 0.0}
            
        # Calculate volatility at different windows
        vols = {}
        for w in windows:
            if len(returns) >= w:
                vols[f'vol_{w}'] = returns.tail(w).std()
                
        # Current vs historical volatility
        current_vol = returns.tail(windows[0]).std()
        hist_vol = returns.std()
        
        vol_ratio = current_vol / hist_vol if hist_vol > 0 else 1.0
        
        # Classify regime
        if vol_ratio < 0.7:
            regime = 'low'
            confidence = min((0.7 - vol_ratio) / 0.3, 1.0)
        elif vol_ratio > 1.3:
            regime = 'high'
            confidence = min((vol_ratio - 1.3) / 0.7, 1.0)
        else:
            regime = 'medium'
            confidence = 1.0 - abs(vol_ratio - 1.0)
            
        return {
            'regime': regime,
            'vol_ratio': float(vol_ratio),
            'current_vol': float(current_vol),
            'historical_vol': float(hist_vol),
            'confidence': float(confidence),
            **{k: float(v) for k, v in vols.items()}
        }
        
    @staticmethod
    def momentum_quality(df: pd.DataFrame, price_col: str = 'price', 
                        window: int = 50) -> Dict:
        """
        Assess momentum quality using consistency and strength metrics.
        High quality = strong, consistent trend.
        """
        if df.empty or price_col not in df.columns or len(df) < window:
            return {}
            
        prices = df[price_col].tail(window)
        returns = prices.pct_change().dropna()
        
        # Trend strength (R² of linear regression)
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Momentum consistency (% of positive returns)
        positive_ratio = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.5
        
        # Momentum score
        momentum_score = r_squared * abs(2 * positive_ratio - 1)  # 0 to 1
        
        return {
            'trend_strength': float(r_squared),
            'positive_return_ratio': float(positive_ratio),
            'momentum_score': float(momentum_score),
            'direction': 'bullish' if positive_ratio > 0.5 else 'bearish',
            'quality': 'high' if momentum_score > 0.7 else 'medium' if momentum_score > 0.4 else 'low'
        }
        
    @staticmethod
    def liquidity_score(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """
        Calculate liquidity score based on volume and tick frequency.
        Higher score = better liquidity.
        """
        if df.empty or len(df) < window:
            return pd.DataFrame()
            
        df = df.copy()
        df = df.sort_values('timestamp')
        
        # Volume metrics
        df['volume_ma'] = df['size'].rolling(window).mean()
        df['volume_ratio'] = df['size'] / df['volume_ma']
        
        # Tick frequency (time between trades)
        df['time_diff'] = pd.to_datetime(df['timestamp']).diff().dt.total_seconds()
        df['avg_time_diff'] = df['time_diff'].rolling(window).mean()
        
        # Liquidity score (higher volume, lower time between trades = higher liquidity)
        df['liquidity_score'] = df['volume_ratio'] / (1 + df['time_diff'] / df['avg_time_diff'])
        
        # Normalize to 0-100
        if df['liquidity_score'].std() > 0:
            df['liquidity_score_normalized'] = (
                (df['liquidity_score'] - df['liquidity_score'].mean()) / 
                df['liquidity_score'].std() * 10 + 50
            ).clip(0, 100)
        else:
            df['liquidity_score_normalized'] = 50
            
        return df[['timestamp', 'liquidity_score', 'liquidity_score_normalized']]