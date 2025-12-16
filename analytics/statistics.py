import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
from utils.logger import get_logger

logger = get_logger()

class StatisticsCalculator:
    """Calculate various statistical measures for price data."""
    
    @staticmethod
    def basic_stats(df: pd.DataFrame, price_col: str = 'close') -> Dict:
        """Calculate basic descriptive statistics."""
        if df.empty or price_col not in df.columns:
            return {}
            
        prices = df[price_col].dropna()
        if len(prices) < 2:
            return {}
            
        return {
            'mean': float(prices.mean()),
            'median': float(prices.median()),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'skew': float(prices.skew()),
            'kurtosis': float(prices.kurtosis()),
            'count': int(len(prices))
        }
        
    @staticmethod
    def returns_analysis(df: pd.DataFrame, price_col: str = 'close') -> Dict:
        """Analyze returns distribution."""
        if df.empty or price_col not in df.columns or len(df) < 2:
            return {}
            
        prices = df[price_col].dropna()
        returns = prices.pct_change().dropna()
        
        if len(returns) < 2:
            return {}
            
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Annualization factor (assuming minute data)
        periods_per_day = 1440  # minutes in a day
        periods_per_year = periods_per_day * 365
        
        return {
            'mean_return': float(returns.mean()),
            'std_return': float(returns.std()),
            'volatility_annual': float(returns.std() * np.sqrt(periods_per_year)),
            'sharpe_approx': float(returns.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0,
            'max_return': float(returns.max()),
            'min_return': float(returns.min()),
            'positive_returns_pct': float((returns > 0).sum() / len(returns) * 100)
        }
        
    @staticmethod
    def rolling_statistics(df: pd.DataFrame, window: int, price_col: str = 'close') -> pd.DataFrame:
        """Calculate rolling statistics."""
        if df.empty or price_col not in df.columns or len(df) < window:
            return pd.DataFrame()
            
        result = pd.DataFrame(index=df.index)
        prices = df[price_col]
        
        result['rolling_mean'] = prices.rolling(window).mean()
        result['rolling_std'] = prices.rolling(window).std()
        result['rolling_min'] = prices.rolling(window).min()
        result['rolling_max'] = prices.rolling(window).max()
        
        # Rolling z-score
        result['z_score'] = (prices - result['rolling_mean']) / result['rolling_std']
        
        # Rolling Sharpe
        returns = prices.pct_change()
        result['rolling_sharpe'] = returns.rolling(window).mean() / returns.rolling(window).std()
        
        return result
        
    @staticmethod
    def correlation_analysis(df1: pd.DataFrame, df2: pd.DataFrame, 
                           price_col: str = 'close', window: int = 50) -> Dict:
        """Calculate correlation between two price series."""
        if df1.empty or df2.empty or len(df1) < 2 or len(df2) < 2:
            return {}
            
        # Align timestamps
        merged = pd.merge(
            df1[['timestamp', price_col]].rename(columns={price_col: 'price1'}),
            df2[['timestamp', price_col]].rename(columns={price_col: 'price2'}),
            on='timestamp',
            how='inner'
        )
        
        if len(merged) < 2:
            return {}
            
        # Pearson correlation
        pearson_corr = merged['price1'].corr(merged['price2'])
        
        # Spearman correlation (rank-based, more robust)
        spearman_corr, _ = stats.spearmanr(merged['price1'], merged['price2'])
        
        # Rolling correlation
        rolling_corr = merged['price1'].rolling(window).corr(merged['price2'])
        
        # Returns correlation
        ret1 = merged['price1'].pct_change().dropna()
        ret2 = merged['price2'].pct_change().dropna()
        returns_corr = ret1.corr(ret2) if len(ret1) > 1 else np.nan
        
        return {
            'pearson': float(pearson_corr),
            'spearman': float(spearman_corr),
            'returns_corr': float(returns_corr),
            'rolling_corr_mean': float(rolling_corr.mean()),
            'rolling_corr_std': float(rolling_corr.std()),
            'data_points': int(len(merged))
        }
        
    @staticmethod
    def volatility_analysis(df: pd.DataFrame, price_col: str = 'close', 
                          windows: list = [10, 20, 50]) -> Dict:
        """Calculate volatility across different windows."""
        if df.empty or price_col not in df.columns:
            return {}
            
        prices = df[price_col].dropna()
        returns = prices.pct_change().dropna()
        
        result = {}
        for window in windows:
            if len(returns) >= window:
                vol = returns.rolling(window).std()
                result[f'vol_{window}'] = {
                    'current': float(vol.iloc[-1]),
                    'mean': float(vol.mean()),
                    'max': float(vol.max()),
                    'min': float(vol.min())
                }
                
        # Parkinson volatility (using high-low range)
        if 'high' in df.columns and 'low' in df.columns:
            high = df['high'].dropna()
            low = df['low'].dropna()
            if len(high) > 1 and len(low) > 1:
                parkinson_vol = np.sqrt(
                    (1 / (4 * len(high) * np.log(2))) * 
                    ((np.log(high / low)) ** 2).sum()
                )
                result['parkinson_vol'] = float(parkinson_vol)
                
        return result
        
    @staticmethod
    def detect_outliers(df: pd.DataFrame, price_col: str = 'close', 
                       threshold: float = 3.0) -> Tuple[pd.DataFrame, int]:
        """Detect outliers using z-score method."""
        if df.empty or price_col not in df.columns or len(df) < 10:
            return df, 0
            
        df = df.copy()
        prices = df[price_col]
        
        mean = prices.mean()
        std = prices.std()
        
        df['z_score'] = (prices - mean) / std
        df['is_outlier'] = abs(df['z_score']) > threshold
        
        outlier_count = df['is_outlier'].sum()
        
        return df, int(outlier_count)