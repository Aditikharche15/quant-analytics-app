import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.linear_model import LinearRegression, HuberRegressor
from typing import Dict, Tuple
from utils.logger import get_logger

logger = get_logger()

class CointegrationAnalyzer:
    """Analyze cointegration and spread between two assets."""
    
    @staticmethod
    def calculate_hedge_ratio(df1: pd.DataFrame, df2: pd.DataFrame, 
                            price_col: str = 'close', 
                            method: str = 'ols') -> Tuple[float, Dict]:
        """
        Calculate hedge ratio using regression.
        
        Args:
            df1: DataFrame for asset 1 (Y)
            df2: DataFrame for asset 2 (X)
            price_col: Price column name
            method: 'ols' or 'huber' (robust regression)
            
        Returns:
            hedge_ratio, stats_dict
        """
        # Align data
        merged = pd.merge(
            df1[['timestamp', price_col]].rename(columns={price_col: 'y'}),
            df2[['timestamp', price_col]].rename(columns={price_col: 'x'}),
            on='timestamp',
            how='inner'
        )
        
        if len(merged) < 30:
            return np.nan, {}
            
        X = merged['x'].values.reshape(-1, 1)
        y = merged['y'].values
        
        # Regression
        if method == 'huber':
            model = HuberRegressor()
        else:
            model = LinearRegression()
            
        model.fit(X, y)
        
        hedge_ratio = float(model.coef_[0])
        intercept = float(model.intercept_)
        
        # R-squared
        y_pred = model.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        stats = {
            'hedge_ratio': hedge_ratio,
            'intercept': intercept,
            'r_squared': float(r_squared),
            'data_points': int(len(merged)),
            'method': method
        }
        
        return hedge_ratio, stats
        
    @staticmethod
    def calculate_spread(df1: pd.DataFrame, df2: pd.DataFrame, 
                        hedge_ratio: float, price_col: str = 'close') -> pd.DataFrame:
        """Calculate spread: S = Y - hedge_ratio * X"""
        merged = pd.merge(
            df1[['timestamp', price_col]].rename(columns={price_col: 'price1'}),
            df2[['timestamp', price_col]].rename(columns={price_col: 'price2'}),
            on='timestamp',
            how='inner'
        )
        
        if merged.empty:
            return pd.DataFrame()
            
        merged['spread'] = merged['price1'] - hedge_ratio * merged['price2']
        return merged
        
    @staticmethod
    def calculate_zscore(spread_df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """Calculate rolling z-score of spread."""
        if spread_df.empty or 'spread' not in spread_df.columns:
            return spread_df
            
        df = spread_df.copy()
        
        # Rolling stats
        df['spread_mean'] = df['spread'].rolling(window).mean()
        df['spread_std'] = df['spread'].rolling(window).std()
        
        # Z-score
        df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
        
        return df
        
    @staticmethod
    def adf_test(series: pd.Series, max_lag: int = 10) -> Dict:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Returns:
            Dict with test statistic, p-value, and conclusion
        """
        if len(series) < 30:
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'critical_values': {},
                'is_stationary': False,
                'note': 'Insufficient data (need >= 30 points)'
            }
            
        try:
            result = adfuller(series.dropna(), maxlag=max_lag)
            
            return {
                'test_statistic': float(result[0]),
                'p_value': float(result[1]),
                'critical_values': {k: float(v) for k, v in result[4].items()},
                'is_stationary': result[1] < 0.05,  # 5% significance
                'lags_used': int(result[2]),
                'n_obs': int(result[3])
            }
        except Exception as e:
            logger.error(f"ADF test error: {e}")
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'critical_values': {},
                'is_stationary': False,
                'error': str(e)
            }
            
    @staticmethod
    def cointegration_test(df1: pd.DataFrame, df2: pd.DataFrame, 
                          price_col: str = 'close') -> Dict:
        """Test for cointegration between two price series."""
        merged = pd.merge(
            df1[['timestamp', price_col]].rename(columns={price_col: 'y1'}),
            df2[['timestamp', price_col]].rename(columns={price_col: 'y2'}),
            on='timestamp',
            how='inner'
        )
        
        if len(merged) < 30:
            return {
                'cointegrated': False,
                'note': 'Insufficient data'
            }
            
        try:
            score, pvalue, _ = coint(merged['y1'], merged['y2'])
            
            return {
                'test_statistic': float(score),
                'p_value': float(pvalue),
                'cointegrated': pvalue < 0.05,
                'data_points': int(len(merged))
            }
        except Exception as e:
            logger.error(f"Cointegration test error: {e}")
            return {
                'cointegrated': False,
                'error': str(e)
            }
            
    @staticmethod
    def rolling_correlation(df1: pd.DataFrame, df2: pd.DataFrame, 
                          window: int = 50, price_col: str = 'close') -> pd.DataFrame:
        """Calculate rolling correlation between two assets."""
        merged = pd.merge(
            df1[['timestamp', price_col]].rename(columns={price_col: 'price1'}),
            df2[['timestamp', price_col]].rename(columns={price_col: 'price2'}),
            on='timestamp',
            how='inner'
        )
        
        if merged.empty or len(merged) < window:
            return pd.DataFrame()
            
        merged['rolling_corr'] = merged['price1'].rolling(window).corr(merged['price2'])
        
        # Returns correlation
        merged['ret1'] = merged['price1'].pct_change()
        merged['ret2'] = merged['price2'].pct_change()
        merged['rolling_corr_returns'] = merged['ret1'].rolling(window).corr(merged['ret2'])
        
        return merged
        
    @staticmethod
    def half_life(spread: pd.Series) -> float:
        """
        Calculate mean reversion half-life using Ornstein-Uhlenbeck process.
        
        Returns half-life in number of periods (bars).
        """
        if len(spread) < 10:
            return np.nan
            
        try:
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Align
            spread_lag = spread_lag[spread_diff.index]
            
            if len(spread_lag) < 10:
                return np.nan
                
            # Regression: Δy_t = λ * y_{t-1} + ε
            X = spread_lag.values.reshape(-1, 1)
            y = spread_diff.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            lambda_param = model.coef_[0]
            
            if lambda_param >= 0:
                return np.inf  # No mean reversion
                
            half_life = -np.log(2) / lambda_param
            return float(half_life)
            
        except Exception as e:
            logger.error(f"Half-life calculation error: {e}")
            return np.nan