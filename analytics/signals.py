import pandas as pd
import numpy as np
from typing import Dict, List
from utils.logger import get_logger

logger = get_logger()

class SignalGenerator:
    """Generate trading signals and opportunity scores."""
    
    @staticmethod
    def mean_reversion_signal(zscore: float, entry_threshold: float = 2.0, 
                             exit_threshold: float = 0.5) -> Dict:
        """
        Generate mean reversion trading signal based on z-score.
        
        Returns:
            signal: 'long', 'short', 'exit', 'neutral'
            strength: 0-1 confidence score
        """
        if pd.isna(zscore):
            return {'signal': 'neutral', 'strength': 0.0, 'zscore': np.nan}
            
        abs_z = abs(zscore)
        
        if zscore < -entry_threshold:
            # Oversold, go long
            signal = 'long'
            strength = min(abs_z / entry_threshold / 2, 1.0)
        elif zscore > entry_threshold:
            # Overbought, go short
            signal = 'short'
            strength = min(abs_z / entry_threshold / 2, 1.0)
        elif abs_z < exit_threshold:
            # Near mean, exit positions
            signal = 'exit'
            strength = 1.0 - abs_z / exit_threshold
        else:
            signal = 'neutral'
            strength = 0.0
            
        return {
            'signal': signal,
            'strength': float(strength),
            'zscore': float(zscore)
        }
        
    @staticmethod
    def momentum_signal(df: pd.DataFrame, fast_window: int = 12, 
                       slow_window: int = 26) -> Dict:
        """Generate momentum signal using EMA crossover."""
        if df.empty or 'close' not in df.columns or len(df) < slow_window:
            return {'signal': 'neutral', 'strength': 0.0}
            
        close = df['close']
        
        ema_fast = close.ewm(span=fast_window).mean()
        ema_slow = close.ewm(span=slow_window).mean()
        
        current_diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]
        prev_diff = ema_fast.iloc[-2] - ema_slow.iloc[-2] if len(ema_fast) > 1 else 0
        
        # Crossover detection
        if current_diff > 0 and prev_diff <= 0:
            signal = 'long'
            strength = 0.9
        elif current_diff < 0 and prev_diff >= 0:
            signal = 'short'
            strength = 0.9
        elif current_diff > 0:
            signal = 'long'
            strength = min(abs(current_diff) / close.iloc[-1] * 100, 1.0)
        elif current_diff < 0:
            signal = 'short'
            strength = min(abs(current_diff) / close.iloc[-1] * 100, 1.0)
        else:
            signal = 'neutral'
            strength = 0.0
            
        return {
            'signal': signal,
            'strength': float(strength),
            'ema_fast': float(ema_fast.iloc[-1]),
            'ema_slow': float(ema_slow.iloc[-1])
        }
        
    @staticmethod
    def opportunity_score(df: pd.DataFrame, zscore: float = None,
                         vol_regime: str = None, momentum_quality: float = None,
                         liquidity: float = None) -> Dict:
        """
        Calculate overall trading opportunity score combining multiple factors.
        
        Score components:
        - Mean reversion potential (z-score)
        - Volatility regime (prefer medium)
        - Momentum quality
        - Liquidity
        
        Returns score 0-100
        """
        score_components = {}
        weights = {}
        
        # Z-score component (30%)
        if zscore is not None and not pd.isna(zscore):
            abs_z = abs(zscore)
            if abs_z > 2:
                z_score = min((abs_z - 2) / 2 * 100, 100)  # Extreme z-score = high opportunity
            else:
                z_score = 0
            score_components['zscore'] = z_score
            weights['zscore'] = 0.30
        else:
            score_components['zscore'] = 0
            weights['zscore'] = 0
            
        # Volatility regime component (25%)
        vol_score = 0
        if vol_regime:
            if vol_regime == 'medium':
                vol_score = 100  # Best for trading
            elif vol_regime == 'low':
                vol_score = 40  # Less opportunity
            elif vol_regime == 'high':
                vol_score = 60  # Risky but opportunity exists
        score_components['volatility'] = vol_score
        weights['volatility'] = 0.25
        
        # Momentum quality component (25%)
        if momentum_quality is not None:
            mom_score = momentum_quality * 100
            score_components['momentum'] = mom_score
            weights['momentum'] = 0.25
        else:
            score_components['momentum'] = 50
            weights['momentum'] = 0.25
            
        # Liquidity component (20%)
        if liquidity is not None:
            liq_score = liquidity  # Already 0-100
            score_components['liquidity'] = liq_score
            weights['liquidity'] = 0.20
        else:
            score_components['liquidity'] = 50
            weights['liquidity'] = 0.20
            
        # Calculate weighted score
        total_score = sum(
            score_components[k] * weights[k] 
            for k in score_components.keys()
        )
        
        # Quality rating
        if total_score >= 75:
            quality = 'Excellent'
        elif total_score >= 60:
            quality = 'Good'
        elif total_score >= 40:
            quality = 'Fair'
        else:
            quality = 'Poor'
            
        return {
            'total_score': float(total_score),
            'quality': quality,
            'components': {k: float(v) for k, v in score_components.items()},
            'weights': weights
        }
        
    @staticmethod
    def generate_alerts(df: pd.DataFrame, conditions: List[Dict]) -> List[Dict]:
        """
        Generate alerts based on custom conditions.
        
        conditions format:
        [
            {'field': 'zscore', 'operator': '>', 'value': 2.0},
            {'field': 'rsi', 'operator': '<', 'value': 30}
        ]
        """
        alerts = []
        
        if df.empty:
            return alerts
            
        latest = df.iloc[-1]
        
        for condition in conditions:
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if field not in latest.index:
                continue
                
            current_value = latest[field]
            
            triggered = False
            if operator == '>':
                triggered = current_value > value
            elif operator == '<':
                triggered = current_value < value
            elif operator == '==':
                triggered = current_value == value
            elif operator == '>=':
                triggered = current_value >= value
            elif operator == '<=':
                triggered = current_value <= value
                
            if triggered:
                alerts.append({
                    'timestamp': latest['timestamp'] if 'timestamp' in latest.index else pd.Timestamp.now(),
                    'field': field,
                    'condition': f"{field} {operator} {value}",
                    'current_value': float(current_value),
                    'threshold': float(value)
                })
                
        return alerts
        
    @staticmethod
    def backtest_simple_strategy(spread_df: pd.DataFrame, 
                                entry_z: float = 2.0,
                                exit_z: float = 0.0) -> Dict:
        """
        Simple backtest of z-score mean reversion strategy.
        
        Strategy:
        - Long when z < -entry_z
        - Short when z > entry_z
        - Exit when z crosses exit_z
        """
        if spread_df.empty or 'zscore' not in spread_df.columns:
            return {}
            
        df = spread_df.copy()
        df = df.dropna(subset=['zscore'])
        
        if len(df) < 10:
            return {'note': 'Insufficient data for backtest'}
            
        position = 0  # 0: flat, 1: long, -1: short
        entry_price = 0
        trades = []
        equity = [100.0]  # Start with $100
        
        for i in range(1, len(df)):
            z = df.iloc[i]['zscore']
            spread = df.iloc[i]['spread']
            
            # Entry logic
            if position == 0:
                if z < -entry_z:
                    position = 1
                    entry_price = spread
                elif z > entry_z:
                    position = -1
                    entry_price = spread
                    
            # Exit logic
            elif position != 0:
                if (position == 1 and z > exit_z) or (position == -1 and z < exit_z):
                    # Close position
                    pnl = position * (spread - entry_price)
                    trades.append({
                        'entry_z': df.iloc[i-1]['zscore'],
                        'exit_z': z,
                        'pnl': pnl,
                        'position': 'long' if position == 1 else 'short'
                    })
                    equity.append(equity[-1] + pnl)
                    position = 0
                    
        # Calculate metrics
        if len(trades) > 0:
            pnls = [t['pnl'] for t in trades]
            win_rate = sum(1 for pnl in pnls if pnl > 0) / len(pnls)
            avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
            avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
            
            return {
                'total_trades': len(trades),
                'win_rate': float(win_rate),
                'total_pnl': float(sum(pnls)),
                'avg_pnl': float(np.mean(pnls)),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'final_equity': float(equity[-1]),
                'return_pct': float((equity[-1] - 100) / 100 * 100)
            }
        else:
            return {'total_trades': 0, 'note': 'No trades executed'}