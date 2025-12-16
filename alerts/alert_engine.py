import threading
import time
from datetime import datetime
from collections import deque
from typing import List, Dict, Callable
from utils.logger import get_logger

logger = get_logger()

class AlertRule:
    """Represents a single alert rule."""
    
    def __init__(self, rule_id: str, name: str, symbol: str,
                 metric: str, operator: str, threshold: float,
                 enabled: bool = True):
        self.rule_id = rule_id
        self.name = name
        self.symbol = symbol
        self.metric = metric  # e.g., 'zscore', 'price', 'volume'
        self.operator = operator  # '>', '<', '==', '>=', '<='
        self.threshold = threshold
        self.enabled = enabled
        self.last_triggered = None
        self.trigger_count = 0
        
    def check(self, value: float) -> bool:
        """Check if alert condition is met."""
        if not self.enabled:
            return False
            
        try:
            if self.operator == '>':
                return value > self.threshold
            elif self.operator == '<':
                return value < self.threshold
            elif self.operator == '==':
                return abs(value - self.threshold) < 0.0001
            elif self.operator == '>=':
                return value >= self.threshold
            elif self.operator == '<=':
                return value <= self.threshold
            return False
        except:
            return False
            
    def trigger(self, cooldown_sec: int = 10):
      now = datetime.now()

      if self.last_triggered:
        if (now - self.last_triggered).total_seconds() < cooldown_sec:
          return  # prevent spam

      self.last_triggered = now
      self.trigger_count += 1

        
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'symbol': self.symbol,
            'metric': self.metric,
            'operator': self.operator,
            'threshold': self.threshold,
            'enabled': self.enabled,
            'trigger_count': self.trigger_count,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None
        }

class AlertEngine:
    """Manages alert rules and notifications."""
    
    def __init__(self, max_history: int = 1000):
        self.rules = {}  # rule_id -> AlertRule
        self.triggered_alerts = deque(maxlen=max_history)
        self.callbacks = []  # Functions to call when alert triggers
        self.running = False
        self.check_thread = None
        self._lock = threading.Lock()
        
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self._lock:
            self.rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.name}")
            
    def remove_rule(self, rule_id: str):
        """Remove an alert rule."""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")
                
    def enable_rule(self, rule_id: str):
        """Enable an alert rule."""
        with self._lock:
            if rule_id in self.rules:
                self.rules[rule_id].enabled = True
                
    def disable_rule(self, rule_id: str):
        """Disable an alert rule."""
        with self._lock:
            if rule_id in self.rules:
                self.rules[rule_id].enabled = False
                
    def add_callback(self, callback: Callable):
        """Add callback function to call when alert triggers."""
        self.callbacks.append(callback)
        
    def check_value(self, symbol: str, metric: str, value: float):
        """Check a metric value against all relevant rules."""
        with self._lock:
            for rule in self.rules.values():
                if rule.symbol == symbol and rule.metric == metric:
                    if rule.check(value):
                        rule.trigger()
                        
                        alert = {
                            'timestamp': datetime.now(),
                            'rule_id': rule.rule_id,
                            'rule_name': rule.name,
                            'symbol': symbol,
                            'metric': metric,
                            'value': value,
                            'threshold': rule.threshold,
                            'operator': rule.operator,
                            'message': f"{rule.name}: {metric} {rule.operator} {rule.threshold} (current: {value:.4f})"
                        }
                        
                        self.triggered_alerts.append(alert)
                        
                        # Call callbacks
                        for callback in self.callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                logger.error(f"Alert callback error: {e}")
                                
    def get_rules(self) -> List[Dict]:
        """Get all rules as dictionaries."""
        with self._lock:
            return [rule.to_dict() for rule in self.rules.values()]
            
    def get_alerts(self, n: int = 50) -> List[Dict]:
        """Get recent triggered alerts."""
        with self._lock:
            alerts = list(self.triggered_alerts)[-n:]
            return [{
                'timestamp': a['timestamp'].isoformat(),
                'rule_name': a['rule_name'],
                'symbol': a['symbol'],
                'metric': a['metric'],
                'value': a['value'],
                'threshold': a['threshold'],
                'operator': a['operator'],
                'message': a['message']
            } for a in alerts]
            
    def clear_alerts(self):
        """Clear alert history."""
        with self._lock:
            self.triggered_alerts.clear()
            
    def get_stats(self) -> Dict:
        """Get alert statistics."""
        with self._lock:
            total_rules = len(self.rules)
            enabled_rules = sum(1 for r in self.rules.values() if r.enabled)
            total_triggered = len(self.triggered_alerts)
            
            return {
                'total_rules': total_rules,
                'enabled_rules': enabled_rules,
                'disabled_rules': total_rules - enabled_rules,
                'total_alerts_triggered': total_triggered
            }

# Global alert engine instance
alert_engine = AlertEngine()