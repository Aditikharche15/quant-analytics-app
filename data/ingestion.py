import json
import threading
import time
from datetime import datetime
from typing import List, Callable
import websocket
from config import BINANCE_WS_URL, RECONNECT_DELAY
from utils.logger import get_logger
from data.storage import storage

logger = get_logger()

class BinanceWSCollector:
    """Collects real-time tick data from Binance WebSocket."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = [s.lower().strip() for s in symbols]
        self.connections = {}
        self.running = False
        self.threads = []
        self.tick_callbacks = []  # For real-time processing
        
    def add_callback(self, callback: Callable):
        """Add callback function called on each tick."""
        self.tick_callbacks.append(callback)
        
    def _on_message(self, ws, message, symbol):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            if data.get('e') == 'trade':
                tick = {
                    'timestamp': datetime.fromtimestamp(data['T'] / 1000).isoformat(),
                    'symbol': symbol,
                    'price': float(data['p']),
                    'size': float(data['q'])
                }
                
                # Store tick
                storage.insert_tick(tick)
                
                # Call callbacks for real-time processing
                for callback in self.tick_callbacks:
                    try:
                        callback(tick)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
        except Exception as e:
            logger.error(f"Message parsing error for {symbol}: {e}")
            
    def _on_error(self, ws, error, symbol):
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error for {symbol}: {error}")
        
    def _on_close(self, ws, close_status_code, close_msg, symbol):
        """Handle WebSocket closure."""
        logger.warning(f"WebSocket closed for {symbol}: {close_status_code} - {close_msg}")
        
        # Auto-reconnect if still running
        if self.running and symbol in self.symbols:
            logger.info(f"Reconnecting {symbol} in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)
            if self.running:
                self._connect_symbol(symbol)
                
    def _on_open(self, ws, symbol):
        """Handle WebSocket connection opened."""
        logger.info(f"WebSocket connected: {symbol}")
        
    def _connect_symbol(self, symbol: str):
        """Connect to WebSocket for a single symbol."""
        url = f"{BINANCE_WS_URL}/{symbol}@trade"
        
        ws = websocket.WebSocketApp(
            url,
            on_message=lambda ws, msg: self._on_message(ws, msg, symbol),
            on_error=lambda ws, err: self._on_error(ws, err, symbol),
            on_close=lambda ws, code, msg: self._on_close(ws, code, msg, symbol),
            on_open=lambda ws: self._on_open(ws, symbol)
        )
        
        self.connections[symbol] = ws
        
        # Run in separate thread
        thread = threading.Thread(
            target=ws.run_forever,
            kwargs={'reconnect': 5},
            daemon=True
        )
        thread.start()
        self.threads.append(thread)
        
    def start(self):
        """Start collecting data for all symbols."""
        if self.running:
            logger.warning("Collector already running")
            return
            
        self.running = True
        logger.info(f"Starting collection for symbols: {self.symbols}")
        
        for symbol in self.symbols:
            self._connect_symbol(symbol)
            time.sleep(0.5)  # Stagger connections
            
    def stop(self):
        """Stop all WebSocket connections."""
        self.running = False
        logger.info("Stopping collector...")
        
        for symbol, ws in self.connections.items():
            try:
                ws.close()
            except:
                pass
                
        self.connections.clear()
        logger.info("Collector stopped")
        
    def add_symbol(self, symbol: str):
        """Add a new symbol to collection."""
        symbol = symbol.lower().strip()
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            if self.running:
                self._connect_symbol(symbol)
                
    def remove_symbol(self, symbol: str):
        """Remove symbol from collection."""
        symbol = symbol.lower().strip()
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            if symbol in self.connections:
                self.connections[symbol].close()
                del self.connections[symbol]