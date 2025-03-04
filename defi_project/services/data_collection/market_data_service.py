"""
Market Data Service Module
Provides cryptocurrency market data collection and processing functionality
"""
import logging
import os
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import threading
import datetime
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
import requests
from dotenv import load_dotenv

from defi_project.utils.helpers import save_json_file, load_json_file, format_datetime

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class MarketDataService:
    """
    Market Data Service Class
    Provides cryptocurrency market data collection and processing functionality
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize market data service
        
        Args:
            config: Configuration dictionary containing exchange API information
        """
        self.config = config
        self.exchange_id = config.get('EXCHANGE_ID', os.getenv('EXCHANGE_ID', 'binance'))
        self.api_key = config.get('EXCHANGE_API_KEY', os.getenv('EXCHANGE_API_KEY'))
        self.api_secret = config.get('EXCHANGE_SECRET', os.getenv('EXCHANGE_SECRET'))
        
        # Data storage path
        self.data_dir = config.get('DATA_DIR', 'defi_project/data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize exchange
        self.exchange = self._init_exchange()
        
        # Default trading pairs
        self.default_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
        
        # Data collection thread
        self.collection_thread = None
        self.is_collecting = False
        
        # Cache
        self.price_cache = {}
        self.price_cache_time = {}
        self.price_cache_ttl = 60  # seconds
        
        logger.info(f"Market data service initialized, using exchange: {self.exchange_id}")
    
    def _init_exchange(self) -> ccxt.Exchange:
        """
        Initialize exchange
        
        Returns:
            Exchange instance
        """
        try:
            # Get exchange class
            exchange_class = getattr(ccxt, self.exchange_id)
            
            # Create exchange instance
            exchange_params = {
                'enableRateLimit': True,  # Enable request rate limit
            }
            
            if self.api_key and self.api_secret:
                exchange_params.update({
                    'apiKey': self.api_key,
                    'secret': self.api_secret
                })
            
            exchange = exchange_class(exchange_params)
            
            # Load markets
            exchange.load_markets()
            
            return exchange
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {str(e)}")
            raise
    
    def get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices
        
        Returns:
            Price dictionary, key is trading pair, value is price
        """
        try:
            result = {}
            
            # Check if there is a valid cache
            current_time = time.time()
            if self.price_cache and all(
                current_time - self.price_cache_time.get(symbol, 0) < self.price_cache_ttl 
                for symbol in self.default_symbols
            ):
                # Use cache
                for symbol in self.default_symbols:
                    if symbol in self.price_cache:
                        result[symbol] = self.price_cache[symbol]
                return result
            
            # Get tickers for all trading pairs
            tickers = self.exchange.fetch_tickers(self.default_symbols)
            
            # Extract prices
            for symbol in self.default_symbols:
                if symbol in tickers:
                    price = tickers[symbol]['last']
                    result[symbol] = price
                    
                    # Update cache
                    self.price_cache[symbol] = price
                    self.price_cache_time[symbol] = current_time
            
            return result
        except Exception as e:
            logger.error(f"Failed to get current prices: {str(e)}")
            # If there is a cache, return cached prices
            if self.price_cache:
                logger.info("Using cached price data")
                return {s: p for s, p in self.price_cache.items() if s in self.default_symbols}
            raise
    
    def get_historical_ohlcv(self, symbol: str, timeframe: str = '1d', 
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get historical K-line data
        
        Args:
            symbol: Trading pair
            timeframe: Time period, e.g., 1m, 5m, 15m, 1h, 4h, 1d
            limit: Number of K-lines to get
            
        Returns:
            K-line data DataFrame
        """
        try:
            # Check trading pair format
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            # Get K-line data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to date time
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Save to file
            file_path = os.path.join(self.data_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
            df.to_csv(file_path)
            
            return df
        except Exception as e:
            logger.error(f"Failed to get historical K-line data: {str(e)}")
            
            # Try loading from file
            try:
                file_path = os.path.join(self.data_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
                if os.path.exists(file_path):
                    logger.info(f"Loading historical data from file: {file_path}")
                    df = pd.read_csv(file_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    return df
            except Exception as load_error:
                logger.error(f"Failed to load historical data from file: {str(load_error)}")
            
            raise
    
    def get_market_depth(self, symbol: str, limit: int = 20) -> Dict[str, List]:
        """
        Get market depth
        
        Args:
            symbol: Trading pair
            limit: Depth limit
            
        Returns:
            Market depth dictionary, containing bids and asks
        """
        try:
            # Check trading pair format
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            # Get order book
            order_book = self.exchange.fetch_order_book(symbol, limit)
            
            return {
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'timestamp': order_book['timestamp'],
                'datetime': order_book['datetime'],
                'symbol': symbol
            }
        except Exception as e:
            logger.error(f"Failed to get market depth: {str(e)}")
            raise
    
    def get_trading_volume(self, symbol: str, timeframe: str = '1d', days: int = 7) -> float:
        """
        Get trading volume
        
        Args:
            symbol: Trading pair
            timeframe: Time period
            days: Number of days
            
        Returns:
            Trading volume (unit: base currency)
        """
        try:
            # Get historical K-line data
            df = self.get_historical_ohlcv(symbol, timeframe, limit=days)
            
            # Calculate total trading volume
            total_volume = df['volume'].sum()
            
            return float(total_volume)
        except Exception as e:
            logger.error(f"Failed to get trading volume: {str(e)}")
            raise
    
    def get_market_sentiment(self, tokens: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get market sentiment
        Calculate simple market sentiment indicators based on price changes and trading volume
        
        Args:
            tokens: List of tokens, default is default trading pairs
            
        Returns:
            Market sentiment dictionary, values between -1 and 1, -1 represents extreme fear, 1 represents extreme greed
        """
        if not tokens:
            tokens = [s.split('/')[0] for s in self.default_symbols]
        
        result = {}
        
        try:
            for token in tokens:
                symbol = f"{token}/USDT"
                
                # Get historical data
                df = self.get_historical_ohlcv(symbol, '1d', limit=14)
                
                if df.empty:
                    continue
                
                # Calculate price change
                price_change = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
                
                # Calculate trading volume change
                volume_change = (df['volume'].iloc[-1] / df['volume'].iloc[0]) - 1
                
                # Calculate volatility
                volatility = df['close'].pct_change().std()
                
                # Calculate RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                # Calculate sentiment indicator
                sentiment = 0.4 * np.tanh(price_change * 5) + 0.3 * np.tanh(volume_change * 2) + 0.3 * ((rsi.iloc[-1] / 100) * 2 - 1)
                
                # Limit between -1 and 1
                sentiment = max(min(sentiment, 1), -1)
                
                result[token] = float(sentiment)
            
            # Calculate overall market sentiment
            if result:
                result['MARKET'] = sum(result.values()) / len(result)
            
            return result
        except Exception as e:
            logger.error(f"Failed to get market sentiment: {str(e)}")
            return {'MARKET': 0}  # Default neutral
    
    def get_defi_stats(self) -> Dict[str, Any]:
        """
        Get DeFi statistics
        Use simulated data, in actual applications, use DeFi Pulse API or other data sources
        
        Returns:
            DeFi statistics dictionary
        """
        try:
            # Here use simulated data, in actual applications, should use real API
            stats = {
                'total_value_locked': 45.67,  # Unit: ten billion dollars
                'ethereum_dominance': 62.5,   # Unit: %
                'top_protocols': [
                    {'name': 'Aave', 'tvl': 6.2, 'change_24h': 1.5},
                    {'name': 'MakerDAO', 'tvl': 5.8, 'change_24h': -0.8},
                    {'name': 'Curve', 'tvl': 4.9, 'change_24h': 0.3},
                    {'name': 'Compound', 'tvl': 3.7, 'change_24h': -1.2},
                    {'name': 'Uniswap', 'tvl': 3.5, 'change_24h': 2.1}
                ],
                'categories': {
                    'lending': 18.2,
                    'dex': 12.5,
                    'derivatives': 5.8,
                    'assets': 4.3,
                    'others': 4.9
                },
                'updated_at': datetime.now().isoformat()
            }
            
            # Save to file
            file_path = os.path.join(self.data_dir, 'defi_stats.json')
            save_json_file(file_path, stats)
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get DeFi statistics: {str(e)}")
            
            # Try loading from file
            try:
                file_path = os.path.join(self.data_dir, 'defi_stats.json')
                if os.path.exists(file_path):
                    return load_json_file(file_path)
            except:
                pass
            
            return {
                'total_value_locked': 0,
                'ethereum_dominance': 0,
                'top_protocols': [],
                'categories': {},
                'updated_at': datetime.now().isoformat()
            }
    
    def calculate_technical_indicators(self, symbol: str, timeframe: str = '1d') -> Dict[str, float]:
        """
        Calculate technical indicators
        
        Args:
            symbol: Trading pair
            timeframe: Time period
            
        Returns:
            Technical indicators dictionary
        """
        try:
            # Get historical data
            df = self.get_historical_ohlcv(symbol, timeframe, limit=100)
            
            if df.empty:
                return {}
            
            # Calculate moving averages
            df['MA7'] = df['close'].rolling(window=7).mean()
            df['MA25'] = df['close'].rolling(window=25).mean()
            df['MA99'] = df['close'].rolling(window=99).mean()
            
            # Calculate MACD
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['Histogram'] = df['MACD'] - df['Signal']
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands
            df['Middle_Band'] = df['close'].rolling(window=20).mean()
            df['STD'] = df['close'].rolling(window=20).std()
            df['Upper_Band'] = df['Middle_Band'] + (df['STD'] * 2)
            df['Lower_Band'] = df['Middle_Band'] - (df['STD'] * 2)
            
            # Calculate ATR
            df['TR'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['ATR'] = df['TR'].rolling(window=14).mean()
            
            # Get latest value
            latest = df.iloc[-1]
            
            # Calculate price position relative to moving averages
            ma7_position = (latest['close'] / latest['MA7'] - 1) * 100
            ma25_position = (latest['close'] / latest['MA25'] - 1) * 100
            ma99_position = (latest['close'] / latest['MA99'] - 1) * 100
            
            # Calculate Bollinger Bands position
            bb_position = (latest['close'] - latest['Lower_Band']) / (latest['Upper_Band'] - latest['Lower_Band'])
            
            # Return result
            return {
                'price': float(latest['close']),
                'ma7': float(latest['MA7']),
                'ma25': float(latest['MA25']),
                'ma99': float(latest['MA99']),
                'ma7_position': float(ma7_position),
                'ma25_position': float(ma25_position),
                'ma99_position': float(ma99_position),
                'macd': float(latest['MACD']),
                'macd_signal': float(latest['Signal']),
                'macd_histogram': float(latest['Histogram']),
                'rsi': float(latest['RSI']),
                'upper_band': float(latest['Upper_Band']),
                'middle_band': float(latest['Middle_Band']),
                'lower_band': float(latest['Lower_Band']),
                'bb_position': float(bb_position),
                'atr': float(latest['ATR']),
                'atr_percent': float(latest['ATR'] / latest['close'] * 100),
                'updated_at': format_datetime(datetime.now())
            }
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {str(e)}")
            return {}
    
    def start_data_collection(self, interval: Optional[int] = None):
        """
        Start data collection
        
        Args:
            interval: Collection interval (seconds), default is 300 seconds (5 minutes)
        """
        if self.is_collecting:
            logger.warning("Data collection is already running")
            return
        
        if interval is None:
            interval = 300  # Default 5 minutes
        
        self.is_collecting = True
        
        def collection_task():
            while self.is_collecting:
                try:
                    logger.info("Starting to collect market data")
                    
                    # Collect current prices
                    prices = self.get_current_prices()
                    
                    # Collect historical data
                    for symbol in self.default_symbols:
                        self.get_historical_ohlcv(symbol, '1d', limit=30)
                        time.sleep(1)  # Avoid too frequent requests
                    
                    # Collect DeFi statistics
                    self.get_defi_stats()
                    
                    # Calculate market sentiment
                    self.get_market_sentiment()
                    
                    logger.info(f"Data collection completed, waiting {interval} seconds before collecting again")
                except Exception as e:
                    logger.error(f"Error occurred during data collection: {str(e)}")
                
                # Wait for next collection
                time.sleep(interval)
        
        # Start collection thread
        self.collection_thread = threading.Thread(target=collection_task)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        logger.info(f"Data collection started, interval: {interval} seconds")
    
    def stop_data_collection(self):
        """
        Stop data collection
        """
        if not self.is_collecting:
            logger.warning("Data collection is not running")
            return
        
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1)
        
        logger.info("Data collection stopped")
    
    def export_data_to_csv(self, symbol: str, timeframe: str = '1d', 
                          filename: Optional[str] = None) -> str:
        """
        Export data to CSV file
        
        Args:
            symbol: Trading pair
            timeframe: Time period
            filename: File name, default is auto-generated
            
        Returns:
            File path
        """
        try:
            # Get historical data
            df = self.get_historical_ohlcv(symbol, timeframe)
            
            # Generate file name
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{symbol.replace('/', '_')}_{timeframe}_{timestamp}.csv"
            
            # Ensure file name has .csv suffix
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Export to file
            file_path = os.path.join(self.data_dir, filename)
            df.to_csv(file_path)
            
            logger.info(f"Data exported to: {file_path}")
            
            return file_path
        except Exception as e:
            logger.error(f"Failed to export data: {str(e)}")
            raise 