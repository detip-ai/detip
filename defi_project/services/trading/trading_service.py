"""
Trading Service Module
Provides trading strategy execution and order management functionality
"""
import logging
import os
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from defi_project.utils.helpers import save_json_file, load_json_file, format_datetime, retry

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class TradingService:
    """
    Trading Service Class
    Provides trading strategy execution and order management functionality
    """
    
    def __init__(self, config: Dict[str, Any], blockchain_service: Any, market_data_service: Any):
        """
        Initialize trading service
        
        Args:
            config: Configuration dictionary
            blockchain_service: Blockchain service instance
            market_data_service: Market data service instance
        """
        self.config = config
        self.blockchain_service = blockchain_service
        self.market_data_service = market_data_service
        
        # Data storage path
        self.data_dir = config.get('DATA_DIR', 'defi_project/data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Trade history file
        self.trade_history_file = os.path.join(self.data_dir, 'trade_history.json')
        
        # Trading strategy models
        self.price_model = None
        self.trend_model = None
        self.scaler = None
        
        # Trading parameters
        self.risk_level = config.get('RISK_LEVEL', 'medium')  # low, medium, high
        self.max_position_size = config.get('MAX_POSITION_SIZE', 0.5)  # Maximum position size ratio
        self.stop_loss_pct = config.get('STOP_LOSS_PCT', 0.05)  # Stop loss percentage
        self.take_profit_pct = config.get('TAKE_PROFIT_PCT', 0.1)  # Take profit percentage
        
        # Adjust parameters based on risk level
        self._adjust_params_by_risk()
        
        # Strategy execution thread
        self.strategy_thread = None
        self.is_running = False
        
        logger.info("Trading service initialized")
    
    def _adjust_params_by_risk(self):
        """
        Adjust trading parameters based on risk level
        """
        if self.risk_level == 'low':
            self.max_position_size = min(self.max_position_size, 0.3)
            self.stop_loss_pct = min(self.stop_loss_pct, 0.03)
            self.take_profit_pct = max(self.take_profit_pct, 0.08)
        elif self.risk_level == 'high':
            self.max_position_size = max(self.max_position_size, 0.7)
            self.stop_loss_pct = max(self.stop_loss_pct, 0.08)
            self.take_profit_pct = min(self.take_profit_pct, 0.15)
        
        logger.info(f"Trading parameters adjusted based on risk level ({self.risk_level}): "
                   f"Maximum position size={self.max_position_size}, "
                   f"Stop loss={self.stop_loss_pct*100}%, "
                   f"Take profit={self.take_profit_pct*100}%")
    
    def load_models(self) -> bool:
        """
        Load trading strategy models
        
        Returns:
            Whether models are successfully loaded
        """
        try:
            # Import model-related libraries
            from joblib import load
            import tensorflow as tf
            
            # Load models
            models_dir = os.path.join('defi_project', 'models', 'trading')
            
            # Load price prediction model
            price_model_path = os.path.join(models_dir, 'price_model.h5')
            if os.path.exists(price_model_path):
                self.price_model = tf.keras.models.load_model(price_model_path)
            
            # Load trend prediction model
            trend_model_path = os.path.join(models_dir, 'trend_model.pkl')
            if os.path.exists(trend_model_path):
                self.trend_model = load(trend_model_path)
            
            # Load feature scaler
            scaler_path = os.path.join(models_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = load(scaler_path)
            
            if self.price_model and self.trend_model and self.scaler:
                logger.info("Trading strategy models loaded successfully")
                return True
            else:
                logger.warning("Some models were not loaded, which may affect prediction accuracy")
                return False
        except Exception as e:
            logger.error(f"Failed to load trading strategy models: {str(e)}")
            return False
    
    def predict_price(self, symbol: str) -> Dict[str, Any]:
        """
        Predict price and trend
        
        Args:
            symbol: Trading pair
            
        Returns:
            Prediction result dictionary
        """
        try:
            # Check if models are loaded
            if not self.price_model or not self.trend_model or not self.scaler:
                if not self.load_models():
                    return {
                        'success': False,
                        'error': 'Models not loaded',
                        'timestamp': format_datetime(datetime.now())
                    }
            
            # Get historical data
            df = self.market_data_service.get_historical_ohlcv(symbol, '1d', limit=30)
            
            if df.empty:
                return {
                    'success': False,
                    'error': 'Unable to get historical data',
                    'timestamp': format_datetime(datetime.now())
                }
            
            # Calculate technical indicators
            indicators = self.market_data_service.calculate_technical_indicators(symbol)
            
            # Prepare features
            features = self._prepare_features(df, indicators)
            
            # Predict price
            price_prediction = self._predict_price(features)
            
            # Predict trend
            trend_prediction = self._predict_trend(features)
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            return {
                'success': True,
                'symbol': symbol,
                'current_price': float(current_price),
                'predicted_price': float(price_prediction),
                'price_change_pct': float((price_prediction / current_price - 1) * 100),
                'trend': trend_prediction,
                'confidence': 0.75,  # Example confidence
                'timestamp': format_datetime(datetime.now())
            }
        except Exception as e:
            logger.error(f"Failed to predict price: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': format_datetime(datetime.now())
            }
    
    def get_trading_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading signal
        
        Args:
            symbol: Trading pair
            
        Returns:
            Trading signal dictionary
        """
        try:
            # Get price prediction
            prediction = self.predict_price(symbol)
            
            if not prediction['success']:
                return {
                    'success': False,
                    'error': prediction.get('error', 'Prediction failed'),
                    'timestamp': format_datetime(datetime.now())
                }
            
            # Get current holdings
            current_holdings = self._get_current_holdings(symbol)
            
            # Generate trading signal
            price_change_pct = prediction['price_change_pct']
            trend = prediction['trend']
            
            # Generate signal based on prediction and current holdings
            if price_change_pct > 3 and trend == 'up':
                action = 'buy'
                confidence = min(0.5 + price_change_pct / 20, 0.95)
                size = self.max_position_size * confidence
            elif price_change_pct < -3 and trend == 'down':
                action = 'sell'
                confidence = min(0.5 + abs(price_change_pct) / 20, 0.95)
                size = current_holdings
            else:
                action = 'hold'
                confidence = 0.5
                size = 0
            
            return {
                'success': True,
                'symbol': symbol,
                'action': action,
                'size': float(size),
                'confidence': float(confidence),
                'current_price': prediction['current_price'],
                'predicted_price': prediction['predicted_price'],
                'price_change_pct': prediction['price_change_pct'],
                'trend': prediction['trend'],
                'current_holdings': float(current_holdings),
                'timestamp': format_datetime(datetime.now())
            }
        except Exception as e:
            logger.error(f"Failed to get trading signal: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': format_datetime(datetime.now())
            }
    
    def _get_current_holdings(self, symbol: str) -> float:
        """
        Get current holdings
        
        Args:
            symbol: Trading pair
            
        Returns:
            Holdings amount
        """
        try:
            # Extract token symbol from trading pair
            token = symbol.split('/')[0] if '/' in symbol else symbol
            
            # Get token balance
            balance = self.blockchain_service.get_token_balance(token)
            
            return balance
        except Exception as e:
            logger.error(f"Failed to get current holdings: {str(e)}")
            return 0.0
    
    def execute_trade(self, symbol: str, signal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute trade
        
        Args:
            symbol: Trading pair
            signal: Trading signal, if None, get signal automatically
            
        Returns:
            Trade result dictionary
        """
        try:
            # If signal is not provided, get signal
            if not signal:
                signal = self.get_trading_signal(symbol)
                
                if not signal['success']:
                    return {
                        'success': False,
                        'error': signal.get('error', 'Failed to get trading signal'),
                        'timestamp': format_datetime(datetime.now())
                    }
            
            # Get trade parameters
            action = signal['action']
            size = signal['size']
            current_price = signal['current_price']
            
            # If holding, do not execute trade
            if action == 'hold' or size <= 0:
                return {
                    'success': True,
                    'executed': False,
                    'reason': 'Signal is hold or trade size is zero',
                    'timestamp': format_datetime(datetime.now())
                }
            
            # Execute trade
            if action == 'buy':
                result = self._execute_buy(symbol, size, current_price)
            elif action == 'sell':
                result = self._execute_sell(symbol, size, current_price)
            else:
                return {
                    'success': False,
                    'error': f'Unknown trade action: {action}',
                    'timestamp': format_datetime(datetime.now())
                }
            
            # Record trade history
            self._record_trade(symbol, action, size, current_price, result)
            
            return result
        except Exception as e:
            logger.error(f"Failed to execute trade: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': format_datetime(datetime.now())
            }
    
    def execute_manual_trade(self, symbol: str, action: str, amount: float) -> Dict[str, Any]:
        """
        Execute manual trade
        
        Args:
            symbol: Trading pair
            action: Trade action (buy/sell)
            amount: Trade amount
            
        Returns:
            Trade result dictionary
        """
        try:
            # Validate parameters
            if action not in ['buy', 'sell']:
                return {
                    'success': False,
                    'error': f'Invalid trade action: {action}',
                    'timestamp': format_datetime(datetime.now())
                }
            
            if amount <= 0:
                return {
                    'success': False,
                    'error': 'Trade amount must be greater than 0',
                    'timestamp': format_datetime(datetime.now())
                }
            
            # Get current price
            current_price = self._get_current_price(symbol)
            
            if not current_price:
                return {
                    'success': False,
                    'error': 'Unable to get current price',
                    'timestamp': format_datetime(datetime.now())
                }
            
            # Execute trade
            if action == 'buy':
                result = self._execute_buy(symbol, amount, current_price)
            else:  # sell
                result = self._execute_sell(symbol, amount, current_price)
            
            # Record trade history
            self._record_trade(symbol, action, amount, current_price, result)
            
            return result
        except Exception as e:
            logger.error(f"Failed to execute manual trade: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': format_datetime(datetime.now())
            }
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price
        
        Args:
            symbol: Trading pair
            
        Returns:
            Current price
        """
        try:
            # Get current price
            prices = self.market_data_service.get_current_prices()
            
            # Check trading pair format
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            if symbol in prices:
                return prices[symbol]
            
            return None
        except Exception as e:
            logger.error(f"Failed to get current price: {str(e)}")
            return None
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get trade history
        
        Returns:
            Trade history list
        """
        try:
            if os.path.exists(self.trade_history_file):
                return load_json_file(self.trade_history_file)
            return []
        except Exception as e:
            logger.error(f"Failed to get trade history: {str(e)}")
            return []
    
    def start_strategy_execution(self, interval: int = 3600):
        """
        Start strategy execution
        
        Args:
            interval: Execution interval (seconds), default is 1 hour
        """
        if self.is_running:
            logger.warning("Strategy execution is already running")
            return
        
        self.is_running = True
        
        def strategy_task():
            while self.is_running:
                try:
                    logger.info("Starting strategy execution")
                    
                    # Get signals for all trading pairs
                    for symbol in self.market_data_service.default_symbols:
                        try:
                            # Get trading signal
                            signal = self.get_trading_signal(symbol)
                            
                            if signal['success']:
                                # If signal suggests trading, execute
                                if signal['action'] != 'hold':
                                    logger.info(f"Executing trade: {symbol} {signal['action']} {signal['size']}")
                                    self.execute_trade(symbol, signal)
                                else:
                                    logger.info(f"Signal is hold: {symbol}")
                            else:
                                logger.warning(f"Failed to get trading signal: {symbol}")
                            
                            # Avoid too frequent requests
                            time.sleep(5)
                        except Exception as e:
                            logger.error(f"Error processing trading pair {symbol}: {str(e)}")
                    
                    logger.info(f"Strategy execution completed, waiting {interval} seconds before executing again")
                except Exception as e:
                    logger.error(f"Error in strategy execution: {str(e)}")
                
                # Wait for next execution
                time.sleep(interval)
        
        # Start strategy thread
        self.strategy_thread = threading.Thread(target=strategy_task)
        self.strategy_thread.daemon = True
        self.strategy_thread.start()
        
        logger.info(f"Strategy execution started, interval: {interval} seconds")
    
    def stop_strategy_execution(self):
        """
        Stop strategy execution
        """
        if not self.is_running:
            logger.warning("Strategy execution is not running")
            return
        
        self.is_running = False
        if self.strategy_thread:
            self.strategy_thread.join(timeout=1)
        
        logger.info("Strategy execution stopped")
    
    def backtest_strategy(self, symbol: str, start_date: str, end_date: str, 
                         initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Backtest strategy
        
        Args:
            symbol: Trading pair
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial capital
            
        Returns:
            Backtest result dictionary
        """
        try:
            # Get historical data
            df = self.market_data_service.get_historical_ohlcv(symbol, '1d')
            
            if df.empty:
                return {
                    'success': False,
                    'error': 'Unable to get historical data',
                    'timestamp': format_datetime(datetime.now())
                }
            
            # Filter date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if df.empty:
                return {
                    'success': False,
                    'error': 'No data found in specified date range',
                    'timestamp': format_datetime(datetime.now())
                }
            
            # Backtest logic
            # Here, simplified processing, actual backtesting should be more complex
            
            # Return result
            return {
                'success': True,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'final_capital': initial_capital * 1.2,  # Example result
                'return_pct': 20.0,  # Example result
                'max_drawdown_pct': 10.0,  # Example result
                'sharpe_ratio': 1.5,  # Example result
                'trade_count': 10,  # Example result
                'win_rate': 0.6,  # Example result
                'timestamp': format_datetime(datetime.now())
            }
        except Exception as e:
            logger.error(f"Failed to backtest strategy: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': format_datetime(datetime.now())
            }
    
    def _prepare_features(self, df: pd.DataFrame, indicators: Dict[str, float]) -> np.ndarray:
        """
        Prepare model features
        
        Args:
            df: Historical data DataFrame
            indicators: Technical indicators dictionary
            
        Returns:
            Feature array
        """
        # Here, simplified processing, actual application should be more complex
        features = np.array([
            indicators.get('rsi', 50),
            indicators.get('ma7_position', 0),
            indicators.get('ma25_position', 0),
            indicators.get('macd', 0),
            indicators.get('bb_position', 0.5)
        ]).reshape(1, -1)
        
        # Apply scaling
        if self.scaler:
            features = self.scaler.transform(features)
        
        return features
    
    def _predict_price(self, features: np.ndarray) -> float:
        """
        Predict price
        
        Args:
            features: Feature array
            
        Returns:
            Predicted price
        """
        # If model is not loaded, return simulated prediction
        if not self.price_model:
            return 0.0
        
        # Use model to predict
        prediction = self.price_model.predict(features)
        
        return float(prediction[0][0])
    
    def _predict_trend(self, features: np.ndarray) -> str:
        """
        Predict trend
        
        Args:
            features: Feature array
            
        Returns:
            Trend prediction (up/down/sideways)
        """
        # If model is not loaded, return simulated prediction
        if not self.trend_model:
            return 'sideways'
        
        # Use model to predict
        prediction = self.trend_model.predict(features)
        
        # Convert prediction to trend
        if prediction[0] > 0.5:
            return 'up'
        elif prediction[0] < -0.5:
            return 'down'
        else:
            return 'sideways'
    
    def _execute_buy(self, symbol: str, size: float, price: float) -> Dict[str, Any]:
        """
        Execute buy
        
        Args:
            symbol: Trading pair
            size: Buy amount
            price: Current price
            
        Returns:
            Trade result dictionary
        """
        # Here, simplified processing, actual application should call exchange API or DEX
        # Simulate successful trade
        return {
            'success': True,
            'executed': True,
            'symbol': symbol,
            'action': 'buy',
            'size': size,
            'price': price,
            'total': size * price,
            'timestamp': format_datetime(datetime.now()),
            'tx_hash': f"0x{os.urandom(32).hex()}"  # Simulated trade hash
        }
    
    def _execute_sell(self, symbol: str, size: float, price: float) -> Dict[str, Any]:
        """
        Execute sell
        
        Args:
            symbol: Trading pair
            size: Sell amount
            price: Current price
            
        Returns:
            Trade result dictionary
        """
        # Here, simplified processing, actual application should call exchange API or DEX
        # Simulate successful trade
        return {
            'success': True,
            'executed': True,
            'symbol': symbol,
            'action': 'sell',
            'size': size,
            'price': price,
            'total': size * price,
            'timestamp': format_datetime(datetime.now()),
            'tx_hash': f"0x{os.urandom(32).hex()}"  # Simulated trade hash
        }
    
    def _record_trade(self, symbol: str, action: str, size: float, price: float, result: Dict[str, Any]):
        """
        Record trade history
        
        Args:
            symbol: Trading pair
            action: Trade action
            size: Trade amount
            price: Trade price
            result: Trade result
        """
        try:
            # Get existing trade history
            trade_history = self.get_trade_history()
            
            # Create trade record
            trade_record = {
                'symbol': symbol,
                'action': action,
                'size': size,
                'price': price,
                'total': size * price,
                'timestamp': format_datetime(datetime.now()),
                'success': result['success'],
                'tx_hash': result.get('tx_hash', '')
            }
            
            # Add to history record
            trade_history.append(trade_record)
            
            # Save to file
            save_json_file(self.trade_history_file, trade_history)
            
            logger.info(f"Trade record saved: {symbol} {action} {size} @ {price}")
        except Exception as e:
            logger.error(f"Failed to record trade history: {str(e)}")
    
    def create_market_buy_order(self, symbol: str, amount: float, params: Dict = None) -> Dict[str, Any]:
        """
        Create market buy order
        
        Args:
            symbol: Trading pair
            amount: Buy amount
            params: Additional parameters
            
        Returns:
            Order result dictionary
        """
        # Here, simplified processing, actual application should call exchange API or DEX
        price = self._get_current_price(symbol)
        
        if not price:
            return {
                'success': False,
                'error': 'Unable to get current price',
                'timestamp': format_datetime(datetime.now())
            }
        
        result = self._execute_buy(symbol, amount, price)
        
        if result['success']:
            self._record_trade(symbol, 'buy', amount, price, result)
        
        return result
    
    def create_market_sell_order(self, symbol: str, amount: float, params: Dict = None) -> Dict[str, Any]:
        """
        Create market sell order
        
        Args:
            symbol: Trading pair
            amount: Sell amount
            params: Additional parameters
            
        Returns:
            Order result dictionary
        """
        # Here, simplified processing, actual application should call exchange API or DEX
        price = self._get_current_price(symbol)
        
        if not price:
            return {
                'success': False,
                'error': 'Unable to get current price',
                'timestamp': format_datetime(datetime.now())
            }
        
        result = self._execute_sell(symbol, amount, price)
        
        if result['success']:
            self._record_trade(symbol, 'sell', amount, price, result)
        
        return result 