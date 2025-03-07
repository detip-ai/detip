import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingStrategy:
    """Trading Strategy Class, provides AI-driven trading strategy optimization and execution"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trading strategy
        
        Args:
            config: Configuration information
        """
        self.max_position_size = config.get('max_position_size', 0.1)
        self.stop_loss_percentage = config.get('stop_loss_percentage', 0.05)
        self.take_profit_percentage = config.get('take_profit_percentage', 0.1)
        self.update_interval = config.get('strategy_update_interval', 3600)
        self.risk_level = config.get('risk_level', 'medium')
        
        # Adjust parameters based on risk level
        self._adjust_params_by_risk()
        
        # Initialize models
        self.price_prediction_model = None
        self.trend_classification_model = None
        self.scaler = StandardScaler()
        
        # Store current positions
        self.current_positions = {}
        
        # Store trade history
        self.trade_history = []
        
        logger.info(f"Trading strategy initialized, risk level: {self.risk_level}")
    
    def _adjust_params_by_risk(self):
        """Adjust trading parameters based on risk level"""
        if self.risk_level == 'low':
            self.max_position_size = min(self.max_position_size, 0.05)
            self.stop_loss_percentage = min(self.stop_loss_percentage, 0.03)
            self.take_profit_percentage = max(self.take_profit_percentage, 0.15)
        elif self.risk_level == 'high':
            self.max_position_size = max(self.max_position_size, 0.2)
            self.stop_loss_percentage = max(self.stop_loss_percentage, 0.1)
            self.take_profit_percentage = min(self.take_profit_percentage, 0.05)
        
        logger.info(f"Adjusted parameters based on risk level ({self.risk_level}): "
                   f"Max position size={self.max_position_size}, "
                   f"Stop loss percentage={self.stop_loss_percentage}, "
                   f"Take profit percentage={self.take_profit_percentage}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature data
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            DataFrame containing features
        """
        # Ensure data is complete
        if df.empty:
            return pd.DataFrame()
        
        # Copy data to avoid modifying original data
        features = df.copy()
        
        # Calculate price changes
        features['price_change'] = features['close'].pct_change()
        features['price_change_1d'] = features['close'].pct_change(periods=1)
        features['price_change_3d'] = features['close'].pct_change(periods=3)
        features['price_change_7d'] = features['close'].pct_change(periods=7)
        
        # Calculate moving averages
        features['ma7'] = features['close'].rolling(window=7).mean()
        features['ma14'] = features['close'].rolling(window=14).mean()
        features['ma30'] = features['close'].rolling(window=30).mean()
        
        # Calculate moving average differences
        features['ma7_14_diff'] = features['ma7'] - features['ma14']
        features['ma7_30_diff'] = features['ma7'] - features['ma30']
        
        # Calculate volatility
        features['volatility_3d'] = features['price_change'].rolling(window=3).std()
        features['volatility_7d'] = features['price_change'].rolling(window=7).std()
        features['volatility_14d'] = features['price_change'].rolling(window=14).std()
        
        # Calculate volume changes
        features['volume_change'] = features['volume'].pct_change()
        features['volume_ma7'] = features['volume'].rolling(window=7).mean()
        features['volume_ma7_diff'] = features['volume'] / features['volume_ma7'] - 1
        
        # Calculate relative strength index (RSI)
        delta = features['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        features['ema12'] = features['close'].ewm(span=12, adjust=False).mean()
        features['ema26'] = features['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = features['ema12'] - features['ema26']
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def create_labels(self, df: pd.DataFrame, prediction_horizon: int = 3) -> pd.DataFrame:
        """
        Create label data
        
        Args:
            df: DataFrame containing price data
            prediction_horizon: Prediction time range (days)
            
        Returns:
            DataFrame containing labels
        """
        # Copy data
        labeled_df = df.copy()
        
        # Create future price change label (for regression)
        labeled_df['future_return'] = labeled_df['close'].pct_change(periods=prediction_horizon).shift(-prediction_horizon)
        
        # Create trend direction label (for classification)
        labeled_df['trend_direction'] = np.where(labeled_df['future_return'] > 0, 1, 0)
        
        # Remove NaN values
        labeled_df = labeled_df.dropna()
        
        return labeled_df
    
    def train_models(self, features_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Train prediction models
        
        Args:
            features_df: DataFrame containing features and labels
            test_size: Test set proportion
            random_state: Random seed
        """
        if features_df.empty:
            logger.error("Training data is empty, cannot train models")
            return
        
        # Prepare features and labels
        feature_columns = [
            'price_change', 'price_change_1d', 'price_change_3d', 'price_change_7d',
            'ma7_14_diff', 'ma7_30_diff', 'volatility_3d', 'volatility_7d', 'volatility_14d',
            'volume_change', 'volume_ma7_diff', 'rsi', 'macd', 'macd_signal', 'macd_hist'
        ]
        
        X = features_df[feature_columns]
        y_reg = features_df['future_return']
        y_cls = features_df['trend_direction']
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split training set and test set
        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
            X_scaled, y_reg, y_cls, test_size=test_size, random_state=random_state
        )
        
        # Train price prediction model (regression)
        logger.info("Training price prediction model...")
        self.price_prediction_model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        )
        self.price_prediction_model.fit(X_train, y_reg_train)
        
        # Evaluate price prediction model
        y_reg_pred = self.price_prediction_model.predict(X_test)
        reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
        logger.info(f"Price prediction model MSE: {reg_mse:.6f}")
        
        # Train trend classification model (classification)
        logger.info("Training trend classification model...")
        self.trend_classification_model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=random_state
        )
        self.trend_classification_model.fit(X_train, y_cls_train)
        
        # Evaluate trend classification model
        y_cls_pred = self.trend_classification_model.predict(X_test)
        cls_accuracy = accuracy_score(y_cls_test, y_cls_pred)
        logger.info(f"Trend classification model accuracy: {cls_accuracy:.4f}")
    
    def save_models(self, price_model_path: str = 'models/trading/price_model.pkl',
                   trend_model_path: str = 'models/trading/trend_model.pkl',
                   scaler_path: str = 'models/trading/scaler.pkl'):
        """
        Save trained models
        
        Args:
            price_model_path: Path to save price prediction model
            trend_model_path: Path to save trend classification model
            scaler_path: Path to save feature scaler
        """
        if self.price_prediction_model:
            joblib.dump(self.price_prediction_model, price_model_path)
            logger.info(f"Price prediction model saved to {price_model_path}")
        
        if self.trend_classification_model:
            joblib.dump(self.trend_classification_model, trend_model_path)
            logger.info(f"Trend classification model saved to {trend_model_path}")
        
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Feature scaler saved to {scaler_path}")
    
    def load_models(self, price_model_path: str = 'models/trading/price_model.pkl',
                   trend_model_path: str = 'models/trading/trend_model.pkl',
                   scaler_path: str = 'models/trading/scaler.pkl'):
        """
        Load trained models
        
        Args:
            price_model_path: Path to price prediction model
            trend_model_path: Path to trend classification model
            scaler_path: Path to feature scaler
        """
        try:
            self.price_prediction_model = joblib.load(price_model_path)
            logger.info(f"Loaded price prediction model: {price_model_path}")
            
            self.trend_classification_model = joblib.load(trend_model_path)
            logger.info(f"Loaded trend classification model: {trend_model_path}")
            
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded feature scaler: {scaler_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Use models to predict
        
        Args:
            features_df: DataFrame containing features
            
        Returns:
            Dictionary containing prediction results
        """
        if features_df.empty:
            logger.error("Feature data is empty, cannot make prediction")
            return {}
        
        if not self.price_prediction_model or not self.trend_classification_model:
            logger.error("Models not trained, cannot make prediction")
            return {}
        
        # Prepare features
        feature_columns = [
            'price_change', 'price_change_1d', 'price_change_3d', 'price_change_7d',
            'ma7_14_diff', 'ma7_30_diff', 'volatility_3d', 'volatility_7d', 'volatility_14d',
            'volume_change', 'volume_ma7_diff', 'rsi', 'macd', 'macd_signal', 'macd_hist'
        ]
        
        # Get latest feature data
        latest_features = features_df[feature_columns].iloc[-1:].values
        
        # Standardize features
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Predict price change
        predicted_return = self.price_prediction_model.predict(latest_features_scaled)[0]
        
        # Predict trend direction
        trend_probability = self.trend_classification_model.predict_proba(latest_features_scaled)[0]
        predicted_trend = self.trend_classification_model.predict(latest_features_scaled)[0]
        
        # Get current price
        current_price = features_df['close'].iloc[-1]
        
        # Calculate predicted price
        predicted_price = current_price * (1 + predicted_return)
        
        return {
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'predicted_return': float(predicted_return),
            'predicted_trend': int(predicted_trend),
            'up_probability': float(trend_probability[1]),
            'down_probability': float(trend_probability[0]),
            'prediction_time': datetime.now().isoformat()
        }
    
    def generate_trading_signal(self, prediction: Dict[str, Any], 
                               current_holdings: float = 0.0) -> Dict[str, Any]:
        """
        Generate trading signal based on prediction
        
        Args:
            prediction: Dictionary containing prediction results
            current_holdings: Current holdings quantity
            
        Returns:
            Dictionary containing trading signal
        """
        if not prediction:
            return {'action': 'hold', 'confidence': 0.0}
        
        # Extract prediction results
        predicted_return = prediction['predicted_return']
        predicted_trend = prediction['predicted_trend']
        up_probability = prediction['up_probability']
        current_price = prediction['current_price']
        
        # Set signal thresholds
        buy_threshold = 0.6  # Buy probability threshold
        sell_threshold = 0.6  # Sell probability threshold
        
        # Generate trading signal based on prediction results
        if predicted_trend == 1 and up_probability > buy_threshold:
            # Bullish signal
            if current_holdings <= 0:
                action = 'buy'
                confidence = up_probability
                # Calculate suggested amount of buy
                suggested_amount = self.max_position_size
            else:
                # Existing holdings, consider whether to increase
                if up_probability > 0.8:  # Very bullish
                    action = 'increase'
                    confidence = up_probability
                    suggested_amount = self.max_position_size - current_holdings
                else:
                    action = 'hold'
                    confidence = up_probability
                    suggested_amount = 0.0
        elif predicted_trend == 0 and (1 - up_probability) > sell_threshold:
            # Bearish signal
            if current_holdings > 0:
                action = 'sell'
                confidence = 1 - up_probability
                suggested_amount = current_holdings
            else:
                # No holdings, consider whether to short
                if self.risk_level == 'high' and (1 - up_probability) > 0.8:
                    action = 'short'
                    confidence = 1 - up_probability
                    suggested_amount = self.max_position_size
                else:
                    action = 'hold'
                    confidence = 1 - up_probability
                    suggested_amount = 0.0
        else:
            # Signal unclear, maintain current state
            action = 'hold'
            confidence = max(up_probability, 1 - up_probability)
            suggested_amount = 0.0
        
        # Calculate stop loss and take profit prices
        if action in ['buy', 'increase']:
            stop_loss_price = current_price * (1 - self.stop_loss_percentage)
            take_profit_price = current_price * (1 + self.take_profit_percentage)
        elif action == 'short':
            stop_loss_price = current_price * (1 + self.stop_loss_percentage)
            take_profit_price = current_price * (1 - self.take_profit_percentage)
        else:
            stop_loss_price = None
            take_profit_price = None
        
        return {
            'action': action,
            'confidence': float(confidence),
            'suggested_amount': float(suggested_amount),
            'stop_loss_price': float(stop_loss_price) if stop_loss_price else None,
            'take_profit_price': float(take_profit_price) if take_profit_price else None,
            'signal_time': datetime.now().isoformat()
        }
    
    def execute_trade(self, symbol: str, signal: Dict[str, Any], 
                     exchange_service: Any) -> Dict[str, Any]:
        """
        Execute trade
        
        Args:
            symbol: Trading pair, e.g., 'BTC/USDT'
            signal: Trading signal
            exchange_service: Exchange service instance
            
        Returns:
            Dictionary containing trade results
        """
        if not signal or signal['action'] == 'hold':
            return {'status': 'no_action', 'message': 'No trading action'}
        
        action = signal['action']
        amount = signal['suggested_amount']
        
        if amount <= 0:
            return {'status': 'invalid_amount', 'message': 'Invalid trade amount'}
        
        try:
            # Execute trade based on trading signal
            if action == 'buy':
                result = exchange_service.create_market_buy_order(symbol, amount)
                trade_type = 'buy'
            elif action == 'sell':
                result = exchange_service.create_market_sell_order(symbol, amount)
                trade_type = 'sell'
            elif action == 'increase':
                result = exchange_service.create_market_buy_order(symbol, amount)
                trade_type = 'increase'
            elif action == 'short':
                # Note: Shorting requires margin trading support from the exchange
                result = exchange_service.create_market_sell_order(symbol, amount, {'type': 'margin'})
                trade_type = 'short'
            else:
                return {'status': 'invalid_action', 'message': f'Invalid trading action: {action}'}
            
            # Record trade history
            trade_record = {
                'symbol': symbol,
                'type': trade_type,
                'amount': amount,
                'price': result.get('price', 0),
                'time': datetime.now().isoformat(),
                'stop_loss': signal.get('stop_loss_price'),
                'take_profit': signal.get('take_profit_price'),
                'confidence': signal.get('confidence'),
                'order_id': result.get('id')
            }
            
            self.trade_history.append(trade_record)
            
            # Update current holdings
            if symbol not in self.current_positions:
                self.current_positions[symbol] = 0
            
            if trade_type in ['buy', 'increase']:
                self.current_positions[symbol] += amount
            elif trade_type in ['sell', 'short']:
                self.current_positions[symbol] -= amount
            
            logger.info(f"Trade executed successfully: {trade_type} {amount} {symbol} @ {result.get('price', 'market')}")
            
            return {
                'status': 'success',
                'message': 'Trade executed successfully',
                'trade': trade_record
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return {
                'status': 'error',
                'message': f'Trade execution failed: {str(e)}'
            }
    
    def backtest_strategy(self, historical_data: pd.DataFrame, 
                         initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Backtest trading strategy
        
        Args:
            historical_data: Historical price data
            initial_capital: Initial capital
            
        Returns:
            Dictionary containing backtest results
        """
        if historical_data.empty:
            logger.error("Historical data is empty, cannot perform backtest")
            return {}
        
        # Prepare features and labels
        features_df = self.prepare_features(historical_data)
        labeled_df = self.create_labels(features_df)
        
        # Train models
        self.train_models(labeled_df)
        
        # Initialize backtest variables
        capital = initial_capital
        holdings = 0.0
        trades = []
        equity_curve = []
        
        # Iterate through historical data
        for i in range(100, len(features_df)):
            # Get current data
            current_data = features_df.iloc[:i]
            current_price = current_data['close'].iloc[-1]
            
            # Generate prediction
            prediction = self.predict(current_data)
            
            # Generate trading signal
            signal = self.generate_trading_signal(prediction, holdings)
            
            # Simulate trade execution
            if signal['action'] == 'buy' and capital > 0:
                # Calculate maximum amount that can be bought
                max_amount = capital / current_price
                amount = min(max_amount, signal['suggested_amount'])
                
                # Execute buy
                cost = amount * current_price
                capital -= cost
                holdings += amount
                
                trades.append({
                    'time': features_df.index[i],
                    'action': 'buy',
                    'price': current_price,
                    'amount': amount,
                    'cost': cost,
                    'capital': capital,
                    'holdings': holdings,
                    'equity': capital + holdings * current_price
                })
                
            elif signal['action'] == 'sell' and holdings > 0:
                # Execute sell
                amount = min(holdings, signal['suggested_amount'])
                revenue = amount * current_price
                capital += revenue
                holdings -= amount
                
                trades.append({
                    'time': features_df.index[i],
                    'action': 'sell',
                    'price': current_price,
                    'amount': amount,
                    'revenue': revenue,
                    'capital': capital,
                    'holdings': holdings,
                    'equity': capital + holdings * current_price
                })
            
            # Record equity curve
            equity = capital + holdings * current_price
            equity_curve.append({
                'time': features_df.index[i],
                'price': current_price,
                'capital': capital,
                'holdings': holdings,
                'equity': equity
            })
        
        # Calculate backtest metrics
        if equity_curve:
            initial_equity = equity_curve[0]['equity']
            final_equity = equity_curve[-1]['equity']
            total_return = (final_equity / initial_equity - 1) * 100
            
            # Calculate daily return rate
            equity_df = pd.DataFrame(equity_curve)
            equity_df['return'] = equity_df['equity'].pct_change()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            sharpe_ratio = equity_df['return'].mean() / equity_df['return'].std() * np.sqrt(252)
            
            # Calculate maximum drawdown
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['cummax'] - equity_df['equity']) / equity_df['cummax']
            max_drawdown = equity_df['drawdown'].max() * 100
            
            return {
                'initial_capital': initial_capital,
                'final_equity': final_equity,
                'total_return_pct': total_return,
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown_pct': float(max_drawdown),
                'trade_count': len(trades),
                'trades': trades,
                'equity_curve': equity_curve
            }
        else:
            return {
                'initial_capital': initial_capital,
                'final_equity': initial_capital,
                'total_return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'trade_count': 0,
                'trades': [],
                'equity_curve': []
            } 
