"""
DeFi Smart Investment and Credit Scoring System
Main Application Entry Point
"""
import os
import logging
import threading
import time
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import configurations
from config.settings import (
    BLOCKCHAIN_CONFIG, EXCHANGE_CONFIG, DATA_COLLECTION_CONFIG,
    TRADING_CONFIG, CREDIT_SCORING_CONFIG, API_CONFIG, LOG_CONFIG
)

# Import services
from services.blockchain.blockchain_service import BlockchainService
from services.data_collection.market_data_service import MarketDataService
from services.trading.trading_service import TradingService
from services.credit_scoring.credit_scoring_service import CreditScoringService

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['log_level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = API_CONFIG['secret_key']

# Initialize services
blockchain_service = BlockchainService(BLOCKCHAIN_CONFIG)
market_data_service = MarketDataService(DATA_COLLECTION_CONFIG)
trading_service = TradingService(TRADING_CONFIG, blockchain_service, market_data_service)
credit_scoring_service = CreditScoringService(CREDIT_SCORING_CONFIG, blockchain_service)

# Global variables
services_initialized = False
data_collection_thread = None
trading_strategy_thread = None

def initialize_services():
    """Initialize all services"""
    global services_initialized
    
    try:
        logger.info("Initializing services...")
        
        # Load models
        trading_service.load_models()
        credit_scoring_service.load_model()
        
        services_initialized = True
        logger.info("Services initialization completed")
        
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        services_initialized = False

def start_data_collection():
    """Start data collection thread"""
    global data_collection_thread
    
    if data_collection_thread and data_collection_thread.is_alive():
        logger.info("Data collection thread is already running")
        return
    
    try:
        logger.info("Starting data collection thread...")
        data_collection_thread = threading.Thread(
            target=market_data_service.start_data_collection,
            daemon=True
        )
        data_collection_thread.start()
        logger.info("Data collection thread started")
        
    except Exception as e:
        logger.error(f"Error starting data collection thread: {str(e)}")

def start_trading_strategy():
    """Start trading strategy thread"""
    global trading_strategy_thread
    
    if trading_strategy_thread and trading_strategy_thread.is_alive():
        logger.info("Trading strategy thread is already running")
        return
    
    try:
        logger.info("Starting trading strategy thread...")
        trading_strategy_thread = threading.Thread(
            target=trading_service.start_strategy_execution,
            daemon=True
        )
        trading_strategy_thread.start()
        logger.info("Trading strategy thread started")
        
    except Exception as e:
        logger.error(f"Error starting trading strategy thread: {str(e)}")

# API routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'services_initialized': services_initialized,
        'data_collection_running': data_collection_thread is not None and data_collection_thread.is_alive(),
        'trading_strategy_running': trading_strategy_thread is not None and trading_strategy_thread.is_alive()
    })

@app.route('/api/market/prices', methods=['GET'])
def get_market_prices():
    """Get market prices endpoint"""
    try:
        prices = market_data_service.get_current_prices()
        return jsonify({
            'status': 'success',
            'data': prices
        })
    except Exception as e:
        logger.error(f"Error getting market prices: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/market/indicators/<symbol>', methods=['GET'])
def get_technical_indicators(symbol):
    """获取技术指标接口"""
    try:
        timeframe = request.args.get('timeframe', '1d')
        indicators = market_data_service.calculate_technical_indicators(symbol, timeframe)
        return jsonify({
            'status': 'success',
            'data': indicators
        })
    except Exception as e:
        logger.error(f"获取技术指标时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/trading/predict/<symbol>', methods=['GET'])
def predict_price(symbol):
    """价格预测接口"""
    try:
        prediction = trading_service.predict_price(symbol)
        return jsonify({
            'status': 'success',
            'data': prediction
        })
    except Exception as e:
        logger.error(f"预测价格时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/trading/signal/<symbol>', methods=['GET'])
def get_trading_signal(symbol):
    """获取交易信号接口"""
    try:
        signal = trading_service.get_trading_signal(symbol)
        return jsonify({
            'status': 'success',
            'data': signal
        })
    except Exception as e:
        logger.error(f"获取交易信号时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/trading/execute', methods=['POST'])
def execute_trade():
    """执行交易接口"""
    try:
        data = request.json
        symbol = data.get('symbol')
        action = data.get('action')
        amount = data.get('amount')
        
        if not symbol or not action or not amount:
            return jsonify({
                'status': 'error',
                'message': '缺少必要参数'
            }), 400
        
        result = trading_service.execute_manual_trade(symbol, action, amount)
        return jsonify({
            'status': 'success',
            'data': result
        })
    except Exception as e:
        logger.error(f"执行交易时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/trading/history', methods=['GET'])
def get_trade_history():
    """获取交易历史接口"""
    try:
        history = trading_service.get_trade_history()
        return jsonify({
            'status': 'success',
            'data': history
        })
    except Exception as e:
        logger.error(f"获取交易历史时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/credit/score/<address>', methods=['GET'])
def get_credit_score(address):
    """获取信用评分接口"""
    try:
        score = credit_scoring_service.get_user_score(address)
        return jsonify({
            'status': 'success',
            'data': score
        })
    except Exception as e:
        logger.error(f"获取信用评分时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/credit/calculate', methods=['POST'])
def calculate_credit_score():
    """计算信用评分接口"""
    try:
        user_data = request.json
        if not user_data or 'address' not in user_data:
            return jsonify({
                'status': 'error',
                'message': '缺少必要参数'
            }), 400
        
        score = credit_scoring_service.calculate_credit_score(user_data)
        return jsonify({
            'status': 'success',
            'data': score
        })
    except Exception as e:
        logger.error(f"计算信用评分时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/blockchain/balance/<address>', methods=['GET'])
def get_balance(address):
    """获取余额接口"""
    try:
        token_address = request.args.get('token')
        
        if token_address:
            balance = blockchain_service.get_token_balance(token_address, address)
            token_symbol = request.args.get('symbol', 'TOKEN')
            return jsonify({
                'status': 'success',
                'data': {
                    'address': address,
                    'token_address': token_address,
                    'token_symbol': token_symbol,
                    'balance': balance
                }
            })
        else:
            balance = blockchain_service.get_eth_balance(address)
            return jsonify({
                'status': 'success',
                'data': {
                    'address': address,
                    'eth_balance': balance
                }
            })
    except Exception as e:
        logger.error(f"获取余额时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/blockchain/transaction', methods=['POST'])
def send_transaction():
    """发送交易接口"""
    try:
        data = request.json
        to_address = data.get('to_address')
        amount = data.get('amount')
        token_address = data.get('token_address')
        
        if not to_address or not amount:
            return jsonify({
                'status': 'error',
                'message': '缺少必要参数'
            }), 400
        
        tx_hash = blockchain_service.send_transaction(
            to_address=to_address,
            amount=amount,
            token_address=token_address
        )
        
        return jsonify({
            'status': 'success',
            'data': {
                'tx_hash': tx_hash
            }
        })
    except Exception as e:
        logger.error(f"发送交易时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/start_services', methods=['POST'])
def start_services():
    """启动服务接口"""
    try:
        initialize_services()
        start_data_collection()
        start_trading_strategy()
        
        return jsonify({
            'status': 'success',
            'message': '服务已启动'
        })
    except Exception as e:
        logger.error(f"启动服务时出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def main():
    """Main function"""
    try:
        # Initialize services
        initialize_services()
        
        # Start data collection
        start_data_collection()
        
        # Start trading strategy
        start_trading_strategy()
        
        # Start API server
        app.run(
            host=API_CONFIG['host'],
            port=API_CONFIG['port'],
            debug=API_CONFIG['debug']
        )
    
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Error running program: {str(e)}")
    finally:
        logger.info("Program exiting")

if __name__ == '__main__':
    main() 