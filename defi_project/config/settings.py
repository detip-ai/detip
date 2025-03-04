"""
DeFi Smart Investment and Credit Scoring System Configuration File
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Blockchain configuration
BLOCKCHAIN_CONFIG = {
    'provider_uri': os.getenv('WEB3_PROVIDER_URI', 'https://mainnet.infura.io/v3/your-infura-key'),
    'private_key': os.getenv('PRIVATE_KEY', ''),
    'gas_limit': 3000000,
    'gas_price': 'auto',  # Auto set gas price
}

# Exchange configuration
EXCHANGE_CONFIG = {
    'api_key': os.getenv('EXCHANGE_API_KEY', ''),
    'api_secret': os.getenv('EXCHANGE_SECRET', ''),
    'exchange_id': os.getenv('EXCHANGE_ID', 'binance'),  # Default to Binance
}

# Data collection configuration
DATA_COLLECTION_CONFIG = {
    'price_update_interval': 60,  # Price data update interval (seconds)
    'historical_days': 30,  # Historical data retrieval days
    'tokens_to_track': [
        'ETH', 'BTC', 'USDT', 'USDC', 'DAI', 'AAVE', 'UNI', 'COMP'
    ],
}

# Trading strategy configuration
TRADING_CONFIG = {
    'max_position_size': 0.1,  # Maximum position size (percentage of total assets)
    'stop_loss_percentage': 0.05,  # Stop loss percentage
    'take_profit_percentage': 0.1,  # Take profit percentage
    'strategy_update_interval': 3600,  # Strategy update interval (seconds)
    'risk_level': 'medium',  # Risk level: low, medium, high
}

# Credit scoring configuration
CREDIT_SCORING_CONFIG = {
    'min_transactions': 10,  # Minimum required transactions
    'min_account_age_days': 30,  # Minimum account age (days)
    'score_update_interval': 86400,  # Score update interval (seconds)
    'score_range': (300, 850),  # Credit score range
    'model_path': 'models/credit_scoring/model.pkl',  # Model save path
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'secret_key': os.getenv('API_SECRET_KEY', 'your-secret-key'),
}

# Logging configuration
LOG_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'logs/defi_app.log',
    'max_log_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# Data storage configuration
DATA_STORAGE_CONFIG = {
    'database_uri': os.getenv('DATABASE_URI', 'sqlite:///data/defi_data.db'),
    'backup_interval': 86400,  # Backup interval (seconds)
} 