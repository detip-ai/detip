"""
Initialization Script
Used to create necessary directories and files
"""
import os
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directory(path):
    """Create directory"""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")
    else:
        logger.info(f"Directory already exists: {path}")

def create_file(path, content=""):
    """Create file"""
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Created file: {path}")
    else:
        logger.info(f"File already exists: {path}")

def create_init_file(directory):
    """Create __init__.py file"""
    init_path = os.path.join(directory, "__init__.py")
    create_file(init_path, '"""Initialize module"""')

def create_sample_data():
    """Create sample data"""
    # Create sample transaction data
    sample_transactions = [
        {
            "hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "from": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "to": "0x123456789012345678901234567890123456789a",
            "value": 1.5,
            "gas_used": 21000,
            "timestamp": datetime.now().isoformat(),
            "type": "transfer",
            "status": True
        },
        {
            "hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "from": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "to": "0x0123456789abcdef0123456789abcdef01234567",
            "value": 0.5,
            "gas_used": 150000,
            "timestamp": datetime.now().isoformat(),
            "type": "contract_interaction",
            "status": True
        }
    ]
    
    create_file(
        "defi_project/data/sample_transactions.json",
        json.dumps(sample_transactions, indent=2)
    )
    
    # Create sample user data
    sample_user = {
        "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        "transactions": sample_transactions,
        "balances": {
            "ETH": {
                "balance": 5.0,
                "value_usd": 15000.0,
                "token_address": None
            },
            "DAI": {
                "balance": 1000.0,
                "value_usd": 1000.0,
                "token_address": "0x6b175474e89094c44da98b954eedeac495271d0f"
            }
        },
        "defi_activities": [
            {
                "protocol": "Aave",
                "type": "loan",
                "subtype": "supply",
                "status": "active",
                "amount": 500.0,
                "amount_usd": 500.0,
                "timestamp": datetime.now().isoformat(),
                "tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
            }
        ],
        "offchain_data": {
            "credit_score": 750,
            "income": 75000,
            "employment_status": "employed",
            "education_level": "bachelor"
        }
    }
    
    create_file(
        "defi_project/data/sample_user.json",
        json.dumps(sample_user, indent=2, default=str)
    )
    
    # Create sample market data
    sample_market_data = {
        "prices": {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "DAI": 1.0,
            "USDC": 1.0,
            "USDT": 1.0,
            "AAVE": 300.0,
            "UNI": 20.0,
            "COMP": 150.0
        },
        "timestamp": datetime.now().isoformat()
    }
    
    create_file(
        "defi_project/data/sample_market_data.json",
        json.dumps(sample_market_data, indent=2, default=str)
    )

def main():
    """Main function"""
    logger.info("Starting project initialization...")
    
    # Create directory structure
    directories = [
        "defi_project/config",
        "defi_project/data",
        "defi_project/models/credit_scoring",
        "defi_project/models/trading",
        "defi_project/services/blockchain",
        "defi_project/services/data_collection",
        "defi_project/services/trading",
        "defi_project/services/credit_scoring",
        "defi_project/utils",
        "defi_project/api",
        "logs"
    ]
    
    for directory in directories:
        create_directory(directory)
        create_init_file(directory)
    
    # Create sample data
    create_sample_data()
    
    # Create log file
    create_file("logs/defi_app.log")
    
    logger.info("Project initialization completed")

if __name__ == "__main__":
    main() 