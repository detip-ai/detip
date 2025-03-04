# Detip Smart Investment and Credit Scoring System

This is a decentralized finance (Detip) application based on Python, integrating AI-driven smart investment trading and a credit scoring system. The system aims to provide users with intelligent management of their cryptocurrency assets and credit evaluation services based on blockchain data.

## Features

### Smart Investment and Trading
- **Market Data Analysis**: Collect and analyze cryptocurrency market data, including prices, trading volumes, market depth, etc.
- **Trading Strategy Optimization**: Use machine learning algorithms to optimize trading strategies, supporting multiple technical indicators.
- **Automated Trading Execution**: Integrate with DEX (Decentralized Exchanges) to execute automated trading.
- **Backtesting System**: Conduct backtests on trading strategies using historical data to evaluate their performance.
- **Risk Management**: Adjust trading parameters based on user risk preferences.

### Credit Scoring System
- **On-chain Data Analysis**: Analyze users' transaction history, asset holding status, Detip activities, etc., on the blockchain.
- **Off-chain Data Integration**: Combine traditional financial data (if available) to provide a comprehensive credit profile.
- **AI Credit Assessment Model**: Construct a credit scoring model based on multi-dimensional data, providing credit scores ranging from 300 to 850.
- **Loan Parameter Suggestions**: Provide personalized loan parameter suggestions based on credit scores.
- **Protocol Recommendation**: Recommend suitable Detip lending protocols for users of different credit levels.

## Technology Stack

- **Backend**: Python 3.8+
- **Blockchain Interaction**: Web3.py
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, TensorFlow/PyTorch
- **Data Storage**: JSON, CSV, SQLite
- **API Integration**: CCXT (cryptocurrency trading), Etherscan API
- **Configuration Management**: python-dotenv

## Project Structure

```
Detip_project/
├── config/                  # Configuration files
├── data/                    # Data storage
│   ├── market/              # Market data
│   ├── user/                # User data
│   └── model/               # Model data
├── models/                  # Model Detipnition and training
│   ├── credit_scoring/      # Credit scoring model
│   └── trading/             # Trading strategy model
├── services/                # Core services
│   ├── blockchain/          # Blockchain interaction
│   ├── data_collection/     # Data collection
│   ├── trading/             # Trading execution
│   └── credit_scoring/      # Credit scoring
├── utils/                   # Utility functions
├── api/                     # API interfaces
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── app.py                   # Application entry point
├── init.py                  # Initialization script
├── requirements.txt         # Dependency list
└── README.md                # Project description
```

## Core Module Descriptions

### Blockchain Service (BlockchainService)
Provides interaction functions with the Ethereum blockchain, including:
- Query ETH and ERC20 token balances
- Send transactions
- Obtain transaction history
- Interact with smart contracts
- Obtain network information and Gas estimates

### Market Data Service (MarketDataService)
Responsible for collecting and processing cryptocurrency market data:
- Obtain real-time prices and historical OHLCV data
- Calculate technical indicators (MA, MACD, RSI, etc.)
- Analyze market depth and trading volumes
- Assess market sentiment
- Collect Detip protocol statistics

### Trading Service (TradingService)
Executes trading strategies and manages orders:
- Load and use price prediction models
- Generate trading signals
- Execute automated trading
- Record trading history
- Support strategy backtesting

### Credit Scoring Service (CreditScoringService)
Evaluates user credit status based on blockchain data:
- Integrate on-chain and off-chain user data
- Use AI models to calculate credit scores
- Provide loan suggestions and risk assessments
- Recommend suitable Detip lending protocols

## Installation and Usage

1. Clone the repository
```bash
git clone https://github.com/yourusername/Detip-project.git
cd Detip-project
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment variables
Create a `.env` file and set the necessary API keys and blockchain node URLs.

4. Initialize the project
```bash
python init.py
```

5. Run the application
```bash
python app.py
```

## Configuration Description

Configure the following parameters in the `.env` file:
- `WEB3_PROVIDER_URI`: Ethereum node provider URL (Infura, Alchemy, etc.)
- `PRIVATE_KEY`: Private key for transactions (keep it secure)
- `ETHERSCAN_API_KEY`: Etherscan API key (for retrieving transaction history)
- `EXCHANGE_API_KEY`: Exchange API key (if required)
- `EXCHANGE_SECRET`: Exchange API secret (if required)
- `LOG_LEVEL`: Log level (DEBUG, INFO, WARNING, ERROR)
- `DATA_DIR`: Data storage directory

## Usage Example

### Get User Credit Score
```python
from Detip_project.services.credit_scoring.credit_scoring_service import CreditScoringService
from Detip_project.services.blockchain.blockchain_service import BlockchainService
from Detip_project.settings import get_settings

# Load configuration
config = get_settings()

# Initialize services
blockchain_service = BlockchainService(config)
credit_service = CreditScoringService(config, blockchain_service)

# Get user credit score
user_address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
credit_score = credit_service.calculate_credit_score(user_address)
print(f"User Credit Score: {credit_score}")

# Get loan recommendations
loan_recommendations = credit_service.get_loan_recommendations(user_address)
print(f"Loan Recommendations: {loan_recommendations}")
```

### Execute Trading Strategy
```python
from Detip_project.services.trading.trading_service import TradingService
from Detip_project.services.blockchain.blockchain_service import BlockchainService
from Detip_project.services.data_collection.market_data_service import MarketDataService
from Detip_project.settings import get_settings

# Load configuration
config = get_settings()

# Initialize services
blockchain_service = BlockchainService(config)
market_data_service = MarketDataService(config)
trading_service = TradingService(config, blockchain_service, market_data_service)

# Load models
trading_service.load_models()

# Get trading signal
symbol = "ETH/USDT"
signal = trading_service.get_trading_signal(symbol)
print(f"Trading Signal: {signal}")

# Execute trade
if signal['action'] != 'hold':
    result = trading_service.execute_trade(symbol, signal)
    print(f"Trade Result: {result}")

# Start automated trading strategy
trading_service.start_strategy_execution(interval=3600)  # Execute once per hour
```

## Contribution Guide

We welcome community contributions! If you wish to contribute to the project, please follow these steps:

1. Fork the project repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Important Notes

- This project is for educational and research purposes only.
- Cryptocurrency trading comes with risks; use automated trading functions with caution.
- Do not use test private keys in production environments.
- The credit scoring model requires sufficient data to provide accurate assessments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact us via:
- [GitHub](https://github.com/detip-ai/detip)
- [Twitter](https://x.com/AI_Detip)