"""
Credit Scoring Service
Integrates credit scoring model with blockchain data
"""
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from models.credit_scoring.credit_score_model import CreditScoreModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditScoringService:
    """
    Credit Scoring Service Class
    Integrates credit scoring model with blockchain data
    """
    
    def __init__(self, config: Dict[str, Any], blockchain_service: Any):
        """
        Initialize credit scoring service
        
        Args:
            config: Credit scoring configuration
            blockchain_service: Blockchain service instance
        """
        self.config = config
        self.blockchain_service = blockchain_service
        
        # Initialize credit scoring model
        self.credit_model = CreditScoreModel(config)
        
        # Store user data cache
        self.user_data_cache = {}
        
        # Store user score cache
        self.user_score_cache = {}
        
        logger.info("Credit scoring service initialized")
    
    def load_model(self) -> bool:
        """
        Load credit scoring model
        
        Returns:
            Whether the model was successfully loaded
        """
        try:
            success = self.credit_model.load_model()
            if success:
                logger.info("Credit scoring model loaded successfully")
            else:
                logger.warning("Credit scoring model failed to load, using new trained model")
                # Here you can add model training logic
            return success
        except Exception as e:
            logger.error(f"Error loading credit scoring model: {str(e)}")
            return False
    
    def get_user_data(self, address: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get user data
        
        Args:
            address: User address
            force_refresh: Whether to force refresh data
            
        Returns:
            User data dictionary
        """
        # Check cache
        if not force_refresh and address in self.user_data_cache:
            cache_time = self.user_data_cache[address].get('cache_time', datetime.now() - timedelta(days=1))
            if (datetime.now() - cache_time).total_seconds() < 3600:  # Cache for 1 hour
                return self.user_data_cache[address]
        
        try:
            # Get blockchain data
            transactions = self._get_user_transactions(address)
            balances = self._get_user_balances(address)
            defi_activities = self._get_user_defi_activities(address)
            
            # Get offchain data (if any)
            offchain_data = self._get_user_offchain_data(address)
            
            # Combine user data
            user_data = {
                'address': address,
                'transactions': transactions,
                'balances': balances,
                'defi_activities': defi_activities,
                'offchain_data': offchain_data,
                'cache_time': datetime.now()
            }
            
            # Update cache
            self.user_data_cache[address] = user_data
            
            return user_data
        
        except Exception as e:
            logger.error(f"Error getting user {address} data: {str(e)}")
            return {'address': address, 'error': str(e)}
    
    def _get_user_transactions(self, address: str) -> List[Dict[str, Any]]:
        """
        Get user transaction history
        
        Args:
            address: User address
            
        Returns:
            Transaction history list
        """
        try:
            # Get transaction history from blockchain service
            transactions = self.blockchain_service.get_transaction_history(address)
            
            # Process transaction data
            processed_txs = []
            for tx in transactions:
                # Determine transaction type
                tx_type = self._determine_transaction_type(tx)
                
                # Process timestamp
                if isinstance(tx.get('timestamp', None), int):
                    timestamp = datetime.fromtimestamp(tx['timestamp'])
                else:
                    timestamp = datetime.now()  # Default to current time
                
                processed_tx = {
                    'hash': tx.get('hash', ''),
                    'from': tx.get('from', ''),
                    'to': tx.get('to', ''),
                    'value': float(self.blockchain_service.web3.from_wei(tx.get('value', 0), 'ether')),
                    'gas_used': tx.get('gas', 0),
                    'timestamp': timestamp,
                    'type': tx_type,
                    'status': tx.get('status', True)
                }
                
                processed_txs.append(processed_tx)
            
            return processed_txs
        
        except Exception as e:
            logger.error(f"Error getting user {address} transaction history: {str(e)}")
            return []
    
    def _determine_transaction_type(self, tx: Dict[str, Any]) -> str:
        """
        Determine transaction type
        
        Args:
            tx: Transaction data
            
        Returns:
            Transaction type
        """
        # Here you can determine transaction type based on transaction data
        # For example: transfer, contract interaction, DEX transaction, etc.
        if tx.get('to') is None:
            return 'contract_creation'
        elif tx.get('input', '0x') != '0x':
            return 'contract_interaction'
        else:
            return 'transfer'
    
    def _get_user_balances(self, address: str) -> Dict[str, Dict[str, Any]]:
        """
        Get user balances
        
        Args:
            address: User address
            
        Returns:
            Balances dictionary
        """
        try:
            # Get ETH balance
            eth_balance = self.blockchain_service.get_eth_balance(address)
            
            # Here you should get all token balances of the user
            # In a real application, you might need to query all tokens the user holds
            # Here we simplify it to just get ETH and some common tokens
            
            balances = {
                'ETH': {
                    'balance': eth_balance,
                    'value_usd': eth_balance * self._get_token_price('ETH'),
                    'token_address': None
                }
            }
            
            # Get common token balances
            common_tokens = [
                {'symbol': 'DAI', 'address': '0x6b175474e89094c44da98b954eedeac495271d0f'},
                {'symbol': 'USDC', 'address': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'},
                {'symbol': 'USDT', 'address': '0xdac17f958d2ee523a2206206994597c13d831ec7'}
            ]
            
            for token in common_tokens:
                try:
                    token_balance = self.blockchain_service.get_token_balance(token['address'], address)
                    if token_balance > 0:
                        balances[token['symbol']] = {
                            'balance': token_balance,
                            'value_usd': token_balance * self._get_token_price(token['symbol']),
                            'token_address': token['address']
                        }
                except Exception as e:
                    logger.warning(f"Error getting {token['symbol']} balance: {str(e)}")
            
            return balances
        
        except Exception as e:
            logger.error(f"Error getting user {address} balances: {str(e)}")
            return {}
    
    def _get_token_price(self, symbol: str) -> float:
        """
        Get token price
        
        Args:
            symbol: Token symbol
            
        Returns:
            Token price (USD)
        """
        # Here you should get real-time price from price API
        # In demo project, using simulated price
        prices = {
            'ETH': 3000.0,
            'BTC': 50000.0,
            'DAI': 1.0,
            'USDC': 1.0,
            'USDT': 1.0,
            'AAVE': 300.0,
            'UNI': 20.0,
            'COMP': 150.0
        }
        
        return prices.get(symbol, 0.0)
    
    def _get_user_defi_activities(self, address: str) -> List[Dict[str, Any]]:
        """
        Get user DeFi activities
        
        Args:
            address: User address
            
        Returns:
            DeFi activities list
        """
        # Here you should get user activity data from various DeFi protocols
        # In demo project, using simulated data
        
        # Simulate some DeFi activities
        activities = []
        
        # Simulate loan activities
        loan_protocols = ['Aave', 'Compound', 'MakerDAO']
        loan_statuses = ['active', 'repaid', 'defaulted', 'overdue']
        loan_types = ['borrow', 'supply', 'repay']
        
        # Generate pseudo-random activities based on address
        address_hash = int(address[-4:], 16)  # Use last 4 characters of address as seed
        
        # Generate loan activities
        num_activities = (address_hash % 10) + 1  # 1-10 activities
        
        for i in range(num_activities):
            protocol_idx = (address_hash + i) % len(loan_protocols)
            status_idx = (address_hash + i * 2) % len(loan_statuses)
            type_idx = (address_hash + i * 3) % len(loan_types)
            
            amount = ((address_hash + i * 100) % 1000) + 100  # 100-1100
            
            activity = {
                'protocol': loan_protocols[protocol_idx],
                'type': 'loan',
                'subtype': loan_types[type_idx],
                'status': loan_statuses[status_idx],
                'amount': amount,
                'amount_usd': amount,  # Assume stablecoin
                'timestamp': datetime.now() - timedelta(days=(address_hash + i * 7) % 90),
                'tx_hash': f"0x{hex(address_hash + i)[2:].zfill(64)}"
            }
            
            activities.append(activity)
        
        return activities
    
    def _get_user_offchain_data(self, address: str) -> Dict[str, Any]:
        """
        Get user offchain data
        
        Args:
            address: User address
            
        Returns:
            Offchain data dictionary
        """
        # Here you should get user offchain data from external API or database
        # In demo project, using simulated data
        
        # Generate pseudo-random data based on address
        address_hash = int(address[-4:], 16)
        
        # Simulate traditional credit score (300-850)
        credit_score = 300 + (address_hash % 550)
        
        # Simulate income (20000-100000)
        income = 20000 + (address_hash % 80000)
        
        # Simulate employment status
        employment_statuses = ['employed', 'self_employed', 'unemployed', 'student']
        employment_idx = address_hash % len(employment_statuses)
        
        # Simulate education level
        education_levels = ['high_school', 'bachelor', 'master', 'phd']
        education_idx = (address_hash // 10) % len(education_levels)
        
        return {
            'credit_score': credit_score,
            'income': income,
            'employment_status': employment_statuses[employment_idx],
            'education_level': education_levels[education_idx]
        }
    
    def calculate_credit_score(self, user_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate user credit score
        
        Args:
            user_data: User data, if None get data
            
        Returns:
            Credit score result
        """
        try:
            # If no user data provided, get data
            if not user_data or 'address' not in user_data:
                logger.error("No valid user data provided")
                return {'status': 'error', 'message': 'No valid user data provided'}
            
            address = user_data.get('address')
            
            # If only address provided, get full user data
            if len(user_data) == 1:
                user_data = self.get_user_data(address)
            
            # Use model to predict credit score
            score_result = self.credit_model.predict_credit_score(user_data)
            
            # Update user score cache
            if score_result.get('status') == 'success':
                self.user_score_cache[address] = score_result
            
            return score_result
        
        except Exception as e:
            logger.error(f"Error calculating credit score for user {user_data.get('address', 'unknown')}: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_user_score(self, address: str, force_recalculate: bool = False) -> Dict[str, Any]:
        """
        Get user credit score
        
        Args:
            address: User address
            force_recalculate: Whether to force recalculate
            
        Returns:
            Credit score result
        """
        try:
            # Check cache
            if not force_recalculate and address in self.user_score_cache:
                cache_time = datetime.fromisoformat(self.user_score_cache[address].get('timestamp', '2000-01-01T00:00:00'))
                if (datetime.now() - cache_time).total_seconds() < 86400:  # Cache for 1 day
                    return self.user_score_cache[address]
            
            # Get user data
            user_data = self.get_user_data(address, force_refresh=force_recalculate)
            
            # Calculate credit score
            score_result = self.calculate_credit_score(user_data)
            
            return score_result
        
        except Exception as e:
            logger.error(f"Error getting user {address} credit score: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_loan_recommendations(self, address: str) -> Dict[str, Any]:
        """
        Get loan recommendations
        
        Args:
            address: User address
            
        Returns:
            Loan recommendations
        """
        try:
            # Get user credit score
            score_result = self.get_user_score(address)
            
            if score_result.get('status') != 'success':
                return {'status': 'error', 'message': 'Unable to get credit score'}
            
            # Get loan parameters
            loan_params = score_result.get('loan_parameters', {})
            
            # Get user balances
            user_data = self.get_user_data(address)
            balances = user_data.get('balances', {})
            
            # Calculate total asset value
            total_assets = sum(balance.get('value_usd', 0) for balance in balances.values())
            
            # Generate loan recommendations
            recommendations = {
                'credit_score': score_result.get('score'),
                'credit_grade': score_result.get('grade'),
                'risk_level': score_result.get('risk_assessment', {}).get('risk_level'),
                'max_loan_amount': loan_params.get('max_loan_amount'),
                'interest_rate': loan_params.get('interest_rate'),
                'collateral_requirement': loan_params.get('collateral_requirement'),
                'max_term_days': loan_params.get('max_term_days'),
                'total_assets': total_assets,
                'recommended_protocols': self._get_recommended_protocols(score_result)
            }
            
            return {
                'status': 'success',
                'data': recommendations
            }
        
        except Exception as e:
            logger.error(f"Error getting user {address} loan recommendations: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _get_recommended_protocols(self, score_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get recommended DeFi protocols
        
        Args:
            score_result: Credit score result
            
        Returns:
            Recommended protocols list
        """
        risk_level = score_result.get('risk_assessment', {}).get('risk_level', 'high')
        
        # Recommend different protocols based on risk level
        if risk_level in ['very_low', 'low']:
            # Low risk users can use all protocols
            return [
                {
                    'name': 'Aave',
                    'type': 'lending',
                    'url': 'https://aave.com',
                    'description': 'Decentralized lending platform, providing lending services for various assets'
                },
                {
                    'name': 'Compound',
                    'type': 'lending',
                    'url': 'https://compound.finance',
                    'description': 'Automated interest rate protocol, allowing users to lend and borrow encrypted assets'
                },
                {
                    'name': 'MakerDAO',
                    'type': 'lending',
                    'url': 'https://makerdao.com',
                    'description': 'Decentralized stablecoin system, allowing users to generate DAI through collateral'
                }
            ]
        elif risk_level == 'medium':
            # Medium risk users recommend mainstream protocols
            return [
                {
                    'name': 'Aave',
                    'type': 'lending',
                    'url': 'https://aave.com',
                    'description': 'Decentralized lending platform, providing lending services for various assets'
                },
                {
                    'name': 'Compound',
                    'type': 'lending',
                    'url': 'https://compound.finance',
                    'description': 'Automated interest rate protocol, allowing users to lend and borrow encrypted assets'
                }
            ]
        else:
            # High risk users only recommend high collateral rate protocols
            return [
                {
                    'name': 'MakerDAO',
                    'type': 'lending',
                    'url': 'https://makerdao.com',
                    'description': 'Decentralized stablecoin system, allowing users to generate DAI through collateral'
                }
            ] 