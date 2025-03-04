"""
Blockchain Service Module
Provides interaction functionality with Ethereum blockchain
"""
import json
import logging
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from web3 import Web3, HTTPProvider
from web3.exceptions import TransactionNotFound, BlockNotFound, ContractLogicError
from web3.middleware import geth_poa_middleware
import time
from dotenv import load_dotenv
import requests

from defi_project.utils.helpers import is_valid_ethereum_address, retry, format_number

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class BlockchainService:
    """
    Blockchain Service Class
    Provides interaction functionality with Ethereum blockchain
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize blockchain service
        
        Args:
            config: Configuration dictionary containing blockchain connection information
        """
        self.config = config
        self.provider_uri = config.get('WEB3_PROVIDER_URI', os.getenv('WEB3_PROVIDER_URI'))
        self.private_key = config.get('PRIVATE_KEY', os.getenv('PRIVATE_KEY'))
        
        if not self.provider_uri:
            raise ValueError("Web3 provider URI not provided, please set WEB3_PROVIDER_URI in config or environment variables")
        
        # Initialize Web3 connection
        self.w3 = Web3(HTTPProvider(self.provider_uri))
        
        # Support POA chains (like BSC, Polygon, etc.)
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Check connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Unable to connect to Ethereum node: {self.provider_uri}")
        
        # Set account
        if self.private_key:
            self.account = self.w3.eth.account.from_key(self.private_key)
            self.address = self.account.address
            logger.info(f"Account set: {self.address}")
        else:
            self.account = None
            self.address = None
            logger.warning("Private key not provided, some functions will be unavailable")
        
        # Common contract ABI
        self.erc20_abi = json.loads('''[{"constant":true,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_from","type":"address"},{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transferFrom","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"payable":true,"stateMutability":"payable","type":"fallback"},{"anonymous":false,"inputs":[{"indexed":true,"name":"owner","type":"address"},{"indexed":true,"name":"spender","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"name":"from","type":"address"},{"indexed":true,"name":"to","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],"name":"Transfer","type":"event"}]''')
        
        # Common token addresses
        self.token_addresses = {
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
            'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
            'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
            'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599'
        }
        
        logger.info(f"Blockchain service initialized, connected to: {self.provider_uri}")
    
    def get_eth_balance(self, address: Optional[str] = None) -> float:
        """
        Get ETH Balance
        
        Args:
            address: Address to query, defaults to current account address
            
        Returns:
            ETH balance (unit: ETH)
        """
        if not address:
            if not self.address:
                raise ValueError("Address not provided and default account not set")
            address = self.address
        
        if not is_valid_ethereum_address(address):
            raise ValueError(f"Invalid Ethereum address: {address}")
        
        try:
            balance_wei = self.w3.eth.get_balance(address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            return float(balance_eth)
        except Exception as e:
            logger.error(f"Failed to get ETH balance: {str(e)}")
            raise
    
    def get_token_balance(self, token_address: str, address: Optional[str] = None) -> float:
        """
        Get ERC20 Token Balance
        
        Args:
            token_address: Token contract address or token symbol (USDT, USDC, DAI, WETH, WBTC)
            address: Address to query, defaults to current account address
            
        Returns:
            Token balance (already considering decimal places)
        """
        if not address:
            if not self.address:
                raise ValueError("Address not provided and default account not set")
            address = self.address
        
        if not is_valid_ethereum_address(address):
            raise ValueError(f"Invalid Ethereum address: {address}")
        
        # Check if using token symbol
        if token_address in self.token_addresses:
            token_address = self.token_addresses[token_address]
        
        if not is_valid_ethereum_address(token_address):
            raise ValueError(f"Invalid token address: {token_address}")
        
        try:
            # Create contract instance
            token_contract = self.w3.eth.contract(address=token_address, abi=self.erc20_abi)
            
            # Get token decimal places
            decimals = token_contract.functions.decimals().call()
            
            # Get balance
            balance = token_contract.functions.balanceOf(address).call()
            
            # Convert to balance with decimal places
            adjusted_balance = balance / (10 ** decimals)
            
            return float(adjusted_balance)
        except Exception as e:
            logger.error(f"Failed to get token balance: {str(e)}")
            raise
    
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get ERC20 Token Information
        
        Args:
            token_address: Token contract address or token symbol
            
        Returns:
            Token information dictionary
        """
        # Check if using token symbol
        if token_address in self.token_addresses:
            token_address = self.token_addresses[token_address]
        
        if not is_valid_ethereum_address(token_address):
            raise ValueError(f"Invalid token address: {token_address}")
        
        try:
            # Create contract instance
            token_contract = self.w3.eth.contract(address=token_address, abi=self.erc20_abi)
            
            # Get token information
            name = token_contract.functions.name().call()
            symbol = token_contract.functions.symbol().call()
            decimals = token_contract.functions.decimals().call()
            total_supply = token_contract.functions.totalSupply().call() / (10 ** decimals)
            
            return {
                'address': token_address,
                'name': name,
                'symbol': symbol,
                'decimals': decimals,
                'total_supply': float(total_supply)
            }
        except Exception as e:
            logger.error(f"Failed to get token information: {str(e)}")
            raise
    
    def get_transaction_history(self, address: Optional[str] = None, 
                               block_start: int = 0, 
                               block_end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get Transaction History
        
        Args:
            address: Address to query, defaults to current account address
            block_start: Start block
            block_end: End block, defaults to latest block
            
        Returns:
            Transaction history list
        """
        if not address:
            if not self.address:
                raise ValueError("Address not provided and default account not set")
            address = self.address
        
        if not is_valid_ethereum_address(address):
            raise ValueError(f"Invalid Ethereum address: {address}")
        
        if block_end is None:
            block_end = self.w3.eth.block_number
        
        # Note: This method is inefficient on mainnet, should use Etherscan API or other indexing service
        # This is just an example implementation
        
        try:
            transactions = []
            address_lower = address.lower()
            
            # Since traversing all blocks is inefficient, limit query range
            max_blocks = 1000
            if block_end - block_start > max_blocks:
                block_start = block_end - max_blocks
                logger.warning(f"Query range too large, limited to recent {max_blocks} blocks")
            
            for block_num in range(block_start, block_end + 1):
                try:
                    block = self.w3.eth.get_block(block_num, full_transactions=True)
                    
                    for tx in block.transactions:
                        tx_dict = dict(tx)
                        # Check if transaction is related to address
                        if (tx_dict.get('from', '').lower() == address_lower or 
                            tx_dict.get('to', '').lower() == address_lower):
                            
                            # Add block information
                            tx_dict['block_number'] = block.number
                            tx_dict['block_timestamp'] = block.timestamp
                            
                            # Add transaction receipt information
                            try:
                                receipt = self.w3.eth.get_transaction_receipt(tx_dict['hash'])
                                tx_dict['status'] = receipt.status
                                tx_dict['gas_used'] = receipt.gasUsed
                            except:
                                tx_dict['status'] = None
                                tx_dict['gas_used'] = None
                            
                            transactions.append(tx_dict)
                except Exception as e:
                    logger.warning(f"Failed to get block {block_num} information: {str(e)}")
                    continue
            
            return transactions
        except Exception as e:
            logger.error(f"Failed to get transaction history: {str(e)}")
            raise
    
    def send_transaction(self, to_address: str, amount: float, 
                        data: str = "", token_address: Optional[str] = None) -> str:
        """
        Send Transaction (ETH or ERC20 Token)
        
        Args:
            to_address: Receiving address
            amount: Sending amount
            data: Transaction data (hex string)
            token_address: Token address, if None send ETH
            
        Returns:
            Transaction hash
        """
        if not self.account:
            raise ValueError("Private key not set, unable to send transaction")
        
        if not is_valid_ethereum_address(to_address):
            raise ValueError(f"Invalid receiving address: {to_address}")
        
        try:
            # Get nonce
            nonce = self.w3.eth.get_transaction_count(self.address)
            
            # Get gas price
            gas_price = self.w3.eth.gas_price
            
            if token_address:
                # Send ERC20 token
                # Check if using token symbol
                if token_address in self.token_addresses:
                    token_address = self.token_addresses[token_address]
                
                if not is_valid_ethereum_address(token_address):
                    raise ValueError(f"Invalid token address: {token_address}")
                
                # Create contract instance
                token_contract = self.w3.eth.contract(address=token_address, abi=self.erc20_abi)
                
                # Get token decimal places
                decimals = token_contract.functions.decimals().call()
                
                # Convert to token minimum unit
                token_amount = int(amount * (10 ** decimals))
                
                # Create transaction
                tx = token_contract.functions.transfer(
                    to_address, 
                    token_amount
                ).build_transaction({
                    'chainId': self.w3.eth.chain_id,
                    'gas': 100000,  # Estimate gas
                    'gasPrice': gas_price,
                    'nonce': nonce,
                })
                
                # Estimate gas
                try:
                    gas_estimate = self.w3.eth.estimate_gas(tx)
                    tx['gas'] = gas_estimate
                except Exception as e:
                    logger.warning(f"Gas estimate failed, using default value: {str(e)}")
            else:
                # Send ETH
                # Convert to wei
                value_wei = self.w3.to_wei(amount, 'ether')
                
                # Create transaction
                tx = {
                    'to': to_address,
                    'value': value_wei,
                    'gas': 21000,  # Basic ETH transfer gas
                    'gasPrice': gas_price,
                    'nonce': nonce,
                    'chainId': self.w3.eth.chain_id,
                }
                
                # If there is data, add to transaction
                if data:
                    tx['data'] = data
                    # Re-estimate gas
                    try:
                        gas_estimate = self.w3.eth.estimate_gas(tx)
                        tx['gas'] = gas_estimate
                    except Exception as e:
                        logger.warning(f"Gas estimate failed, using default value: {str(e)}")
            
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Return transaction hash
            return self.w3.to_hex(tx_hash)
        except Exception as e:
            logger.error(f"Failed to send transaction: {str(e)}")
            raise
    
    def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """
        Get Transaction Status
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction status dictionary
        """
        try:
            # Get transaction
            tx = self.w3.eth.get_transaction(tx_hash)
            
            # Get transaction receipt
            try:
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                status = receipt.status
                gas_used = receipt.gasUsed
                block_number = receipt.blockNumber
                
                # Get block
                block = self.w3.eth.get_block(block_number)
                block_timestamp = block.timestamp
                
                # Calculate confirmation count
                current_block = self.w3.eth.block_number
                confirmations = current_block - block_number + 1 if block_number else 0
                
                return {
                    'hash': tx_hash,
                    'from': tx['from'],
                    'to': tx['to'],
                    'value': self.w3.from_wei(tx['value'], 'ether'),
                    'status': status,
                    'success': status == 1,
                    'gas_price': self.w3.from_wei(tx['gasPrice'], 'gwei'),
                    'gas_used': gas_used,
                    'gas_cost_eth': self.w3.from_wei(tx['gasPrice'] * gas_used, 'ether') if gas_used else None,
                    'block_number': block_number,
                    'block_timestamp': block_timestamp,
                    'confirmations': confirmations,
                    'pending': False
                }
            except TransactionNotFound:
                # Transaction not yet mined
                return {
                    'hash': tx_hash,
                    'from': tx['from'],
                    'to': tx['to'],
                    'value': self.w3.from_wei(tx['value'], 'ether'),
                    'status': None,
                    'success': None,
                    'gas_price': self.w3.from_wei(tx['gasPrice'], 'gwei'),
                    'gas_used': None,
                    'gas_cost_eth': None,
                    'block_number': None,
                    'block_timestamp': None,
                    'confirmations': 0,
                    'pending': True
                }
        except Exception as e:
            logger.error(f"Failed to get transaction status: {str(e)}")
            raise
    
    def wait_for_transaction_receipt(self, tx_hash: str, timeout: int = 120, poll_interval: float = 0.1) -> Dict[str, Any]:
        """
        Wait for Transaction Receipt
        
        Args:
            tx_hash: Transaction hash
            timeout: Timeout time (seconds)
            poll_interval: Query interval (seconds)
            
        Returns:
            Transaction status dictionary
        """
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout, poll_latency=poll_interval)
            return self.get_transaction_status(tx_hash)
        except Exception as e:
            logger.error(f"Failed to wait for transaction receipt: {str(e)}")
            raise
    
    def get_contract_events(self, contract_address: str, abi: List[Dict], 
                           event_name: str, from_block: int, to_block: Optional[int] = None) -> List[Dict]:
        """
        Get Contract Events
        
        Args:
            contract_address: Contract address
            abi: Contract ABI
            event_name: Event name
            from_block: Start block
            to_block: End block, defaults to latest block
            
        Returns:
            Event list
        """
        if not is_valid_ethereum_address(contract_address):
            raise ValueError(f"Invalid contract address: {contract_address}")
        
        if to_block is None:
            to_block = self.w3.eth.block_number
        
        try:
            # Create contract instance
            contract = self.w3.eth.contract(address=contract_address, abi=abi)
            
            # Get event object
            event = getattr(contract.events, event_name)
            
            # Get event logs
            logs = event.get_logs(fromBlock=from_block, toBlock=to_block)
            
            # Process logs
            events = []
            for log in logs:
                event_data = dict(log.args)
                event_data['block_number'] = log.blockNumber
                event_data['transaction_hash'] = self.w3.to_hex(log.transactionHash)
                event_data['log_index'] = log.logIndex
                
                # Get block timestamp
                try:
                    block = self.w3.eth.get_block(log.blockNumber)
                    event_data['block_timestamp'] = block.timestamp
                except:
                    event_data['block_timestamp'] = None
                
                events.append(event_data)
            
            return events
        except Exception as e:
            logger.error(f"Failed to get contract events: {str(e)}")
            raise
    
    def estimate_gas(self, tx: Dict[str, Any]) -> int:
        """
        Estimate Transaction Gas
        
        Args:
            tx: Transaction dictionary
            
        Returns:
            Estimated gas value
        """
        try:
            gas = self.w3.eth.estimate_gas(tx)
            return gas
        except Exception as e:
            logger.error(f"Failed to estimate gas: {str(e)}")
            raise
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get Network Information
        
        Returns:
            Network information dictionary
        """
        try:
            chain_id = self.w3.eth.chain_id
            gas_price = self.w3.eth.gas_price
            block_number = self.w3.eth.block_number
            
            # Get latest block
            latest_block = self.w3.eth.get_block('latest')
            
            return {
                'chain_id': chain_id,
                'gas_price_wei': gas_price,
                'gas_price_gwei': self.w3.from_wei(gas_price, 'gwei'),
                'latest_block': block_number,
                'block_timestamp': latest_block.timestamp,
                'is_connected': self.w3.is_connected()
            }
        except Exception as e:
            logger.error(f"Failed to get network information: {str(e)}")
            raise
    
    def get_eth_price(self) -> Optional[float]:
        """
        Get ETH Price (USD)
        Use CoinGecko API
        
        Returns:
            ETH price (USD)
        """
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
            response = requests.get(url)
            data = response.json()
            return data['ethereum']['usd']
        except Exception as e:
            logger.error(f"Failed to get ETH price: {str(e)}")
            return None
    
    def get_token_price(self, token_address: str) -> Optional[float]:
        """
        Get Token Price (USD)
        Use CoinGecko API
        
        Args:
            token_address: Token address or symbol
            
        Returns:
            Token price (USD)
        """
        # Check if using token symbol
        if token_address in self.token_addresses:
            token_address = self.token_addresses[token_address]
        
        try:
            url = f"https://api.coingecko.com/api/v3/simple/token_price/ethereum?contract_addresses={token_address}&vs_currencies=usd"
            response = requests.get(url)
            data = response.json()
            return data[token_address.lower()]['usd']
        except Exception as e:
            logger.error(f"Failed to get token price: {str(e)}")
            return None
    
    def get_gas_estimate(self) -> Dict[str, Any]:
        """
        Get Gas Estimate
        
        Returns:
            Gas estimate dictionary
        """
        try:
            gas_price = self.w3.eth.gas_price
            
            # Calculate different speed gas prices
            slow_gas_price = int(gas_price * 0.8)
            fast_gas_price = int(gas_price * 1.2)
            rapid_gas_price = int(gas_price * 1.5)
            
            return {
                'slow': {
                    'gas_price_wei': slow_gas_price,
                    'gas_price_gwei': self.w3.from_wei(slow_gas_price, 'gwei'),
                    'estimated_seconds': 120
                },
                'average': {
                    'gas_price_wei': gas_price,
                    'gas_price_gwei': self.w3.from_wei(gas_price, 'gwei'),
                    'estimated_seconds': 60
                },
                'fast': {
                    'gas_price_wei': fast_gas_price,
                    'gas_price_gwei': self.w3.from_wei(fast_gas_price, 'gwei'),
                    'estimated_seconds': 30
                },
                'rapid': {
                    'gas_price_wei': rapid_gas_price,
                    'gas_price_gwei': self.w3.from_wei(rapid_gas_price, 'gwei'),
                    'estimated_seconds': 15
                }
            }
        except Exception as e:
            logger.error(f"Failed to get gas estimate: {str(e)}")
            raise 