"""
Credit Scoring Model Module
Provides credit scoring functionality based on blockchain data
"""
import logging
import os
import json
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from defi_project.utils.helpers import save_json_file, load_json_file, format_datetime

# Set up logging
logger = logging.getLogger(__name__)

class CreditScoreModel:
    """
    Credit Scoring Model Class
    Provides credit scoring functionality based on blockchain data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize credit scoring model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Data storage path
        self.data_dir = config.get('DATA_DIR', 'defi_project/data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Model storage path
        self.models_dir = os.path.join('defi_project', 'models', 'credit_scoring')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default model path
        self.default_model_path = os.path.join(self.models_dir, 'credit_score_model.pkl')
        self.default_preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        
        # Initialize model
        self.model = None
        self.preprocessor = None
        
        # Feature columns
        self.numerical_features = [
            'eth_balance', 
            'token_balance_total_usd',
            'transaction_count',
            'avg_transaction_value',
            'max_transaction_value',
            'unique_interaction_addresses',
            'days_since_first_tx',
            'active_days',
            'defi_protocol_interactions',
            'successful_tx_ratio'
        ]
        
        self.categorical_features = [
            'most_used_defi_protocol',
            'transaction_frequency',
            'has_ens_name'
        ]
        
        # Credit score range
        self.min_score = 300
        self.max_score = 850
        
        logger.info("Credit Scoring Model Initialized")
    
    def prepare_features(self, user_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare Model Features
        
        Args:
            user_data: User data dictionary
            
        Returns:
            Feature DataFrame
        """
        try:
            # Extract transaction history
            transactions = user_data.get('transactions', [])
            
            # Extract token balance
            balances = user_data.get('balances', {})
            
            # Extract DeFi activities
            defi_activities = user_data.get('defi_activities', [])
            
            # Extract offchain data
            offchain_data = user_data.get('offchain_data', {})
            
            # Calculate features
            features = {}
            
            # Balance-related features
            features['eth_balance'] = balances.get('ETH', {}).get('balance', 0)
            
            # Calculate token total value (USD)
            token_balance_total_usd = 0
            for token, data in balances.items():
                if token != 'ETH':
                    token_balance_total_usd += data.get('balance_usd', 0)
            features['token_balance_total_usd'] = token_balance_total_usd
            
            # Transaction-related features
            features['transaction_count'] = len(transactions)
            
            if transactions:
                # Transaction amount statistics
                tx_values = [tx.get('value', 0) for tx in transactions]
                features['avg_transaction_value'] = sum(tx_values) / len(tx_values) if tx_values else 0
                features['max_transaction_value'] = max(tx_values) if tx_values else 0
                
                # Transaction address statistics
                interaction_addresses = set()
                for tx in transactions:
                    if tx.get('from'):
                        interaction_addresses.add(tx['from'])
                    if tx.get('to'):
                        interaction_addresses.add(tx['to'])
                features['unique_interaction_addresses'] = len(interaction_addresses)
                
                # Account activity statistics
                tx_timestamps = [tx.get('block_timestamp', 0) for tx in transactions if tx.get('block_timestamp')]
                if tx_timestamps:
                    first_tx_time = min(tx_timestamps)
                    last_tx_time = max(tx_timestamps)
                    account_age_days = (last_tx_time - first_tx_time) / (24 * 3600) + 1
                    features['days_since_first_tx'] = account_age_days
                    
                    # Calculate active days
                    unique_days = set()
                    for ts in tx_timestamps:
                        day = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        unique_days.add(day)
                    features['active_days'] = len(unique_days)
                    
                    # Transaction frequency classification
                    tx_frequency = features['transaction_count'] / account_age_days if account_age_days > 0 else 0
                    if tx_frequency < 0.1:
                        features['transaction_frequency'] = 'very_low'
                    elif tx_frequency < 0.5:
                        features['transaction_frequency'] = 'low'
                    elif tx_frequency < 2:
                        features['transaction_frequency'] = 'medium'
                    elif tx_frequency < 5:
                        features['transaction_frequency'] = 'high'
                    else:
                        features['transaction_frequency'] = 'very_high'
                else:
                    features['days_since_first_tx'] = 0
                    features['active_days'] = 0
                    features['transaction_frequency'] = 'none'
                
                # Transaction success rate
                successful_txs = sum(1 for tx in transactions if tx.get('status') == 1)
                features['successful_tx_ratio'] = successful_txs / len(transactions) if transactions else 0
                
                # Gas fee statistics
                gas_fees = [tx.get('gas_used', 0) * tx.get('gas_price', 0) for tx in transactions 
                           if tx.get('gas_used') and tx.get('gas_price')]
                features['avg_gas_paid'] = sum(gas_fees) / len(gas_fees) if gas_fees else 0
            else:
                features['avg_transaction_value'] = 0
                features['max_transaction_value'] = 0
                features['unique_interaction_addresses'] = 0
                features['days_since_first_tx'] = 0
                features['active_days'] = 0
                features['transaction_frequency'] = 'none'
                features['successful_tx_ratio'] = 0
                features['avg_gas_paid'] = 0
            
            # DeFi activity features
            features['defi_protocol_interactions'] = len(defi_activities)
            
            if defi_activities:
                # Statistics of protocol usage
                protocol_counts = {}
                total_volume = 0
                lending_volume = 0
                borrowing_volume = 0
                repayment_amount = 0
                borrowed_amount = 0
                liquidation_count = 0
                
                for activity in defi_activities:
                    protocol = activity.get('protocol', '')
                    activity_type = activity.get('type', '')
                    amount = activity.get('amount', 0)
                    
                    # Statistics of protocol usage
                    protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
                    
                    # Statistics of transaction volume
                    total_volume += amount
                    
                    # Statistics of lending volume
                    if activity_type == 'lend':
                        lending_volume += amount
                    elif activity_type == 'borrow':
                        borrowing_volume += amount
                        borrowed_amount += amount
                    elif activity_type == 'repay':
                        repayment_amount += amount
                    elif activity_type == 'liquidation':
                        liquidation_count += 1
                
                # Most used DeFi protocol
                features['most_used_defi_protocol'] = max(protocol_counts.items(), key=lambda x: x[1])[0] if protocol_counts else 'none'
                
                # Transaction volume statistics
                features['total_volume_traded'] = total_volume
                features['lending_volume'] = lending_volume
                features['borrowing_volume'] = borrowing_volume
                
                # Repayment ratio
                features['repayment_ratio'] = repayment_amount / borrowed_amount if borrowed_amount > 0 else 0
                
                # Liquidation count
                features['liquidation_count'] = liquidation_count
            else:
                features['most_used_defi_protocol'] = 'none'
                features['total_volume_traded'] = 0
                features['lending_volume'] = 0
                features['borrowing_volume'] = 0
                features['repayment_ratio'] = 0
                features['liquidation_count'] = 0
            
            # Offchain data features
            features['has_ens_name'] = 'yes' if offchain_data.get('has_ens_name', False) else 'no'
            
            # Create DataFrame
            df = pd.DataFrame([features])
            
            return df
        except Exception as e:
            logger.error(f"Failed to prepare features: {str(e)}")
            # Return empty DataFrame, including all feature columns
            columns = self.numerical_features + self.categorical_features
            return pd.DataFrame(columns=columns)
    
    def _create_preprocessor(self, features_df: pd.DataFrame) -> ColumnTransformer:
        """
        Create Feature Preprocessor
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            Column Transformer
        """
        # Numerical feature processing
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Categorical feature processing
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combined transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Drop other columns
        )
        
        return preprocessor
    
    def train_model(self, training_data: List[Dict[str, Any]], 
                   credit_scores: List[float], test_size: float = 0.2, 
                   random_state: int = 42) -> float:
        """
        Train Credit Scoring Model
        
        Args:
            training_data: Training data list, each element is user data dictionary
            credit_scores: Corresponding credit score list
            test_size: Test set ratio
            random_state: Random seed
            
        Returns:
            Model R² Score
        """
        try:
            # Prepare features
            features_list = []
            for user_data in training_data:
                features = self.prepare_features(user_data)
                features_list.append(features)
            
            # Merge features
            X = pd.concat(features_list, ignore_index=True)
            y = np.array(credit_scores)
            
            # Split training set and test set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Create preprocessor
            self.preprocessor = self._create_preprocessor(X)
            
            # Create model
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            )
            
            # Train model
            self.model.fit(self.preprocessor.fit_transform(X_train), y_train)
            
            # Evaluate model
            y_pred = self.model.predict(self.preprocessor.transform(X_test))
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model training completed, MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Save model
            self.save_model()
            
            return r2
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            return 0.0
    
    def save_model(self, model_path: Optional[str] = None):
        """
        Save Model
        
        Args:
            model_path: Model save path, default to preset path
        """
        if not model_path:
            model_path = self.default_model_path
        
        preprocessor_path = model_path.replace('credit_score_model.pkl', 'preprocessor.pkl')
        
        try:
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save preprocessor
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            
            logger.info(f"Model saved to: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load Model
        
        Args:
            model_path: Model load path, default to preset path
            
        Returns:
            Whether loading is successful
        """
        if not model_path:
            model_path = self.default_model_path
        
        preprocessor_path = model_path.replace('credit_score_model.pkl', 'preprocessor.pkl')
        
        try:
            # Load model
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                logger.warning(f"Model file does not exist: {model_path}")
                return False
            
            # Load preprocessor
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
            else:
                logger.warning(f"Preprocessor file does not exist: {preprocessor_path}")
                return False
            
            logger.info(f"Model loaded: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def predict_credit_score(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict User Credit Score
        
        Args:
            user_data: User data dictionary
            
        Returns:
            Credit score result dictionary
        """
        try:
            # Check if model is loaded
            if not self.model or not self.preprocessor:
                if not self.load_model():
                    # If model loading fails, return default score
                    return {
                        'success': False,
                        'error': 'Model not loaded',
                        'score': 500,  # Default medium score
                        'grade': 'C',
                        'timestamp': format_datetime(datetime.now())
                    }
            
            # Prepare features
            features = self.prepare_features(user_data)
            
            # Predict score
            X = self.preprocessor.transform(features)
            raw_score = self.model.predict(X)[0]
            
            # Limit score within range
            score = max(min(raw_score, self.max_score), self.min_score)
            
            # Calculate credit grade
            grade = self._calculate_credit_grade(score)
            
            # Calculate risk assessment
            risk_assessment = self._calculate_risk_assessment(score)
            
            # Generate loan parameter suggestions
            loan_suggestions = self._suggest_loan_parameters(score, user_data)
            
            return {
                'success': True,
                'address': user_data.get('address', ''),
                'score': int(score),
                'grade': grade,
                'risk_assessment': risk_assessment,
                'loan_suggestions': loan_suggestions,
                'timestamp': format_datetime(datetime.now())
            }
        except Exception as e:
            logger.error(f"Failed to predict credit score: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'score': 500,  # Default medium score
                'grade': 'C',
                'timestamp': format_datetime(datetime.now())
            }
    
    def _calculate_credit_grade(self, score: float) -> str:
        """
        Calculate Credit Grade
        
        Args:
            score: Credit score
            
        Returns:
            Credit grade (A+, A, B+, B, C+, C, D, F)
        """
        if score >= 800:
            return 'A+'
        elif score >= 750:
            return 'A'
        elif score >= 700:
            return 'B+'
        elif score >= 650:
            return 'B'
        elif score >= 600:
            return 'C+'
        elif score >= 550:
            return 'C'
        elif score >= 500:
            return 'D'
        else:
            return 'F'
    
    def _calculate_risk_assessment(self, score: float) -> Dict[str, Any]:
        """
        Calculate Risk Assessment
        
        Args:
            score: Credit score
            
        Returns:
            Risk assessment dictionary
        """
        # Calculate risk level based on score
        if score >= 750:
            risk_level = 'very_low'
            default_probability = round((850 - score) / 850 * 0.05, 4)
        elif score >= 700:
            risk_level = 'low'
            default_probability = round((750 - score) / 750 * 0.10 + 0.05, 4)
        elif score >= 650:
            risk_level = 'moderate'
            default_probability = round((700 - score) / 700 * 0.15 + 0.10, 4)
        elif score >= 600:
            risk_level = 'moderate_high'
            default_probability = round((650 - score) / 650 * 0.20 + 0.15, 4)
        elif score >= 550:
            risk_level = 'high'
            default_probability = round((600 - score) / 600 * 0.25 + 0.20, 4)
        else:
            risk_level = 'very_high'
            default_probability = round((550 - score) / 550 * 0.30 + 0.25, 4)
        
        # Get risk description
        risk_description = self._get_risk_description(risk_level)
        
        return {
            'risk_level': risk_level,
            'default_probability': default_probability,
            'description': risk_description
        }
    
    def _get_risk_description(self, risk_level: str) -> str:
        """
        Get Risk Level Description
        
        Args:
            risk_level: Risk level
            
        Returns:
            Risk description
        """
        descriptions = {
            'very_low': 'Borrower has very low default risk, excellent credit history, able to repay loans on time.',
            'low': 'Borrower has low default risk, good credit history, usually able to repay loans on time.',
            'moderate': 'Borrower has moderate default risk, average credit history, possibly occasionally delayed loan repayment.',
            'moderate_high': 'Borrower has moderate-high default risk, poor credit history, has history of delayed loan repayment.',
            'high': 'Borrower has high default risk, poor credit history, frequently delayed loan repayment or has outstanding debt.',
            'very_high': 'Borrower has very high default risk, poor credit history, has multiple default history.'
        }
        
        return descriptions.get(risk_level, 'Unknown risk level')
    
    def _suggest_loan_parameters(self, score: float, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest Loan Parameters
        
        Args:
            score: Credit score
            user_data: User data
            
        Returns:
            Loan parameters suggestion dictionary
        """
        # Get user asset total value
        balances = user_data.get('balances', {})
        eth_balance = balances.get('ETH', {}).get('balance_usd', 0)
        
        token_balance_total_usd = 0
        for token, data in balances.items():
            if token != 'ETH':
                token_balance_total_usd += data.get('balance_usd', 0)
        
        total_asset_value = eth_balance + token_balance_total_usd
        
        # Calculate loan parameters based on score and asset
        if score >= 750:  # A+, A
            max_loan_amount = total_asset_value * 0.8
            interest_rate = 0.05
            collateral_ratio = 1.2
            loan_term_days = 90
        elif score >= 700:  # B+
            max_loan_amount = total_asset_value * 0.7
            interest_rate = 0.07
            collateral_ratio = 1.3
            loan_term_days = 60
        elif score >= 650:  # B
            max_loan_amount = total_asset_value * 0.6
            interest_rate = 0.09
            collateral_ratio = 1.4
            loan_term_days = 45
        elif score >= 600:  # C+
            max_loan_amount = total_asset_value * 0.5
            interest_rate = 0.12
            collateral_ratio = 1.5
            loan_term_days = 30
        elif score >= 550:  # C
            max_loan_amount = total_asset_value * 0.4
            interest_rate = 0.15
            collateral_ratio = 1.6
            loan_term_days = 21
        elif score >= 500:  # D
            max_loan_amount = total_asset_value * 0.3
            interest_rate = 0.18
            collateral_ratio = 1.8
            loan_term_days = 14
        else:  # F
            max_loan_amount = total_asset_value * 0.2
            interest_rate = 0.25
            collateral_ratio = 2.0
            loan_term_days = 7
        
        # Limit maximum loan amount
        max_loan_amount = min(max_loan_amount, 100000)
        
        # Recommended DeFi protocols
        recommended_protocols = self._get_recommended_protocols({
            'score': score,
            'grade': self._calculate_credit_grade(score)
        })
        
        return {
            'max_loan_amount_usd': round(max_loan_amount, 2),
            'suggested_interest_rate': round(interest_rate, 4),
            'required_collateral_ratio': round(collateral_ratio, 2),
            'max_loan_term_days': loan_term_days,
            'recommended_protocols': recommended_protocols
        }
    
    def get_user_score(self, address: str) -> Dict[str, Any]:
        """
        Get User Score
        
        Args:
            address: User address
            
        Returns:
            User score dictionary
        """
        # Load user score from file
        score_file = os.path.join(self.data_dir, f"credit_score_{address}.json")
        
        if os.path.exists(score_file):
            return load_json_file(score_file)
        else:
            return {
                'success': False,
                'error': 'User score not found',
                'address': address,
                'timestamp': format_datetime(datetime.now())
            }
    
    def update_user_score(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update User Score
        
        Args:
            user_data: User data
            
        Returns:
            Updated score dictionary
        """
        try:
            # Predict score
            score_result = self.predict_credit_score(user_data)
            
            if score_result['success']:
                # Save score result
                address = user_data.get('address', '')
                if address:
                    score_file = os.path.join(self.data_dir, f"credit_score_{address}.json")
                    save_json_file(score_file, score_result)
            
            return score_result
        except Exception as e:
            logger.error(f"Failed to update user score: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'address': user_data.get('address', ''),
                'timestamp': format_datetime(datetime.now())
            }
    
    def _get_recommended_protocols(self, score_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get Recommended DeFi Protocols
        
        Args:
            score_result: Score result
            
        Returns:
            Recommended protocol list
        """
        grade = score_result.get('grade', 'C')
        
        # Recommend different protocols based on credit grade
        if grade in ['A+', 'A']:
            return [
                {'name': 'Aave', 'type': 'lending', 'url': 'https://aave.com'},
                {'name': 'Compound', 'type': 'lending', 'url': 'https://compound.finance'},
                {'name': 'MakerDAO', 'type': 'lending', 'url': 'https://makerdao.com'}
            ]
        elif grade in ['B+', 'B']:
            return [
                {'name': 'Compound', 'type': 'lending', 'url': 'https://compound.finance'},
                {'name': 'MakerDAO', 'type': 'lending', 'url': 'https://makerdao.com'},
                {'name': 'Euler', 'type': 'lending', 'url': 'https://euler.finance'}
            ]
        elif grade in ['C+', 'C']:
            return [
                {'name': 'Euler', 'type': 'lending', 'url': 'https://euler.finance'},
                {'name': 'TrueFi', 'type': 'lending', 'url': 'https://truefi.io'}
            ]
        else:
            return [
                {'name': 'TrueFi', 'type': 'lending', 'url': 'https://truefi.io'}
            ] 