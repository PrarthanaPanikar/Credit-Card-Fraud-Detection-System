"""
Prediction Module for Credit Card Fraud Detection
Handles single transactions and batch predictions.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Union, List, Dict, Any
import os

from src.features import add_features


class FraudPredictor:
    """Predictor class for fraud detection inference."""
    
    def __init__(self, model_path='models/fraud_detection_model.joblib'):
        """Initialize predictor with trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        # Load model bundle
        bundle = joblib.load(model_path)
        self.model = bundle['model']
        self.model_name = bundle.get('model_name', 'unknown')
        self.threshold = bundle.get('threshold', 0.5)
        self.metrics = bundle.get('metrics', {})
        self.version = bundle.get('version', '1.0.0')
        
        print(f"Loaded model: {self.model_name} (v{self.version})")
        print(f"Decision threshold: {self.threshold:.3f}")
    
    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud probability for a single transaction.
        
        Args:
            transaction: Dictionary with transaction features
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Add features
        df = add_features(df)
        
        # Drop non-feature columns
        drop_cols = ['tx_id', 'ts', 'merchant_id_hash', 'card_id_hash', 'is_fraud']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        # Predict
        proba = self.model.predict_proba(X)[0, 1]
        
        # Decision based on threshold
        decision = "REVIEW" if proba >= self.threshold else "ALLOW"
        
        return {
            'transaction_id': transaction.get('tx_id', 'unknown'),
            'fraud_probability': float(proba),
            'risk_score': int(proba * 100),
            'decision': decision,
            'threshold': float(self.threshold),
            'model_version': self.version
        }
    
    def predict_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict fraud probability for multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for tx in transactions:
            result = self.predict_single(tx)
            results.append(result)
        return results
    
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud for a DataFrame of transactions.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with predictions added
        """
        # Add features
        df_features = add_features(df.copy())
        
        # Drop non-feature columns
        drop_cols = ['tx_id', 'ts', 'merchant_id_hash', 'card_id_hash', 'is_fraud']
        X = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns], errors='ignore')
        
        # Predict
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Add results
        df['fraud_probability'] = probabilities
        df['risk_score'] = (probabilities * 100).astype(int)
        df['decision'] = np.where(probabilities >= self.threshold, 'REVIEW', 'ALLOW')
        df['predicted_fraud'] = (probabilities >= self.threshold).astype(int)
        
        return df
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'version': self.version,
            'threshold': self.threshold,
            'metrics': self.metrics
        }


def demonstrate_prediction():
    """Demonstrate prediction on sample transactions."""
    # Initialize predictor
    predictor = FraudPredictor()
    
    # Sample normal transaction
    normal_tx = {
        'tx_id': 'TX123456789',
        'ts': '2024-06-15 14:30:00',
        'amount': 1250.00,
        'merchant_cat': 'grocery',
        'merchant_id_hash': 'MERCH001',
        'card_id_hash': 'CARD001',
        'city': 'Mumbai',
        'country': 'IN',
        'device_type': 'mobile',
        'channel': 'app',
        'hour': 14,
        'dayofweek': 2,
        'is_international': False,
        'is_night': False,
        'prev_24h_tx_count_card': 3.0,
        'prev_24h_amt_card': 2500.0,
        'prev_1h_tx_count_card': 0.0,
        'velocity_amt_1h': 0.0
    }
    
    # Sample suspicious transaction
    fraud_tx = {
        'tx_id': 'TX987654321',
        'ts': '2024-06-15 03:30:00',
        'amount': 50000.00,
        'merchant_cat': 'travel',
        'merchant_id_hash': 'MERCH099',
        'card_id_hash': 'CARD001',
        'city': 'Dubai',
        'country': 'AE',
        'device_type': 'desktop',
        'channel': 'online',
        'hour': 3,
        'dayofweek': 2,
        'is_international': True,
        'is_night': True,
        'prev_24h_tx_count_card': 8.0,
        'prev_24h_amt_card': 75000.0,
        'prev_1h_tx_count_card': 4.0,
        'velocity_amt_1h': 35000.0
    }
    
    print("\n" + "="*60)
    print("PREDICTION DEMONSTRATION")
    print("="*60)
    
    # Predict normal transaction
    print("\n1. NORMAL TRANSACTION:")
    result = predictor.predict_single(normal_tx)
    print(f"   Transaction ID: {result['transaction_id']}")
    print(f"   Amount: ₹{normal_tx['amount']:,.2f}")
    print(f"   Fraud Probability: {result['fraud_probability']:.2%}")
    print(f"   Risk Score: {result['risk_score']}/100")
    print(f"   Decision: {result['decision']}")
    
    # Predict suspicious transaction
    print("\n2. SUSPICIOUS TRANSACTION:")
    result = predictor.predict_single(fraud_tx)
    print(f"   Transaction ID: {result['transaction_id']}")
    print(f"   Amount: ₹{fraud_tx['amount']:,.2f}")
    print(f"   Fraud Probability: {result['fraud_probability']:.2%}")
    print(f"   Risk Score: {result['risk_score']}/100")
    print(f"   Decision: {result['decision']}")
    
    # Batch prediction
    print("\n3. BATCH PREDICTION:")
    batch_results = predictor.predict_batch([normal_tx, fraud_tx])
    for i, r in enumerate(batch_results):
        print(f"   Transaction {i+1}: Risk={r['risk_score']}/100, Decision={r['decision']}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    demonstrate_prediction()
