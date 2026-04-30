"""
Feature Engineering Module for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for fraud detection feature engineering."""
    
    def __init__(self):
        self.merchant_cat_counts = None
        
    def fit(self, X, y=None):
        # Store merchant category frequencies for rare category detection
        if 'merchant_cat' in X.columns:
            self.merchant_cat_counts = X['merchant_cat'].value_counts()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Log transform amount (handles skewness)
        X['log_amount'] = np.log1p(X['amount'])
        
        # Average transaction amount in last 24h
        X['avg_tx_amt_24h'] = X['prev_24h_amt_card'] / (X['prev_24h_tx_count_card'] + 1e-3)
        
        # Velocity ratio: 1h amount vs 24h average
        X['velocity_ratio'] = X['velocity_amt_1h'] / (X['avg_tx_amt_24h'] + 1e-3)
        X['velocity_ratio'] = X['velocity_ratio'].replace([np.inf, -np.inf], 0)
        
        # Time-based features
        X['is_weekend'] = X['dayofweek'].isin([5, 6]).astype(int)
        X['is_night'] = X['is_night'].astype(int)
        X['is_international'] = X['is_international'].astype(int)
        
        # Hour buckets
        X['hour_bucket'] = pd.cut(X['hour'], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['night', 'morning', 'afternoon', 'evening'],
                                   include_lowest=True)
        
        # Amount relative to card's history
        X['amount_vs_avg'] = X['amount'] / (X['avg_tx_amt_24h'] + 1e-3)
        X['amount_vs_avg'] = X['amount_vs_avg'].replace([np.inf, -np.inf], 0)
        
        # Transaction frequency features
        X['tx_intensity_1h'] = X['prev_1h_tx_count_card']
        X['tx_intensity_24h'] = X['prev_24h_tx_count_card']
        
        # Weekend + night combination (high risk)
        X['weekend_night'] = X['is_weekend'] * X['is_night']
        
        # International + high amount
        X['international_high_amount'] = X['is_international'] * (X['amount'] > X['amount'].quantile(0.9)).astype(int)
        
        # Rare merchant category flag
        if self.merchant_cat_counts is not None:
            X['merchant_cat_rare'] = X['merchant_cat'].map(
                lambda x: int(self.merchant_cat_counts.get(x, 0) < 50)
            )
        else:
            X['merchant_cat_rare'] = 0
        
        # Channel + device combinations
        X['channel_device'] = X['channel'].astype(str) + '_' + X['device_type'].astype(str)
        
        return X


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the dataset.
    Standalone function for use in notebooks.
    """
    engineer = FeatureEngineer()
    engineer.fit(df)
    return engineer.transform(df)


# Feature groups for modeling
NUMERIC_FEATURES = [
    'amount', 'log_amount',
    'prev_24h_tx_count_card', 'prev_24h_amt_card',
    'prev_1h_tx_count_card', 'velocity_amt_1h',
    'avg_tx_amt_24h', 'velocity_ratio',
    'amount_vs_avg', 'tx_intensity_1h', 'tx_intensity_24h',
    'hour', 'dayofweek'
]

CATEGORICAL_FEATURES = [
    'merchant_cat', 'city', 'country',
    'device_type', 'channel', 'hour_bucket', 'channel_device',
    'is_international', 'is_night', 'is_weekend', 
    'merchant_cat_rare', 'weekend_night', 'international_high_amount'
]

DROP_FEATURES = [
    'tx_id', 'ts', 'merchant_id_hash', 'card_id_hash'
]

TARGET = 'is_fraud'
