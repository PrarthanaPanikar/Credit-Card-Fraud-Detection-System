"""
Synthetic Credit Card Transaction Data Generator
Generates realistic PII-safe transaction data for fraud detection training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import os

# Configuration
N_SAMPLES = 100000
FRAUD_RATIO = 0.015  # 1.5% fraud rate (typical for credit cards)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def generate_card_id():
    """Generate anonymized card ID hash."""
    return hashlib.sha256(str(np.random.randint(1, 10000)).encode()).hexdigest()[:16]


def generate_merchant_id():
    """Generate anonymized merchant ID hash."""
    return hashlib.sha256(str(np.random.randint(1, 5000)).encode()).hexdigest()[:16]


def generate_normal_transaction(card_id, merchant_cat, base_time):
    """Generate a normal (non-fraud) transaction."""
    # Normal amounts follow log-normal distribution
    amount = np.random.lognormal(mean=4.0, sigma=1.0)
    
    # Normal spending patterns
    hour = np.random.choice(
        range(24), 
        p=[0.02, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 
           0.10, 0.10, 0.09, 0.09, 0.08, 0.07, 0.06, 0.06,
           0.06, 0.05, 0.05, 0.06, 0.07, 0.08, 0.06, 0.04]
    )
    
    dayofweek = np.random.randint(0, 7)
    
    # Device and channel preferences (consistent per card)
    device_type = np.random.choice(['mobile', 'desktop', 'tablet', 'pos'], p=[0.4, 0.3, 0.1, 0.2])
    channel = np.random.choice(['online', 'in_store', 'app', 'phone'], p=[0.35, 0.40, 0.20, 0.05])
    
    # Geographic patterns
    city = np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata'])
    country = 'IN'
    is_international = np.random.choice([True, False], p=[0.05, 0.95])
    
    # Time features
    is_night = 1 if hour < 6 or hour > 23 else 0
    
    return {
        'tx_id': f'TX{np.random.randint(100000000, 999999999)}',
        'ts': base_time,
        'amount': round(amount, 2),
        'merchant_cat': merchant_cat,
        'merchant_id_hash': generate_merchant_id(),
        'card_id_hash': card_id,
        'city': city,
        'country': country,
        'device_type': device_type,
        'channel': channel,
        'hour': hour,
        'dayofweek': dayofweek,
        'is_international': is_international,
        'is_night': is_night,
        'is_fraud': 0
    }


def generate_fraud_transaction(card_id, merchant_cat, base_time, normal_profile):
    """Generate a fraudulent transaction with suspicious patterns."""
    # Fraud amounts are often unusual (very high or testing small amounts)
    amount = np.random.choice([
        np.random.lognormal(mean=2.0, sigma=0.5),  # Small test transactions
        np.random.lognormal(mean=6.0, sigma=0.8),  # Large fraudulent purchases
        normal_profile['avg_amount'] * np.random.uniform(2, 5)  # Unusual multiples
    ], p=[0.3, 0.4, 0.3])
    
    # Fraud often happens at unusual hours
    hour = np.random.choice(
        range(24),
        p=[0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.04,
           0.05, 0.06, 0.06, 0.06, 0.05, 0.04, 0.03, 0.03,
           0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.12]
    )
    
    dayofweek = np.random.randint(0, 7)
    
    # Fraud often uses different devices/channels
    device_type = np.random.choice(['mobile', 'desktop', 'tablet', 'pos'], p=[0.5, 0.3, 0.15, 0.05])
    channel = np.random.choice(['online', 'in_store', 'app', 'phone'], p=[0.60, 0.15, 0.20, 0.05])
    
    # Geographic anomalies
    if np.random.random() < 0.4:  # 40% chance of different city
        city = np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata'])
    else:
        city = normal_profile['city']
    
    country = np.random.choice(['IN', 'US', 'GB', 'SG', 'AE'], p=[0.6, 0.15, 0.10, 0.10, 0.05])
    is_international = country != 'IN'
    
    is_night = 1 if hour < 6 or hour > 23 else 0
    
    return {
        'tx_id': f'TX{np.random.randint(100000000, 999999999)}',
        'ts': base_time,
        'amount': round(amount, 2),
        'merchant_cat': merchant_cat,
        'merchant_id_hash': generate_merchant_id(),
        'card_id_hash': card_id,
        'city': city,
        'country': country,
        'device_type': device_type,
        'channel': channel,
        'hour': hour,
        'dayofweek': dayofweek,
        'is_international': is_international,
        'is_night': is_night,
        'is_fraud': 1
    }


def generate_dataset(n_samples=N_SAMPLES, fraud_ratio=FRAUD_RATIO):
    """Generate complete synthetic transaction dataset."""
    print(f"Generating {n_samples} synthetic transactions...")
    
    transactions = []
    base_date = datetime(2024, 1, 1)
    
    # Create card profiles for consistency
    card_profiles = {}
    n_cards = n_samples // 20  # ~20 transactions per card on average
    
    for i in range(n_cards):
        card_id = generate_card_id()
        card_profiles[card_id] = {
            'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata']),
            'avg_amount': np.random.lognormal(mean=4.0, sigma=1.0),
            'merchant_prefs': np.random.choice(['grocery', 'fuel', 'retail', 'travel', 'food', 'utilities', 'entertainment'], 
                                               size=np.random.randint(3, 8), replace=False)
        }
    
    # Merchant categories
    merchant_cats = ['grocery', 'fuel', 'retail', 'travel', 'food', 'utilities', 'entertainment', 'healthcare', 'education']
    
    # Determine fraud indices
    fraud_indices = set(np.random.choice(n_samples, size=int(n_samples * fraud_ratio), replace=False))
    
    for i in range(n_samples):
        # Select a card (some cards more active than others)
        card_id = np.random.choice(list(card_profiles.keys()))
        profile = card_profiles[card_id]
        
        # Timestamp with realistic patterns (more transactions during day)
        ts_offset = timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        ts = base_date + ts_offset
        
        # Select merchant category
        if np.random.random() < 0.7:
            merchant_cat = np.random.choice(profile['merchant_prefs'])
        else:
            merchant_cat = np.random.choice(merchant_cats)
        
        # Generate transaction
        if i in fraud_indices:
            tx = generate_fraud_transaction(card_id, merchant_cat, ts, profile)
        else:
            tx = generate_normal_transaction(card_id, merchant_cat, ts)
        
        transactions.append(tx)
    
    # Sort by timestamp
    transactions.sort(key=lambda x: x['ts'])
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Add velocity features (rolling window calculations)
    print("Computing velocity features...")
    df = add_velocity_features(df)
    
    print(f"Dataset generated: {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    
    return df


def add_velocity_features(df):
    """Add velocity and behavioral features based on historical data."""
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Initialize velocity columns
    df['prev_24h_tx_count_card'] = 0.0
    df['prev_24h_amt_card'] = 0.0
    df['prev_1h_tx_count_card'] = 0.0
    df['velocity_amt_1h'] = 0.0
    
    # Group by card and calculate rolling features
    for card_id in df['card_id_hash'].unique():
        card_mask = df['card_id_hash'] == card_id
        card_df = df[card_mask].copy()
        
        if len(card_df) < 2:
            continue
        
        # For each transaction, look back 24h and 1h
        for idx in card_df.index:
            current_ts = df.loc[idx, 'ts']
            
            # 24h window
            mask_24h = (df['card_id_hash'] == card_id) & \
                       (df['ts'] < current_ts) & \
                       (df['ts'] >= current_ts - timedelta(hours=24))
            
            # 1h window
            mask_1h = (df['card_id_hash'] == card_id) & \
                      (df['ts'] < current_ts) & \
                      (df['ts'] >= current_ts - timedelta(hours=1))
            
            df.loc[idx, 'prev_24h_tx_count_card'] = mask_24h.sum()
            df.loc[idx, 'prev_24h_amt_card'] = df.loc[mask_24h, 'amount'].sum()
            df.loc[idx, 'prev_1h_tx_count_card'] = mask_1h.sum()
            df.loc[idx, 'velocity_amt_1h'] = df.loc[mask_1h, 'amount'].sum()
    
    return df


def save_dataset(df, output_dir='data'):
    """Save dataset to CSV and Parquet formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV for easy viewing
    csv_path = os.path.join(output_dir, 'transactions.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Parquet for efficient storage
    parquet_path = os.path.join(output_dir, 'transactions.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path}")
    
    return csv_path, parquet_path


if __name__ == '__main__':
    # Generate dataset
    df = generate_dataset()
    
    # Display statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total transactions: {len(df):,}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum():,}")
    print(f"Normal transactions: {(df['is_fraud'] == 0).sum():,}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"\nAmount statistics:")
    print(df.groupby('is_fraud')['amount'].describe())
    print(f"\nTop merchant categories:")
    print(df['merchant_cat'].value_counts().head())
    
    # Save dataset
    save_dataset(df)
    print("\nDataset generation complete!")
