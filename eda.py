"""
Exploratory Data Analysis Module for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(data_path='data/transactions.parquet'):
    """Load transaction data."""
    df = pd.read_parquet(data_path)
    df['ts'] = pd.to_datetime(df['ts'])
    return df


def basic_stats(df):
    """Generate basic statistics."""
    print("="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Total Transactions: {len(df):,}")
    print(f"Fraudulent Transactions: {df['is_fraud'].sum():,}")
    print(f"Normal Transactions: {(df['is_fraud'] == 0).sum():,}")
    print(f"Fraud Rate: {df['is_fraud'].mean():.2%}")
    
    print("\nDate Range:")
    print(f"  From: {df['ts'].min()}")
    print(f"  To: {df['ts'].max()}")
    
    print("\nAmount Statistics by Class:")
    print(df.groupby('is_fraud')['amount'].describe())
    
    return {
        'total': len(df),
        'fraud_count': df['is_fraud'].sum(),
        'fraud_rate': df['is_fraud'].mean()
    }


def plot_fraud_distribution(df, output_dir='outputs'):
    """Plot fraud distribution."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    fraud_counts = df['is_fraud'].value_counts()
    labels = ['Normal', 'Fraud']
    colors = ['#2ecc71', '#e74c3c']
    
    axes[0].pie(fraud_counts, labels=labels, autopct='%1.2f%%', 
                colors=colors, startangle=90, explode=(0, 0.1))
    axes[0].set_title('Transaction Distribution')
    
    # Bar chart
    axes[1].bar(labels, fraud_counts.values, color=colors)
    axes[1].set_ylabel('Count')
    axes[1].set_title('Transaction Count by Class')
    axes[1].set_yscale('log')  # Log scale due to imbalance
    
    for i, v in enumerate(fraud_counts.values):
        axes[1].text(i, v, f'{v:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fraud_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/fraud_distribution.png")


def plot_amount_distribution(df, output_dir='outputs'):
    """Plot transaction amount distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribution by class
    normal = df[df['is_fraud'] == 0]['amount']
    fraud = df[df['is_fraud'] == 1]['amount']
    
    # Histogram
    axes[0, 0].hist(normal, bins=50, alpha=0.7, label='Normal', color='green', density=True)
    axes[0, 0].hist(fraud, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
    axes[0, 0].set_xlabel('Amount')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Transaction Amount Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, df['amount'].quantile(0.99))
    
    # Box plot
    df_plot = df[df['amount'] <= df['amount'].quantile(0.95)]  # Remove extreme outliers
    sns.boxplot(data=df_plot, x='is_fraud', y='amount', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Class (0=Normal, 1=Fraud)')
    axes[0, 1].set_ylabel('Amount')
    axes[0, 1].set_title('Amount Distribution by Class')
    
    # Log scale histogram
    axes[1, 0].hist(np.log1p(normal), bins=50, alpha=0.7, label='Normal', color='green', density=True)
    axes[1, 0].hist(np.log1p(fraud), bins=50, alpha=0.7, label='Fraud', color='red', density=True)
    axes[1, 0].set_xlabel('Log(Amount + 1)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Log-Scaled Amount Distribution')
    axes[1, 0].legend()
    
    # Violin plot
    sns.violinplot(data=df_plot, x='is_fraud', y='amount', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Class (0=Normal, 1=Fraud)')
    axes[1, 1].set_ylabel('Amount')
    axes[1, 1].set_title('Amount Distribution Shape by Class')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'amount_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/amount_distribution.png")


def plot_time_patterns(df, output_dir='outputs'):
    """Plot time-based patterns."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Hour of day
    hour_fraud = df.groupby('hour')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
    
    ax1 = axes[0, 0]
    ax2 = ax1.twinx()
    ax1.bar(hour_fraud['hour'], hour_fraud['count'], alpha=0.6, color='steelblue', label='Total')
    ax2.plot(hour_fraud['hour'], hour_fraud['mean'] * 100, color='red', marker='o', linewidth=2, label='Fraud %')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Transaction Count', color='steelblue')
    ax2.set_ylabel('Fraud Rate (%)', color='red')
    ax1.set_title('Transactions & Fraud Rate by Hour')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Day of week
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_fraud = df.groupby('dayofweek')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
    
    ax3 = axes[0, 1]
    ax4 = ax3.twinx()
    ax3.bar(day_fraud['dayofweek'], day_fraud['count'], alpha=0.6, color='steelblue')
    ax4.plot(day_fraud['dayofweek'], day_fraud['mean'] * 100, color='red', marker='o', linewidth=2)
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('Transaction Count', color='steelblue')
    ax4.set_ylabel('Fraud Rate (%)', color='red')
    ax3.set_title('Transactions & Fraud Rate by Day')
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(day_names)
    
    # Night transactions
    night_fraud = df.groupby('is_night')['is_fraud'].mean() * 100
    axes[1, 0].bar(['Day (6AM-12AM)', 'Night (12AM-6AM)'], night_fraud.values, 
                   color=['green', 'darkred'])
    axes[1, 0].set_ylabel('Fraud Rate (%)')
    axes[1, 0].set_title('Fraud Rate: Day vs Night')
    for i, v in enumerate(night_fraud.values):
        axes[1, 0].text(i, v, f'{v:.2f}%', ha='center', va='bottom')
    
    # Weekend vs Weekday
    df['is_weekend'] = df['dayofweek'].isin([5, 6])
    weekend_fraud = df.groupby('is_weekend')['is_fraud'].mean() * 100
    axes[1, 1].bar(['Weekday', 'Weekend'], weekend_fraud.values, 
                   color=['steelblue', 'orange'])
    axes[1, 1].set_ylabel('Fraud Rate (%)')
    axes[1, 1].set_title('Fraud Rate: Weekday vs Weekend')
    for i, v in enumerate(weekend_fraud.values):
        axes[1, 1].text(i, v, f'{v:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/time_patterns.png")


def plot_categorical_patterns(df, output_dir='outputs'):
    """Plot categorical feature patterns."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Merchant category
    cat_fraud = df.groupby('merchant_cat')['is_fraud'].agg(['count', 'mean']).reset_index()
    cat_fraud = cat_fraud.sort_values('mean', ascending=False)
    
    axes[0, 0].barh(cat_fraud['merchant_cat'], cat_fraud['mean'] * 100, color='coral')
    axes[0, 0].set_xlabel('Fraud Rate (%)')
    axes[0, 0].set_title('Fraud Rate by Merchant Category')
    
    # Device type
    device_fraud = df.groupby('device_type')['is_fraud'].agg(['count', 'mean']).reset_index()
    axes[0, 1].bar(device_fraud['device_type'], device_fraud['mean'] * 100, color='mediumpurple')
    axes[0, 1].set_ylabel('Fraud Rate (%)')
    axes[0, 1].set_title('Fraud Rate by Device Type')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Channel
    channel_fraud = df.groupby('channel')['is_fraud'].agg(['count', 'mean']).reset_index()
    axes[1, 0].bar(channel_fraud['channel'], channel_fraud['mean'] * 100, color='seagreen')
    axes[1, 0].set_ylabel('Fraud Rate (%)')
    axes[1, 0].set_title('Fraud Rate by Channel')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # International
    intl_fraud = df.groupby('is_international')['is_fraud'].mean() * 100
    axes[1, 1].bar(['Domestic', 'International'], intl_fraud.values, 
                   color=['teal', 'crimson'])
    axes[1, 1].set_ylabel('Fraud Rate (%)')
    axes[1, 1].set_title('Fraud Rate: Domestic vs International')
    for i, v in enumerate(intl_fraud.values):
        axes[1, 1].text(i, v, f'{v:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'categorical_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/categorical_patterns.png")


def plot_velocity_features(df, output_dir='outputs'):
    """Plot velocity feature distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    velocity_cols = ['prev_24h_tx_count_card', 'prev_1h_tx_count_card', 
                     'velocity_amt_1h', 'avg_tx_amt_24h']
    titles = ['24h Transaction Count', '1h Transaction Count', 
              '1h Velocity Amount', '24h Average Amount']
    
    for idx, (col, title) in enumerate(zip(velocity_cols, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Cap extreme values for visualization
        normal_vals = df[df['is_fraud'] == 0][col]
        fraud_vals = df[df['is_fraud'] == 1][col]
        
        q99 = df[col].quantile(0.99)
        normal_vals = normal_vals[normal_vals <= q99]
        fraud_vals = fraud_vals[fraud_vals <= q99]
        
        ax.hist(normal_vals, bins=30, alpha=0.6, label='Normal', color='green', density=True)
        ax.hist(fraud_vals, bins=30, alpha=0.6, label='Fraud', color='red', density=True)
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.set_title(f'{title} Distribution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/velocity_features.png")


def plot_correlation_matrix(df, output_dir='outputs'):
    """Plot correlation matrix of numeric features."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'ts' and 'hash' not in c]
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/correlation_matrix.png")


def generate_full_eda(data_path='data/transactions.parquet', output_dir='outputs'):
    """Generate complete EDA report."""
    print("="*60)
    print("GENERATING EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data(data_path)
    
    # Generate all plots
    stats = basic_stats(df)
    plot_fraud_distribution(df, output_dir)
    plot_amount_distribution(df, output_dir)
    plot_time_patterns(df, output_dir)
    plot_categorical_patterns(df, output_dir)
    plot_velocity_features(df, output_dir)
    plot_correlation_matrix(df, output_dir)
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print(f"All plots saved to: {output_dir}/")
    print("="*60)
    
    return stats


if __name__ == '__main__':
    generate_full_eda()
