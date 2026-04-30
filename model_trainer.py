"""
Model Training Module for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score, roc_auc_score,
    precision_recall_curve, f1_score,
    accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

from src.pipeline import get_all_pipelines
from src.features import add_features, DROP_FEATURES, TARGET


class FraudDetectionTrainer:
    """Trainer class for fraud detection models."""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, df):
        """Prepare data for training with time-based split."""
        print("Preparing data...")
        
        # Add features
        df = add_features(df)
        
        # Time-based split (80/20)
        df = df.sort_values('ts').reset_index(drop=True)
        cutoff = int(len(df) * 0.8)
        
        train_df = df.iloc[:cutoff].copy()
        test_df = df.iloc[cutoff:].copy()
        
        print(f"Train size: {len(train_df):,}")
        print(f"Test size: {len(test_df):,}")
        print(f"Train fraud rate: {train_df[TARGET].mean():.2%}")
        print(f"Test fraud rate: {test_df[TARGET].mean():.2%}")
        
        # Prepare features
        drop_cols = DROP_FEATURES + [TARGET, 'ts'] if 'ts' in train_df.columns else DROP_FEATURES + [TARGET]
        
        X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
        y_train = train_df[TARGET]
        
        X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
        y_test = test_df[TARGET]
        
        return X_train, X_test, y_train, y_test
    
    def train_all_models(self, X_train, X_test, y_train, y_test, use_smote=True):
        """Train all available models."""
        pipelines = get_all_pipelines(use_smote=use_smote)
        
        for name, pipeline in pipelines.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print(f"{'='*50}")
            
            try:
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Predict
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                
                # Evaluate
                results = self._evaluate_model(y_test, y_pred, y_proba, name)
                
                # Store
                self.models[name] = pipeline
                self.results[name] = results
                
                print(f"\nResults for {name}:")
                print(f"  PR-AUC: {results['pr_auc']:.4f}")
                print(f"  ROC-AUC: {results['roc_auc']:.4f}")
                print(f"  F1 Score: {results['f1']:.4f}")
                print(f"  Recall: {results['recall']:.4f}")
                print(f"  Precision: {results['precision']:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        # Select best model based on PR-AUC
        if self.results:
            self.best_model_name = max(self.results, key=lambda x: self.results[x]['pr_auc'])
            self.best_model = self.models[self.best_model_name]
            print(f"\n{'='*50}")
            print(f"Best Model: {self.best_model_name}")
            print(f"Best PR-AUC: {self.results[self.best_model_name]['pr_auc']:.4f}")
            print(f"{'='*50}")
        
        return self.results
    
    def _evaluate_model(self, y_true, y_pred, y_proba, model_name):
        """Evaluate model performance."""
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'pr_auc': average_precision_score(y_true, y_proba),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        return results
    
    def find_optimal_threshold(self, model_name=None, fn_cost=5000, fp_cost=50):
        """Find cost-optimal threshold for classification."""
        if model_name is None:
            model_name = self.best_model_name
        
        results = self.results[model_name]
        y_true = results['y_true']
        y_proba = results['y_proba']
        
        best_threshold = 0.5
        best_cost = float('inf')
        
        thresholds = np.linspace(0.01, 0.99, 99)
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate cost
            cost = fn * fn_cost + fp * fp_cost
            
            if cost < best_cost:
                best_cost = cost
                best_threshold = threshold
        
        print(f"\nOptimal Threshold for {model_name}:")
        print(f"  Threshold: {best_threshold:.3f}")
        print(f"  Estimated Cost: ₹{best_cost:,.0f}")
        print(f"  (FN Cost: ₹{fn_cost}, FP Cost: ₹{fp_cost})")
        
        return best_threshold, best_cost
    
    def save_best_model(self, threshold=None):
        """Save the best model with metadata."""
        if self.best_model is None:
            raise ValueError("No model trained yet!")
        
        # Find optimal threshold if not provided
        if threshold is None:
            threshold, _ = self.find_optimal_threshold()
        
        # Prepare model bundle
        model_bundle = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'threshold': threshold,
            'metrics': self.results[self.best_model_name],
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        # Save
        model_path = os.path.join(self.model_dir, 'fraud_detection_model.joblib')
        joblib.dump(model_bundle, model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Also save individual models
        for name, model in self.models.items():
            path = os.path.join(self.model_dir, f'{name}.joblib')
            joblib.dump(model, path)
        
        return model_path
    
    def plot_results(self, output_dir='outputs'):
        """Generate evaluation plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Model Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # PR-AUC Comparison
        ax1 = axes[0, 0]
        names = list(self.results.keys())
        pr_aucs = [self.results[n]['pr_auc'] for n in names]
        ax1.barh(names, pr_aucs, color='steelblue')
        ax1.set_xlabel('PR-AUC')
        ax1.set_title('Model Comparison - PR-AUC (Higher is Better)')
        ax1.set_xlim(0, 1)
        for i, v in enumerate(pr_aucs):
            ax1.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # ROC-AUC Comparison
        ax2 = axes[0, 1]
        roc_aucs = [self.results[n]['roc_auc'] for n in names]
        ax2.barh(names, roc_aucs, color='forestgreen')
        ax2.set_xlabel('ROC-AUC')
        ax2.set_title('Model Comparison - ROC-AUC')
        ax2.set_xlim(0, 1)
        for i, v in enumerate(roc_aucs):
            ax2.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # F1 Score Comparison
        ax3 = axes[1, 0]
        f1s = [self.results[n]['f1'] for n in names]
        ax3.barh(names, f1s, color='coral')
        ax3.set_xlabel('F1 Score')
        ax3.set_title('Model Comparison - F1 Score')
        ax3.set_xlim(0, 1)
        for i, v in enumerate(f1s):
            ax3.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Recall Comparison
        ax4 = axes[1, 1]
        recalls = [self.results[n]['recall'] for n in names]
        ax4.barh(names, recalls, color='mediumpurple')
        ax4.set_xlabel('Recall')
        ax4.set_title('Model Comparison - Recall (Sensitivity)')
        ax4.set_xlim(0, 1)
        for i, v in enumerate(recalls):
            ax4.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix for Best Model
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = self.results[self.best_model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {self.best_model_name}')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curves
        fig, ax = plt.subplots(figsize=(10, 8))
        for name, results in self.results.items():
            precision, recall, _ = precision_recall_curve(
                results['y_true'], results['y_proba']
            )
            ax.plot(recall, precision, label=f"{name} (AP={results['pr_auc']:.3f})")
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPlots saved to {output_dir}/")
        return output_dir


def train_fraud_detection_model(data_path='data/transactions.parquet', 
                               model_dir='models',
                               output_dir='outputs',
                               use_smote=True):
    """
    Complete training pipeline.
    
    Args:
        data_path: Path to transaction data
        model_dir: Directory to save models
        output_dir: Directory to save outputs
        use_smote: Whether to use SMOTE for balancing
    
    Returns:
        trainer: Trained FraudDetectionTrainer instance
    """
    # Load data
    print("Loading data...")
    df = pd.read_parquet(data_path)
    
    # Initialize trainer
    trainer = FraudDetectionTrainer(model_dir=model_dir)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Train all models
    results = trainer.train_all_models(X_train, X_test, y_train, y_test, use_smote=use_smote)
    
    # Find optimal threshold
    threshold, _ = trainer.find_optimal_threshold()
    
    # Save model
    trainer.save_best_model(threshold=threshold)
    
    # Generate plots
    trainer.plot_results(output_dir=output_dir)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    
    return trainer


if __name__ == '__main__':
    trainer = train_fraud_detection_model()
