"""
ML Pipeline for Credit Card Fraud Detection
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
import numpy as np

from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FeatureEngineer


def create_preprocessor():
    """Create preprocessing pipeline for features."""
    
    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
    ], remainder='drop')
    
    return preprocessor


def create_logistic_regression_pipeline(use_smote=True):
    """Create Logistic Regression pipeline with SMOTE."""
    preprocessor = create_preprocessor()
    
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    if use_smote:
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('classifier', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
    
    return pipeline


def create_random_forest_pipeline(use_smote=True):
    """Create Random Forest pipeline with SMOTE."""
    preprocessor = create_preprocessor()
    
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    if use_smote:
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('classifier', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
    
    return pipeline


def create_xgboost_pipeline(use_smote=True):
    """Create XGBoost pipeline with SMOTE."""
    preprocessor = create_preprocessor()
    
    classifier = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=50,  # Handle imbalance
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        verbosity=0
    )
    
    if use_smote:
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('classifier', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
    
    return pipeline


def create_lightgbm_pipeline(use_smote=True):
    """Create LightGBM pipeline with SMOTE."""
    preprocessor = create_preprocessor()
    
    classifier = LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    if use_smote:
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('classifier', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
    
    return pipeline


def create_gradient_boosting_pipeline(use_smote=True):
    """Create Gradient Boosting pipeline with SMOTE."""
    preprocessor = create_preprocessor()
    
    classifier = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    if use_smote:
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('classifier', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
    
    return pipeline


def get_all_pipelines(use_smote=True):
    """Get dictionary of all available pipelines."""
    return {
        'logistic_regression': create_logistic_regression_pipeline(use_smote),
        'random_forest': create_random_forest_pipeline(use_smote),
        'xgboost': create_xgboost_pipeline(use_smote),
        'lightgbm': create_lightgbm_pipeline(use_smote),
        'gradient_boosting': create_gradient_boosting_pipeline(use_smote)
    }
