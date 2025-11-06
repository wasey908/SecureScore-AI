"""ML pipeline tests"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ml.features import FeatureEngineer


def test_feature_engineer_initialization():
    """Test FeatureEngineer initialization"""
    fe = FeatureEngineer()
    assert fe.scaler is not None
    assert fe.feature_columns is None


def test_feature_engineering():
    """Test feature engineering produces correct output"""
    # Create sample data
    df = pd.DataFrame({
        'Time': [1000, 2000, 3000],
        'Amount': [50.0, 150.0, 300.0],
        'V1': [1.0, -1.0, 0.5],
        'V2': [0.5, 1.5, -0.5],
        'V4': [0.2, -0.3, 0.1],
        'V11': [1.1, -1.2, 0.8],
        'Class': [0, 1, 0]
    })
    
    fe = FeatureEngineer()
    df_engineered = fe.engineer_features(df, fit=True)
    
    # Check new features are created
    assert 'hour' in df_engineered.columns
    assert 'hour_sin' in df_engineered.columns
    assert 'hour_cos' in df_engineered.columns
    assert 'log_amount' in df_engineered.columns
    assert 'amount_bucket' in df_engineered.columns
    
    # Check no NaN values
    assert not df_engineered.isnull().any().any()
    
    # Check shapes
    assert len(df_engineered) == len(df)


def test_prepare_features():
    """Test feature preparation for model input"""
    df = pd.DataFrame({
        'Time': [1000, 2000, 3000],
        'Amount': [50.0, 150.0, 300.0],
        'V1': [1.0, -1.0, 0.5],
        'V2': [0.5, 1.5, -0.5],
        'Class': [0, 1, 0]
    })
    
    fe = FeatureEngineer()
    X, y = fe.prepare_features(df, target_col='Class', fit=True)
    
    # Check separation of features and target
    assert 'Class' not in X.columns
    assert len(y) == len(df)
    assert len(X) == len(df)
    
    # Check no NaN values
    assert not X.isnull().any().any()
    
    # Check feature columns are stored
    assert fe.feature_columns is not None
    assert len(fe.feature_columns) > 0


def test_feature_names():
    """Test getting feature names"""
    df = pd.DataFrame({
        'Time': [1000],
        'Amount': [50.0],
        'V1': [1.0],
        'Class': [0]
    })
    
    fe = FeatureEngineer()
    X, y = fe.prepare_features(df, target_col='Class', fit=True)
    
    feature_names = fe.get_feature_importance_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert len(feature_names) == len(X.columns)
