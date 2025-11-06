"""Scoring service for fraud detection"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import logging

from app.models import TransactionRequest, PredictionResponse

logger = logging.getLogger(__name__)


class ScoringService:
    """Service for scoring transactions with fraud detection model"""
    
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.metadata = None
        self.model_type = "Unknown"
        self.training_date = "Unknown"
        self.version = "Unknown"
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model and preprocessing pipeline"""
        ml_dir = Path(__file__).parent.parent / 'ml'
        
        try:
            # Load model
            model_path = ml_dir / 'model.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model file not found: {model_path}")
                return
            
            # Load feature engineer
            preprocess_path = ml_dir / 'preprocess.pkl'
            if preprocess_path.exists():
                with open(preprocess_path, 'rb') as f:
                    self.feature_engineer = pickle.load(f)
                logger.info("Feature engineer loaded successfully")
            
            # Load metadata
            metadata_path = ml_dir / 'metadata.pkl'
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                    self.model_type = self.metadata.get('model_type', 'XGBoost')
                    self.training_date = self.metadata.get('training_date', 'Unknown')
                    self.version = self.metadata.get('version', '1.0.0')
                logger.info(f"Loaded {self.model_type} model version {self.version}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def score_transaction(self, transaction: TransactionRequest) -> PredictionResponse:
        """Score a single transaction for fraud risk"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert request to dataframe
        transaction_dict = transaction.model_dump()
        
        # Map field names to match training data
        feature_dict = {
            'Time': transaction_dict['time'],
            'Amount': transaction_dict['amount'],
        }
        
        # Add V1-V28 features
        for i in range(1, 29):
            feature_dict[f'V{i}'] = transaction_dict.get(f'v{i}', 0.0)
        
        df = pd.DataFrame([feature_dict])
        
        # Engineer features
        if self.feature_engineer:
            X, _ = self.feature_engineer.prepare_features(df, target_col='Class', fit=False)
        else:
            X = df
        
        # Predict
        fraud_proba = self.model.predict_proba(X)[0, 1]
        
        # Calculate risk score (0-100)
        risk_score = int(fraud_proba * 100)
        
        # Determine risk label
        if fraud_proba >= 0.2:
            risk_label = "HIGH_RISK"
        elif fraud_proba >= 0.05:
            risk_label = "MEDIUM_RISK"
        else:
            risk_label = "LOW_RISK"
        
        return PredictionResponse(
            fraud_probability=round(float(fraud_proba), 4),
            risk_score=risk_score,
            risk_label=risk_label,
            model_version=self.version,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
