"""Training script for fraud detection models"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
from pathlib import Path
import logging
from datetime import datetime, timezone
import sys

from app.ml.features import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionTrainer:
    """Train and evaluate fraud detection models"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load fraud detection dataset"""
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                "Please download the Kaggle Credit Card Fraud dataset and place it in data/raw/creditcard.csv"
            )
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} transactions")
        logger.info(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess and split data"""
        logger.info("Preprocessing data...")
        
        # Prepare features
        X, y = self.feature_engineer.prepare_features(df, target_col='Class', fit=True)
        
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Number of features: {len(X.columns)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Save test sample for demo
        test_sample = X_test.head(100).copy()
        test_sample['Class'] = y_test.head(100)
        test_sample_path = self.data_path.parent.parent / 'processed' / 'test_sample.csv'
        test_sample_path.parent.mkdir(parents=True, exist_ok=True)
        test_sample.to_csv(test_sample_path, index=False)
        logger.info(f"Saved test sample to {test_sample_path}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        """Handle class imbalance using SMOTE"""
        logger.info("Applying SMOTE to handle class imbalance...")
        
        original_counts = y_train.value_counts()
        logger.info(f"Original class distribution: {original_counts.to_dict()}")
        
        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        new_counts = pd.Series(y_resampled).value_counts()
        logger.info(f"After SMOTE: {new_counts.to_dict()}")
        
        return X_resampled, y_resampled
    
    def train_baseline(self, X_train, y_train, X_test, y_test):
        """Train baseline Logistic Regression model"""
        logger.info("\n" + "="*50)
        logger.info("Training Baseline: Logistic Regression")
        logger.info("="*50)
        
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = self.evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        return model, results
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        logger.info("\n" + "="*50)
        logger.info("Training Primary Model: XGBoost")
        logger.info("="*50)
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            use_label_encoder=False,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = self.evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
        
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        return model, results
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba, model_name: str) -> dict:
        """Evaluate model performance"""
        results = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall: {results['recall']:.4f}")
        logger.info(f"  F1-Score: {results['f1_score']:.4f}")
        
        logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")
        logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
        
        return results
    
    def save_models(self, output_dir: Path):
        """Save trained models and preprocessing"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best model (XGBoost)
        model_path = output_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.models['xgboost'], f)
        logger.info(f"Saved XGBoost model to {model_path}")
        
        # Save feature engineer
        preprocess_path = output_dir / 'preprocess.pkl'
        with open(preprocess_path, 'wb') as f:
            pickle.dump(self.feature_engineer, f)
        logger.info(f"Saved feature engineer to {preprocess_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'XGBoost',
            'training_date': datetime.now(timezone.utc).isoformat(),
            'version': '1.0.0',
            'results': self.results,
            'feature_count': len(self.feature_engineer.feature_columns)
        }
        
        metadata_path = output_dir / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")
        
        logger.info("\nModel training completed successfully!")
    
    def train_all(self):
        """Complete training pipeline"""
        # Load data
        df = self.load_data()
        
        # Preprocess
        X_train, X_test, y_train, y_test = self.preprocess_data(df)
        
        # Handle imbalance
        X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
        
        # Train models
        self.train_baseline(X_train_balanced, y_train_balanced, X_test, y_test)
        self.train_xgboost(X_train_balanced, y_train_balanced, X_test, y_test)
        
        # Compare models
        logger.info("\n" + "="*50)
        logger.info("Model Comparison")
        logger.info("="*50)
        for model_name, results in self.results.items():
            logger.info(f"\n{model_name}:")
            for metric, value in results.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Save models
        output_dir = Path(__file__).parent
        self.save_models(output_dir)


def main():
    """Main training script"""
    # Default data path
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'creditcard.csv'
    
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    
    try:
        trainer = FraudDetectionTrainer(data_path)
        trainer.train_all()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("\nTo download the dataset:")
        logger.info("1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        logger.info(f"2. Download creditcard.csv and place it at: {data_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
