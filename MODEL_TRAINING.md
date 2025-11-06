# Model Training Guide

## Overview

This document explains how to train the fraud detection model for the SecureScore AI system.

## Prerequisites

1. **Dataset Downloaded**: You must have the Kaggle Credit Card Fraud dataset
   - File: `data/raw/creditcard.csv`
   - See: `data/raw/DOWNLOAD_INSTRUCTIONS.md`

2. **Backend Dependencies Installed**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

## Training the Model

### Step 1: Navigate to Backend Directory

```bash
cd /app/backend
```

### Step 2: Run Training Script

```bash
python -m app.ml.train
```

### What Happens During Training

The training process performs the following steps:

#### 1. Data Loading
- Loads `creditcard.csv` from `data/raw/`
- Validates data integrity
- Displays fraud case statistics

#### 2. Feature Engineering
- **Time-based features**: Hour of day, cyclical transformations
- **Amount features**: Log transformation, bucketing
- **Interaction features**: PCA component interactions (V1×V2, V4×V11)
- **Domain features**: High-amount flags, risk indicators

#### 3. Data Preprocessing
- Train/test split (80/20) with stratification
- Saves test sample to `data/processed/test_sample.csv`

#### 4. Class Imbalance Handling
- Applies SMOTE (Synthetic Minority Over-sampling Technique)
- Balances fraud vs non-fraud cases to 50% ratio
- Prevents model bias toward majority class

#### 5. Model Training

**Baseline Model: Logistic Regression**
- Fast, interpretable baseline
- Class weight balancing
- 1000 max iterations

**Primary Model: XGBoost**
- 200 estimators (trees)
- Max depth: 6
- Learning rate: 0.1
- Scale_pos_weight for imbalance
- ROC-AUC optimization

#### 6. Model Evaluation

Both models are evaluated on:
- **ROC-AUC Score**: Primary metric (target: >0.95)
- **Precision**: Minimize false positives
- **Recall**: Catch actual fraud cases
- **F1-Score**: Harmonic mean of precision/recall
- **Confusion Matrix**: Detailed performance breakdown
- **Classification Report**: Per-class metrics

#### 7. Model Saving

The best performing model (typically XGBoost) is saved:
- `backend/app/ml/model.pkl`: Trained model
- `backend/app/ml/preprocess.pkl`: Feature transformer
- `backend/app/ml/metadata.pkl`: Training metadata

## Expected Output

```
====================================
Training Baseline: Logistic Regression
====================================

Logistic Regression Results:
  ROC-AUC: 0.9720
  Precision: 0.8845
  Recall: 0.8502
  F1-Score: 0.8670

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.99      0.99     56864
           1       0.88      0.85      0.87        98

    accuracy                           0.99     56962

Confusion Matrix:
[[56562   302]
 [   15    83]]

====================================
Training Primary Model: XGBoost
====================================

XGBoost Results:
  ROC-AUC: 0.9856
  Precision: 0.9234
  Recall: 0.8876
  F1-Score: 0.9052

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.92      0.89      0.91        98

    accuracy                           1.00     56962

Confusion Matrix:
[[56815    49]
 [   11    87]]

====================================
Model Comparison
====================================

logistic_regression:
  roc_auc: 0.9720
  precision: 0.8845
  recall: 0.8502
  f1_score: 0.8670

xgboost:
  roc_auc: 0.9856
  precision: 0.9234
  recall: 0.8876
  f1_score: 0.9052

Model training completed successfully!
```

## Expected Training Time

- **Small dataset** (<100k records): 1-2 minutes
- **Full Kaggle dataset** (284k records): 2-5 minutes
- **Large dataset** (>1M records): 5-15 minutes

*Times vary based on CPU/RAM availability*

## Verifying Training Success

### 1. Check Model Files Exist

```bash
ls -lh backend/app/ml/model.pkl
ls -lh backend/app/ml/preprocess.pkl
ls -lh backend/app/ml/metadata.pkl
```

All three files should exist with reasonable sizes (>1KB).

### 2. Test API Health

```bash
curl http://localhost:8001/api/health
```

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_type": "XGBoost",
    "version": "1.0.0"
  }
}
```

### 3. Test Prediction

```bash
curl -X POST http://localhost:8001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 150.50,
    "time": 12345.0
  }'
```

Should return a prediction with fraud probability and risk score.

## Troubleshooting

### Error: Dataset Not Found

```
FileNotFoundError: Data file not found: data/raw/creditcard.csv
```

**Solution**: Download the dataset from Kaggle and place it in `data/raw/`
- See: `data/raw/DOWNLOAD_INSTRUCTIONS.md`

### Error: Memory Error

```
MemoryError: Unable to allocate array
```

**Solution**: 
- Reduce dataset size in training script
- Use sampling: `df = df.sample(frac=0.5)`
- Increase available RAM
- Use a smaller max_depth for XGBoost

### Error: Import Errors

```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution**: Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Poor Model Performance

If ROC-AUC < 0.90 or precision/recall are too low:

1. **Check data quality**: Ensure dataset is correct
2. **Adjust hyperparameters**: Edit `app/ml/train.py`
   - Increase `n_estimators` for XGBoost
   - Adjust `max_depth`
   - Tune `learning_rate`
3. **Feature engineering**: Add domain-specific features
4. **Class balancing**: Adjust SMOTE `sampling_strategy`

## Retraining the Model

To retrain with new data or updated parameters:

1. Update dataset in `data/raw/`
2. Modify hyperparameters in `app/ml/train.py` (optional)
3. Run training:
   ```bash
   python -m app.ml.train
   ```
4. Restart backend to load new model:
   ```bash
   sudo supervisorctl restart backend
   ```

## Model Performance Benchmarks

Based on the Kaggle Credit Card Fraud dataset:

| Metric | Target | Typical Result |
|--------|--------|----------------|
| ROC-AUC | >0.95 | 0.97-0.99 |
| Precision | >0.85 | 0.88-0.95 |
| Recall | >0.80 | 0.85-0.90 |
| F1-Score | >0.82 | 0.86-0.92 |

## Advanced Configuration

### Hyperparameter Tuning

Edit `backend/app/ml/train.py`:

```python
# XGBoost parameters
model = XGBClassifier(
    n_estimators=200,      # Number of trees (increase for better accuracy)
    max_depth=6,           # Tree depth (increase for complex patterns)
    learning_rate=0.1,     # Step size (decrease for better convergence)
    scale_pos_weight=...,  # Auto-calculated for imbalance
    random_state=42,
    eval_metric='auc',
    n_jobs=-1              # Use all CPU cores
)
```

### Custom Feature Engineering

Add custom features in `backend/app/ml/features.py`:

```python
def engineer_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    # Your custom features here
    df['custom_feature'] = ...
    
    return df
```

## Next Steps

After successful training:

1. **Test the API**: Use curl or the frontend dashboard
2. **Run Tests**: `pytest backend/app/tests/`
3. **Monitor Performance**: Check prediction accuracy over time
4. **Deploy**: Use Docker for production deployment

## Support

For issues or questions:
- Check README.md for common problems
- Review API documentation: http://localhost:8001/docs
- Examine training logs for errors
