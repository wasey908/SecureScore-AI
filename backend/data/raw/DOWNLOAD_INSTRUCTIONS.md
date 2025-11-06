# Dataset Download Instructions

This fraud detection system uses the **Credit Card Fraud Detection** dataset from Kaggle.

## Option 1: Manual Download (Recommended)

1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click "Download" button (requires free Kaggle account)
3. Extract `creditcard.csv` from the downloaded zip
4. Place `creditcard.csv` in this directory (`/app/data/raw/creditcard.csv`)

## Option 2: Using Kaggle API

If you have Kaggle CLI set up:

```bash
cd /app
bash scripts/download_dataset.sh
```

## Dataset Information

- **Size**: ~150MB
- **Records**: 284,807 transactions
- **Fraud Cases**: 492 (0.172%)
- **Features**: 30 (Time, Amount, V1-V28 PCA components)
- **Format**: CSV

## After Download

Once you have `creditcard.csv` in this directory, train the model:

```bash
cd /app/backend
python -m app.ml.train
```

This will:
1. Load and preprocess the data
2. Train XGBoost and Logistic Regression models
3. Save the best model to `backend/app/ml/model.pkl`
4. Create test samples in `data/processed/`

Expected training time: 2-5 minutes on standard hardware.
