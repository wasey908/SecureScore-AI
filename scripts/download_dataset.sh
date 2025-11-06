#!/bin/bash

# Script to download Kaggle Credit Card Fraud dataset

echo "===================================="
echo "Fraud Detection Dataset Downloader"
echo "===================================="
echo ""

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null
then
    echo "Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check for Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle credentials not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. This downloads kaggle.json"
    echo "5. Place it in ~/.kaggle/kaggle.json"
    echo "6. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "OR download manually:"
    echo "1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    echo "2. Download creditcard.csv"
    echo "3. Place it in: data/raw/creditcard.csv"
    exit 1
fi

# Create data directory
mkdir -p data/raw
mkdir -p data/processed

echo "Downloading dataset from Kaggle..."
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/

# Unzip
echo "Extracting dataset..."
cd data/raw
unzip -o creditcardfraud.zip
rm creditcardfraud.zip
cd ../..

echo ""
echo "✅ Dataset downloaded successfully!"
echo "Location: data/raw/creditcard.csv"
echo ""
echo "Next steps:"
echo "1. cd backend"
echo "2. python -m app.ml.train"
echo ""
