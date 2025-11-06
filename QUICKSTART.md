# Quick Start Guide - SecureScore AI

Get up and running with the fraud detection system in minutes!

## 🚀 Fast Track Setup

### 1. Download Dataset (5 minutes)

Visit https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and download `creditcard.csv`

Place it here: `/app/data/raw/creditcard.csv`

### 2. Train Model (3-5 minutes)

```bash
cd /app/backend
python -m app.ml.train
```

Wait for training to complete. You should see:
- Logistic Regression results
- XGBoost results  
- Model saved successfully message

### 3. Start Backend (if not running)

```bash
# Check if backend is running
curl http://localhost:8001/api/health

# If not, start it
cd /app/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### 4. Start Frontend (if not running)

```bash
# In a new terminal
cd /app/frontend
yarn start
```

### 5. Access Dashboard

Open browser: **http://localhost:3000**

## ✅ Verify Everything Works

### Test 1: Health Check

```bash
curl http://localhost:8001/api/health
```

Expected: `"model_loaded": true`

### Test 2: Make a Prediction

In the dashboard:
1. Enter **Amount**: 150.50
2. Enter **Time**: 12345
3. Click "Check Risk Score"

You should see a prediction result!

### Test 3: Run Tests

```bash
cd /app/backend
pytest app/tests/ -v
```

Expected: All tests pass ✅

## 🎯 What to Try Next

### Low-Risk Transaction
- Amount: $50
- Time: 5000
- Expected: LOW_RISK

### High-Risk Transaction  
- Amount: $5000
- Time: 3600
- Expected: Likely MEDIUM or HIGH_RISK

### Check Analytics
- Submit multiple predictions
- View charts update in real-time
- Check flagged transactions table

## 📊 Key Features to Explore

1. **Transaction Analysis Form**: Left panel
2. **Risk Analytics Charts**: Right panel - pie chart and line chart
3. **Statistics Cards**: Top row - total predictions, high risk count, etc.
4. **Flagged Transactions Table**: Appears when you have medium/high-risk transactions
5. **Model Status**: Top right - shows model health

## 🐛 Quick Troubleshooting

### Model Not Loaded?
```bash
cd /app/backend
python -m app.ml.train
sudo supervisorctl restart backend
```

### Frontend Not Loading?
```bash
sudo supervisorctl restart frontend
# Wait 30 seconds, then refresh browser
```

### Dataset Missing?
See `/app/data/raw/DOWNLOAD_INSTRUCTIONS.md`

## 🔗 Important Links

- **Dashboard**: http://localhost:3000
- **API Docs**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/api/health

## 📚 Documentation

- `README.md` - Complete documentation
- `MODEL_TRAINING.md` - Detailed training guide
- `data/raw/DOWNLOAD_INSTRUCTIONS.md` - Dataset download help

## 💡 Pro Tips

1. **Clear helpful warning**: The yellow warning box tells you if model needs training
2. **Toast notifications**: Green = low risk, Yellow = medium, Red = high risk
3. **Real-time updates**: Charts update automatically after each prediction
4. **Responsive**: Try resizing your browser window
5. **Dark theme**: Professional fintech-style UI optimized for analysts

## 🎓 For Portfolio/CV

This project demonstrates:
- ✅ End-to-end ML pipeline (data → training → deployment)
- ✅ Production-grade FastAPI backend
- ✅ Professional React dashboard
- ✅ Docker deployment capability
- ✅ Comprehensive testing
- ✅ Real-world fraud detection use case
- ✅ Clean code architecture
- ✅ Technical documentation



**Need Help?** Check the main README.md or MODEL_TRAINING.md for detailed guides.
