from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List
import pickle
import numpy as np
import pandas as pd

from app.models import (
    TransactionRequest,
    PredictionResponse,
    HealthResponse,
    SampleTransaction,
    ModelInfo
)
from app.services.scoring_service import ScoringService

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(
    title="Fraud Detection & Risk Scoring API",
    description="Real-time fraud detection and risk scoring system for financial transactions",
    version="1.0.0"
)

# Create API router with /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize scoring service
scoring_service = None

@app.on_event("startup")
async def startup_event():
    global scoring_service
    try:
        scoring_service = ScoringService()
        logger.info("Fraud detection model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start but /predict endpoint will not work until model is trained")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# Health check endpoint
@api_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    model_loaded = scoring_service is not None and scoring_service.model is not None
    
    model_info = None
    if model_loaded:
        model_info = ModelInfo(
            model_type=scoring_service.model_type,
            training_date=scoring_service.training_date,
            version=scoring_service.version
        )
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_loaded=model_loaded,
        model_info=model_info
    )

# Prediction endpoint
@api_router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    """Predict fraud probability and risk score for a transaction"""
    if scoring_service is None or scoring_service.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running: python -m app.ml.train"
        )
    
    try:
        prediction = scoring_service.score_transaction(transaction)
        
        # Store prediction in database for analytics
        prediction_doc = prediction.model_dump()
        prediction_doc['created_at'] = datetime.now(timezone.utc).isoformat()
        await db.predictions.insert_one(prediction_doc)
        
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Sample transactions endpoint
@api_router.get("/sample-transactions", response_model=List[SampleTransaction])
async def get_sample_transactions(limit: int = 10):
    """Get sample transactions for testing and demonstration"""
    try:
        # Try to load from processed data
        data_path = ROOT_DIR / "data" / "processed" / "test_sample.csv"
        
        if not data_path.exists():
            # Return synthetic examples if no data available
            return [
                SampleTransaction(
                    amount=120.50,
                    time=1234.0,
                    v1=-1.5,
                    v2=2.3,
                    is_fraud=False
                ),
                SampleTransaction(
                    amount=950.00,
                    time=5678.0,
                    v1=3.2,
                    v2=-2.1,
                    is_fraud=True
                )
            ]
        
        df = pd.read_csv(data_path)
        samples = df.head(limit).to_dict('records')
        
        return [
            SampleTransaction(
                amount=row.get('Amount', 0.0),
                time=row.get('Time', 0.0),
                v1=row.get('V1', 0.0),
                v2=row.get('V2', 0.0),
                is_fraud=bool(row.get('Class', 0))
            )
            for row in samples
        ]
    except Exception as e:
        logger.error(f"Error loading sample transactions: {e}")
        return []

# Analytics endpoint
@api_router.get("/analytics/risk-distribution")
async def get_risk_distribution():
    """Get distribution of risk scores from recent predictions"""
    try:
        predictions = await db.predictions.find(
            {},
            {"_id": 0, "risk_score": 1, "fraud_probability": 1, "risk_label": 1, "created_at": 1}
        ).sort("created_at", -1).limit(1000).to_list(1000)
        
        if not predictions:
            return {
                "total_predictions": 0,
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "average_risk_score": 0
            }
        
        high_risk = sum(1 for p in predictions if p.get('risk_label') == 'HIGH_RISK')
        medium_risk = sum(1 for p in predictions if p.get('risk_label') == 'MEDIUM_RISK')
        low_risk = sum(1 for p in predictions if p.get('risk_label') == 'LOW_RISK')
        avg_score = sum(p.get('risk_score', 0) for p in predictions) / len(predictions)
        
        return {
            "total_predictions": len(predictions),
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk,
            "average_risk_score": round(avg_score, 2),
            "recent_predictions": predictions[:20]
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return {"error": str(e)}

# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)
