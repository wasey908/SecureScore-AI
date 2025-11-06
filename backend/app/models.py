"""Pydantic models for API requests and responses"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime


class TransactionRequest(BaseModel):
    """Request model for transaction fraud prediction"""
    model_config = ConfigDict(extra="forbid")
    
    amount: float = Field(..., description="Transaction amount", ge=0)
    time: float = Field(..., description="Time in seconds from first transaction", ge=0)
    
    # PCA components from Kaggle dataset (V1-V28)
    v1: Optional[float] = Field(default=0.0, description="PCA component 1")
    v2: Optional[float] = Field(default=0.0, description="PCA component 2")
    v3: Optional[float] = Field(default=0.0, description="PCA component 3")
    v4: Optional[float] = Field(default=0.0, description="PCA component 4")
    v5: Optional[float] = Field(default=0.0, description="PCA component 5")
    v6: Optional[float] = Field(default=0.0, description="PCA component 6")
    v7: Optional[float] = Field(default=0.0, description="PCA component 7")
    v8: Optional[float] = Field(default=0.0, description="PCA component 8")
    v9: Optional[float] = Field(default=0.0, description="PCA component 9")
    v10: Optional[float] = Field(default=0.0, description="PCA component 10")
    v11: Optional[float] = Field(default=0.0, description="PCA component 11")
    v12: Optional[float] = Field(default=0.0, description="PCA component 12")
    v13: Optional[float] = Field(default=0.0, description="PCA component 13")
    v14: Optional[float] = Field(default=0.0, description="PCA component 14")
    v15: Optional[float] = Field(default=0.0, description="PCA component 15")
    v16: Optional[float] = Field(default=0.0, description="PCA component 16")
    v17: Optional[float] = Field(default=0.0, description="PCA component 17")
    v18: Optional[float] = Field(default=0.0, description="PCA component 18")
    v19: Optional[float] = Field(default=0.0, description="PCA component 19")
    v20: Optional[float] = Field(default=0.0, description="PCA component 20")
    v21: Optional[float] = Field(default=0.0, description="PCA component 21")
    v22: Optional[float] = Field(default=0.0, description="PCA component 22")
    v23: Optional[float] = Field(default=0.0, description="PCA component 23")
    v24: Optional[float] = Field(default=0.0, description="PCA component 24")
    v25: Optional[float] = Field(default=0.0, description="PCA component 25")
    v26: Optional[float] = Field(default=0.0, description="PCA component 26")
    v27: Optional[float] = Field(default=0.0, description="PCA component 27")
    v28: Optional[float] = Field(default=0.0, description="PCA component 28")


class PredictionResponse(BaseModel):
    """Response model for fraud prediction"""
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    risk_score: int = Field(..., description="Risk score (0-100)")
    risk_label: str = Field(..., description="Risk classification: LOW_RISK, MEDIUM_RISK, or HIGH_RISK")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")


class ModelInfo(BaseModel):
    """Model information"""
    model_type: str = Field(..., description="Type of ML model used")
    training_date: str = Field(..., description="Date when model was trained")
    version: str = Field(..., description="Model version")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status: healthy, degraded, or unhealthy")
    timestamp: str = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    model_info: Optional[ModelInfo] = Field(None, description="Information about loaded model")


class SampleTransaction(BaseModel):
    """Sample transaction for demonstration"""
    amount: float
    time: float
    v1: float
    v2: float
    is_fraud: bool
