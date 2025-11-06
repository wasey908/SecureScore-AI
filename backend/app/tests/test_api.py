"""API endpoint tests"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "model_loaded" in data
    assert data["status"] in ["healthy", "degraded", "unhealthy"]


def test_predict_endpoint_structure():
    """Test predict endpoint with valid payload structure"""
    payload = {
        "amount": 150.50,
        "time": 12345.0,
        "v1": -1.5,
        "v2": 2.3,
        "v3": 0.5,
        "v4": 1.2
    }
    
    response = client.post("/api/predict", json=payload)
    
    # Should return 503 if model not trained, 200 if trained
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "fraud_probability" in data
        assert "risk_score" in data
        assert "risk_label" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert 0 <= data["fraud_probability"] <= 1
        assert 0 <= data["risk_score"] <= 100
        assert data["risk_label"] in ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]


def test_predict_invalid_payload():
    """Test predict endpoint with invalid payload"""
    # Missing required fields
    payload = {
        "amount": 150.50
        # Missing 'time' field
    }
    
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_negative_amount():
    """Test predict endpoint with negative amount"""
    payload = {
        "amount": -100.0,  # Invalid: negative amount
        "time": 12345.0
    }
    
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_sample_transactions():
    """Test sample transactions endpoint"""
    response = client.get("/api/sample-transactions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    
    if len(data) > 0:
        transaction = data[0]
        assert "amount" in transaction
        assert "time" in transaction
        assert "is_fraud" in transaction


def test_analytics_endpoint():
    """Test analytics endpoint"""
    response = client.get("/api/analytics/risk-distribution")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
