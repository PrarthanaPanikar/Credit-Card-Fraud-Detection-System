"""
FastAPI Application for Credit Card Fraud Detection
Provides REST API endpoints for real-time fraud scoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import os
import sys
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.predictor import FraudPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection API for credit card transactions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (lazy loading)
predictor = None

def get_predictor():
    """Get or initialize fraud predictor."""
    global predictor
    if predictor is None:
        model_path = os.environ.get('MODEL_PATH', 'models/fraud_detection_model.joblib')
        predictor = FraudPredictor(model_path)
    return predictor


# Pydantic models
class Transaction(BaseModel):
    """Transaction data model."""
    tx_id: str = Field(..., description="Unique transaction ID")
    ts: Optional[str] = Field(None, description="Transaction timestamp")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_cat: str = Field(..., description="Merchant category")
    merchant_id_hash: str = Field(..., description="Hashed merchant ID")
    card_id_hash: str = Field(..., description="Hashed card ID")
    city: str = Field(..., description="City")
    country: str = Field(..., description="Country code")
    device_type: str = Field(..., description="Device type (mobile/desktop/tablet/pos)")
    channel: str = Field(..., description="Channel (online/in_store/app/phone)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    dayofweek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    is_international: bool = Field(default=False, description="International transaction flag")
    is_night: bool = Field(default=False, description="Night time transaction flag")
    prev_24h_tx_count_card: Optional[float] = Field(default=0.0, description="Card transaction count in last 24h")
    prev_24h_amt_card: Optional[float] = Field(default=0.0, description="Card transaction amount in last 24h")
    prev_1h_tx_count_card: Optional[float] = Field(default=0.0, description="Card transaction count in last 1h")
    velocity_amt_1h: Optional[float] = Field(default=0.0, description="Velocity amount in last 1h")


class PredictionResult(BaseModel):
    """Prediction result model."""
    transaction_id: str
    fraud_probability: float
    risk_score: int
    decision: str
    threshold: float
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    transactions: List[Transaction]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    results: List[PredictionResult]
    total: int
    flagged: int


class Alert(BaseModel):
    """Fraud alert model."""
    alert_id: str
    transaction_id: str
    risk_score: int
    decision: str
    timestamp: str
    amount: float
    reason: List[str]


# Store alerts (in production, use a database)
alerts_store: List[Alert] = []


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        pred = get_predictor()
        return {
            "status": "healthy",
            "model": pred.model_name,
            "version": pred.version,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/model/info")
async def model_info():
    """Get model information."""
    try:
        pred = get_predictor()
        info = pred.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score", response_model=PredictionResult)
async def score_transaction(transaction: Transaction):
    """
    Score a single transaction for fraud risk.
    
    Returns fraud probability, risk score, and recommended action.
    """
    try:
        pred = get_predictor()
        
        # Convert to dict
        tx_dict = transaction.dict()
        
        # Get prediction
        result = pred.predict_single(tx_dict)
        
        # Create alert if high risk
        if result['risk_score'] >= 70:
            alert = Alert(
                alert_id=f"ALT{datetime.now().strftime('%Y%m%d%H%M%S')}",
                transaction_id=result['transaction_id'],
                risk_score=result['risk_score'],
                decision=result['decision'],
                timestamp=datetime.now().isoformat(),
                amount=transaction.amount,
                reason=generate_alert_reasons(transaction, result['risk_score'])
            )
            alerts_store.append(alert)
        
        return PredictionResult(
            transaction_id=result['transaction_id'],
            fraud_probability=result['fraud_probability'],
            risk_score=result['risk_score'],
            decision=result['decision'],
            threshold=result['threshold'],
            model_version=result['model_version'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/batch", response_model=BatchPredictionResponse)
async def score_batch(request: BatchPredictionRequest):
    """
    Score multiple transactions in batch.
    
    Efficient for processing multiple transactions at once.
    """
    try:
        pred = get_predictor()
        
        # Convert transactions to dicts
        tx_dicts = [t.dict() for t in request.transactions]
        
        # Batch prediction
        results = pred.predict_batch(tx_dicts)
        
        # Convert to response model
        prediction_results = []
        flagged = 0
        
        for tx, result in zip(request.transactions, results):
            if result['risk_score'] >= 70:
                flagged += 1
            
            prediction_results.append(PredictionResult(
                transaction_id=result['transaction_id'],
                fraud_probability=result['fraud_probability'],
                risk_score=result['risk_score'],
                decision=result['decision'],
                threshold=result['threshold'],
                model_version=result['model_version'],
                timestamp=datetime.now().isoformat()
            ))
        
        return BatchPredictionResponse(
            results=prediction_results,
            total=len(prediction_results),
            flagged=flagged
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """Get recent fraud alerts."""
    return {
        "alerts": alerts_store[-limit:][::-1],
        "total": len(alerts_store)
    }


@app.post("/stream")
async def stream_transaction(transaction: Transaction):
    """
    Stream endpoint for real-time processing.
    
    Similar to /score but designed for webhook/Kafka integration.
    """
    try:
        pred = get_predictor()
        result = pred.predict_single(transaction.dict())
        
        return {
            "transaction_id": result['transaction_id'],
            "probability": result['fraud_probability'],
            "decision": result['decision'],
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_alert_reasons(transaction: Transaction, risk_score: int) -> List[str]:
    """Generate human-readable reasons for high-risk alerts."""
    reasons = []
    
    if risk_score >= 90:
        reasons.append("Very high fraud probability")
    elif risk_score >= 80:
        reasons.append("High fraud probability")
    else:
        reasons.append("Elevated risk detected")
    
    if transaction.amount > 25000:
        reasons.append(f"Large transaction amount: ₹{transaction.amount:,.0f}")
    
    if transaction.is_international:
        reasons.append("International transaction")
    
    if transaction.is_night:
        reasons.append("Unusual nighttime transaction")
    
    if transaction.velocity_amt_1h and transaction.velocity_amt_1h > 50000:
        reasons.append("High velocity in recent transactions")
    
    if transaction.prev_1h_tx_count_card and transaction.prev_1h_tx_count_card > 5:
        reasons.append("Unusual transaction frequency")
    
    return reasons


if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
