"""
serve.py  —  Online Model Serving (REST API)
=============================================
Serves the production LightGBM model as a REST endpoint. One session
record in, one purchase prediction out.

HOW TO RUN
----------
    # 1. Train the model first
    python main.py

    # 2. Start the server
    uvicorn serve:app --reload --port 8000

    # 3. Open the auto-generated Swagger docs
    http://localhost:8000/docs

    # 4. POST a single session for scoring
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{
               "Administrative": 2,
               "Administrative_Duration": 80.0,
               "Informational": 0,
               "Informational_Duration": 0.0,
               "ProductRelated": 15,
               "ProductRelated_Duration": 450.0,
               "BounceRates": 0.02,
               "ExitRates": 0.04,
               "PageValues": 12.5,
               "SpecialDay": 0.0,
               "Month": "Nov",
               "OperatingSystems": 2,
               "Browser": 2,
               "Region": 1,
               "TrafficType": 2,
               "VisitorType": "Returning_Visitor",
               "Weekend": false
             }'

ONLINE vs BATCH INFERENCE
--------------------------
Online (this file):
  * One session at a time
  * Low latency is the goal (< 100ms per prediction)
  * Triggered by a real-time event (session active, visitor browsing)
  * Model loaded ONCE at startup, not per request

Batch (batch_infer.py):
  * Thousands of rows at once from a CSV
  * Throughput is the goal, not latency
  * Scheduled or triggered by data file arrival (e.g., end-of-day export)

WHY LOAD FROM REGISTRY
-----------------------
A common mistake is loading the newest .pkl file in the models/ directory.
A candidate model (not yet approved) might be the newest file.
The correct approach: load from registry.json → production entry only.
Models explicitly promoted to 'production' are the only ones served.
"""

import time
import logging

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.paths import create_directories
from src.features.build_features import build_features
from src.registry.versioning import get_production_model
from src.monitoring.logging import log_predictions, log_latency

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="E-Commerce Purchase Prediction API",
    description=(
        "Predicts whether an active browsing session will end in a purchase. "
        "Run `python main.py` to train a model before starting the server."
    ),
    version="1.0.0",
)

_model = None
_meta  = None


@app.on_event("startup")
def load_model_on_startup():
    """
    Load the production model once when the server starts.
    Loading per-request would add 50–500ms of disk I/O to every prediction.
    """
    global _model, _meta
    create_directories()
    try:
        _model, _meta = get_production_model()
        logger.info(f"Production model loaded: {_meta['model_file']}")
        logger.info(f"  F1={_meta['metrics'].get('f1', 'n/a')}  "
                    f"Features={len(_meta.get('feature_cols', []))}")
    except (RuntimeError, FileNotFoundError) as e:
        logger.warning(f"Could not load model: {e}")
        logger.warning("Run `python main.py` to train a model, then restart the server.")


# ── request / response schemas ────────────────────────────────────────────────

class SessionRecord(BaseModel):
    """
    Input schema for a single e-commerce browsing session.
    Fields match the UCI Online Shoppers Purchasing Intention dataset columns.
    All fields are required; missing values are imputed server-side using
    the training-time medians saved in models/feature_pipeline.json.
    """
    Administrative:             float = Field(..., ge=0,  example=2)
    Administrative_Duration:    float = Field(..., ge=0,  example=80.0)
    Informational:              float = Field(..., ge=0,  example=0)
    Informational_Duration:     float = Field(..., ge=0,  example=0.0)
    ProductRelated:             float = Field(..., ge=0,  example=15)
    ProductRelated_Duration:    float = Field(..., ge=0,  example=450.0)
    BounceRates:                float = Field(..., ge=0, le=1, example=0.02)
    ExitRates:                  float = Field(..., ge=0, le=1, example=0.04)
    PageValues:                 float = Field(..., ge=0,  example=12.5)
    SpecialDay:                 float = Field(..., ge=0, le=1, example=0.0)
    Month:                      str   = Field(..., example="Nov")
    OperatingSystems:           int   = Field(..., ge=1,  example=2)
    Browser:                    int   = Field(..., ge=1,  example=2)
    Region:                     int   = Field(..., ge=1,  example=1)
    TrafficType:                int   = Field(..., ge=1,  example=2)
    VisitorType:                str   = Field(..., example="Returning_Visitor")
    Weekend:                    bool  = Field(..., example=False)


class PredictionResponse(BaseModel):
    predicted_purchase: int    # 0 = predicted no purchase, 1 = predicted purchase
    probability:        float  # P(purchase = 1) — useful for threshold tuning
    intent_level:       str    # "low" / "medium" / "high" — human-readable label
    model_version:      str
    latency_ms:         float


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health_check():
    """
    Health probe endpoint. Useful for load balancers and Kubernetes liveness checks.
    Returns whether a production model is currently loaded.
    """
    return {
        "status":        "ok",
        "model_loaded":  _model is not None,
        "model_version": _meta["version"] if _meta else None,
        "model_file":    _meta["model_file"] if _meta else None,
    }


@app.get("/model-info", tags=["Meta"])
def model_info():
    """Return the full registry entry for the currently loaded model."""
    if _meta is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    return _meta


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_purchase(record: SessionRecord):
    """
    Predict purchase intent for a single active browsing session.

    Returns:
    * `predicted_purchase` : 0 or 1  (binary prediction)
    * `probability`        : float   (P(purchase=1), useful for offer targeting)
    * `intent_level`       : "low" / "medium" / "high"
    * `latency_ms`         : end-to-end inference time in milliseconds
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No production model is loaded. "
                "Run `python main.py` to train a model, then restart the server."
            ),
        )

    start = time.time()

    row_df = pd.DataFrame([record.dict()])
    row_df["Weekend"] = row_df["Weekend"].astype(str)

    # Use saved training-time medians — same pipeline as training
    X, _, feature_cols = build_features(row_df, training=False)

    # Align columns to the exact feature set the model was trained on
    if _meta and "feature_cols" in _meta:
        trained_cols = _meta["feature_cols"]
        X_df = pd.DataFrame(X, columns=feature_cols)
        for c in trained_cols:
            if c not in X_df.columns:
                X_df[c] = 0.0
        X = X_df[trained_cols].values.astype(np.float32)

    pred    = int(_model.predict(X)[0])
    prob    = float(_model.predict_proba(X)[0][1])
    latency = (time.time() - start) * 1000

    # Map probability to a human-readable intent tier
    # Thresholds reflect the cost asymmetry: FN=$85, FP=$3
    if prob < 0.25:
        intent_level = "low"
    elif prob < 0.55:
        intent_level = "medium"
    else:
        intent_level = "high"

    logger.info(
        f"Prediction: {pred}  prob={prob:.3f}  intent={intent_level}  "
        f"latency={latency:.1f}ms"
    )

    log_latency(start)
    log_predictions(np.array([pred]))

    return PredictionResponse(
        predicted_purchase = pred,
        probability        = round(prob, 4),
        intent_level       = intent_level,
        model_version      = _meta["version"] if _meta else "unknown",
        latency_ms         = round(latency, 2),
    )
