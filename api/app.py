"""
Vehicle Telemetry ML API — with LLM Explanation Layer
Architecture:
    POST /predict/explain
      → ML Model (soot prediction)
      → Feature Context Builder
      → RAG (FAISS vector search)
      → LLM (Gemini Flash explanation)
      → JSON response

Also serves the frontend UI at GET /
"""

import os
import json
import numpy as np
import joblib
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from api.rag.context_builder import build_feature_context, build_rag_query
from api.rag.knowledge_base import retrieve
from api.llm.explainer import generate_explanation

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DPF Telemetry ML + LLM API",
    description="Predicts DPF soot load and generates AI-powered explanations",
    version="2.0.0",
)

# Serve static frontend
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# ── Model Loading ─────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent.parent / "models"

_regressor = None
_classifier = None
_feature_cols = []
_metrics = {"regression": {}, "classification": {}}


def load_models():
    global _regressor, _classifier, _feature_cols, _metrics
    reg_path = MODEL_DIR / "regressor.pkl"
    clf_path = MODEL_DIR / "classifier.pkl"

    if not reg_path.exists() or not clf_path.exists():
        raise FileNotFoundError(
            f"Model files not found in {MODEL_DIR}. "
            "Run: python models/train.py"
        )
    _regressor = joblib.load(reg_path)
    _classifier = joblib.load(clf_path)

    # Load feature_cols if available
    fc_path = MODEL_DIR / "feature_cols.pkl"
    if fc_path.exists():
        _feature_cols = joblib.load(fc_path)

    # Load metrics if available
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                _metrics = json.load(f)
        except Exception:
            pass


@app.on_event("startup")
async def startup_event():
    try:
        load_models()
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
        print("   API will run but predictions will return mock data.")


# ── Request / Response Schemas ────────────────────────────────────────────────
class TelemetryInput(BaseModel):
    engine_load: float = Field(..., ge=0.0, le=1.0, description="Engine load (0.0–1.0)")
    rpm: float = Field(..., ge=0, le=8000, description="Engine RPM")
    speed: float = Field(..., ge=0, le=300, description="Vehicle speed km/h")
    exhaust_temp_pre: float = Field(..., ge=0, le=900, description="Pre-DPF exhaust temp °C")
    exhaust_temp_post: float = Field(..., ge=0, le=900, description="Post-DPF exhaust temp °C")
    flow_rate: float = Field(..., ge=0, le=500, description="Exhaust flow rate g/s")
    ambient_temp: float = Field(..., ge=-40, le=60, description="Ambient temperature °C")
    temp_roll_mean_10: float = Field(..., description="Rolling mean exhaust temp (10 min) °C")
    temp_roll_mean_60: float = Field(..., description="Rolling mean exhaust temp (60 min) °C")
    temp_delta: float = Field(..., description="Temp delta (instantaneous - rolling mean) °C")
    idle_ratio_30: float = Field(..., ge=0.0, le=1.0, description="Fraction of last 30 min at idle")
    high_load_ratio_30: float = Field(..., ge=0.0, le=1.0, description="Fraction of last 30 min at high load")

    # --- derived state flags (optional with sensible defaults) ---
    is_regen: int = Field(default=0, ge=0, le=1, description="Is active regeneration occurring? (0 or 1)")
    idle: int = Field(default=0, ge=0, le=1, description="Is engine idling? (0 or 1)")
    high_load: int = Field(default=0, ge=0, le=1, description="Is engine under high load? (0 or 1)")

    # --- trip features (optional with sensible defaults) ---
    trip_duration_min: float = Field(default=0.0, ge=0, description="Current trip duration in minutes")
    trip_avg_speed: float = Field(default=0.0, ge=0, le=300, description="Average speed this trip km/h")


# Must match model's feature_names_in_ exactly
FEATURE_ORDER = [
    "engine_load", "rpm", "speed", "exhaust_temp_pre", "exhaust_temp_post",
    "flow_rate", "ambient_temp", "temp_roll_mean_10", "temp_roll_mean_60",
    "temp_delta", "is_regen", "idle", "high_load",
    "idle_ratio_30", "high_load_ratio_30", "trip_duration_min", "trip_avg_speed",
]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "DPF Telemetry API v2.0 — frontend not found"})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": _regressor is not None,
        "version": "2.0.0"
    }


@app.get("/model/info")
def model_info():
    """Model metadata endpoint — preserved for backward compatibility."""
    return {
        "model_type": "RandomForest",
        **_metrics,  # flattens so regression/classification keys exist
        "feature_count": len(_feature_cols),
        "timestamp": str(datetime.utcnow())
    }


@app.post("/predict/soot-load")
async def predict_soot_load(data: TelemetryInput):
    """Original prediction endpoint (ML only, no LLM)."""
    prediction = _run_ml_prediction(data)
    return prediction


@app.post("/predict/batch")
async def predict_batch(data: list = Body(...)):
    """Batch prediction endpoint — preserved for backward compatibility."""
    results = []
    for item in data:
        try:
            parsed = TelemetryInput(**item)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))
        results.append(_run_ml_prediction(parsed))
    return {"predictions": results}


@app.post("/predict/explain")
async def predict_with_explanation(data: TelemetryInput):
    """
    Full pipeline: ML prediction → Feature Context → RAG → LLM explanation.
    Returns prediction + AI-generated explanation.
    """
    # Step 1: ML Prediction
    prediction = _run_ml_prediction(data)

    # Step 2: Build feature context
    inputs_dict = data.model_dump()
    feature_context = build_feature_context(inputs_dict, prediction)

    # Step 3: RAG retrieval
    rag_query = build_rag_query(inputs_dict, prediction)
    rag_chunks = retrieve(rag_query, top_k=5)

    # Step 4: LLM explanation
    llm_result = await generate_explanation(feature_context, rag_chunks, prediction)

    return {
        **prediction,
        "feature_context": feature_context,
        "rag_chunks_used": rag_chunks,
        "explanation": llm_result["explanation"],
        "model_used": llm_result["model_used"],
    }


# ── ML Prediction Helper ──────────────────────────────────────────────────────
def _run_ml_prediction(data: TelemetryInput) -> dict:
    if _regressor is None or _classifier is None:
        # Mock prediction for demo when models aren't trained yet
        import random
        soot = round(random.uniform(20, 95), 1)
        return {
            "soot_load_percent": soot,
            "confidence_interval": round(random.uniform(2, 8), 1),
            "regen_recommended": soot > 70,
            "_note": "Mock prediction — run models/train.py to use real models"
        }

    feature_vec = np.array([[getattr(data, f) for f in FEATURE_ORDER]])

    soot_load = float(np.clip(_regressor.predict(feature_vec)[0], 0, 100))

    if hasattr(_classifier, "predict_proba"):
        regen_proba = float(_classifier.predict_proba(feature_vec)[0][1])
        regen_recommended = regen_proba > 0.5
    else:
        regen_recommended = bool(_classifier.predict(feature_vec)[0])
        regen_proba = 1.0 if regen_recommended else 0.0

    # Confidence interval via RF tree variance
    if hasattr(_regressor, "estimators_"):
        tree_preds = np.array([t.predict(feature_vec)[0] for t in _regressor.estimators_])
        ci = float(np.std(tree_preds) * 1.96)
    else:
        ci = 0.0

    return {
        "soot_load_percent": round(soot_load, 2),
        "confidence_interval": round(ci, 2),
        "regen_recommended": regen_recommended,
        "regen_probability": round(regen_proba, 3),
    }


# ── Mount static AFTER routes ─────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")