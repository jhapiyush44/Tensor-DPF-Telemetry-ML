# =====================================================
# DPF Soot Load Prediction API (Production Ready)
# =====================================================

from fastapi import FastAPI, Body, HTTPException
import pandas as pd
import joblib
import numpy as np
import json
from datetime import datetime
from pathlib import Path

app = FastAPI(title="DPF Soot Load Prediction API")


# =====================================================
# Paths
# =====================================================

MODEL_DIR = Path("models")


# =====================================================
# Safe model loading (CI friendly ⭐)
# =====================================================

def safe_load(path, default=None):
    try:
        return joblib.load(path)
    except Exception:
        return default


reg_model = safe_load(MODEL_DIR / "regressor.pkl")
clf_model = safe_load(MODEL_DIR / "classifier.pkl")
feature_cols = safe_load(MODEL_DIR / "feature_cols.pkl", [])
metrics = {}

try:
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
except Exception:
    metrics = {}


# =====================================================
# Feature preparation (shared logic)
# =====================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies same transformations used during training.
    Guarantees schema consistency.
    """

    # safety clipping
    df["speed"] = df["speed"].clip(0, 120)
    df["rpm"] = df["rpm"].clip(500, 4000)

    # derived features
    df["idle"] = (df["speed"] < 5).astype(int)
    df["high_load"] = (df["engine_load"] > 0.7).astype(int)
    df["is_regen"] = 0

    # enforce exact training schema
    df = df.reindex(columns=feature_cols, fill_value=0)

    return df


# =====================================================
# Helper: regression with confidence interval ⭐
# =====================================================

def predict_with_uncertainty(df: pd.DataFrame):
    """
    RandomForest uncertainty estimation using tree variance.
    """

    tree_preds = np.array([t.predict(df)[0] for t in reg_model.estimators_])

    mean = tree_preds.mean()
    std = tree_preds.std()

    ci = 1.96 * std  # 95% CI

    return float(mean), float(ci)


# =====================================================
# Health
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "reg_model_loaded": reg_model is not None,
        "clf_model_loaded": clf_model is not None,
        "timestamp": str(datetime.utcnow())
    }


# =====================================================
# Model info
# =====================================================

@app.get("/model/info")
def model_info():
    return {
        "model_type": "RandomForest",
        "metrics": metrics,
        "feature_count": len(feature_cols),
        "timestamp": str(datetime.utcnow())
    }


# =====================================================
# Required fields
# =====================================================

REQUIRED_COLS = [
    "engine_load", "rpm", "speed",
    "exhaust_temp_pre", "exhaust_temp_post",
    "flow_rate", "ambient_temp", "diff_pressure",
    "temp_roll_mean_10", "temp_roll_mean_60",
    "temp_delta", "minutes_since_regen",
    "idle_ratio_30", "high_load_ratio_30"
]


def validate_payload(data: dict):
    missing = [c for c in REQUIRED_COLS if c not in data]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing fields: {missing}"
        )


# =====================================================
# Single prediction
# =====================================================

@app.post("/predict/soot-load")
def predict_single(data: dict = Body(...)):

    if reg_model is None or clf_model is None:
        raise HTTPException(503, "Models not loaded")

    validate_payload(data)

    df = pd.DataFrame([data])
    df = prepare_features(df)

    soot_mean, soot_ci = predict_with_uncertainty(df)
    regen_pred = int(clf_model.predict(df)[0])

    return {
        "soot_load_percent": round(soot_mean * 100, 2),
        "confidence_interval": round(soot_ci * 100, 2),
        "regen_recommended": bool(regen_pred)
    }


# =====================================================
# Batch prediction
# =====================================================

@app.post("/predict/batch")
def predict_batch(data: list = Body(...)):

    if reg_model is None or clf_model is None:
        raise HTTPException(503, "Models not loaded")

    df = pd.DataFrame(data)
    df = prepare_features(df)

    results = []

    for i in range(len(df)):
        row = df.iloc[[i]]

        mean, ci = predict_with_uncertainty(row)
        regen = int(clf_model.predict(row)[0])

        results.append({
            "soot_load_percent": round(mean * 100, 2),
            "confidence_interval": round(ci * 100, 2),
            "regen_recommended": bool(regen)
        })

    return {"predictions": results}
