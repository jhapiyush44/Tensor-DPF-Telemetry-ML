from fastapi import FastAPI, HTTPException, Body
import joblib
import pandas as pd
import json
import numpy as np
from datetime import datetime
from .monitoring import log_prediction
from pathlib import Path

app = FastAPI(title="DPF Soot Load Prediction API")


# =================================================
# Load models
# =================================================
reg_model = joblib.load("models/regressor.pkl")
clf_model = joblib.load("models/classifier.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

metrics_path = Path("models/metrics.json")

if metrics_path.exists():
    metrics = json.load(open(metrics_path))
else:
    # CI / test fallback
    metrics = {
        "version": "test",
        "trained_at": "N/A",
        "regression": {},
        "classification": {}
    }



# =================================================
# Feature prep
# =================================================
def prepare_features(df: pd.DataFrame):

    df["speed"] = df["speed"].clip(0, 120)
    df["rpm"] = df["rpm"].clip(500, 4000)

    df["idle"] = (df["speed"] < 5).astype(int)
    df["high_load"] = (df["engine_load"] > 0.7).astype(int)
    df["is_regen"] = 0

    return df.reindex(columns=feature_cols, fill_value=0)


# =================================================
# Health
# =================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }


# =================================================
# Model info (now includes metrics ⭐)
# =================================================
@app.get("/model/info")
def model_info():
    return metrics


# =================================================
# Single prediction
# =================================================
@app.post("/predict/soot-load")
def predict_single(data: dict):

    required = [
        "engine_load","rpm","speed","exhaust_temp_pre","exhaust_temp_post",
        "flow_rate","ambient_temp","diff_pressure",
        "temp_roll_mean_10","temp_roll_mean_60","temp_delta",
        "minutes_since_regen","idle_ratio_30","high_load_ratio_30"
    ]

    missing = [c for c in required if c not in data]
    if missing:
        raise HTTPException(422, f"Missing fields: {missing}")

    df = prepare_features(pd.DataFrame([data]))

    # ⭐ confidence interval using tree std
    preds = np.array([t.predict(df)[0] for t in reg_model.estimators_])
    mean = preds.mean()
    std = preds.std()

    log_prediction(mean)

    return {
        "soot_load_percent": round(mean * 100, 2),
        "confidence_interval": [
            round((mean - 2 * std) * 100, 2),
            round((mean + 2 * std) * 100, 2)
        ],
        "regen_recommended": bool(clf_model.predict(df)[0])
    }


# =================================================
# Batch
# =================================================
@app.post("/predict/batch")
def predict_batch(data: list = Body(...)):

    df = prepare_features(pd.DataFrame(data))

    soot = reg_model.predict(df)
    regen = clf_model.predict(df)

    for s in soot:
        log_prediction(s)

    return {
        "predictions": [
            {
                "soot_load_percent": round(float(s * 100), 2),
                "regen_recommended": bool(r)
            }
            for s, r in zip(soot, regen)
        ]
    }
