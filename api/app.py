from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path


app = FastAPI(title="DPF Soot Load Prediction API")


# =================================================
# Load models at startup
# =================================================

MODEL_DIR = Path("models")

try:
    reg_model = joblib.load(MODEL_DIR / "regressor.pkl")
    clf_model = joblib.load(MODEL_DIR / "classifier.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
except Exception as e:
    raise RuntimeError(
        "Models not found. Train models first using models/train.py"
    ) from e


# =================================================
# Request Schema (⭐ professional upgrade)
# =================================================

class TelemetryInput(BaseModel):
    engine_load: float
    rpm: float
    speed: float
    exhaust_temp_pre: float
    exhaust_temp_post: float
    flow_rate: float
    ambient_temp: float
    diff_pressure: float
    temp_roll_mean_10: float
    temp_roll_mean_60: float
    temp_delta: float
    minutes_since_regen: float
    idle_ratio_30: float
    high_load_ratio_30: float


# =================================================
# Feature preparation
# =================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies same feature engineering used during training.
    Guarantees training == inference consistency.
    """

    df = df.copy()

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


# =================================================
# Health
# =================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": reg_model is not None and clf_model is not None
    }


# =================================================
# Model info
# =================================================

@app.get("/model/info")
def model_info():
    return {
        "model_type": "RandomForest",
        "version": "v1.0",
        "timestamp": str(datetime.now())
    }


# =================================================
# Single prediction
# =================================================

@app.post("/predict/soot-load")
def predict_single(data: TelemetryInput):

    df = pd.DataFrame([data.dict()])
    df = prepare_features(df)

    soot_pred = float(reg_model.predict(df)[0])
    regen_pred = int(clf_model.predict(df)[0])

    return {
        "soot_load_percent": round(soot_pred * 100, 2),
        "regen_recommended": bool(regen_pred)
    }


# =================================================
# Batch prediction
# =================================================

@app.post("/predict/batch")
def predict_batch(data: list[TelemetryInput] = Body(...)):

    if not data:
        raise HTTPException(status_code=400, detail="Empty batch payload")

    df = pd.DataFrame([d.dict() for d in data])
    df = prepare_features(df)

    soot = reg_model.predict(df)
    regen = clf_model.predict(df)

    results = [
        {
            "soot_load_percent": round(float(s * 100), 2),
            "regen_recommended": bool(r)
        }
        for s, r in zip(soot, regen)
    ]

    return {"predictions": results}
