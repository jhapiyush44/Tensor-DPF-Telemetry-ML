from fastapi import FastAPI
import joblib
import pandas as pd
from datetime import datetime
from fastapi import Body

app = FastAPI(title="DPF Soot Load Prediction API")


# -------------------------------------------------
# Load models at startup
# -------------------------------------------------

reg_model = joblib.load("models/regressor.pkl")
clf_model = joblib.load("models/classifier.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")


# -------------------------------------------------
# Feature preparation (⭐ shared logic)
# -------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies same feature engineering used during training.
    This guarantees training == inference consistency.
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


# -------------------------------------------------
# Health
# -------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------------------------
# Model info
# -------------------------------------------------

@app.get("/model/info")
def model_info():
    return {
        "model_type": "RandomForest",
        "version": "v1.0",
        "timestamp": str(datetime.now())
    }


# -------------------------------------------------
# Single prediction
# -------------------------------------------------

@app.post("/predict/soot-load")
def predict_single(data: dict):

    df = pd.DataFrame([data])
    df = prepare_features(df)

    soot_pred = float(reg_model.predict(df)[0])
    regen_pred = int(clf_model.predict(df)[0])

    return {
        "soot_load_percent": round(soot_pred * 100, 2),
        "regen_recommended": bool(regen_pred)
    }


# -------------------------------------------------
# Batch prediction (FIXED ⭐)
# -------------------------------------------------

@app.post("/predict/batch")
def predict_batch(data: list = Body(...)):

    df = pd.DataFrame(data)
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

