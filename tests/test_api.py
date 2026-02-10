"""
API tests

Key idea:
---------
We do NOT commit real .pkl models to git.
So tests create tiny dummy models automatically if missing.

This keeps:
✔ CI fast
✔ repo clean
✔ no large binaries
✔ production unchanged
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# =================================================
# Create dummy models ONLY for testing
# =================================================

def create_dummy_models():

    os.makedirs("models", exist_ok=True)

    feature_cols = [
        "engine_load", "rpm", "speed",
        "exhaust_temp_pre", "exhaust_temp_post",
        "flow_rate", "ambient_temp", "diff_pressure",
        "temp_roll_mean_10", "temp_roll_mean_60",
        "temp_delta", "minutes_since_regen",
        "idle_ratio_30", "high_load_ratio_30",
        "idle", "high_load", "is_regen"
    ]

    X = pd.DataFrame(np.random.rand(20, len(feature_cols)), columns=feature_cols)
    y_reg = np.random.rand(20)
    y_clf = (y_reg > 0.5).astype(int)

    reg = RandomForestRegressor(n_estimators=2)
    reg.fit(X, y_reg)

    clf = RandomForestClassifier(n_estimators=2)
    clf.fit(X, y_clf)

    joblib.dump(reg, "models/regressor.pkl")
    joblib.dump(clf, "models/classifier.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")


# Create models only if missing (CI case)
if not os.path.exists("models/regressor.pkl"):
    create_dummy_models()


# =================================================
# NOW import app (models exist)
# =================================================

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


# =================================================
# Helper payload
# =================================================

def sample_payload():
    return {
        "engine_load": 0.6,
        "rpm": 2000,
        "speed": 60,
        "exhaust_temp_pre": 500,
        "exhaust_temp_post": 470,
        "flow_rate": 50,
        "ambient_temp": 30,
        "diff_pressure": 10,
        "temp_roll_mean_10": 480,
        "temp_roll_mean_60": 470,
        "temp_delta": 30,
        "minutes_since_regen": 120,
        "idle_ratio_30": 0.1,
        "high_load_ratio_30": 0.5
    }


# =================================================
# Tests
# =================================================

def test_health():

    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_single_prediction():

    r = client.post("/predict/soot-load", json=sample_payload())

    data = r.json()

    assert r.status_code == 200
    assert "soot_load_percent" in data
    assert "regen_recommended" in data
    assert isinstance(data["regen_recommended"], bool)


def test_batch_prediction():

    r = client.post(
        "/predict/batch",
        json=[sample_payload(), sample_payload()]
    )

    data = r.json()

    assert r.status_code == 200
    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_missing_fields():

    r = client.post("/predict/soot-load", json={})

    assert r.status_code != 200
