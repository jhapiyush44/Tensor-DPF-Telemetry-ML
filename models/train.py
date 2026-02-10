import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

DATA_PATH = Path("data/features.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# =================================================
# Config (tracked automatically)
# =================================================
PARAMS = {
    "n_estimators": 40,
    "max_depth": 10,
    "random_state": 42
}


# =================================================
# Main
# =================================================
def main():

    print("Loading features...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["vehicle_id", "timestamp"])

    split_date = df["timestamp"].quantile(0.9)

    train = df[df["timestamp"] < split_date]
    test = df[df["timestamp"] >= split_date]

    drop_cols = ["timestamp", "vehicle_id", "soot_load", "regen_needed"]

    X_train = train.drop(columns=drop_cols)
    X_test = test.drop(columns=drop_cols)

    y_train_reg = train["soot_load"]
    y_test_reg = test["soot_load"]

    y_train_clf = train["regen_needed"]
    y_test_clf = test["regen_needed"]

    feature_cols = X_train.columns.tolist()
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

    # =============================
    # Regression
    # =============================
    reg = RandomForestRegressor(**PARAMS, n_jobs=-1)
    reg.fit(X_train, y_train_reg)

    pred_reg = reg.predict(X_test)

    mae = mean_absolute_error(y_test_reg, pred_reg)
    r2 = r2_score(y_test_reg, pred_reg)

    joblib.dump(reg, MODEL_DIR / "regressor.pkl")

    # =============================
    # Classification
    # =============================
    clf = RandomForestClassifier(**PARAMS, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train_clf)

    pred_clf = clf.predict(X_test)

    acc = accuracy_score(y_test_clf, pred_clf)
    prec = precision_score(y_test_clf, pred_clf)
    rec = recall_score(y_test_clf, pred_clf)
    f1 = f1_score(y_test_clf, pred_clf)

    joblib.dump(clf, MODEL_DIR / "classifier.pkl")

    # =============================
    # Metrics + Versioning ⭐
    # =============================
    metrics = {
        "version": "v1.1",
        "trained_at": str(datetime.now()),
        "params": PARAMS,
        "regression": {"mae": float(mae), "r2": float(r2)},
        "classification": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        }
    }

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved models + metrics.json")


if __name__ == "__main__":
    main()
