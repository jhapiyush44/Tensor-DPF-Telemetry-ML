import pandas as pd
import joblib
import json
from pathlib import Path
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
# Training
# =================================================

def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "features.parquet not found. Run feature_engineering.py first."
        )

    print("Loading features...")

    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["vehicle_id", "timestamp"])

    # -------------------------------------------------
    # Time-based split
    # -------------------------------------------------

    split_date = df["timestamp"].quantile(0.9)

    train = df[df["timestamp"] < split_date].copy()
    test = df[df["timestamp"] >= split_date].copy()

    print(f"Train size: {len(train):,}")
    print(f"Test size : {len(test):,}")

    # -------------------------------------------------
    # Features
    # -------------------------------------------------

    drop_cols = ["timestamp", "vehicle_id", "soot_load", "regen_needed"]

    X_train = train.drop(columns=drop_cols)
    X_test = test.drop(columns=drop_cols)

    y_train_reg = train["soot_load"]
    y_test_reg = test["soot_load"]

    y_train_clf = train["regen_needed"]
    y_test_clf = test["regen_needed"]

    # save schema for inference
    feature_cols = X_train.columns.tolist()
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

    # enforce consistent ordering
    X_test = X_test[feature_cols]

    print(f"Number of features: {len(feature_cols)}")

    # -------------------------------------------------
    # Regression
    # -------------------------------------------------

    reg = RandomForestRegressor(
        n_estimators=40,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )

    reg.fit(X_train, y_train_reg)

    pred_reg = reg.predict(X_test)

    mae = mean_absolute_error(y_test_reg, pred_reg)
    r2 = r2_score(y_test_reg, pred_reg)

    print("\n=== Regression ===")
    print("MAE:", mae)
    print("R2 :", r2)

    joblib.dump(reg, MODEL_DIR / "regressor.pkl", compress=3)

    # -------------------------------------------------
    # Classification
    # -------------------------------------------------

    clf = RandomForestClassifier(
        n_estimators=40,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    clf.fit(X_train, y_train_clf)

    pred_clf = clf.predict(X_test)

    acc = accuracy_score(y_test_clf, pred_clf)
    prec = precision_score(y_test_clf, pred_clf, zero_division=0)
    rec = recall_score(y_test_clf, pred_clf, zero_division=0)
    f1 = f1_score(y_test_clf, pred_clf, zero_division=0)

    print("\n=== Classification ===")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1       :", f1)

    joblib.dump(clf, MODEL_DIR / "classifier.pkl", compress=3)

    # -------------------------------------------------
    # Save metrics
    # -------------------------------------------------

    metrics = {
        "regression": {
            "mae": float(mae),
            "r2": float(r2)
        },
        "classification": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1)
        }
    }

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved models + metrics.")


# =================================================

if __name__ == "__main__":
    main()
