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

from xgboost import XGBRegressor, XGBClassifier

def convert_to_python_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    elif hasattr(obj, "item"):  # numpy types
        return obj.item()
    else:
        return obj


DATA_PATH = Path("data/features.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# =================================================
# Config
# =================================================
RF_PARAMS = {
    "n_estimators": 40,
    "max_depth": 10,
    "random_state": 42
}


# =================================================
# Helper: Feature Importance
# =================================================
def get_feature_importance(model, feature_names):
    try:
        importance = model.feature_importances_
        return dict(sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        ))
    except AttributeError:
        return {}


# =================================================
# Main
# =================================================
def main():

    print("Loading features...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["vehicle_id", "timestamp"])

    # =============================
    # Class Distribution
    # =============================
    print("\nClass Distribution:")
    print(df["regen_needed"].value_counts(normalize=True))

    # =============================
    # Time-based Split
    # =============================
    q80 = df["timestamp"].quantile(0.8)
    q90 = df["timestamp"].quantile(0.9)

    train = df[df["timestamp"] < q80]
    val   = df[(df["timestamp"] >= q80) & (df["timestamp"] < q90)]
    test  = df[df["timestamp"] >= q90]

    drop_cols = ["timestamp", "vehicle_id", "soot_load", "regen_needed", "diff_pressure", "minutes_since_regen"]

    X_train = train.drop(columns=drop_cols)
    X_val   = val.drop(columns=drop_cols)
    X_test  = test.drop(columns=drop_cols)

    y_train_reg = train["soot_load"]
    y_val_reg   = val["soot_load"]
    y_test_reg  = test["soot_load"]

    y_train_clf = train["regen_needed"]
    y_val_clf   = val["regen_needed"]
    y_test_clf  = test["regen_needed"]

    # Save feature columns
    feature_cols = X_train.columns.tolist()
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

    # =================================================
    # RANDOM FOREST
    # =================================================
    print("\nTraining RandomForest...")

    rf_reg = RandomForestRegressor(**RF_PARAMS, n_jobs=-1)
    rf_reg.fit(X_train, y_train_reg)

    rf_pred_reg = rf_reg.predict(X_val)
    rf_mae = mean_absolute_error(y_val_reg, rf_pred_reg)
    rf_r2 = r2_score(y_val_reg, rf_pred_reg)

    rf_clf = RandomForestClassifier(**RF_PARAMS, class_weight="balanced", n_jobs=-1)
    rf_clf.fit(X_train, y_train_clf)

    rf_pred_clf = rf_clf.predict(X_val)
    rf_acc = accuracy_score(y_val_clf, rf_pred_clf)
    rf_prec = precision_score(y_val_clf, rf_pred_clf)
    rf_rec = recall_score(y_val_clf, rf_pred_clf)
    rf_f1 = f1_score(y_val_clf, rf_pred_clf)

    # =================================================
    # XGBOOST
    # =================================================
    print("\nTraining XGBoost...")

    xgb_reg = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_reg.fit(X_train, y_train_reg)

    xgb_pred_reg = xgb_reg.predict(X_val)
    xgb_mae = mean_absolute_error(y_val_reg, xgb_pred_reg)
    xgb_r2 = r2_score(y_val_reg, xgb_pred_reg)

    xgb_clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train_clf)

    xgb_pred_clf = xgb_clf.predict(X_val)
    xgb_acc = accuracy_score(y_val_clf, xgb_pred_clf)
    xgb_prec = precision_score(y_val_clf, xgb_pred_clf)
    xgb_rec = recall_score(y_val_clf, xgb_pred_clf)
    xgb_f1 = f1_score(y_val_clf, xgb_pred_clf)

    # =================================================
    # Model Comparison
    # =================================================
    print("\nModel Comparison (Validation):")
    print(f"RF  -> MAE: {rf_mae:.4f}, F1: {rf_f1:.4f}")
    print(f"XGB -> MAE: {xgb_mae:.4f}, F1: {xgb_f1:.4f}")

    # =================================================
    # Select Best Models
    # =================================================
    best_reg = xgb_reg if xgb_mae < rf_mae else rf_reg
    best_clf = xgb_clf if xgb_f1 > rf_f1 else rf_clf

    best_reg_name = "xgboost" if xgb_mae < rf_mae else "random_forest"
    best_clf_name = "xgboost" if xgb_f1 > rf_f1 else "random_forest"

    joblib.dump(best_reg, MODEL_DIR / "regressor.pkl")
    joblib.dump(best_clf, MODEL_DIR / "classifier.pkl")

    # =================================================
    # Feature Importance
    # =================================================
    reg_feature_importance = get_feature_importance(best_reg, feature_cols)
    clf_feature_importance = get_feature_importance(best_clf, feature_cols)

    print("\nTop Features (Regressor):")
    for k, v in list(reg_feature_importance.items())[:10]:
        print(f"{k}: {v:.4f}")

    print("\nTop Features (Classifier):")
    for k, v in list(clf_feature_importance.items())[:10]:
        print(f"{k}: {v:.4f}")

    # Save full feature importance
    with open(MODEL_DIR / "feature_importance.json", "w") as f:
        json.dump({
            "regressor": {k: float(v) for k, v in reg_feature_importance.items()},
            "classifier": {k: float(v) for k, v in clf_feature_importance.items()}
        }, f, indent=2)

    # =================================================
    # Final Test Evaluation
    # =================================================
    final_pred_reg = best_reg.predict(X_test)
    final_mae = mean_absolute_error(y_test_reg, final_pred_reg)
    final_r2 = r2_score(y_test_reg, final_pred_reg)

    final_pred_clf = best_clf.predict(X_test)
    final_f1 = f1_score(y_test_clf, final_pred_clf)

    # =================================================
    # Metrics Logging
    # =================================================
    metrics = {
        "version": "v3.0",
        "trained_at": str(datetime.now()),

        "random_forest": {
            "regression": {"mae": float(rf_mae), "r2": float(rf_r2)},
            "classification": {
                "accuracy": float(rf_acc),
                "precision": float(rf_prec),
                "recall": float(rf_rec),
                "f1": float(rf_f1)
            }
        },

        "xgboost": {
            "regression": {"mae": float(xgb_mae), "r2": float(xgb_r2)},
            "classification": {
                "accuracy": float(xgb_acc),
                "precision": float(xgb_prec),
                "recall": float(xgb_rec),
                "f1": float(xgb_f1)
            }
        },

        "selected_model": {
            "regressor": best_reg_name,
            "classifier": best_clf_name
        },

        "final_test_performance": {
            "regression": {"mae": float(final_mae), "r2": float(final_r2)},
            "classification": {"f1": float(final_f1)}
        },

        "feature_importance": {
            "regressor_top10": dict(list(reg_feature_importance.items())[:10]),
            "classifier_top10": dict(list(clf_feature_importance.items())[:10])
        }
    }

    with open(MODEL_DIR / "metrics.json", "w") as f:
        metrics_clean = convert_to_python_types(metrics)

        json.dump(metrics_clean, f, indent=2)
        
    print("\nSaved best models + metrics.json + feature_importance.json")


if __name__ == "__main__":
    main()