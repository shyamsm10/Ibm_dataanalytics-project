"""
Model training, evaluation, and persistence for Cafe Sales Prediction.
"""
import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_processor import prepare_dataset, get_feature_columns

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def train_all_models() -> dict:
    """Train multiple models, compare, save best. Returns full results dict."""
    X, y, _ = prepare_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = evaluate(y_test, preds)
        results[name] = {"metrics": metrics, "model": pipeline}
        print(f"{name}: MAE={metrics['MAE']} RMSE={metrics['RMSE']} R2={metrics['R2']}")

    # Select best by R2
    best_name = max(results, key=lambda k: results[k]["metrics"]["R2"])
    best_model = results[best_name]["model"]

    print(f"\nBest model: {best_name}")

    # Save
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    # Save metrics (no model objects)
    metrics_out = {
        name: {**data["metrics"]} for name, data in results.items()
    }
    metrics_out["best_model"] = best_name
    metrics_out["feature_columns"] = get_feature_columns()
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    return metrics_out


def load_model():
    """Load the saved best model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_all_models() first."
        )
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    """Load saved model metrics."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Metrics file not found. Run training first.")
    with open(METRICS_PATH) as f:
        return json.load(f)


def predict_single(features: dict) -> float:
    """Predict Total Spent for a single sample given a feature dict."""
    import pandas as pd
    model = load_model()
    feature_cols = get_feature_columns()
    df = pd.DataFrame([features])[feature_cols]
    return float(model.predict(df)[0])
