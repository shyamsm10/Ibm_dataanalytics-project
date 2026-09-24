"""
Tests for data processor and model trainer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from src.data_processor import (
    load_raw,
    clean,
    engineer_features,
    prepare_dataset,
    get_feature_columns,
    VALID_ITEMS,
    ITEM_CATEGORY,
    ITEM_PRICE_MAP,
)


# ── Data processor tests ─────────────────────────────────────────────────────

def test_load_raw_shape():
    df = load_raw()
    assert df.shape[0] > 0
    assert df.shape[1] == 6


def test_load_raw_columns():
    df = load_raw()
    expected = {"Transaction ID", "Item", "Quantity", "Price Per Unit", "Total Spent", "Transaction Date"}
    assert expected == set(df.columns)


def test_clean_removes_noise_labels():
    df = load_raw()
    df_clean = clean(df)
    assert "unknown" not in df_clean["Item"].values
    assert "error" not in df_clean["Item"].values


def test_clean_all_valid_items():
    df = load_raw()
    df_clean = clean(df)
    assert set(df_clean["Item"].unique()).issubset(VALID_ITEMS)


def test_clean_no_nulls():
    df = load_raw()
    df_clean = clean(df)
    assert df_clean.isnull().sum().sum() == 0


def test_clean_date_parsed():
    df = load_raw()
    df_clean = clean(df)
    assert pd.api.types.is_datetime64_any_dtype(df_clean["Transaction Date"])


def test_engineer_features_columns():
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    for col in ["month", "day_of_week", "week_of_year", "quarter", "is_weekend",
                "day_of_month", "category", "item_encoded", "category_encoded"]:
        assert col in df.columns, f"Missing feature column: {col}"


def test_engineer_is_weekend_values():
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    assert set(df["is_weekend"].unique()).issubset({0, 1})


def test_engineer_category_values():
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    assert set(df["category"].unique()).issubset({"beverage", "food"})


def test_prepare_dataset_shapes():
    X, y, df = prepare_dataset()
    assert len(X) == len(y)
    assert X.shape[1] == len(get_feature_columns())


def test_prepare_dataset_no_leakage():
    """Total Spent must NOT appear in X."""
    X, y, df = prepare_dataset()
    assert "Total Spent" not in X.columns


def test_feature_columns_list():
    cols = get_feature_columns()
    assert isinstance(cols, list)
    assert len(cols) == 10
    assert "item_encoded" in cols
    assert "Quantity" in cols


def test_item_category_coverage():
    for item in VALID_ITEMS:
        assert item in ITEM_CATEGORY, f"{item} missing from ITEM_CATEGORY"


def test_item_price_coverage():
    for item in VALID_ITEMS:
        assert item in ITEM_PRICE_MAP, f"{item} missing from ITEM_PRICE_MAP"


# ── Model tests ──────────────────────────────────────────────────────────────

def test_model_file_exists_after_training():
    from src.model_trainer import MODEL_PATH, train_all_models
    train_all_models()
    assert MODEL_PATH.exists()


def test_metrics_file_exists_after_training():
    from src.model_trainer import METRICS_PATH, train_all_models
    train_all_models()
    assert METRICS_PATH.exists()


def test_load_model_returns_predictor():
    from src.model_trainer import load_model, train_all_models
    train_all_models()
    model = load_model()
    assert hasattr(model, "predict")


def test_predict_single_reasonable_range():
    from src.model_trainer import predict_single, train_all_models
    import pandas as pd
    train_all_models()
    features = {
        "item_encoded": 1,       # coffee
        "category_encoded": 0,   # beverage
        "Quantity": 2,
        "Price Per Unit": 2.0,
        "month": 6,
        "day_of_week": 2,
        "week_of_year": 24,
        "quarter": 2,
        "is_weekend": 0,
        "day_of_month": 15,
    }
    pred = predict_single(features)
    assert 1.0 <= pred <= 25.0, f"Prediction out of expected range: {pred}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
