"""
Data cleaning, preprocessing, and feature engineering for Cafe Sales dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "Cleaned_DataSet.csv"

# Valid item labels (exclude noise labels)
VALID_ITEMS = {"juice", "coffee", "cake", "sandwich", "smoothie", "cookie", "tea", "salad"}

# Item category mapping
ITEM_CATEGORY = {
    "coffee": "beverage",
    "tea": "beverage",
    "juice": "beverage",
    "smoothie": "beverage",
    "cake": "food",
    "cookie": "food",
    "sandwich": "food",
    "salad": "food",
}

# Known price map per item (for imputation reference)
ITEM_PRICE_MAP = {
    "coffee": 2.0,
    "tea": 1.5,
    "juice": 3.0,
    "smoothie": 4.0,
    "cake": 3.0,
    "cookie": 1.0,
    "sandwich": 4.0,
    "salad": 5.0,
}


def load_raw() -> pd.DataFrame:
    """Load raw CSV without any modification."""
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Drop rows with noisy item labels ('unknown', 'error')
    - Parse Transaction Date to datetime
    - Cast Quantity to int
    - Validate Total Spent consistency
    """
    df = df.copy()

    # Normalise item to lowercase strip
    df["Item"] = df["Item"].str.strip().str.lower()

    # Remove noisy rows
    df = df[df["Item"].isin(VALID_ITEMS)].reset_index(drop=True)

    # Parse date
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])

    # Ensure numeric types
    df["Quantity"] = df["Quantity"].astype(int)
    df["Price Per Unit"] = df["Price Per Unit"].astype(float)
    df["Total Spent"] = df["Total Spent"].astype(float)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and categorical features:
    - month, day_of_week, week_of_year, quarter, is_weekend
    - category (beverage / food)
    - item encoded (ordinal)
    """
    df = df.copy()

    # Date features
    df["month"] = df["Transaction Date"].dt.month
    df["day_of_week"] = df["Transaction Date"].dt.dayofweek  # 0=Mon
    df["week_of_year"] = df["Transaction Date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["Transaction Date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_month"] = df["Transaction Date"].dt.day

    # Category
    df["category"] = df["Item"].map(ITEM_CATEGORY)

    # Encode categorical columns (label encoding — stable order for training)
    item_list = sorted(VALID_ITEMS)
    df["item_encoded"] = df["Item"].apply(lambda x: item_list.index(x))
    df["category_encoded"] = (df["category"] == "food").astype(int)

    return df


def get_feature_columns() -> list:
    """Return the feature column list used for model training."""
    return [
        "item_encoded",
        "category_encoded",
        "Quantity",
        "Price Per Unit",
        "month",
        "day_of_week",
        "week_of_year",
        "quarter",
        "is_weekend",
        "day_of_month",
    ]


def get_target_column() -> str:
    return "Total Spent"


def prepare_dataset() -> tuple:
    """Full pipeline: load → clean → engineer → split X, y."""
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    features = get_feature_columns()
    target = get_target_column()
    X = df[features]
    y = df[target]
    return X, y, df
