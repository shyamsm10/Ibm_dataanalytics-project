"""
FastAPI backend for Cafe Sales Prediction.
"""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data_processor import (
    load_raw, clean, engineer_features,
    get_feature_columns, VALID_ITEMS, ITEM_CATEGORY, ITEM_PRICE_MAP,
)
from src.eda import (
    compute_kpis,
    fig_monthly_revenue,
    fig_item_revenue,
    fig_item_transactions,
    fig_category_revenue,
    fig_daily_trend,
    fig_weekday_revenue,
    fig_heatmap_month_item,
)
from src.model_trainer import load_model, load_metrics, predict_single

app = FastAPI(
    title="Cafe Sales Prediction API",
    description="Predict cafe transaction totals and explore sales analytics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ─────────────────────────────────────────────────────────

ITEM_LIST = sorted(VALID_ITEMS)


class PredictRequest(BaseModel):
    item: str = Field(..., description=f"One of: {', '.join(ITEM_LIST)}")
    quantity: int = Field(..., ge=1, le=5, description="Number of items (1-5)")
    transaction_date: str = Field(..., description="Date in YYYY-MM-DD format")


class PredictResponse(BaseModel):
    predicted_total_spent: float
    item: str
    quantity: int
    price_per_unit: float
    transaction_date: str


# ── Helper ───────────────────────────────────────────────────────────────────

def _build_features(item: str, quantity: int, date_str: str) -> dict:
    item = item.strip().lower()
    if item not in VALID_ITEMS:
        raise HTTPException(status_code=400, detail=f"Unknown item '{item}'. Valid items: {ITEM_LIST}")
    date = pd.to_datetime(date_str)
    price = ITEM_PRICE_MAP[item]
    item_list = ITEM_LIST
    category = ITEM_CATEGORY[item]
    return {
        "item_encoded": item_list.index(item),
        "category_encoded": int(category == "food"),
        "Quantity": quantity,
        "Price Per Unit": price,
        "month": date.month,
        "day_of_week": date.dayofweek,
        "week_of_year": date.isocalendar()[1],
        "quarter": date.quarter,
        "is_weekend": int(date.dayofweek >= 5),
        "day_of_month": date.day,
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["root"])
def root():
    return {
        "message": "Cafe Sales Prediction API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/analytics", "/model-info"],
    }


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest):
    """Predict Total Spent for a cafe transaction."""
    item = req.item.strip().lower()
    features = _build_features(item, req.quantity, req.transaction_date)
    try:
        prediction = predict_single(features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return PredictResponse(
        predicted_total_spent=round(prediction, 2),
        item=item,
        quantity=req.quantity,
        price_per_unit=ITEM_PRICE_MAP[item],
        transaction_date=req.transaction_date,
    )


@app.get("/analytics", tags=["analytics"])
def analytics():
    """Return KPIs and chart JSON for all EDA charts."""
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)

    kpis = compute_kpis(df)

    charts = {
        "monthly_revenue": fig_monthly_revenue(df).to_json(),
        "item_revenue": fig_item_revenue(df).to_json(),
        "item_transactions": fig_item_transactions(df).to_json(),
        "category_revenue": fig_category_revenue(df).to_json(),
        "daily_trend": fig_daily_trend(df).to_json(),
        "weekday_revenue": fig_weekday_revenue(df).to_json(),
        "heatmap_month_item": fig_heatmap_month_item(df).to_json(),
    }

    return {"kpis": kpis, "charts": charts}


@app.get("/model-info", tags=["model"])
def model_info():
    """Return saved model metrics and metadata."""
    try:
        metrics = load_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return metrics
