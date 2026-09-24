"""
Tests for FastAPI backend endpoints.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

# Train model before backend tests
from src.model_trainer import train_all_models
train_all_models()

from backend.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "message" in body
    assert "Cafe Sales" in body["message"]


def test_predict_valid():
    payload = {
        "item": "coffee",
        "quantity": 2,
        "transaction_date": "2023-06-15",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_total_spent" in body
    assert body["item"] == "coffee"
    assert body["quantity"] == 2
    assert body["price_per_unit"] == 2.0
    assert isinstance(body["predicted_total_spent"], float)


def test_predict_invalid_item():
    payload = {
        "item": "pizza",
        "quantity": 1,
        "transaction_date": "2023-06-15",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 400


def test_predict_invalid_quantity():
    payload = {
        "item": "coffee",
        "quantity": 10,   # > 5
        "transaction_date": "2023-06-15",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_analytics():
    r = client.get("/analytics")
    assert r.status_code == 200
    body = r.json()
    assert "kpis" in body
    assert "charts" in body
    kpis = body["kpis"]
    assert "total_revenue" in kpis
    assert "total_transactions" in kpis
    assert kpis["total_transactions"] > 0


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "best_model" in body
    assert "feature_columns" in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
