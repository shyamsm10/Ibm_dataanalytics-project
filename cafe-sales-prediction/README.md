# ☕ Cafe Sales Prediction & Analytics

**Author:** Shyam Sarath  
**Project:** End-to-end ML pipeline for cafe sales prediction and interactive analytics

> Predicts cafe transaction totals using machine learning and delivers a fully interactive sales analytics dashboard — built with FastAPI, Streamlit, Scikit-learn, and Plotly.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Dashboard Pages](#dashboard-pages)
- [ML Models & Results](#ml-models--results)
- [Feature Engineering](#feature-engineering)
- [Running Tests](#running-tests)

---

## Overview

This project demonstrates a complete production-style ML pipeline:

1. **Data cleaning** — removes noise labels (`unknown`, `error`), parses dates, validates types
2. **Feature engineering** — extracts time features (month, weekday, week, quarter, weekend flag) and categorical encodings
3. **Model training** — trains and compares four regression models, selects the best by R²
4. **FastAPI backend** — serves predictions and analytics over a REST API
5. **Streamlit frontend** — a dark-themed interactive dashboard with KPI cards, revenue leaderboard, trend charts, and a live ML prediction form

---

## Dataset

| Property | Detail |
|---|---|
| File | `data/Cleaned_DataSet.csv` |
| Source | [Kaggle — Cafe Sales Dataset](https://www.kaggle.com/) |
| Raw rows | 9,741 |
| Clean rows | 9,121 (after removing `unknown` / `error` labels) |
| Date range | 2023-01-01 — 2023-12-31 |
| Target column | `Total Spent` (regression) |

**Columns**

| Column | Type | Description |
|---|---|---|
| `Transaction ID` | string | Unique transaction identifier |
| `Item` | string | Cafe item ordered |
| `Quantity` | int | Number of items (1–5) |
| `Price Per Unit` | float | Unit price ($1–$5) |
| `Total Spent` | float | Transaction total (target) |
| `Transaction Date` | date | Date of transaction |

**Valid items** — coffee · tea · juice · smoothie · cake · cookie · sandwich · salad

**Key insight** — Sandwich is the highest-revenue item ($13,484), while Juice leads by transaction count (1,499 transactions).

---

## Project Structure

```
cafe-sales-prediction/
├── data/
│   └── Cleaned_DataSet.csv          # Source dataset
│
├── models/
│   ├── best_model.pkl               # Saved best model (Gradient Boosting)
│   └── model_metrics.json           # Training metrics for all models
│
├── src/
│   ├── __init__.py
│   ├── data_processor.py            # Loading, cleaning, feature engineering
│   ├── eda.py                       # KPI computation + Plotly chart builders
│   ├── model_trainer.py             # Training, evaluation, model persistence
│   └── train.py                     # Training entry-point script
│
├── backend/
│   └── main.py                      # FastAPI application (4 endpoints)
│
├── frontend/
│   └── app.py                       # Streamlit dashboard (6 pages)
│
├── tests/
│   ├── __init__.py
│   ├── test_data_and_model.py       # 18 data processor + model tests
│   └── test_backend.py              # 6 API endpoint tests
│
├── ShyamSarath_CafeSalesPrediction.ipynb   # Project notebook
├── ShyamSarath_ProjectReport.docx          # Full project report
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Library | Version | Purpose |
|---|---|---|---|
| Data processing | `pandas` | 3.0.6 | Cleaning, transformation, feature engineering |
| Numerical | `numpy` | 2.5.0 | Array operations |
| Machine learning | `scikit-learn` | 1.9.0 | Model training, evaluation, pipelines |
| Visualisation | `plotly` | 7.1.0 | Interactive charts |
| Backend API | `fastapi` + `uvicorn` | 0.135.3 / 0.44.0 | REST API serving predictions and analytics |
| Frontend | `streamlit` | 1.64.0 | Interactive dashboard UI |
| Model storage | `joblib` | 1.5.3 | Serialise / deserialise trained model |
| HTTP client | `requests` | 2.33.1 | Frontend → backend communication |
| Testing | `pytest` | 9.x | Automated test suite |

---

## Setup & Installation

**Requirements:** Python 3.10+

```bash
# 1. Navigate to the project directory
cd cafe-sales-prediction

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 — Train the model

Run **once** from the project root. Skip if `models/best_model.pkl` already exists.

```bash
python src/train.py
```

Expected output:
```
Ridge Regression:   MAE=1.5702  RMSE=2.1901  R2=0.8258
Decision Tree:      MAE=0.6623  RMSE=1.5949  R2=0.9076
Random Forest:      MAE=0.6027  RMSE=1.4679  R2=0.9217
Gradient Boosting:  MAE=0.6230  RMSE=1.4400  R2=0.9247

Best model: Gradient Boosting
Model saved to models/best_model.pkl
```

### Step 2 — Start the backend

Open a terminal and run from the project root:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

- API base URL: `http://127.0.0.1:8000`
- Swagger UI docs: `http://127.0.0.1:8000/docs`

### Step 3 — Start the frontend

Open a **second** terminal and run:

```bash
python -m streamlit run frontend/app.py
```

- Dashboard URL: `http://localhost:8501`
- The browser opens automatically.

> Both terminals must stay running simultaneously.

---

## API Reference

### `GET /`
Returns API metadata.

### `POST /predict`
Predicts `Total Spent` for a single transaction.

**Request body**
```json
{
  "item": "sandwich",
  "quantity": 2,
  "transaction_date": "2023-06-15"
}
```

**Response**
```json
{
  "predicted_total_spent": 8.12,
  "item": "sandwich",
  "quantity": 2,
  "price_per_unit": 4.0,
  "transaction_date": "2023-06-15"
}
```

### `GET /analytics`
Returns KPIs (including revenue leaderboard) and all Plotly chart JSON.

### `GET /model-info`
Returns metrics for all trained models and the best model name.

---

## Dashboard Pages

| Page | Content |
|---|---|
| **📊 Dashboard** | 5 KPI cards, monthly revenue trend, revenue leaderboard, item revenue bar, category donut |
| **📈 Sales Trends** | Daily revenue line, weekday average bar, item × month heatmap |
| **🛒 Product Analysis** | Transaction share pie, revenue by item bar, heatmap |
| **🔮 Predict Sales** | Item selector, quantity slider, date picker, ML prediction card |
| **🤖 Model Performance** | Metrics table, MAE/RMSE grouped bar, R² chart, feature pills |
| **ℹ️ About** | Dataset summary, architecture table, stack details, run instructions |

---

## ML Models & Results

All models trained on **80% of clean data**, evaluated on **20% hold-out test set** (`random_state=42`).

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|
| Ridge Regression | 1.5702 | 2.1901 | 0.8258 |
| Decision Tree | 0.6623 | 1.5949 | 0.9076 |
| Random Forest | 0.6027 | 1.4679 | 0.9217 |
| **Gradient Boosting** ⭐ | **0.6230** | **1.4400** | **0.9247** |

> Gradient Boosting selected as best model (highest R²). Saved to `models/best_model.pkl`.

---

## Feature Engineering

| Feature | Source | Description |
|---|---|---|
| `item_encoded` | `Item` | Ordinal label encoding (alphabetical order) |
| `category_encoded` | `Item` | Binary: `1` = food, `0` = beverage |
| `Quantity` | `Quantity` | Number of items ordered |
| `Price Per Unit` | `Price Per Unit` | Unit price of the item |
| `month` | `Transaction Date` | Month number (1–12) |
| `day_of_week` | `Transaction Date` | Day of week (0=Mon, 6=Sun) |
| `week_of_year` | `Transaction Date` | ISO week number (1–53) |
| `quarter` | `Transaction Date` | Quarter (1–4) |
| `is_weekend` | `Transaction Date` | Binary: `1` if Saturday or Sunday |
| `day_of_month` | `Transaction Date` | Day of month (1–31) |

> `Total Spent` is the **target only** — never used as a feature (no data leakage).

---

## Running Tests

```bash
pytest tests/ -v
```

**24 tests — all passing:**
- `tests/test_data_and_model.py` — 18 tests: data loading, cleaning, feature engineering, model training, prediction range
- `tests/test_backend.py` — 6 tests: all API endpoints (root, predict valid/invalid, analytics, model-info)

---

*Built by Shyam Sarath · Python · FastAPI · Streamlit · Scikit-learn · Plotly*
