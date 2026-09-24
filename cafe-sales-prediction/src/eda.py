"""
EDA module — returns Plotly figures based on the cleaned dataset.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_processor import load_raw, clean, engineer_features


def get_full_df() -> pd.DataFrame:
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    return df


# ── KPI helpers ─────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    item_rev = df.groupby("Item")["Total Spent"].sum().sort_values(ascending=False)
    top_item = item_rev.idxmax()
    top_item_revenue = round(float(item_rev.max()), 2)
    item_revenues = [
        {"item": item, "revenue": round(float(rev), 2)}
        for item, rev in item_rev.items()
    ]
    return {
        "total_revenue": round(df["Total Spent"].sum(), 2),
        "total_transactions": len(df),
        "avg_order_value": round(df["Total Spent"].mean(), 2),
        "top_item": top_item,
        "top_item_revenue": top_item_revenue,
        "item_revenues": item_revenues,
        "top_category": df["category"].value_counts().idxmax(),
        "avg_quantity": round(df["Quantity"].mean(), 2),
    }


# ── Chart builders ───────────────────────────────────────────────────────────

def fig_monthly_revenue(df: pd.DataFrame) -> go.Figure:
    monthly = df.groupby("month")["Total Spent"].sum().reset_index()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m - 1])
    fig = px.bar(
        monthly, x="month_name", y="Total Spent",
        title="Monthly Revenue (2023)",
        labels={"Total Spent": "Revenue ($)", "month_name": "Month"},
        color="Total Spent",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def fig_item_revenue(df: pd.DataFrame) -> go.Figure:
    item_rev = df.groupby("Item")["Total Spent"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(
        item_rev, x="Item", y="Total Spent",
        title="Revenue by Item",
        labels={"Total Spent": "Total Revenue ($)"},
        color="Item",
    )
    return fig


def fig_item_transactions(df: pd.DataFrame) -> go.Figure:
    item_cnt = df["Item"].value_counts().reset_index()
    item_cnt.columns = ["Item", "Transactions"]
    fig = px.pie(
        item_cnt, names="Item", values="Transactions",
        title="Transaction Share by Item",
        hole=0.4,
    )
    return fig


def fig_category_revenue(df: pd.DataFrame) -> go.Figure:
    cat_rev = df.groupby("category")["Total Spent"].sum().reset_index()
    fig = px.pie(
        cat_rev, names="category", values="Total Spent",
        title="Revenue Split: Beverage vs Food",
        color_discrete_map={"beverage": "#3b82f6", "food": "#f59e0b"},
        hole=0.4,
    )
    return fig


def fig_daily_trend(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby("Transaction Date")["Total Spent"].sum().reset_index()
    fig = px.line(
        daily, x="Transaction Date", y="Total Spent",
        title="Daily Revenue Trend",
        labels={"Total Spent": "Revenue ($)", "Transaction Date": "Date"},
    )
    fig.update_traces(line_color="#3b82f6")
    return fig


def fig_weekday_revenue(df: pd.DataFrame) -> go.Figure:
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    wd = df.groupby("day_of_week")["Total Spent"].mean().reset_index()
    wd["day_name"] = wd["day_of_week"].apply(lambda d: day_names[d])
    fig = px.bar(
        wd, x="day_name", y="Total Spent",
        title="Average Revenue by Day of Week",
        labels={"Total Spent": "Avg Revenue ($)", "day_name": "Day"},
        color="Total Spent",
        color_continuous_scale="Greens",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def fig_quantity_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="Quantity", nbins=5,
        title="Quantity Distribution",
        labels={"Quantity": "Items per Transaction"},
        color_discrete_sequence=["#7c5cd8"],
    )
    return fig


def fig_price_vs_total(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(1000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Price Per Unit", y="Total Spent",
        color="Item", size="Quantity",
        title="Price Per Unit vs Total Spent",
        labels={"Price Per Unit": "Price/Unit ($)", "Total Spent": "Total ($)"},
        opacity=0.7,
    )
    return fig


def fig_heatmap_month_item(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(
        index="Item", columns="month", values="Total Spent", aggfunc="sum"
    ).fillna(0)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[month_names[m - 1] for m in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="Blues",
        text=pivot.values.round(0),
        texttemplate="%{text}",
    ))
    fig.update_layout(title="Revenue Heatmap: Item × Month")
    return fig
