"""
Streamlit frontend for Cafe Sales Prediction & Analytics — Premium UI.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Any

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from datetime import date

# ── Page config ───────────────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Cafe Sales Prediction",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base & font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem;
    padding: 6px 0;
    cursor: pointer;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}
[data-testid="stSidebar"] .stCaption { color: #94a3b8 !important; }

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,179,237,0.15);
}
.kpi-icon { font-size: 2rem; margin-bottom: 0.4rem; }
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1.1;
}
.kpi-sub { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; }

/* ── Section headers ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1.4rem 0 0.6rem 0;
    padding-left: 0.8rem;
    border-left: 3px solid #3b82f6;
}

/* ── Page title ── */
.page-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.page-subtitle {
    color: #64748b;
    font-size: 0.92rem;
    margin-bottom: 1.4rem;
}

/* ── Prediction card ── */
.pred-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a1a2e 100%);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 20px;
    padding: 2rem 2.4rem;
    margin-top: 1.2rem;
    text-align: center;
}
.pred-amount {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #63b3ed, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.pred-label {
    font-size: 0.85rem;
    color: #94a3b8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.pred-detail {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1.4rem;
    flex-wrap: wrap;
}
.pred-detail-item { text-align: center; }
.pred-detail-val { font-size: 1.2rem; font-weight: 600; color: #e2e8f0; }
.pred-detail-lbl { font-size: 0.73rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Metric comparison table ── */
.best-badge {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── Feature pill ── */
.feature-pill {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #93c5fd;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    margin: 3px;
}

/* ── About card ── */
.about-card {
    background: #1e293b;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
}
.about-card h4 { color: #63b3ed; margin-top: 0; font-size: 0.95rem; }
.about-card p, .about-card li { color: #94a3b8; font-size: 0.88rem; line-height: 1.7; }

/* ── Revenue leaderboard ── */
.rev-leader {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    height: 100%;
}
.rev-leader-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.9rem;
}
.rev-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.rev-row:last-child { border-bottom: none; }
.rev-rank {
    font-size: 0.72rem;
    font-weight: 700;
    color: #475569;
    width: 18px;
    flex-shrink: 0;
    text-align: center;
}
.rev-rank-1 { color: #f59e0b; }
.rev-rank-2 { color: #94a3b8; }
.rev-rank-3 { color: #b45309; }
.rev-name {
    flex: 1;
    font-size: 0.88rem;
    font-weight: 500;
    color: #e2e8f0;
}
.rev-bar-wrap {
    flex: 2;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.rev-bar { height: 6px; border-radius: 4px; background: #3b82f6; }
.rev-bar-1 { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.rev-bar-2 { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.rev-bar-3 { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.rev-amt {
    font-size: 0.83rem;
    font-weight: 600;
    color: #94a3b8;
    width: 60px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Insight banner ── */
.insight-banner {
    background: linear-gradient(135deg, rgba(59,130,246,0.12) 0%, rgba(167,139,250,0.08) 100%);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 12px;
    padding: 0.75rem 1.1rem;
    font-size: 0.84rem;
    color: #93c5fd;
    margin-bottom: 0.6rem;
}
.insight-banner strong { color: #e2e8f0; }

/* ── Plotly chart background ── */
.js-plotly-plot .plotly { border-radius: 12px; }

/* ── Form tweaks ── */
[data-testid="stForm"] {
    background: #1e293b;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.6rem;
}
.stSelectbox label, .stSlider label, .stDateInput label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

/* ── Status banner ── */
.status-ok {
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    color: #86efac;
    font-size: 0.84rem;
    margin-bottom: 1rem;
}
.status-err {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    color: #fca5a5;
    font-size: 0.84rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme ─────────────────────────────────────────────────────────

PLOTLY_LAYOUT: dict[str, Any] = dict(
    paper_bgcolor="rgba(15,23,42,0)",
    plot_bgcolor="rgba(15,23,42,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    title_font=dict(color="#e2e8f0", size=15, family="Inter, sans-serif"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.1)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    margin=dict(l=10, r=10, t=45, b=10),
)


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def render_chart(json_str: str, key: str | None = None):
    fig = pio.from_json(json_str)
    apply_theme(fig)
    st.plotly_chart(fig, width='stretch', key=key)


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_analytics() -> dict:
    try:
        r = requests.get(f"{API_BASE}/analytics", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def fetch_model_info() -> dict:
    try:
        r = requests.get(f"{API_BASE}/model-info", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 0.5rem 0;'>
        <div style='font-size:3rem;'>☕</div>
        <div style='font-size:1.25rem; font-weight:700; color:#e2e8f0; margin-top:0.3rem;'>Cafe Sales</div>
        <div style='font-size:0.78rem; color:#64748b; letter-spacing:0.1em; text-transform:uppercase;'>Prediction & Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.1); margin:0.8rem 0'/>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊  Dashboard",
            "📈  Sales Trends",
            "🛒  Product Analysis",
            "🔮  Predict Sales",
            "🤖  Model Performance",
            "ℹ️  About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.1); margin:1rem 0 0.6rem 0'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding: 0 0.2rem;'>
        <div style='font-size:0.73rem; color:#475569; margin-bottom:6px;'>DATA SOURCE</div>
        <div style='font-size:0.82rem; color:#64748b;'>2023 Cafe Transactions</div>
        <div style='font-size:0.73rem; color:#475569; margin-top:10px; margin-bottom:4px;'>STACK</div>
        <div style='font-size:0.82rem; color:#64748b;'>FastAPI · Streamlit<br/>Scikit-learn · Plotly</div>
    </div>
    """, unsafe_allow_html=True)


# ── Shared: API connection guard ──────────────────────────────────────────────

def api_error_banner(msg: str):
    st.markdown(f'<div class="status-err">⚠ {msg}</div>', unsafe_allow_html=True)
    st.markdown(
        "> **Start the backend:** `uvicorn backend.main:app --reload --port 8000`",
        unsafe_allow_html=False,
    )


def kpi_card(icon: str, label: str, value: str, sub: str = ""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


# ── Page: Dashboard ───────────────────────────────────────────────────────────

if page == "📊  Dashboard":
    page_header("📊 Sales Dashboard", "Key performance indicators and revenue overview — 2023")

    data = fetch_analytics()
    if "error" in data:
        api_error_banner(f"Cannot connect to API: {data['error']}")
    else:
        kpis = data["kpis"]

        # ── Row 1: KPI cards ──────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi_card("💰", "Total Revenue",    f"${kpis['total_revenue']:,.0f}",    "2023 full year")
        with c2: kpi_card("🧾", "Transactions",     f"{kpis['total_transactions']:,}",   "clean records")
        with c3: kpi_card("🛍️", "Avg Order Value", f"${kpis['avg_order_value']:.2f}",   "per transaction")
        with c4: kpi_card("🏆", "Top Revenue Item", kpis["top_item"].title(),            f"${kpis.get('top_item_revenue', 0):,.0f} revenue")
        with c5: kpi_card("📦", "Avg Qty / Order",  f"{kpis['avg_quantity']:.2f}",       "items per order")

        st.markdown("<br/>", unsafe_allow_html=True)

        # ── Row 2: Monthly trend (wide) + Revenue leaderboard ─────────────────
        col_trend, col_board = st.columns([3, 2])

        with col_trend:
            section("Monthly Revenue Trend")
            render_chart(data["charts"]["monthly_revenue"], key="dash_monthly_revenue")

        with col_board:
            section("Revenue Leaderboard")
            # Data comes directly from KPIs — no chart parsing needed
            item_revenues = kpis.get("item_revenues", [])
            if item_revenues:
                max_rev = item_revenues[0]["revenue"] or 1
                bar_classes  = ["rev-bar-1", "rev-bar-2", "rev-bar-3"]
                rank_classes = ["rev-rank-1", "rev-rank-2", "rev-rank-3"]
                rows = []
                for i, entry in enumerate(item_revenues[:8]):
                    rev      = entry["revenue"]
                    name     = entry["item"]
                    pct      = int(rev / max_rev * 100)
                    rank_cls = rank_classes[i] if i < 3 else ""
                    bar_cls  = bar_classes[i]  if i < 3 else "rev-bar"
                    medal    = ["🥇", "🥈", "🥉"][i] if i < 3 else str(i + 1)
                    amt      = f"&#36;{rev:,.0f}"
                    rows.append(
                        f'<div class="rev-row">'
                        f'<div class="rev-rank {rank_cls}">{medal}</div>'
                        f'<div class="rev-name">{name.title()}</div>'
                        f'<div class="rev-bar-wrap"><div class="{bar_cls}" style="width:{pct}%"></div></div>'
                        f'<div class="rev-amt">{amt}</div>'
                        f'</div>'
                    )
                html = (
                    '<div class="rev-leader">'
                    '<div class="rev-leader-title">Revenue by Item &#8212; Ranked</div>'
                    + "".join(rows)
                    + '</div>'
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                render_chart(data["charts"]["item_revenue"], key="dash_leaderboard_fallback")

        st.markdown("<br/>", unsafe_allow_html=True)

        # ── Row 3: Revenue by Item bar + Category split ───────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            section("Revenue by Item")
            render_chart(data["charts"]["item_revenue"], key="dash_item_revenue")
        with col_b:
            section("Category Split")
            render_chart(data["charts"]["category_revenue"], key="dash_category_revenue")

# ── Page: Sales Trends ────────────────────────────────────────────────────────

elif page == "📈  Sales Trends":
    page_header("📈 Sales Trends", "Daily, weekly, and seasonal revenue patterns")

    data = fetch_analytics()
    if "error" in data:
        api_error_banner(f"Cannot connect to API: {data['error']}")
    else:
        section("Daily Revenue Trend — 2023")
        render_chart(data["charts"]["daily_trend"], key="trends_daily")

        col_a, col_b = st.columns(2)
        with col_a:
            section("Average Revenue by Day of Week")
            render_chart(data["charts"]["weekday_revenue"], key="trends_weekday")
        with col_b:
            section("Revenue Heatmap: Item × Month")
            render_chart(data["charts"]["heatmap_month_item"], key="trends_heatmap")

# ── Page: Product Analysis ────────────────────────────────────────────────────

elif page == "🛒  Product Analysis":
    page_header("🛒 Product & Category Analysis", "Breakdown by item and category")

    data = fetch_analytics()
    if "error" in data:
        api_error_banner(f"Cannot connect to API: {data['error']}")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            section("Transaction Share by Item")
            render_chart(data["charts"]["item_transactions"], key="prod_item_tx")
        with col_b:
            section("Revenue by Item")
            render_chart(data["charts"]["item_revenue"], key="prod_item_revenue")

        section("Revenue Heatmap: Item × Month")
        render_chart(data["charts"]["heatmap_month_item"], key="prod_heatmap")

# ── Page: Predict ─────────────────────────────────────────────────────────────

elif page == "🔮  Predict Sales":
    page_header("🔮 Predict Transaction Total", "Enter transaction details and get an ML-powered revenue forecast")

    VALID_ITEMS = ["cake", "coffee", "cookie", "juice", "salad", "sandwich", "smoothie", "tea"]
    ITEM_ICONS = {"cake": "🎂", "coffee": "☕", "cookie": "🍪", "juice": "🍊",
                  "salad": "🥗", "sandwich": "🥪", "smoothie": "🥤", "tea": "🍵"}
    ITEM_PRICE = {"cake": 3.0, "coffee": 2.0, "cookie": 1.0, "juice": 3.0,
                  "salad": 5.0, "sandwich": 4.0, "smoothie": 4.0, "tea": 1.5}

    with st.form("predict_form", clear_on_submit=False):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            item = st.selectbox(
                "Item",
                options=VALID_ITEMS,
                format_func=lambda x: f"{ITEM_ICONS[x]}  {x.title()}",
            )
        with col2:
            quantity = st.slider("Quantity", min_value=1, max_value=5, value=2)
        with col3:
            transaction_date = st.date_input(
                "Transaction Date",
                value=date(2023, 6, 15),
                min_value=date(2023, 1, 1),
                max_value=date(2024, 12, 31),
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔮  Get Prediction",
            width='stretch',
            type="primary",
        )

    # Live preview below the form
    price = ITEM_PRICE[item]
    naive_total = price * quantity
    col_p, col_q, col_t = st.columns(3)
    col_p.metric("Unit Price", f"${price:.2f}")
    col_q.metric("Quantity", quantity)
    col_t.metric("Simple Total (Qty × Price)", f"${naive_total:.2f}")

    if submitted:
        payload = {
            "item": item,
            "quantity": quantity,
            "transaction_date": str(transaction_date),
        }
        try:
            with st.spinner("Running model prediction..."):
                r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            r.raise_for_status()
            result = r.json()

            pred = result["predicted_total_spent"]
            delta_pct = ((pred - naive_total) / naive_total * 100) if naive_total else 0

            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-label">ML Predicted Total Spent</div>
                <div class="pred-amount">${pred:.2f}</div>
                <div class="pred-detail">
                    <div class="pred-detail-item">
                        <div class="pred-detail-val">{ITEM_ICONS[item]} {result['item'].title()}</div>
                        <div class="pred-detail-lbl">Item</div>
                    </div>
                    <div class="pred-detail-item">
                        <div class="pred-detail-val">{result['quantity']}</div>
                        <div class="pred-detail-lbl">Quantity</div>
                    </div>
                    <div class="pred-detail-item">
                        <div class="pred-detail-val">${result['price_per_unit']:.2f}</div>
                        <div class="pred-detail-lbl">Unit Price</div>
                    </div>
                    <div class="pred-detail-item">
                        <div class="pred-detail-val">{result['transaction_date']}</div>
                        <div class="pred-detail-lbl">Date</div>
                    </div>
                    <div class="pred-detail-item">
                        <div class="pred-detail-val" style="color:{'#86efac' if delta_pct >= 0 else '#fca5a5'}">
                            {'+' if delta_pct >= 0 else ''}{delta_pct:.1f}%
                        </div>
                        <div class="pred-detail-lbl">vs Simple Total</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            api_error_banner("Cannot connect to API. Start the backend first.")
        except Exception as e:
            st.markdown(f'<div class="status-err">Prediction failed: {e}</div>', unsafe_allow_html=True)

# ── Page: Model Performance ───────────────────────────────────────────────────

elif page == "🤖  Model Performance":
    page_header("🤖 Model Performance", "Comparison of all trained ML models on the hold-out test set")

    info = fetch_model_info()
    if "error" in info:
        api_error_banner(f"Cannot load model info: {info['error']}")
    else:
        best = info.get("best_model", "N/A")

        st.markdown(f"""
        <div class="status-ok">
            ✓ &nbsp; Best selected model: <strong>{best}</strong> &nbsp;·&nbsp; Criterion: highest R² on 20% hold-out test set
        </div>
        """, unsafe_allow_html=True)

        model_names = [k for k in info if k not in ("best_model", "feature_columns")]
        rows = []
        for name in model_names:
            m = info[name]
            rows.append({
                "Model": name,
                "MAE ↓": m["MAE"],
                "RMSE ↓": m["RMSE"],
                "R² ↑": m["R2"],
                "Status": "Best" if name == best else "",
            })

        df_metrics = pd.DataFrame(rows).sort_values("R² ↑", ascending=False)

        # Use a plain dataframe for compatibility across Streamlit versions.
        display_df = df_metrics.copy()
        for col in ["MAE ↓", "RMSE ↓", "R² ↑"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
        st.dataframe(display_df, width='stretch', hide_index=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            section("MAE & RMSE — lower is better")
            fig_err = go.Figure()
            for metric, color in [("MAE ↓", "#3b82f6"), ("RMSE ↓", "#f59e0b")]:
                fig_err.add_trace(go.Bar(
                    name=metric.split(" ")[0],
                    x=df_metrics["Model"],
                    y=df_metrics[metric],
                    marker_color=color,
                    marker_line_color="rgba(0,0,0,0)",
                    marker_opacity=0.85,
                ))
            fig_err.update_layout(**PLOTLY_LAYOUT, barmode="group")
            st.plotly_chart(fig_err, width='stretch')

        with col_b:
            section("R² Score — higher is better")
            fig_r2 = go.Figure(go.Bar(
                x=df_metrics["Model"],
                y=df_metrics["R² ↑"],
                marker_color=[
                    "#22c55e" if n == best else "#3b82f6"
                    for n in df_metrics["Model"]
                ],
                marker_line_color="rgba(0,0,0,0)",
                marker_opacity=0.85,
                text=df_metrics["R² ↑"].apply(lambda v: f"{v:.4f}"),
                textposition="outside",
                textfont=dict(color="#94a3b8", size=11),
            ))
            fig_r2.update_layout(**PLOTLY_LAYOUT, yaxis_range=[0, 1.05])
            st.plotly_chart(fig_r2, width='stretch')

        section("Features Used by Model")
        features = info.get("feature_columns", [])
        pills = "".join(f'<span class="feature-pill">{f}</span>' for f in features)
        st.markdown(f"<div style='margin-top:0.5rem'>{pills}</div>", unsafe_allow_html=True)

# ── Page: About ───────────────────────────────────────────────────────────────

elif page == "ℹ️  About":
    page_header("ℹ️ About This Project", "End-to-end ML pipeline for cafe sales prediction")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="about-card">
            <h4>📂 Dataset</h4>
            <ul>
                <li>Source: <code>Cleaned_DataSet.csv</code></li>
                <li>Cleaned cafe transaction dataset</li>
                <li>Date range: Jan – Dec 2023</li>
                <li>Items: coffee, tea, juice, smoothie, cake, cookie, sandwich, salad</li>
                <li>Target: <code>Total Spent</code> (regression)</li>
            </ul>
        </div>
        <div class="about-card">
            <h4>🤖 Models Trained</h4>
            <ul>
                <li>Ridge Regression</li>
                <li>Decision Tree</li>
                <li>Random Forest</li>
                <li>Gradient Boosting</li>
            </ul>
            <p>The trained model information is loaded from the backend.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="about-card">
            <h4>🏗️ Architecture</h4>
            <table style="width:100%; font-size:0.85rem; color:#94a3b8; border-collapse:collapse;">
                <tr><td style="padding:4px 0; color:#63b3ed;">Data Processing</td><td>Pandas, NumPy</td></tr>
                <tr><td style="padding:4px 0; color:#63b3ed;">Machine Learning</td><td>Scikit-learn</td></tr>
                <tr><td style="padding:4px 0; color:#63b3ed;">Visualisation</td><td>Plotly</td></tr>
                <tr><td style="padding:4px 0; color:#63b3ed;">Backend API</td><td>FastAPI + Uvicorn</td></tr>
                <tr><td style="padding:4px 0; color:#63b3ed;">Frontend</td><td>Streamlit</td></tr>
                <tr><td style="padding:4px 0; color:#63b3ed;">Model Storage</td><td>Joblib</td></tr>
            </table>
        </div>
        <div class="about-card">
            <h4>🚀 How to Run</h4>
            <p style="margin:0 0 0.4rem 0;">1. Install dependencies</p>
            <code style="color:#63b3ed; font-size:0.82rem;">pip install -r requirements.txt</code>
            <p style="margin:0.6rem 0 0.4rem 0;">2. Train the model</p>
            <code style="color:#63b3ed; font-size:0.82rem;">python src/train.py</code>
            <p style="margin:0.6rem 0 0.4rem 0;">3. Start backend</p>
            <code style="color:#63b3ed; font-size:0.82rem;">uvicorn backend.main:app --reload --port 8000</code>
            <p style="margin:0.6rem 0 0.4rem 0;">4. Start frontend</p>
            <code style="color:#63b3ed; font-size:0.82rem;">streamlit run frontend/app.py</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:2rem; padding:1.2rem;
         border-top:1px solid rgba(255,255,255,0.07); color:#475569; font-size:0.82rem;'>
        Built with Python &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Scikit-learn &nbsp;·&nbsp; Plotly
    </div>
    """, unsafe_allow_html=True)
