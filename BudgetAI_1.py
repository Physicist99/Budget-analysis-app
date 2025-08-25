# app.py
import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import altair as alt
from datetime import datetime

# =============================
# Page & Theme
# =============================
st.set_page_config(
    page_title="AI Budget Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
        text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .filter-header {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        color: white; padding: 0.75rem; border-radius: 8px; margin: 1rem 0 0.5rem 0;
        font-weight: bold; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8); color: white; border: none;
        border-radius: 8px; padding: 0.5rem 1rem; font-weight: bold; transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(59,130,246,0.3);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(59,130,246,0.4); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9; border-radius: 8px; color: #475569; font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8); color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Config & OpenAI (optional)
# =============================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None
if not api_key:
    st.markdown("""
    <div style="background: linear-gradient(45deg, #f59e0b, #d97706); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        ⚠️ <strong>OpenAI API Key not found:</strong> Add OPENAI_API_KEY to .env or Streamlit secrets to enable GPT features (optional).
    </div>
    """, unsafe_allow_html=True)

# =============================
# Helpers
# =============================
def money_fmt(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return "-"

@st.cache_data
def load_actuals(path: str) -> pd.DataFrame:
    """Load and validate actuals/allocated CSV."""
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        # Fix common typo
        if "Budget_Allcated" in df.columns:
            df = df.rename(columns={"Budget_Allcated": "Budget_Allocated"})

        req = ["Month", "Department", "Category", "Budget_Allocated", "Actual_Spent"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            st.error(f"Missing required columns in actuals data: {miss}")
            return pd.DataFrame()

        # Optional variance column
        if "Variance" not in df.columns:
            df["Variance"] = df["Actual_Spent"] - df["Budget_Allocated"]

        # Parse types
        df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
        if df["Month"].isna().any():
            df["Month"] = pd.to_datetime(df["Month"], errors="coerce")

        for c in ["Budget_Allocated", "Actual_Spent", "Variance"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["Month", "Budget_Allocated", "Actual_Spent"]).copy()
        df["Year"] = df["Month"].dt.year
        df["Quarter"] = df["Month"].dt.quarter
        df["Variance_Percent"] = ((df["Actual_Spent"] - df["Budget_Allocated"]) / df["Budget_Allocated"] * 100).round(2)

        return df.sort_values("Month").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading actuals: {e}")
        return pd.DataFrame()

@st.cache_data
def load_forecast(path: str) -> pd.DataFrame | None:
    """Load optional forecast file."""
    try:
        if not os.path.exists(path):
            return None
        f = pd.read_csv(path)
        f.columns = [c.strip().replace(" ", "_") for c in f.columns]
        # Standardize prediction column name
        if "Predicted_Spent" not in f.columns:
            for alt in ["Forecast", "Forecasted", "yhat", "y_pred", "Prediction"]:
                if alt in f.columns:
                    f = f.rename(columns={alt: "Predicted_Spent"})
                    break

        req = ["Month", "Department", "Category", "Predicted_Spent"]
        miss = [c for c in req if c not in f.columns]
        if miss:
            st.error(f"Missing required columns in forecast data: {miss}")
            return None

        f["Month"] = pd.to_datetime(f["Month"], errors="coerce")
        f["Predicted_Spent"] = pd.to_numeric(f["Predicted_Spent"], errors="coerce")
        f = f.dropna(subset=["Month", "Predicted_Spent"]).copy()

        return f.sort_values("Month").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading forecast: {e}")
        return None

# =============================
# Load Data
# =============================
ACTUALS_PATH = "rich_dummy_budget_data.csv"
FORECAST_PATH = "forecasted_budget_2025.csv"

df = load_actuals(ACTUALS_PATH)
forecast = load_forecast(FORECAST_PATH)

if df.empty:
    st.error("❌ Could not load actuals. Ensure 'rich_dummy_budget_data.csv' exists with required columns.")
    st.stop()

# =============================
# Sidebar Filters
# =============================
with st.sidebar:
    st.markdown('<div class="filter-header">🎛️ Control Panel</div>', unsafe_allow_html=True)

    # Overview
    st.markdown("**📊 Dataset Overview**")
    colA, colB = st.columns(2)
    with colA:
        st.metric("📋 Records", f"{len(df):,}")
        st.metric("🏢 Departments", f"{df['Department'].nunique():,}")
    with colB:
        st.metric("📁 Categories", f"{df['Category'].nunique():,}")
        st.metric("📅 Years", f"{df['Year'].min()}–{df['Year'].max()}")

    st.markdown("---")
    st.markdown('<div class="filter-header">🔍 Primary Filters</div>', unsafe_allow_html=True)

    dept_sel = st.multiselect(
        "🏢 Department(s)",
        options=sorted(df["Department"].unique()),
        default=sorted(df["Department"].unique()),
    )

    cat_sel = st.multiselect(
        "📁 Category(ies)",
        options=sorted(df["Category"].unique()),
        default=sorted(df["Category"].unique()),
    )

    min_m = pd.to_datetime(df["Month"].min()).to_pydatetime()
    max_m = pd.to_datetime(df["Month"].max()).to_pydatetime()
    date_range = st.slider(
        "📅 Month Range",
        value=(min_m, max_m), min_value=min_m, max_value=max_m, format="YYYY-MM"
    )

    st.markdown('<div class="filter-header">⚙️ Advanced Filters</div>', unsafe_allow_html=True)

    year_opts = sorted(df["Year"].unique())
    years_sel = st.multiselect("📅 Years", options=year_opts, default=year_opts)

    vmin, vmax = float(df["Variance_Percent"].min()), float(df["Variance_Percent"].max())
    variance_range = st.slider("📊 Variance % Range", min_value=vmin, max_value=vmax, value=(vmin, vmax), step=1.0)

    amin, amax = float(df["Actual_Spent"].min()), float(df["Actual_Spent"].max())
    amount_range = st.slider("💰 Actual Spent Range", min_value=amin, max_value=amax, value=(amin, amax), step=1000.0, format="$%.0f")

    perf = st.selectbox(
        "🎯 Budget Performance",
        ["All", "Over Budget (>0%)", "Under Budget (<0%)", "On Target (±5%)", "Significant Variance (>±10%)"]
    )

    st.markdown('<div class="filter-header">🔮 Forecast</div>', unsafe_allow_html=True)
    show_forecast = st.checkbox("📈 Show 2025 Forecast", value=(forecast is not None), disabled=(forecast is None))
    st.caption("Place 'forecasted_budget_2025.csv' to enable.")

# Apply filters
mask = (
    df["Department"].isin(dept_sel)
    & df["Category"].isin(cat_sel)
    & (df["Month"] >= pd.to_datetime(date_range[0]))
    & (df["Month"] <= pd.to_datetime(date_range[1]))
    & df["Year"].isin(years_sel)
    & df["Variance_Percent"].between(variance_range[0], variance_range[1])
    & df["Actual_Spent"].between(amount_range[0], amount_range[1])
)

if perf == "Over Budget (>0%)":
    mask &= df["Variance_Percent"] > 0
elif perf == "Under Budget (<0%)":
    mask &= df["Variance_Percent"] < 0
elif perf == "On Target (±5%)":
    mask &= df["Variance_Percent"].between(-5, 5)
elif perf == "Significant Variance (>±10%)":
    mask &= (df["Variance_Percent"] > 10) | (df["Variance_Percent"] < -10)

df_f = df.loc[mask].copy()

f_f = None
if forecast is not None:
    fmask = forecast["Department"].isin(dept_sel) & forecast["Category"].isin(cat_sel)
    f_f = forecast.loc[fmask].copy()

# =============================
# Header & KPIs
# =============================
st.markdown("""
<div class="main-header">
  <h1>💼 AI Budget Forecast & Analysis</h1>
  <p style="font-size: 1.1rem; margin: 0.5rem 0;">Advanced Financial Intelligence Platform</p>
  <p style="font-size: 0.9rem; opacity: 0.9;">Actuals: historical to present · Forecast: 2025 (optional)</p>
</div>
""", unsafe_allow_html=True)

if df_f.empty:
    st.warning("No data matches your current filters. Try expanding the date range or selections.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
total_budget = df_f["Budget_Allocated"].sum()
total_actual = df_f["Actual_Spent"].sum()
total_var = df_f["Variance"].sum()
var_pct = (total_var / total_budget * 100) if total_budget else 0.0

with col1:
    st.metric("💰 Total Budget", money_fmt(total_budget), delta=f"{len(df_f):,} records")
with col2:
    delta_color = "normal" if abs(var_pct) < 5 else "inverse"
    st.metric("💳 Actual Spent", money_fmt(total_actual), delta=f"vs Budget: {var_pct:+.1f}%", delta_color=delta_color)
with col3:
    variance_color = "inverse" if total_var > 0 else "normal"
    st.metric("📊 Net Variance", f"{money_fmt(total_var)}", delta=f"{var_pct:+.2f}%", delta_color=variance_color)
with col4:
    efficiency = max(0.0, 100 - abs(var_pct))
    tag = "Excellent" if efficiency > 95 else "Good" if efficiency > 85 else "Needs Review"
    eff_color = "normal" if efficiency > 90 else "inverse"
    st.metric("🎯 Budget Efficiency", f"{efficiency:.1f}%", delta=tag, delta_color=eff_color)

# Quick stats
c1, c2, c3, c4 = st.columns(4)
over_b = (df_f["Variance_Percent"] > 0).sum()
under_b = (df_f["Variance_Percent"] < 0).sum()
on_tgt = df_f["Variance_Percent"].between(-2, 2).sum()
avg_var = df_f["Variance_Percent"].mean()
with c1: st.metric("🔴 Over Budget", f"{over_b}", f"{over_b/len(df_f)*100:.1f}%")
with c2: st.metric("🟢 Under Budget", f"{under_b}", f"{under_b/len(df_f)*100:.1f}%")
with c3: st.metric("🎯 On Target (±2%)", f"{on_tgt}", f"{on_tgt/len(df_f)*100:.1f}%")
with c4: st.metric("📈 Avg Variance", f"{avg_var:+.1f}%", "Overall")

# =============================
# Visual Analytics
# =============================
st.markdown("### 📊 Visual Analytics")

# Prep: Monthly trend (Allocated vs Spent) + optional forecast
monthly = (
    df_f.groupby("Month", as_index=False)[["Budget_Allocated", "Actual_Spent"]].sum()
    .sort_values("Month")
)
monthly_long = monthly.melt(
    id_vars="Month",
    value_vars=["Budget_Allocated", "Actual_Spent"],
    var_name="Type",
    value_name="Amount"
)
type_map = {"Budget_Allocated": "Allocated", "Actual_Spent": "Spent"}
monthly_long["Type"] = monthly_long["Type"].map(type_map)

if show_forecast and (f_f is not None) and (len(f_f) > 0):
    fc = (
        f_f.groupby("Month", as_index=False)["Predicted_Spent"].sum()
        .rename(columns={"Predicted_Spent": "Amount"})
    )
    fc["Type"] = "Forecast (Spent)"
    monthly_long = pd.concat([monthly_long, fc], ignore_index=True)

# Prep: Department & Category
dept = df_f.groupby("Department", as_index=False)[["Budget_Allocated", "Actual_Spent"]].sum()
dept_long = dept.melt("Department", ["Budget_Allocated", "Actual_Spent"], var_name="Type", value_name="Amount")
dept_long["Type"] = dept_long["Type"].map(type_map)

cat = df_f.groupby("Category", as_index=False)[["Budget_Allocated", "Actual_Spent"]].sum()
cat_long = cat.melt("Category", ["Budget_Allocated", "Actual_Spent"], var_name="Type", value_name="Amount")
cat_long["Type"] = cat_long["Type"].map(type_map)

# Tabs
tab1, tab2, tab3 = st.tabs(["📆 Monthly Trend", "🏢 By Department", "🗂️ By Category"])

with tab1:
    st.caption("Monthly Allocated vs Spent as clustered bars. Hover for values.")
    chart_monthly = (
        alt.Chart(monthly_long)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(Month):O", title="Month", sort=None),
            y=alt.Y("Amount:Q", title="Amount ($)", axis=alt.Axis(format="$,d")),
            color=alt.Color("Type:N", title=""),
            xOffset="Type:N",
            tooltip=[
                alt.Tooltip("yearmonth(Month):O", title="Month"),
                alt.Tooltip("Type:N"),
                alt.Tooltip("Amount:Q", title="Amount", format=",.0f")
            ]
        )
        .properties(height=360)
    )
    st.altair_chart(chart_monthly, use_container_width=True)

with tab2:
    st.caption("Allocated vs Spent by Department (sorted by variance).")
    dept_tot = df_f.groupby("Department", as_index=False).agg(
        Allocated=("Budget_Allocated", "sum"),
        Spent=("Actual_Spent", "sum")
    )
    dept_tot["Variance"] = dept_tot["Spent"] - dept_tot["Allocated"]
    order = dept_tot.sort_values("Variance", ascending=False)["Department"].tolist()

    chart_dept = (
        alt.Chart(dept_long)
        .mark_bar()
        .encode(
            x=alt.X("Department:N", sort=order, title="Department"),
            y=alt.Y("Amount:Q", title="Amount ($)", axis=alt.Axis(format="$,d")),
            color=alt.Color("Type:N", title=""),
            xOffset="Type:N",
            tooltip=[
                alt.Tooltip("Department:N"),
                alt.Tooltip("Type:N"),
                alt.Tooltip("Amount:Q", title="Amount", format=",.0f")
            ]
        )
        .properties(height=420)
    )
    st.altair_chart(chart_dept, use_container_width=True)

with tab3:
    st.caption("Allocated vs Spent by Category (sorted by variance).")
    cat_tot = df_f.groupby("Category", as_index=False).agg(
        Allocated=("Budget_Allocated", "sum"),
        Spent=("Actual_Spent", "sum")
    )
    cat_tot["Variance"] = cat_tot["Spent"] - cat_tot["Allocated"]
    order_cat = cat_tot.sort_values("Variance", ascending=False)["Category"].tolist()

    chart_cat = (
        alt.Chart(cat_long)
        .mark_bar()
        .encode(
            x=alt.X("Category:N", sort=order_cat, title="Category"),
            y=alt.Y("Amount:Q", title="Amount ($)", axis=alt.Axis(format="$,d")),
            color=alt.Color("Type:N", title=""),
            xOffset="Type:N",
            tooltip=[
                alt.Tooltip("Category:N"),
                alt.Tooltip("Type:N"),
                alt.Tooltip("Amount:Q", title="Amount", format=",.0f")
            ]
        )
        .properties(height=420)
    )
    st.altair_chart(chart_cat, use_container_width=True)

# =============================
# Details & Downloads
# =============================
st.markdown("### 📄 Details")
with st.expander("Show filtered data"):
    st.dataframe(
        df_f.sort_values(["Month", "Department", "Category"]),
        use_container_width=True,
        hide_index=True
    )

csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered data (CSV)",
    data=csv,
    file_name=f"budget_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)
