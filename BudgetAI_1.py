import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import altair as alt
from datetime import datetime
import re

st.set_page_config(
    page_title="AI Budget Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    /* Dark theme and professional styling */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e293b;
    }
    
    /* Filter section headers */
    .filter-header {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 1rem 0 0.5rem 0;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px;
        color: #475569;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        color: white;
    }
    
    /* Warning and info boxes */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border-radius: 8px;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    /* Caption styling */
    .caption-style {
        background: linear-gradient(90deg, #f8fafc, #e2e8f0);
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 3px solid #6366f1;
        font-style: italic;
        color: #475569;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === OpenAI client ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None
if not api_key:
    st.markdown("""
    <div style="background: linear-gradient(45deg, #f59e0b, #d97706); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        ⚠️ <strong>OpenAI API Key Required:</strong> Add OPENAI_API_KEY to .env or Streamlit secrets for GPT features.
    </div>
    """, unsafe_allow_html=True)

# === Data Loading Functions ===
@st.cache_data
def load_actuals(path: str):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        # fix common typo
        if "Budget_Allcated" in df.columns:
            df = df.rename(columns={"Budget_Allcated": "Budget_Allocated"})
        req = ["Month", "Department", "Category", "Budget_Allocated", "Actual_Spent", "Variance"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            st.error(f"Missing required columns in actuals data: {missing}")
            return pd.DataFrame()
        df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
        if df["Month"].isna().any():
            df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
        for c in ["Budget_Allocated", "Actual_Spent", "Variance"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Month", "Budget_Allocated", "Actual_Spent"])
        return df.sort_values("Month")
    except Exception as e:
        st.error(f"Error loading actuals data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_forecast(path: str):
    try:
        if not os.path.exists(path):
            return None
        f = pd.read_csv(path)
        f.columns = [c.strip().replace(" ", "_") for c in f.columns]
        if "Predicted_Spent" not in f.columns:
            for altcol in ["Forecast", "Forecasted", "yhat", "y_pred"]:
                if altcol in f.columns:
                    f = f.rename(columns={altcol: "Predicted_Spent"})
                    break
        req = ["Month", "Department", "Category", "Predicted_Spent"]
        missing = [c for c in req if c not in f.columns]
        if missing:
            st.error(f"Missing required columns in forecast data: {missing}")
            return None
        f["Month"] = pd.to_datetime(f["Month"], errors="coerce")
        f["Predicted_Spent"] = pd.to_numeric(f["Predicted_Spent"], errors="coerce")
        f = f.dropna(subset=["Month", "Predicted_Spent"])
        return f.sort_values("Month")
    except Exception as e:
        st.error(f"Error loading forecast data: {e}")
        return None

# === Load Data ===
ACTUALS_PATH = "rich_dummy_budget_data.csv"
FORECAST_PATH = "forecasted_budget_2025.csv"

df = load_actuals(ACTUALS_PATH)
forecast = load_forecast(FORECAST_PATH)

# Check if data loaded successfully
if df.empty:
    st.error("❌ Could not load actuals data. Please check that 'rich_dummy_budget_data.csv' exists and has the correct format.")
    st.stop()

# Add calculated fields for enhanced filtering
df["Variance_Percent"] = ((df["Actual_Spent"] - df["Budget_Allocated"]) / df["Budget_Allocated"] * 100).round(2)
df["Year"] = df["Month"].dt.year
df["Quarter"] = df["Month"].dt.quarter

# === Enhanced Sidebar Filters ===
with st.sidebar:
    st.markdown('<div class="filter-header">🎛️ Control Panel</div>', unsafe_allow_html=True)
    
    # Dataset Overview
    st.markdown("**📊 Dataset Overview**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📋 Records", f"{len(df):,}", help="Total number of budget records")
        st.metric("🏢 Departments", len(df["Department"].unique()))
    with col2:
        st.metric("📁 Categories", len(df["Category"].unique()))
        st.metric("📅 Time Span", f"{df['Year'].min()}-{df['Year'].max()}")
    
    st.markdown("---")
    
    # Primary Filters
    st.markdown('<div class="filter-header">🔍 Primary Filters</div>', unsafe_allow_html=True)
    
    dept_sel = st.multiselect(
        "🏢 Department(s)", 
        sorted(df["Department"].unique()), 
        default=sorted(df["Department"].unique()),
        help="Select one or more departments to analyze"
    )
    
    cat_sel = st.multiselect(
        "📁 Category(ies)", 
        sorted(df["Category"].unique()), 
        default=sorted(df["Category"].unique()),
        help="Choose expense categories to include"
    )
    
    # Enhanced Date Filter
    min_month_py = pd.to_datetime(df["Month"].min()).to_pydatetime()
    max_month_py = pd.to_datetime(df["Month"].max()).to_pydatetime()
    date_range = st.slider(
        "📅 Actuals Month Range",
        value=(min_month_py, max_month_py),
        min_value=min_month_py,
        max_value=max_month_py,
        format="YYYY-MM",
        help="Select the time period for analysis"
    )
    
    # Advanced Filters
    st.markdown('<div class="filter-header">⚙️ Advanced Filters</div>', unsafe_allow_html=True)
    
    # Year Filter
    year_options = sorted(df["Year"].unique())
    selected_years = st.multiselect(
        "📅 Filter by Year(s)",
        year_options,
        default=year_options,
        help="Focus on specific years"
    )
    
    # Variance Filter
    variance_range = st.slider(
        "📊 Variance % Range",
        min_value=float(df["Variance_Percent"].min()),
        max_value=float(df["Variance_Percent"].max()),
        value=(float(df["Variance_Percent"].min()), float(df["Variance_Percent"].max())),
        step=1.0,
        help="Filter by budget variance percentage"
    )
    
    # Amount Filter
    amount_range = st.slider(
        "💰 Spending Range ($)",
        min_value=float(df["Actual_Spent"].min()),
        max_value=float(df["Actual_Spent"].max()),
        value=(float(df["Actual_Spent"].min()), float(df["Actual_Spent"].max())),
        step=1000.0,
        format="$%.0f",
        help="Filter by actual spending amount"
    )
    
    # Budget Performance Filter
    budget_performance = st.selectbox(
        "🎯 Budget Performance",
        ["All", "Over Budget (>0%)", "Under Budget (<0%)", "On Target (±5%)", "Significant Variance (>±10%)"],
        help="Filter by budget adherence"
    )
    
    # Chart Type Selection
    st.markdown('<div class="filter-header">📊 Chart Options</div>', unsafe_allow_html=True)
    chart_type = st.selectbox(
        "📊 Chart Style",
        ["Side-by-Side Bars", "Overlapping Bars", "Variance Focus", "Pie Charts"],
        help="Choose visualization style for budget vs actual comparison"
    )
    
    # Forecast Options
    st.markdown('<div class="filter-header">🔮 Forecast Settings</div>', unsafe_allow_html=True)
    show_forecast = st.checkbox(
        "📈 Show 2025 Forecast", 
        value=(forecast is not None),
        help="Display ML-generated predictions",
        disabled=(forecast is None)
    )
    
    if forecast is None:
        st.info("💡 No forecast file found. Place 'forecasted_budget_2025.csv' in the directory to enable forecasting.")

# === Apply Enhanced Filters ===
mask = (
    df["Department"].isin(dept_sel)
    & df["Category"].isin(cat_sel)
    & (df["Month"] >= pd.to_datetime(date_range[0]))
    & (df["Month"] <= pd.to_datetime(date_range[1]))
    & df["Year"].isin(selected_years)
    & (df["Variance_Percent"] >= variance_range[0])
    & (df["Variance_Percent"] <= variance_range[1])
    & (df["Actual_Spent"] >= amount_range[0])
    & (df["Actual_Spent"] <= amount_range[1])
)

# Apply budget performance filter
if budget_performance == "Over Budget (>0%)":
    mask = mask & (df["Variance_Percent"] > 0)
elif budget_performance == "Under Budget (<0%)":
    mask = mask & (df["Variance_Percent"] < 0)
elif budget_performance == "On Target (±5%)":
    mask = mask & (df["Variance_Percent"].between(-5, 5))
elif budget_performance == "Significant Variance (>±10%)":
    mask = mask & ((df["Variance_Percent"] > 10) | (df["Variance_Percent"] < -10))

df_f = df.loc[mask].copy()

if forecast is not None:
    f_mask = forecast["Department"].isin(dept_sel) & forecast["Category"].isin(cat_sel)
    f_f = forecast.loc[f_mask].copy()
else:
    f_f = None

# === Enhanced Header & KPIs ===
st.markdown("""
<div class="main-header">
    <h1>💼 AI Budget Forecast & Analysis</h1>
    <p style="font-size: 1.1rem; margin: 0.5rem 0;">Advanced Financial Intelligence Platform</p>
    <p style="font-size: 0.9rem; opacity: 0.9;">Actuals: 2021–present | Forecast: 2025 (from ML model)</p>
</div>
""", unsafe_allow_html=True)

# Enhanced KPI Display
if len(df_f) == 0:
    st.markdown("""
    <div style="background: linear-gradient(45deg, #f59e0b, #d97706); color: white; padding: 2rem; border-radius: 12px; text-align: center;">
        <h3>⚠️ No Data Found</h3>
        <p>No records match your current filter criteria. Please adjust the filters to see results.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    col1, col2, col3, col4 = st.columns(4)
    total_budget = df_f["Budget_Allocated"].sum()
    total_actual = df_f["Actual_Spent"].sum()
    total_var = df_f["Variance"].sum()
    var_pct = (total_var / total_budget * 100) if total_budget else 0.0
    
    with col1:
        delta_color = "normal" if abs(var_pct) < 5 else "inverse"
        st.metric(
            "💰 Total Budget",
            f"${total_budget:,.0f}",
            delta=f"{len(df_f)} records",
            help="Total allocated budget for filtered data"
        )
    
    with col2:
        st.metric(
            "💳 Actual Spent",
            f"${total_actual:,.0f}",
            delta=f"vs Budget: {var_pct:+.1f}%",
            delta_color=delta_color,
            help="Total actual spending with variance indicator"
        )
    
    with col3:
        variance_color = "inverse" if total_var > 0 else "normal"
        st.metric(
            "📊 Net Variance",
            f"${total_var:+,.0f}",
            delta=f"{var_pct:+.2f}%",
            delta_color=variance_color,
            help="Total variance: Actual - Budget"
        )
    
    with col4:
        # Budget efficiency score
        efficiency = 100 - abs(var_pct)
        efficiency_color = "normal" if efficiency > 90 else "inverse"
        st.metric(
            "🎯 Budget Efficiency",
            f"{efficiency:.1f}%",
            delta="Excellent" if efficiency > 95 else "Good" if efficiency > 85 else "Needs Review",
            delta_color=efficiency_color,
            help="Budget adherence score (100% = perfect)"
        )

    # Quick Stats Row
    col1, col2, col3, col4 = st.columns(4)
    
    over_budget = len(df_f[df_f["Variance_Percent"] > 0])
    under_budget = len(df_f[df_f["Variance_Percent"] < 0])
    on_target = len(df_f[df_f["Variance_Percent"].between(-2, 2)])
    avg_variance = df_f["Variance_Percent"].mean()
    
    with col1:
        st.metric("🔴 Over Budget", f"{over_budget}", f"{over_budget/len(df_f)*100:.1f}% of items")
    with col2:
        st.metric("🟢 Under Budget", f"{under_budget}", f"{under_budget/len(df_f)*100:.1f}% of items")
    with col3:
        st.metric("🎯 On Target (±2%)", f"{on_target}", f"{on_target/len(df_f)*100:.1f}% of items")
    with col4:
        st.metric("📈 Avg Variance", f"{avg_variance:+.1f}%", "Across all filtered items")

    # === Enhanced Tables ===
    st.markdown("
