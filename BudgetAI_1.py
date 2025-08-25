# app.py
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

# ========== DESIGN (your preferred styling) ==========
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

# ========== OpenAI client ==========
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None
if not api_key:
    st.markdown("""
    <div style="background: linear-gradient(45deg, #f59e0b, #d97706); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        ⚠️ <strong>OpenAI API Key Required:</strong> Add OPENAI_API_KEY to .env or Streamlit secrets for GPT features.
    </div>
    """, unsafe_allow_html=True)

# ========== Helpers ==========
def money_fmt(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return "-"

def call_openai(system_msg: str, user_msg: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
    if not client:
        return "OpenAI API key not configured."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

@st.cache_data
def load_actuals(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    if "Budget_Allcated" in df.columns:
        df = df.rename(columns={"Budget_Allcated": "Budget_Allocated"})
    req = ["Month", "Department", "Category", "Budget_Allocated", "Actual_Spent", "Variance"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Actuals missing columns: {missing}")
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    if df["Month"].isna().any():
        df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    for c in ["Budget_Allocated", "Actual_Spent", "Variance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Month", "Budget_Allocated", "Actual_Spent"])
    return df.sort_values("Month")

@st.cache_data
def load_forecast(path: str):
    if not os.path.exists(path):
        return None
    f = pd.read_csv(path)
    f.columns = [c.strip().replace(" ", "_") for c in f.columns]
    if "Predicted_Spent" not in f.columns:
        for altcol in ["Forecast", "Forecasted", "yhat", "y_pred", "Prediction"]:
            if altcol in f.columns:
                f = f.rename(columns={altcol: "Predicted_Spent"})
                break
    req = ["Month", "Department", "Category", "Predicted_Spent"]
    missing = [c for c in req if c not in f.columns]
    if missing:
        raise ValueError(f"Forecast missing columns: {missing}")
    f["Month"] = pd.to_datetime(f["Month"], errors="coerce")
    f["Predicted_Spent"] = pd.to_numeric(f["Predicted_Spent"], errors="coerce")
    f = f.dropna(subset=["Month", "Predicted_Spent"])
    return f.sort_values("Month")

def tbl(df_):
    return "(none)" if df_.empty else df_.to_string(index=False)

def parse_month_year(text: str):
    # Accept "June 2025", "Jun 2025", "2025-06", "06/2025"
    month_map = {m.lower(): i for i, m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"], 1)}
    m1 = re.search(r'\b([A-Za-z]{3,9})\s+(\d{4})\b', text)
    if m1:
        mname = m1.group(1).lower()
        yr = int(m1.group(2))
        mon = month_map.get(mname)
        if mon is None:
            for full, idx in month_map.items():
                if full.startswith(mname): mon = idx; break
        if mon: return yr, mon
    m2 = re.search(r'\b(20\d{2})[-/](0?[1-9]|1[0-2])\b', text)
    if m2: return int(m2.group(1)), int(m2.group(2))
    m3 = re.search(r'\b(0?[1-9]|1[0-2])[-/](20\d{2})\b', text)
    if m3: return int(m3.group(2)), int(m3.group(1))
    return None, None

def extract_match(text, options):
    text_l = text.lower()
    for opt in options:
        if opt.lower() in text_l:
            return opt
    return None

def build_compact_summary(actuals_df: pd.DataFrame, forecast_df: pd.DataFrame | None):
    # ACTUALS (last 24 months + top10 dept/cat)
    a_month = actuals_df.groupby("Month", as_index=False)[["Actual_Spent","Budget_Allocated"]].sum()
    a_month["Month"] = a_month["Month"].dt.strftime("%Y-%m")
    a_month_tail = a_month.tail(24)
    a_dept = (
        actuals_df.groupby("Department", as_index=False)[["Actual_Spent","Budget_Allocated"]]
        .sum().sort_values("Actual_Spent", ascending=False).head(10)
    )
    a_cat = (
        actuals_df.groupby("Category", as_index=False)[["Actual_Spent","Budget_Allocated"]]
        .sum().sort_values("Actual_Spent", ascending=False).head(10)
    )

    # FORECAST (limit rows)
    if forecast_df is not None and not forecast_df.empty:
        f_month = forecast_df.groupby("Month", as_index=False)["Predicted_Spent"].sum()
        f_month["Month"] = f_month["Month"].dt.strftime("%Y-%m")
        f_month_head = f_month.head(24)
        f_dept = (
            forecast_df.groupby("Department", as_index=False)["Predicted_Spent"]
            .sum().sort_values("Predicted_Spent", ascending=False).head(10)
        )
        f_cat = (
            forecast_df.groupby("Category", as_index=False)["Predicted_Spent"]
            .sum().sort_values("Predicted_Spent", ascending=False).head(10)
        )
    else:
        f_month_head = pd.DataFrame(columns=["Month","Predicted_Spent"])
        f_dept = pd.DataFrame(columns=["Department","Predicted_Spent"])
        f_cat  = pd.DataFrame(columns=["Category","Predicted_Spent"])

    prompt = f"""
Use the following compact summaries to answer the user question succinctly.
If the question is about forecasts, rely on FORECAST; if about history/variance, rely on ACTUALS.
If insufficient info, say so.

ACTUALS — monthly (last 24):
{tbl(a_month_tail)}

ACTUALS — top departments:
{tbl(a_dept)}

ACTUALS — top categories:
{tbl(a_cat)}

FORECAST — monthly:
{tbl(f_month_head)}

FORECAST — top departments:
{tbl(f_dept)}

FORECAST — top categories:
{tbl(f_cat)}
"""
    return prompt

# ========== Load Data ==========
ACTUALS_PATH = "rich_dummy_budget_data.csv"
FORECAST_PATH = "forecasted_budget_2025.csv"

df = load_actuals(ACTUALS_PATH)
forecast = load_forecast(FORECAST_PATH)

# Derived fields
df["Variance_Percent"] = ((df["Actual_Spent"] - df["Budget_Allocated"]) / df["Budget_Allocated"] * 100).round(2)
df["Year"] = df["Month"].dt.year
df["Quarter"] = df["Month"].dt.quarter

# ========== Sidebar ==========
with st.sidebar:
    st.markdown('<div class="filter-header">🎛️ Control Panel</div>', unsafe_allow_html=True)

    # Dataset Overview (fix: nunique() is already int; don't wrap in len())
    st.markdown("**📊 Dataset Overview**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📋 Records", f"{len(df):,}", help="Total number of budget records")
        st.metric("🏢 Departments", f"{df['Department'].nunique():,}")
    with col2:
        st.metric("📁 Categories", f"{df['Category'].nunique():,}")
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

    # Date Filter
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

    year_options = sorted(df["Year"].unique())
    selected_years = st.multiselect(
        "📅 Filter by Year(s)",
        year_options,
        default=year_options,
        help="Focus on specific years"
    )

    variance_range = st.slider(
        "📊 Variance % Range",
        min_value=float(df["Variance_Percent"].min()),
        max_value=float(df["Variance_Percent"].max()),
        value=(float(df["Variance_Percent"].min()), float(df["Variance_Percent"].max())),
        step=1.0,
        help="Filter by budget variance percentage"
    )

    amount_range = st.slider(
        "💰 Spending Range ($)",
        min_value=float(df["Actual_Spent"].min()),
        max_value=float(df["Actual_Spent"].max()),
        value=(float(df["Actual_Spent"].min()), float(df["Actual_Spent"].max())),
        step=1000.0,
        format="$%.0f",
        help="Filter by actual spending amount"
    )

    budget_performance = st.selectbox(
        "🎯 Budget Performance",
        ["All", "Over Budget (>0%)", "Under Budget (<0%)", "On Target (±5%)", "Significant Variance (>±10%)"],
        help="Filter by budget adherence"
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

# Apply filters
mask = (
    df["Department"].isin(dept_sel)
    & df["Category"].isin(cat_sel)
    & (df["Month"] >= pd.to_datetime(date_range[0]))
    & (df["Month"] <= pd.to_datetime(date_range[1]))
    & df["Year"].isin(selected_years)
    & df["Variance_Percent"].between(variance_range[0], variance_range[1])
    & df["Actual_Spent"].between(amount_range[0], amount_range[1])
)

if budget_performance == "Over Budget (>0%)":
    mask &= df["Variance_Percent"] > 0
elif budget_performance == "Under Budget (<0%)":
    mask &= df["Variance_Percent"] < 0
elif budget_performance == "On Target (±5%)":
    mask &= df["Variance_Percent"].between(-5, 5)
elif budget_performance == "Significant Variance (>±10%)":
    mask &= (df["Variance_Percent"] > 10) | (df["Variance_Percent"] < -10)

df_f = df.loc[mask].copy()

if forecast is not None:
    f_mask = forecast["Department"].isin(dept_sel) & forecast["Category"].isin(cat_sel)
    f_f = forecast.loc[f_mask].copy()
else:
    f_f = None

# ========== Header & KPIs ==========
st.markdown("""
<div class="main-header">
    <h1>💼 AI Budget Forecast & Analysis</h1>
    <p style="font-size: 1.1rem; margin: 0.5rem 0;">Advanced Financial Intelligence Platform</p>
    <p style="font-size: 0.9rem; opacity: 0.9;">Actuals: 2021–present | Forecast: 2025 (from ML model)</p>
</div>
""", unsafe_allow_html=True)

if len(df_f) == 0:
    st.markdown("""
    <div style="background: linear-gradient(45deg, #f59e0b, #d97706); color: white; padding: 2rem; border-radius: 12px; text-align: center;">
        <h3>⚠️ No Data Found</h3>
        <p>No records match your current filter criteria. Please adjust the filters to see results.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()
else:
    col1, col2, col3, col4 = st.columns(4)
    total_budget = df_f["Budget_Allocated"].sum()
    total_actual = df_f["Actual_Spent"].sum()
    total_var = (df_f["Actual_Spent"] - df_f["Budget_Allocated"]).sum()
    var_pct = (total_var / total_budget * 100) if total_budget else 0.0

    with col1:
        st.metric("💰 Total Budget", money_fmt(total_budget), delta=f"{len(df_f)} records",
                  help="Total allocated budget for filtered data")

    with col2:
        delta_color = "normal" if abs(var_pct) < 5 else "inverse"
        st.metric("💳 Actual Spent", money_fmt(total_actual), delta=f"vs Budget: {var_pct:+.1f}%",
                  delta_color=delta_color, help="Total actual spending with variance indicator")

    with col3:
        variance_color = "inverse" if total_var > 0 else "normal"
        st.metric("📊 Net Variance", money_fmt(total_var), delta=f"{var_pct:+.2f}%",
                  delta_color=variance_color, help="Total variance: Actual - Budget")

    with col4:
        efficiency = 100 - abs(var_pct)
        efficiency_color = "normal" if efficiency > 90 else "inverse"
        st.metric("🎯 Budget Efficiency", f"{efficiency:.1f}%",
                  delta="Excellent" if efficiency > 95 else "Good" if efficiency > 85 else "Needs Review",
                  delta_color=efficiency_color, help="Budget adherence score (100% = perfect)")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    over_budget = (df_f["Variance_Percent"] > 0).sum()
    under_budget = (df_f["Variance_Percent"] < 0).sum()
    on_target = df_f["Variance_Percent"].between(-2, 2).sum()
    avg_variance = df_f["Variance_Percent"].mean()

    with col1:
        st.metric("🔴 Over Budget", f"{over_budget}", f"{over_budget/len(df_f)*100:.1f}% of items")
    with col2:
        st.metric("🟢 Under Budget", f"{under_budget}", f"{under_budget/len(df_f)*100:.1f}% of items")
    with col3:
        st.metric("🎯 On Target (±2%)", f"{on_target}", f"{on_target/len(df_f)*100:.1f}% of items")
    with col4:
        st.metric("📈 Avg Variance", f"{avg_variance:+.1f}%", "Across all filtered items")

# ========== Detailed Table ==========
st.markdown("---")
st.subheader("📊 Detailed Analysis")

col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"📋 **{len(df_f):,}** records displayed")
with col2:
    st.info(f"🏢 **{df_f['Department'].nunique()}** departments active")
with col3:
    st.info(f"📁 **{df_f['Category'].nunique()}** categories shown")

display_df = df_f.copy()
display_df["Month_Display"] = display_df["Month"].dt.strftime("%Y-%m")
display_df["Budget_Display"] = display_df["Budget_Allocated"].apply(lambda x: f"${x:,.0f}")
display_df["Actual_Display"] = display_df["Actual_Spent"].apply(lambda x: f"${x:,.0f}")
display_df["Variance_Display"] = display_df["Variance"].apply(lambda x: f"${x:+,.0f}")
display_df["Variance_Pct_Display"] = display_df["Variance_Percent"].apply(lambda x: f"{x:+.1f}%")

st.dataframe(
    display_df[["Month_Display", "Department", "Category", "Budget_Display", "Actual_Display", "Variance_Display", "Variance_Pct_Display"]].rename(columns={
        "Month_Display": "Month",
        "Budget_Display": "Budget",
        "Actual_Display": "Actual",
        "Variance_Display": "Variance ($)",
        "Variance_Pct_Display": "Variance (%)"
    }),
    use_container_width=True,
    height=400
)

# Forecast table (optional)
if show_forecast and f_f is not None and not f_f.empty:
    st.subheader("🔮 2025 Forecast Data")
    forecast_total = f_f["Predicted_Spent"].sum()
    forecast_monthly_avg = forecast_total / 12 if len(f_f) > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Total 2025 Forecast", money_fmt(forecast_total))
    with col2:
        st.metric("📅 Monthly Average", money_fmt(forecast_monthly_avg))

    f_display = f_f.copy()
    f_display["Month_Display"] = f_display["Month"].dt.strftime("%Y-%m")
    f_display["Predicted_Display"] = f_display["Predicted_Spent"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(
        f_display[["Month_Display", "Department", "Category", "Predicted_Display"]].rename(columns={
            "Month_Display": "Month",
            "Predicted_Display": "Predicted Spending"
        }),
        use_container_width=True,
        height=300
    )

# ========== Visual Analytics (CLUSTERED BARS) ==========
st.markdown("---")
st.subheader("📈 Visual Analytics")
tab1, tab2, tab3 = st.tabs(["📊 Monthly Trends", "🏢 By Department", "📁 By Category"])

# Colors to match your design
COLOR_ALLOC = "#3b82f6"  # blue
COLOR_SPENT = "#ef4444"  # red
COLOR_FOREC = "#10b981"  # green

with tab1:
    # Monthly clustered bars: Allocated vs Spent (+ optional Forecast)
    monthly = df_f.groupby("Month", as_index=False)[["Budget_Allocated", "Actual_Spent"]].sum().sort_values("Month")
    monthly_long = monthly.melt(id_vars="Month",
                                value_vars=["Budget_Allocated", "Actual_Spent"],
                                var_name="Type", value_name="Amount")
    type_map = {"Budget_Allocated": "Allocated", "Actual_Spent": "Spent"}
    monthly_long["Type"] = monthly_long["Type"].map(type_map)

    if show_forecast and (f_f is not None) and (len(f_f) > 0):
        fc = (f_f.groupby("Month", as_index=False)["Predicted_Spent"].sum()
                .rename(columns={"Predicted_Spent": "Amount"}))
        fc["Type"] = "Forecast (Spent)"
        monthly_long = pd.concat([monthly_long, fc], ignore_index=True)

    color_scale = alt.Scale(
        domain=["Allocated", "Spent", "Forecast (Spent)"],
        range=[COLOR_ALLOC, COLOR_SPENT, COLOR_FOREC]
    )

    chart_monthly = (
        alt.Chart(monthly_long)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(Month):O", title="Month", sort=None),
            y=alt.Y("Amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Type:N", title="", scale=color_scale),
            xOffset="Type:N",
            tooltip=[
                alt.Tooltip("yearmonth(Month):O", title="Month"),
                alt.Tooltip("Type:N"),
                alt.Tooltip("Amount:Q", title="Amount", format=",.0f")
            ]
        )
        .properties(height=380)
    )

    st.altair_chart(chart_monthly, use_container_width=True)

with tab2:
    # By Department clustered bars (sorted by variance desc)
    dept_tot = df_f.groupby("Department", as_index=False).agg(
        Allocated=("Budget_Allocated", "sum"),
        Spent=("Actual_Spent", "sum")
    )
    dept_tot["Variance"] = dept_tot["Spent"] - dept_tot["Allocated"]
    order = dept_tot.sort_values("Variance", ascending=False)["Department"].tolist()

    dept_long = dept_tot.melt(id_vars="Department", value_vars=["Allocated", "Spent"],
                              var_name="Type", value_name="Amount")

    chart_dept = (
        alt.Chart(dept_long)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Department:N", sort=order, axis=alt.Axis(labelAngle=-45), title="Department"),
            y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f"), title="Amount ($)"),
            color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[COLOR_ALLOC, COLOR_SPENT])),
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
    # By Category clustered bars (sorted by variance desc)
    cat_tot = df_f.groupby("Category", as_index=False).agg(
        Allocated=("Budget_Allocated", "sum"),
        Spent=("Actual_Spent", "sum")
    )
    cat_tot["Variance"] = cat_tot["Spent"] - cat_tot["Allocated"]
    order_cat = cat_tot.sort_values("Variance", ascending=False)["Category"].tolist()

    cat_long = cat_tot.melt(id_vars="Category", value_vars=["Allocated", "Spent"],
                            var_name="Type", value_name="Amount")

    chart_cat = (
        alt.Chart(cat_long)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Category:N", sort=order_cat, axis=alt.Axis(labelAngle=-45), title="Category"),
            y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f"), title="Amount ($)"),
            color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[COLOR_ALLOC, COLOR_SPENT])),
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

# ========== AI-Powered Insights (two options) ==========
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
    <h2>🤖 AI-Powered Insights</h2>
    <p>Get intelligent analysis and answers about your budget data</p>
</div>
""", unsafe_allow_html=True)

cols = st.columns([1, 2])

# Quick Analysis (left)
with cols[0]:
    st.markdown("### 📈 Quick Analysis")
    if st.button("🔍 Generate Smart Summary", use_container_width=True):
        if not client:
            st.error("❌ OpenAI API key not configured")
        else:
            compact = build_compact_summary(df_f, f_f if (show_forecast and f_f is not None) else None)
            user_prompt = compact + "\nPlease summarize key trends and give 2–3 actionable recommendations, scoped to the current filters."
            with st.spinner("🧠 AI analyzing your data..."):
                ans = call_openai(
                    system_msg="You are a budget analyst. Be concise, numeric, and practical. Use $ and % formatting.",
                    user_msg=user_prompt,
                    temperature=0.25,
                    max_tokens=600
                )
            st.markdown("""
            <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>✅ Analysis Complete!</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #10b981;">
                {ans}
            </div>
            """, unsafe_allow_html=True)

# Ask Questions (right)
with cols[1]:
    st.markdown("### 💬 Ask Questions")
    q = st.text_input(
        "🔍 Ask about your budget data:",
        placeholder="e.g., 'Actual Spent in 2023-06 for HR', 'Budget for Marketing 2024-11', 'Forecast for 2025-06 IT Software'",
        help="Ask specific questions about actuals, budgets, variance, or forecasts"
    )

    if q:
        # Intent + slicing
        wants_forecast = any(k in q.lower() for k in ["forecast", "predicted", "prediction"])
        wants_actual = ("actual" in q.lower()) or ("spent" in q.lower())
        wants_budget = ("budget" in q.lower()) and not wants_forecast
        wants_variance = "variance" in q.lower()

        year, month = parse_month_year(q)
        dept_in_q = extract_match(q, df["Department"].unique())
        cat_in_q = extract_match(q, df["Category"].unique())

        def df_slice_actuals():
            d = df_f.copy()  # respect current filters
            if year and month:
                ms = f"{year}-{month:02d}"
                d["Month_str"] = d["Month"].dt.strftime("%Y-%m")
                d = d[d["Month_str"] == ms]
            if dept_in_q: d = d[d["Department"] == dept_in_q]
            if cat_in_q:  d = d[d["Category"] == cat_in_q]
            return d

        def df_slice_forecast():
            if f_f is None: return pd.DataFrame()
            d = f_f.copy()
            if year and month:
                ms = f"{year}-{month:02d}"
                d["Month_str"] = d["Month"].dt.strftime("%Y-%m")
                d = d[d["Month_str"] == ms]
            if dept_in_q: d = d[d["Department"] == dept_in_q]
            if cat_in_q:  d = d[d["Category"] == cat_in_q]
            return d

        # Choose primary table
        primary = None
        if wants_forecast:
            primary = "forecast"
            df_q = df_slice_forecast()
        elif wants_actual or wants_budget or wants_variance:
            primary = "actuals"
            df_q = df_slice_actuals()
        else:
            df_q = pd.DataFrame()

        # Early notice if empty
        if primary and df_q.empty:
            target = "forecast" if primary == "forecast" else "actuals"
            detail = []
            if year and month: detail.append(f"month={year}-{month:02d}")
            if dept_in_q: detail.append(f"dept={dept_in_q}")
            if cat_in_q: detail.append(f"cat={cat_in_q}")
            detail_text = ", ".join(detail) if detail else "current filters"
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, #f59e0b, #d97706); color: white; padding: 1rem; border-radius: 8px;">
                <strong>⚠️ No Data Found:</strong> No {target} rows found for {detail_text}. Try adjusting filters or rephrasing your question.
            </div>
            """, unsafe_allow_html=True)
        else:
            if not client:
                st.error("❌ OpenAI API key not configured")
            else:
                MAX_ROWS = 40
                if not df_q.empty:
                    send_cols = (
                        ["Month","Department","Category","Predicted_Spent"] if primary=="forecast"
                        else ["Month","Department","Category","Budget_Allocated","Actual_Spent","Variance"]
                    )
                    slim = df_q[send_cols].copy()
                    slim["Month"] = pd.to_datetime(slim["Month"]).dt.strftime("%Y-%m")
                    slim = slim.head(MAX_ROWS)
                    context_table = slim.to_string(index=False)
                else:
                    context_table = "(no direct rows selected; using summaries)"

                compact = build_compact_summary(df_f, f_f if (show_forecast and f_f is not None) else None)

                final_prompt = f"""
User question: {q}

Primary context ({primary or 'auto'}):
{context_table}

Additional summaries:
{compact}

Instructions:
- If primary context exists, answer using it first.
- If primary is empty or insufficient, use the summaries.
- Show numbers and brief calculations if relevant.
- Keep answers concise and practical, scoped to the current filters.
"""
                with st.spinner("🤔 AI thinking..."):
                    ans = call_openai(
                        system_msg="You are a precise financial analyst. Be concise and numeric.",
                        user_msg=final_prompt,
                        temperature=0.2,
                        max_tokens=500
                    )

                st.markdown(f"""
                <div style="background: #f0f9ff; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #3b82f6;">
                    <strong>💡 AI Response:</strong><br><br>
                    {ans}
                </div>
                """, unsafe_allow_html=True)

# ========== Tips ==========
st.markdown("""
<div class="caption-style">
    <strong>💡 Pro Tips:</strong>
    • Use sidebar filters to focus your analysis
    • Charts update automatically based on your selections
    • Q&A sends only relevant data to AI for faster, more accurate responses
    • Try questions like "Which department overspent the most?" or "Show me Q4 2023 trends"
</div>
""", unsafe_allow_html=True)

# ========== Download ==========
csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered data (CSV)",
    data=csv,
    file_name=f"budget_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)
