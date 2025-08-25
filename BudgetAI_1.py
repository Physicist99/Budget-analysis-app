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
    st.markdown("---")
    st.subheader("📊 Detailed Analysis")
    
    # Filter summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📋 **{len(df_f):,}** records displayed")
    with col2:
        st.info(f"🏢 **{len(df_f['Department'].unique())}** departments active")
    with col3:
        st.info(f"📁 **{len(df_f['Category'].unique())}** categories shown")

    # Enhanced table with better formatting
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

    if show_forecast and f_f is not None and not f_f.empty:
        st.subheader("🔮 2025 Forecast Data")
        
        # Forecast summary
        forecast_total = f_f["Predicted_Spent"].sum()
        forecast_monthly_avg = forecast_total / 12 if len(f_f) > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📈 Total 2025 Forecast", f"${forecast_total:,.0f}")
        with col2:
            st.metric("📅 Monthly Average", f"${forecast_monthly_avg:,.0f}")
        
        # Format forecast table
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

    # === Executive Dashboard Charts ===
    st.markdown("---")
    st.subheader("📈 Visual Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Monthly Trends", "🏢 By Department", "📁 By Category"])

    with tab1:
        monthly_actuals = df_f.groupby("Month", as_index=False)[["Actual_Spent", "Budget_Allocated"]].sum()
        
        if monthly_actuals.empty:
            st.warning("No monthly data to display.")
        else:
            # Calculate variance for analysis
            monthly_actuals["Variance"] = monthly_actuals["Actual_Spent"] - monthly_actuals["Budget_Allocated"]
            monthly_actuals["Variance_Percent"] = (monthly_actuals["Variance"] / monthly_actuals["Budget_Allocated"] * 100).round(1)
            
            # 1. BIG PICTURE: Line Chart with Dual Series
            st.subheader("📈 Long-term Trend Analysis")
            line_chart = (
                alt.Chart(monthly_actuals)
                .transform_fold(["Actual_Spent", "Budget_Allocated"], as_=["Type", "Amount"])
                .mark_line(point=True, strokeWidth=4)
                .encode(
                    x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b %Y")),
                    y=alt.Y("Amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
                    color=alt.Color("Type:N", 
                                  scale=alt.Scale(domain=["Budget_Allocated", "Actual_Spent"], 
                                                range=["#3b82f6", "#ef4444"]),
                                  legend=alt.Legend(title="Trend Lines", orient="top")),
                    tooltip=[
                        alt.Tooltip("Month:T", format="%B %Y"),
                        alt.Tooltip("Type:N", title="Series"), 
                        alt.Tooltip("Amount:Q", format="$,.0f")
                    ]
                )
                .properties(height=350, title="Budget vs Actual - Strategic Overview")
            )
            st.altair_chart(line_chart, use_container_width=True)
            
            # 2. DETAILED VIEW: Clustered Column Chart for Month-to-Month Accountability  
            st.subheader("📊 Monthly Accountability - Clustered Comparison")
            monthly_melt = monthly_actuals.melt("Month", value_vars=["Budget_Allocated", "Actual_Spent"], 
                                              var_name="Type", value_name="Amount")
            
            clustered_chart = (
                alt.Chart(monthly_melt)
                .mark_bar(size=12)
                .encode(
                    x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b %Y")),
                    y=alt.Y("Amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
                    color=alt.Color("Type:N", 
                                  scale=alt.Scale(domain=["Budget_Allocated", "Actual_Spent"], 
                                                range=["#3b82f6", "#ef4444"]),
                                  legend=alt.Legend(title="Comparison", orient="top")),
                    xOffset="Type:N",
                    tooltip=[
                        alt.Tooltip("Month:T", format="%B %Y"),
                        "Type:N",
                        alt.Tooltip("Amount:Q", format="$,.0f")
                    ]
                )
                .properties(height=400, title="Side-by-Side Monthly Performance")
            )
            st.altair_chart(clustered_chart, use_container_width=True)
            
            # 3. COMBINATION CHART: Budget Line + Actual Bars
            st.subheader("🎯 Combination View - Budget Target vs Actual Performance")
            
            # Budget line
            budget_line = (
                alt.Chart(monthly_actuals)
                .mark_line(point=True, strokeWidth=3, color="#3b82f6")
                .encode(
                    x=alt.X("Month:T", title="Month"),
                    y=alt.Y("Budget_Allocated:Q", title="Amount ($)"),
                    tooltip=[
                        alt.Tooltip("Month:T", format="%B %Y"),
                        alt.Tooltip("Budget_Allocated:Q", format="$,.0f", title="Budget Target")
                    ]
                )
            )
            
            # Actual bars
            actual_bars = (
                alt.Chart(monthly_actuals)
                .mark_bar(opacity=0.7, color="#ef4444")
                .encode(
                    x=alt.X("Month:T"),
                    y=alt.Y("Actual_Spent:Q"),
                    tooltip=[
                        alt.Tooltip("Month:T", format="%B %Y"),
                        alt.Tooltip("Actual_Spent:Q", format="$,.0f", title="Actual Spent")
                    ]
                )
            )
            
            combination_chart = (budget_line + actual_bars).resolve_scale(y="shared").properties(
                height=400, title="Budget Targets (Line) vs Actual Performance (Bars)"
            )
            st.altair_chart(combination_chart, use_container_width=True)
            
            # 4. VARIANCE CHART: Quick Diagnostic View for Leadership
            st.subheader("⚡ Executive Summary - Variance Analysis")
            
            variance_chart = (
                alt.Chart(monthly_actuals)
                .mark_bar(size=25)
                .encode(
                    x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b %Y")),
                    y=alt.Y("Variance:Q", title="Variance ($)", scale=alt.Scale(zero=False)),
                    color=alt.condition(
                        alt.datum.Variance > 0,
                        alt.value("#ef4444"),  # Red for over budget
                        alt.value("#10b981")   # Green for under budget
                    ),
                    tooltip=[
                        alt.Tooltip("Month:T", format="%B %Y"),
                        alt.Tooltip("Variance:Q", format="$+,.0f", title="Budget Variance"),
                        alt.Tooltip("Variance_Percent:Q", format="+.1f", title="Variance %")
                    ]
                )
                .properties(height=300, title="Monthly Budget Variance - Quick Diagnostic (Red = Over Budget)")
            )
            st.altair_chart(variance_chart, use_container_width=True)
            
            # Executive Summary Cards
            col1, col2, col3, col4 = st.columns(4)
            avg_budget = monthly_actuals["Budget_Allocated"].mean()
            avg_actual = monthly_actuals["Actual_Spent"].mean()
            avg_variance = monthly_actuals["Variance"].mean()
            months_over = len(monthly_actuals[monthly_actuals["Variance"] > 0])
            
            with col1:
                st.metric("📊 Avg Monthly Budget", f"${avg_budget:,.0f}")
            with col2:
                st.metric("💳 Avg Monthly Actual", f"${avg_actual:,.0f}")
            with col3:
                st.metric("📈 Avg Variance", f"${avg_variance:+,.0f}")
            with col4:
                st.metric("🔴 Months Over Budget", f"{months_over}/{len(monthly_actuals)}")
            
            # Add forecast if available
            if show_forecast and f_f is not None and not f_f.empty:
                st.markdown("---")
                st.subheader("🔮 2025 Forecast Trend")
                monthly_fc = f_f.groupby("Month", as_index=False)["Predicted_Spent"].sum()
                
                fc_chart = (
                    alt.Chart(monthly_fc)
                    .mark_line(point=True, strokeWidth=3, strokeDash=[8, 4])
                    .encode(
                        x=alt.X("Month:T", title="2025 Forecast Period"),
                        y=alt.Y("Predicted_Spent:Q", title="Predicted Spending ($)"),
                        color=alt.value("#10b981"),
                        tooltip=[
                            alt.Tooltip("Month:T", format="%B %Y"),
                            alt.Tooltip("Predicted_Spent:Q", format="$,.0f", title="Forecast")
                        ]
                    )
                    .properties(height=300, title="2025 Monthly Forecast Projection")
                )
                st.altair_chart(fc_chart, use_container_width=True)

    with tab2:
        by_dept = df_f.groupby("Department", as_index=False)[["Actual_Spent", "Budget_Allocated"]].sum()
        
        if by_dept.empty:
            st.warning("No department data to display. Please adjust your filters.")
        else:
            if chart_type == "Side-by-Side Bars":
                st.subheader("Department Budget vs Actual - Side-by-Side Comparison")
                by_dept_melt = by_dept.melt("Department", var_name="Type", value_name="Amount")
                
                dept_chart = (
                    alt.Chart(by_dept_melt)
                    .mark_bar(size=25)
                    .encode(
                        x=alt.X("Department:N", axis=alt.Axis(labelAngle=-45), sort="-y"),
                        y=alt.Y("Amount:Q", title="Amount ($)"),
                        color=alt.Color("Type:N", 
                                      scale=alt.Scale(domain=["Budget_Allocated", "Actual_Spent"], 
                                                    range=["#3b82f6", "#ef4444"]),
                                      legend=alt.Legend(title="Type", orient="top")),
                        xOffset="Type:N",
                        tooltip=["Department", "Type", "Amount:Q"]
                    )
                    .properties(height=450, title="Department Performance Analysis")
                )
                st.altair_chart(dept_chart, use_container_width=True)
                
            elif chart_type == "Overlapping Bars":
                st.subheader("Department Budget vs Actual - Overlapping View")
                
                # Budget bars (background)
                budget_bars = (
                    alt.Chart(by_dept)
                    .mark_bar(size=40, opacity=0.6)
                    .encode(
                        x=alt.X("Department:N", axis=alt.Axis(labelAngle=-45), sort="-y"),
                        y=alt.Y("Budget_Allocated:Q", title="Amount ($)"),
                        color=alt.value("#3b82f6"),
                        tooltip=["Department", alt.Tooltip("Budget_Allocated:Q", format="$,.0f", title="Budget")]
                    )
                )
                
                # Actual bars (foreground)
                actual_bars = (
                    alt.Chart(by_dept)
                    .mark_bar(size=25, opacity=0.9)
                    .encode(
                        x=alt.X("Department:N", sort="-y"),
                        y=alt.Y("Actual_Spent:Q"),
                        color=alt.value("#ef4444"),
                        tooltip=["Department", alt.Tooltip("Actual_Spent:Q", format="$,.0f", title="Actual")]
                    )
                )
                
                overlapping_chart = (budget_bars + actual_bars).resolve_scale(y="shared").properties(
                    height=450, title="Overlapping Budget vs Actual Analysis"
                )
                st.altair_chart(overlapping_chart, use_container_width=True)
                
                st.info("💡 **Chart Guide:** Blue bars = Budget, Red bars = Actual. When red exceeds blue, department is over budget.")
                
            elif chart_type == "Variance Focus":
                st.subheader("Department Performance - Variance Analysis")
                
                by_dept["Variance"] = by_dept["Actual_Spent"] - by_dept["Budget_Allocated"]
                by_dept["Variance_Percent"] = (by_dept["Variance"] / by_dept["Budget_Allocated"] * 100).round(1)
                
                # Variance bar chart
                variance_chart = (
                    alt.Chart(by_dept)
                    .mark_bar(size=30)
                    .encode(
                        x=alt.X("Department:N", axis=alt.Axis(labelAngle=-45), sort="y"),
                        y=alt.Y("Variance:Q", title="Variance ($)", scale=alt.Scale(zero=False)),
                        color=alt.condition(
                            alt.datum.Variance > 0,
                            alt.value("#ef4444"),  # Red for over budget
                            alt.value("#10b981")   # Green for under budget
                        ),
                        tooltip=[
                            "Department", 
                            alt.Tooltip("Variance:Q", format="$+,.0f"),
                            alt.Tooltip("Variance_Percent:Q", format="+.1f", title="Variance %")
                        ]
                    )
                    .properties(height=400, title="Department Budget Variance Analysis")
                )
                st.altair_chart(variance_chart, use_container_width=True)
                
                # Performance tables
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**🟢 Best Performers (Under Budget)**")
                    best = by_dept.nsmallest(5, "Variance")[["Department", "Variance", "Variance_Percent"]]
                    best["Variance"] = best["Variance"].apply(lambda x: f"${x:,.0f}")
                    best["Variance_Percent"] = best["Variance_Percent"].apply(lambda x: f"{x:+.1f}%")
                    st.dataframe(best, use_container_width=True)
                with col2:
                    st.write("**🔴 Needs Attention (Over Budget)**")
                    worst = by_dept.nlargest(5, "Variance")[["Department", "Variance", "Variance_Percent"]]
                    worst["Variance"] = worst["Variance"].apply(lambda x: f"${x:,.0f}")
                    worst["Variance_Percent"] = worst["Variance_Percent"].apply(lambda x: f"{x:+.1f}%")
                    st.dataframe(worst, use_container_width=True)
                    
            else:  # Pie Charts
                st.subheader("Department Budget Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Budget Allocation**")
                    budget_pie = (
                        alt.Chart(by_dept)
                        .mark_arc(innerRadius=40, outerRadius=120)
                        .encode(
                            theta="Budget_Allocated:Q",
                            color=alt.Color("Department:N", scale=alt.Scale(scheme="category20")),
                            tooltip=["Department", alt.Tooltip("Budget_Allocated:Q", format="$,.0f")]
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(budget_pie, use_container_width=True)
                
                with col2:
                    st.write("**Actual Spending**")
                    actual_pie = (
                        alt.Chart(by_dept)
                        .mark_arc(innerRadius=40, outerRadius=120)
                        .encode(
                            theta="Actual_Spent:Q",
                            color=alt.Color("Department:N", scale=alt.Scale(scheme="category20")),
                            tooltip=["Department", alt.Tooltip("Actual_Spent:Q", format="$,.0f")]
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(actual_pie, use_container_width=True)

    with tab3:
        by_cat = df_f.groupby("Category", as_index=False)[["Actual_Spent", "Budget_Allocated"]].sum()
        
        if by_cat.empty:
            st.warning("❌ No category data to display. Please check your category filters!")
        else:
            if chart_type == "Side-by-Side Bars":
                st.subheader("Category Budget vs Actual - Side-by-Side Comparison")
                by_cat_melt = by_cat.melt("Category", var_name="Type", value_name="Amount")
                
                cat_chart = (
                    alt.Chart(by_cat_melt)
                    .mark_bar(size=25)
                    .encode(
                        x=alt.X("Category:N", axis=alt.Axis(labelAngle=-45), sort="-y"),
                        y=alt.Y("Amount:Q", title="Amount ($)"),
                        color=alt.Color("Type:N", 
                                      scale=alt.Scale(domain=["Budget_Allocated", "Actual_Spent"], 
                                                    range=["#3b82f6", "#ef4444"]),
                                      legend=alt.Legend(title="Type", orient="top")),
                        xOffset="Type:N",
                        tooltip=["Category", "Type", "Amount:Q"]
                    )
                    .properties(height=450, title="Category Performance Analysis")
                )
                st.altair_chart(cat_chart, use_container_width=True)
                
            elif chart_type == "Overlapping Bars":
                st.subheader("Category Budget vs Actual - Overlapping View")
                
                # Budget bars (background)
                budget_bars = (
                    alt.Chart(by_cat)
                    .mark_bar(size=40, opacity=0.6)
                    .encode(
                        x=alt.X("Category:N", axis=alt.Axis(labelAngle=-45), sort="-y"),
                        y=alt.Y("Budget_Allocated:Q", title="Amount ($)"),
                        color=alt.value("#3b82f6"),
                        tooltip=["Category", alt.Tooltip("Budget_Allocated:Q", format="$,.0f", title="Budget")]
                    )
                )
                
                # Actual bars (foreground)
                actual_bars = (
                    alt.Chart(by_cat)
                    .mark_bar(size=25, opacity=0.9)
                    .encode(
                        x=alt.X("Category:N", sort="-y"),
                        y=alt.Y("Actual_Spent:Q"),
                        color=alt.value("#ef4444"),
                        tooltip=["Category", alt.Tooltip("Actual_Spent:Q", format="$,.0f", title="Actual")]
                    )
                )
                
                overlapping_chart = (budget_bars + actual_bars).resolve_scale(y="shared").properties(
                    height=450, title="Overlapping Budget vs Actual Analysis"
                )
                st.altair_chart(overlapping_chart, use_container_width=True)
                
                st.info("💡 **Chart Guide:** Blue bars = Budget, Red bars = Actual. When red exceeds blue, category is over budget.")
                
            elif chart_type == "Variance Focus":
                st.subheader("Category Performance - Variance Analysis")
                
                by_cat["Variance"] = by_cat["Actual_Spent"] - by_cat["Budget_Allocated"]
                by_cat["Variance_Percent"] = (by_cat["Variance"] / by_cat["Budget_Allocated"] * 100).round(1)
                
                # Variance bar chart
                variance_chart = (
                    alt.Chart(by_cat)
                    .mark_bar(size=30)
                    .encode(
                        x=alt.X("Category:N", axis=alt.Axis(labelAngle=-45), sort="y"),
                        y=alt.Y("Variance:Q", title="Variance ($)", scale=alt.Scale(zero=False)),
                        color=alt.condition(
                            alt.datum.Variance > 0,
                            alt.value("#ef4444"),  # Red for over budget
                            alt.value("#10b981")   # Green for under budget
                        ),
                        tooltip=[
                            "Category", 
                            alt.Tooltip("Variance:Q", format="$+,.0f"),
                            alt.Tooltip("Variance_Percent:Q", format="+.1f", title="Variance %")
                        ]
                    )
                    .properties(height=400, title="Category Budget Variance Analysis")
                )
                st.altair_chart(variance_chart, use_container_width=True)
                
                # Performance tables
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**🟢 Best Categories (Under Budget)**")
                    best_cat = by_cat.nsmallest(5, "Variance")[["Category", "Variance", "Variance_Percent"]]
                    best_cat["Variance"] = best_cat["Variance"].apply(lambda x: f"${x:,.0f}")
                    best_cat["Variance_Percent"] = best_cat["Variance_Percent"].apply(lambda x: f"{x:+.1f}%")
                    st.dataframe(best_cat, use_container_width=True)
                    
                with col2:
                    st.write("**🔴 Categories Over Budget**")
                    worst_cat = by_cat.nlargest(5, "Variance")[["Category", "Variance", "Variance_Percent"]]
                    worst_cat["Variance"] = worst_cat["Variance"].apply(lambda x: f"${x:,.0f}")
                    worst_cat["Variance_Percent"] = worst_cat["Variance_Percent"].apply(lambda x: f"{x:+.1f}%")
                    st.dataframe(worst_cat, use_container_width=True)
                    
            else:  # Pie Charts
                st.subheader("Category Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Budget Allocation by Category**")
                    budget_pie = (
                        alt.Chart(by_cat)
                        .mark_arc(innerRadius=40, outerRadius=120)
                        .encode(
                            theta="Budget_Allocated:Q",
                            color=alt.Color("Category:N", scale=alt.Scale(scheme="set3")),
                            tooltip=["Category", alt.Tooltip("Budget_Allocated:Q", format="$,.0f")]
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(budget_pie, use_container_width=True)
                
                with col2:
                    st.write("**Actual Spending by Category**")
                    actual_pie = (
                        alt.Chart(by_cat)
                        .mark_arc(innerRadius=40, outerRadius=120)
                        .encode(
                            theta="Actual_Spent:Q",
                            color=alt.Color("Category:N", scale=alt.Scale(scheme="set3")),
                            tooltip=["Category", alt.Tooltip("Actual_Spent:Q", format="$,.0f")]
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(actual_pie, use_container_width=True)
            
            # Summary statistics
            if chart_type != "Pie Charts":
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_budget_cat = by_cat["Budget_Allocated"].sum()
                    st.metric("💰 Total Budget", f"${total_budget_cat:,.0f}")
                with col2:
                    total_actual_cat = by_cat["Actual_Spent"].sum()
                    st.metric("💳 Total Actual", f"${total_actual_cat:,.0f}")
                with col3:
                    variance_cat = total_actual_cat - total_budget_cat
                    variance_pct_cat = (variance_cat / total_budget_cat * 100) if total_budget_cat else 0
                    st.metric("📊 Net Variance", f"${variance_cat:+,.0f}", f"{variance_pct_cat:+.1f}%")

# === Helper Functions ===
def tbl(df_):
    return "(none)" if df_.empty else df_.to_string(index=False)

def parse_month_year(text: str):
    month_map = {m.lower(): i for i, m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"], 1)}
    m1 = re.search(r'\b([A-Za-z]{3,9})\s+(\d{4})\b', text)
    if m1:
        mname = m1.group(1).lower()
        yr = int(m1.group(2))
        mon = month_map.get(mname)
        if mon is None:
            for full, idx in month_map.items():
                if full.startswith(mname): 
                    mon = idx
                    break
        if mon: 
            return yr, mon
    m2 = re.search(r'\b(20\d{2})[-/](0?[1-9]|1[0-2])\b', text)
    if m2: 
        return int(m2.group(1)), int(m2.group(2))
    m3 = re.search(r'\b(0?[1-9]|1[0-2])[-/](20\d{2})\b', text)
    if m3: 
        return int(m3.group(2)), int(m3.group(1))
    return None, None

def extract_match(text, options):
    text_l = text.lower()
    for opt in options:
        if opt.lower() in text_l:
            return opt
    return None

def build_compact_summary(actuals_df: pd.DataFrame, forecast_df: pd.DataFrame):
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

# === Enhanced GPT Insights ===
if len(df_f) > 0:  # Only show AI section if there's data
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center;">
        <h2>🤖 AI-Powered Insights</h2>
        <p>Get intelligent analysis and answers about your budget data</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns([1, 2])

    # ---- Summary button (compact forecast+actuals) ----
    with cols[0]:
        st.markdown("### 📈 Quick Analysis")
        if st.button("🔍 Generate Smart Summary", use_container_width=True):
            if not client:
                st.error("❌ OpenAI API key not configured")
            else:
                compact = build_compact_summary(df_f, f_f if (show_forecast and f_f is not None) else None)
                user_prompt = compact + "\nPlease summarize key trends and give 2–3 actionable recommendations."
                with st.spinner("🧠 AI analyzing your data..."):
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role":"system","content":"You are a budget analyst. Be concise, numeric, and practical."},
                                {"role":"user","content":user_prompt},
                            ],
                            temperature=0.3,
                            max_tokens=500
                        )
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                            <strong>✅ Analysis Complete!</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #10b981;">
                            {resp.choices[0].message.content}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error calling OpenAI API: {e}")

    # ---- Q&A (smart filtering before GPT) ----
    with cols[1]:
        st.markdown("### 💬 Ask Questions")
        q = st.text_input(
            "🔍 Ask about your budget data:",
            placeholder="e.g., 'Actual Spent in June 2022 for HR', 'Budget for Marketing 2023-11', 'Forecast for June 2025 IT Software'",
            help="Ask specific questions about actuals, budgets, or forecasts"
        )
        
        if q:
            wants_forecast = any(k in q.lower() for k in ["forecast", "predicted", "prediction"])
            wants_actual = ("actual" in q.lower()) or ("spent" in q.lower())
            wants_budget = ("budget" in q.lower()) and not wants_forecast
            wants_variance = "variance" in q.lower()

            year, month = parse_month_year(q)
            dept_in_q = extract_match(q, df["Department"].unique())
            cat_in_q = extract_match(q, df["Category"].unique())

            # Build a small, relevant slice for the question
            def df_slice_actuals():
                d = df.copy()
                if year and month:
                    ms = f"{year}-{month:02d}"
                    d["Month_str"] = d["Month"].dt.strftime("%Y-%m")
                    d = d[d["Month_str"] == ms]
                d = d[d["Department"].isin(dept_sel)]
                d = d[d["Category"].isin(cat_sel)]
                if dept_in_q: 
                    d = d[d["Department"] == dept_in_q]
                if cat_in_q: 
                    d = d[d["Category"] == cat_in_q]
                return d

            def df_slice_forecast():
                if f_f is None: 
                    return pd.DataFrame()
                d = f_f.copy()
                if year and month:
                    ms = f"{year}-{month:02d}"
                    d["Month_str"] = d["Month"].dt.strftime("%Y-%m")
                    d = d[d["Month_str"] == ms]
                if dept_in_q: 
                    d = d[d["Department"] == dept_in_q]
                if cat_in_q: 
                    d = d[d["Category"] == cat_in_q]
                return d

            # Decide which table to use primarily
            primary = None
            if wants_forecast:
                primary = "forecast"
                df_q = df_slice_forecast()
            elif wants_actual or wants_budget or wants_variance:
                primary = "actuals"
                df_q = df_slice_actuals()
            else:
                # undefined intent → use summaries only
                df_q = pd.DataFrame()

            # If nothing found, tell user early
            if primary and df_q.empty:
                target = "forecast" if primary == "forecast" else "actuals"
                detail = []
                if year and month: 
                    detail.append(f"month={year}-{month:02d}")
                if dept_in_q: 
                    detail.append(f"dept={dept_in_q}")
                if cat_in_q: 
                    detail.append(f"cat={cat_in_q}")
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
                    # Build tiny prompt: include only up to N rows to stay well below token limits
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
                        context_table = "(no direct rows selected; use summaries below)"

                    # Also include compact summaries for backup context
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
- Keep answers concise and practical.
"""

                    with st.spinner("🤔 AI thinking..."):
                        try:
                            resp = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role":"system","content":"You are a precise financial analyst. Be concise and numeric."},
                                    {"role":"user","content":final_prompt}
                                ],
                                temperature=0.2,
                                max_tokens=450
                            )
                            
                            st.markdown(f"""
                            <div style="background: #f0f9ff; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #3b82f6;">
                                <strong>💡 AI Response:</strong><br><br>
                                {resp.choices[0].message.content}
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error calling OpenAI API: {e}")

# Enhanced tip section
st.markdown("""
<div class="caption-style">
    <strong>💡 Executive Dashboard Tips:</strong> 
    • **Monthly Trends:** Strategic overview (line), accountability (bars), combination view, and variance analysis
    • **Best chart types:** Side-by-Side Bars (easy comparison), Overlapping Bars (space efficient), Variance Focus (shows performance gaps)
    • Charts update automatically based on sidebar filters
    • Use AI-powered Q&A for deeper insights: "Which department overspent the most?" or "Show me Q4 2023 trends"
</div>
""", unsafe_allow_html=True)
