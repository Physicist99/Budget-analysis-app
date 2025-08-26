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

# =============================
# Finance palettes (Executive Light/Dark)
# =============================
THEMES = {
    "Executive Light": {
        "bg": "#F7FAFC",
        "sidebar": "#0B1E3E",
        "panel": "#FFFFFF",
        "card": "#FFFFFF",
        "text": "#0F172A",
        "muted": "#475569",
        "grid": "#E2E8F0",
        "brand1": "#0B1E3E",
        "brand2": "#2F6BFF",
        "alloc": "#2151B8",
        "spent": "#53A2FF",
        "forecast": "#16A34A",
        "warn": "#D97706",
        "ok": "#10B981"
    },
    "Executive Dark": {  # ⬅️ navy background + white text
        "bg": "#0B1E3E",       # NAVY page background
        "sidebar": "#071A34",   # darker navy sidebar
        "panel": "#0E2044",     # panels/cards on navy
        "card": "#0F234B",
        "text": "#FFFFFF",      # WHITE text
        "muted": "#CBD5E1",     # soft slate for secondary text
        "grid": "#14315F",      # blue-ish gridlines
        "brand1": "#0B1E3E",
        "brand2": "#2F6BFF",
        "alloc": "#8FB7FF",     # lighter blues for contrast on navy
        "spent": "#5FB0FF",
        "forecast": "#22C55E",
        "warn": "#F59E0B",
        "ok": "#10B981"
    }
}

alt.themes.register("fin_theme", lambda: _alt_theme(pal))
alt.themes.enable("fin_theme")

# =============================
# Professional CSS (uses palette)
# =============================
st.markdown(f"""
<style>
  .stApp {{ background: {pal['bg']}; }}
  .block-container {{ padding-top: 1rem; }}
  /* Sidebar (robust selector) */
  [data-testid="stSidebar"] > div:first-child {{ background: {pal['sidebar']}; }}
  [data-testid="stSidebar"] * {{ color: #E5E7EB !important; }}
  .stSelectbox div, .stMultiSelect div, .stSlider div {{ color: #111827; }} /* content areas */

  /* Header */
  .main-header {{
    background: linear-gradient(135deg, {pal['brand1']} 0%, {pal['brand2']} 100%);
    padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
    text-align: center; color: white; box-shadow: 0 4px 24px rgba(0,0,0,0.10);
  }}

  /* Section headers */
  .filter-header {{
    background: {pal['panel']}; color: {pal['text']};
    padding: 0.75rem; border-radius: 10px; margin: 1rem 0 0.5rem 0;
    font-weight: 700; text-align: center; border: 1px solid {pal['grid']};
  }}

  /* Buttons */
  .stButton > button {{
    background: {pal['brand2']}; color: white; border: none;
    border-radius: 10px; padding: 0.6rem 1rem; font-weight: 700;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08); transition: all .2s ease;
  }}
  .stButton > button:hover {{ transform: translateY(-1px); filter: brightness(1.03); }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
  .stTabs [data-baseweb="tab"] {{
    background-color: {pal['panel']}; border-radius: 10px; color: {pal['muted']};
    font-weight: 700; border: 1px solid {pal['grid']};
  }}
  .stTabs [aria-selected="true"] {{
    background: {pal['brand2']}; color: #fff; border: 1px solid {pal['brand2']};
  }}

  /* Metrics */
  div[data-testid="stMetricValue"] {{ color: {pal['text']}; }}
  div[data-testid="stMetricDelta"] {{ color: {pal['muted']}; }}

  /* Tables & cards */
  .dataframe, .stDataFrame {{ border-radius: 10px; border: 1px solid {pal['grid']}; background: {pal['card']}; }}

  /* AI boxes */
  .ai-box {{
    border-radius: 12px; padding: 1rem 1.25rem;
    background: linear-gradient(135deg, {pal['brand1']} 0%, {pal['brand2']} 100%);
    color: white; margin-top: 1rem;
  }}
  .ai-result {{
    border-radius: 12px; padding: 1rem 1.25rem; background: {pal['card']}; color: {pal['text']};
    border: 1px solid {pal['grid']}; box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
  }}
  .caption-style {{
    background: linear-gradient(90deg, #f8fafc, #e2e8f0);
    padding: 0.75rem; border-radius: 8px; border-left: 3px solid {pal['brand2']};
    font-style: italic; color: {pal['muted']}; margin: 1rem 0;
  }}
</style>
""", unsafe_allow_html=True)

# =============================
# OpenAI client
# =============================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None
if not api_key:
    st.markdown(f"""
    <div style="background: linear-gradient(45deg, {pal['warn']}, #B45309); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        ⚠️ <strong>OpenAI API Key Required:</strong> Add OPENAI_API_KEY to .env or Streamlit secrets for GPT features.
    </div>
    """, unsafe_allow_html=True)

# =============================
# Helpers
# =============================
def money_fmt(x: float) -> str:
    try: return f"${x:,.0f}"
    except Exception: return "-"

def call_openai(system_msg: str, user_msg: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
    if not client: return "OpenAI API key not configured."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
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
    req = ["Month","Department","Category","Budget_Allocated","Actual_Spent","Variance"]
    missing = [c for c in req if c not in df.columns]
    if missing: raise ValueError(f"Actuals missing columns: {missing}")
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m", errors="coerce")
    if df["Month"].isna().any(): df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    for c in ["Budget_Allocated","Actual_Spent","Variance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Month","Budget_Allocated","Actual_Spent"])
    return df.sort_values("Month")

@st.cache_data
def load_forecast(path: str):
    if not os.path.exists(path): return None
    f = pd.read_csv(path)
    f.columns = [c.strip().replace(" ", "_") for c in f.columns]
    if "Predicted_Spent" not in f.columns:
        for altcol in ["Forecast","Forecasted","yhat","y_pred","Prediction"]:
            if altcol in f.columns: f = f.rename(columns={altcol:"Predicted_Spent"}); break
    req = ["Month","Department","Category","Predicted_Spent"]
    missing = [c for c in req if c not in f.columns]
    if missing: raise ValueError(f"Forecast missing columns: {missing}")
    f["Month"] = pd.to_datetime(f["Month"], errors="coerce")
    f["Predicted_Spent"] = pd.to_numeric(f["Predicted_Spent"], errors="coerce")
    f = f.dropna(subset=["Month","Predicted_Spent"])
    return f.sort_values("Month")

def tbl(df_): return "(none)" if df_.empty else df_.to_string(index=False)

def parse_month_year(text: str):
    month_map = {m.lower(): i for i, m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"], 1)}
    m1 = re.search(r'\b([A-Za-z]{3,9})\s+(\d{4})\b', text)
    if m1:
        mname, yr = m1.group(1).lower(), int(m1.group(2))
        mon = month_map.get(mname)
        if mon is None:
            for full, idx in month_map.items():
                if full.startswith(mname): mon = idx; break
        if mon: return yr, mon
    m2 = re.search(r'\b(20\d{{2}})[-/](0?[1-9]|1[0-2])\b', text)
    if m2: return int(m2.group(1)), int(m2.group(2))
    m3 = re.search(r'\b(0?[1-9]|1[0-2])[-/](20\d{{2}})\b', text)
    if m3: return int(m3.group(2)), int(m3.group(1))
    return None, None

def extract_match(text, options):
    text_l = text.lower()
    for opt in options:
        if opt.lower() in text_l: return opt
    return None

def build_compact_summary(actuals_df: pd.DataFrame, forecast_df: pd.DataFrame | None):
    a_month = actuals_df.groupby("Month", as_index=False)[["Actual_Spent","Budget_Allocated"]].sum()
    a_month["Month"] = a_month["Month"].dt.strftime("%Y-%m")
    a_dept = (actuals_df.groupby("Department", as_index=False)[["Actual_Spent","Budget_Allocated"]]
              .sum().sort_values("Actual_Spent", ascending=False).head(10))
    a_cat = (actuals_df.groupby("Category", as_index=False)[["Actual_Spent","Budget_Allocated"]]
             .sum().sort_values("Actual_Spent", ascending=False).head(10))

    if forecast_df is not None and not forecast_df.empty:
        f_month = forecast_df.groupby("Month", as_index=False)["Predicted_Spent"].sum()
        f_month["Month"] = f_month["Month"].dt.strftime("%Y-%m")
        f_dept = (forecast_df.groupby("Department", as_index=False)["Predicted_Spent"]
                  .sum().sort_values("Predicted_Spent", ascending=False).head(10))
        f_cat = (forecast_df.groupby("Category", as_index=False)["Predicted_Spent"]
                 .sum().sort_values("Predicted_Spent", ascending=False).head(10))
    else:
        f_month = pd.DataFrame(columns=["Month","Predicted_Spent"])
        f_dept = pd.DataFrame(columns=["Department","Predicted_Spent"])
        f_cat  = pd.DataFrame(columns=["Category","Predicted_Spent"])

    prompt = f"""
Use these compact summaries. If the question is about forecast, use FORECAST; if about history/variance, use ACTUALS.

ACTUALS — monthly:
{tbl(a_month.tail(24))}

ACTUALS — top departments:
{tbl(a_dept)}

ACTUALS — top categories:
{tbl(a_cat)}

FORECAST — monthly:
{tbl(f_month.head(24))}

FORECAST — top departments:
{tbl(f_dept)}

FORECAST — top categories:
{tbl(f_cat)}
"""
    return prompt

# =============================
# Load Data
# =============================
ACTUALS_PATH = "rich_dummy_budget_data.csv"
FORECAST_PATH = "forecasted_budget_2025.csv"
df = load_actuals(ACTUALS_PATH)
forecast = load_forecast(FORECAST_PATH)

# Derived fields
df["Variance_Percent"] = ((df["Actual_Spent"] - df["Budget_Allocated"]) / df["Budget_Allocated"] * 100).round(2)
df["Year"] = df["Month"].dt.year
df["Quarter"] = df["Month"].dt.quarter

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.markdown('<div class="filter-header">🎛️ Control Panel</div>', unsafe_allow_html=True)

    st.markdown("**📊 Dataset Overview**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📋 Records", f"{len(df):,}", help="Total number of budget records")
        st.metric("🏢 Departments", f"{df['Department'].nunique():,}")
    with col2:
        st.metric("📁 Categories", f"{df['Category'].nunique():,}")
        st.metric("📅 Time Span", f"{df['Year'].min()}–{df['Year'].max()}")

    st.markdown("---")
    st.markdown('<div class="filter-header">🔍 Primary Filters</div>', unsafe_allow_html=True)
    dept_sel = st.multiselect("🏢 Department(s)", sorted(df["Department"].unique()), default=sorted(df["Department"].unique()))
    cat_sel  = st.multiselect("📁 Category(ies)", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))

    min_month_py = pd.to_datetime(df["Month"].min()).to_pydatetime()
    max_month_py = pd.to_datetime(df["Month"].max()).to_pydatetime()
    date_range = st.slider("📅 Actuals Month Range", value=(min_month_py, max_month_py),
                           min_value=min_month_py, max_value=max_month_py, format="YYYY-MM")

    st.markdown('<div class="filter-header">⚙️ Advanced Filters</div>', unsafe_allow_html=True)
    year_options = sorted(df["Year"].unique())
    selected_years = st.multiselect("📅 Filter by Year(s)", year_options, default=year_options)

    variance_range = st.slider("📊 Variance % Range",
                               min_value=float(df["Variance_Percent"].min()),
                               max_value=float(df["Variance_Percent"].max()),
                               value=(float(df["Variance_Percent"].min()), float(df["Variance_Percent"].max())),
                               step=1.0)

    amount_range = st.slider("💰 Spending Range ($)",
                             min_value=float(df["Actual_Spent"].min()),
                             max_value=float(df["Actual_Spent"].max()),
                             value=(float(df["Actual_Spent"].min()), float(df["Actual_Spent"].max())),
                             step=1000.0, format="$%.0f")

    budget_performance = st.selectbox("🎯 Budget Performance",
                                      ["All","Over Budget (>0%)","Under Budget (<0%)","On Target (±5%)","Significant Variance (>±10%)"])

    st.markdown('<div class="filter-header">🔮 Forecast Settings</div>', unsafe_allow_html=True)
    show_forecast = st.checkbox("📈 Show 2025 Forecast", value=(forecast is not None), disabled=(forecast is None))
    if forecast is None:
        st.info("💡 No forecast file found. Place 'forecasted_budget_2025.csv' in the directory to enable forecasting.")

# Apply filters
mask = (
    df["Department"].isin(dept_sel) &
    df["Category"].isin(cat_sel) &
    (df["Month"] >= pd.to_datetime(date_range[0])) &
    (df["Month"] <= pd.to_datetime(date_range[1])) &
    df["Year"].isin(selected_years) &
    df["Variance_Percent"].between(variance_range[0], variance_range[1]) &
    df["Actual_Spent"].between(amount_range[0], amount_range[1])
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
f_f = forecast.loc[forecast["Department"].isin(dept_sel) & forecast["Category"].isin(cat_sel)].copy() if forecast is not None else None

# =============================
# Header & KPIs
# =============================
st.markdown(f"""
<div class="main-header">
  <h1>AI Budget Forecast & Analysis</h1>
  <p style="font-size:1.05rem;margin:.4rem 0;">Advanced Financial Intelligence Platform</p>
  <p style="font-size:.9rem;opacity:.9;">Actuals: historical to present · Forecast: 2025 (optional)</p>
</div>
""", unsafe_allow_html=True)

if df_f.empty:
    st.markdown(f"""
    <div style="background: linear-gradient(45deg, {pal['warn']}, #B45309); color: white; padding: 1.25rem; border-radius: 12px; text-align: center;">
        <strong>⚠️ No Data Found</strong><br/>Adjust filters to see results.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

col1, col2, col3, col4 = st.columns(4)
total_budget = df_f["Budget_Allocated"].sum()
total_actual = df_f["Actual_Spent"].sum()
total_var = (df_f["Actual_Spent"] - df_f["Budget_Allocated"]).sum()
var_pct = (total_var / total_budget * 100) if total_budget else 0.0

with col1:
    st.metric("Total Budget", money_fmt(total_budget), delta=f"{len(df_f):,} records")
with col2:
    st.metric("Actual Spent", money_fmt(total_actual), delta=f"vs Budget: {var_pct:+.1f}%",
              delta_color="normal" if abs(var_pct) < 5 else "inverse")
with col3:
    st.metric("Net Variance", money_fmt(total_var), delta=f"{var_pct:+.2f}%",
              delta_color="inverse" if total_var > 0 else "normal")
with col4:
    efficiency = max(0.0, 100 - abs(var_pct))
    tag = "Excellent" if efficiency > 95 else "Good" if efficiency > 85 else "Needs Review"
    st.metric("Budget Efficiency", f"{efficiency:.1f}%", delta=tag,
              delta_color="normal" if efficiency > 90 else "inverse")

# Quick stats
c1, c2, c3, c4 = st.columns(4)
over_b = (df_f["Variance_Percent"] > 0).sum()
under_b = (df_f["Variance_Percent"] < 0).sum()
on_tgt = df_f["Variance_Percent"].between(-2, 2).sum()
avg_var = df_f["Variance_Percent"].mean()
with c1: st.metric("Over Budget", f"{over_b}", f"{over_b/len(df_f)*100:.1f}%")
with c2: st.metric("Under Budget", f"{under_b}", f"{under_b/len(df_f)*100:.1f}%")
with c3: st.metric("On Target (±2%)", f"{on_tgt}", f"{on_tgt/len(df_f)*100:.1f}%")
with c4: st.metric("Avg Variance", f"{avg_var:+.1f}%", "Overall")

# =============================
# Detailed table
# =============================
st.markdown("---")
st.subheader("📊 Detailed Analysis")
colA, colB, colC = st.columns(3)
with colA: st.info(f"📋 **{len(df_f):,}** records displayed")
with colB: st.info(f"🏢 **{df_f['Department'].nunique()}** departments active")
with colC: st.info(f"📁 **{df_f['Category'].nunique()}** categories shown")

display_df = df_f.copy()
display_df["Month_Display"] = display_df["Month"].dt.strftime("%Y-%m")
display_df["Budget_Display"] = display_df["Budget_Allocated"].apply(lambda x: f"${x:,.0f}")
display_df["Actual_Display"] = display_df["Actual_Spent"].apply(lambda x: f"${x:,.0f}")
display_df["Variance_Display"] = display_df["Variance"].apply(lambda x: f"${x:+,.0f}")
display_df["Variance_Pct_Display"] = display_df["Variance_Percent"].apply(lambda x: f"{x:+.1f}%")
st.dataframe(
    display_df[["Month_Display","Department","Category","Budget_Display","Actual_Display","Variance_Display","Variance_Pct_Display"]]
      .rename(columns={"Month_Display":"Month","Budget_Display":"Budget","Actual_Display":"Actual",
                       "Variance_Display":"Variance ($)","Variance_Pct_Display":"Variance (%)"}),
    use_container_width=True, height=400
)

# Optional forecast table
if show_forecast and f_f is not None and not f_f.empty:
    st.subheader("🔮 2025 Forecast Data")
    ftot = f_f["Predicted_Spent"].sum()
    favg = ftot / 12 if len(f_f) > 0 else 0
    d1, d2 = st.columns(2)
    with d1: st.metric("Total 2025 Forecast", money_fmt(ftot))
    with d2: st.metric("Monthly Average", money_fmt(favg))
    f_display = f_f.copy()
    f_display["Month_Display"] = f_display["Month"].dt.strftime("%Y-%m")
    f_display["Predicted_Display"] = f_display["Predicted_Spent"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(
        f_display[["Month_Display","Department","Category","Predicted_Display"]]
          .rename(columns={"Month_Display":"Month","Predicted_Display":"Predicted Spending"}),
        use_container_width=True, height=300
    )

# =============================
# Visual Analytics (clustered bars)
# =============================
st.markdown("---")
st.subheader("📈 Visual Analytics")
tab1, tab2, tab3 = st.tabs(["📊 Monthly Trends", "🏢 By Department", "📁 By Category"])

with tab1:
    monthly = df_f.groupby("Month", as_index=False)[["Budget_Allocated","Actual_Spent"]].sum().sort_values("Month")
    m_long = monthly.melt(id_vars="Month", value_vars=["Budget_Allocated","Actual_Spent"], var_name="Type", value_name="Amount")
    m_long["Type"] = m_long["Type"].map({"Budget_Allocated":"Allocated","Actual_Spent":"Spent"})

    if show_forecast and (f_f is not None) and (len(f_f) > 0):
        fc = f_f.groupby("Month", as_index=False)["Predicted_Spent"].sum().rename(columns={"Predicted_Spent":"Amount"})
        fc["Type"] = "Forecast (Spent)"
        m_long = pd.concat([m_long, fc], ignore_index=True)

    color_scale = alt.Scale(domain=["Allocated","Spent","Forecast (Spent)"],
                            range=[pal["alloc"], pal["spent"], pal["forecast"]])

    chart_monthly = (
        alt.Chart(m_long)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(Month):O", title="Month", sort=None),
            y=alt.Y("Amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Type:N", title="", scale=color_scale),
            xOffset="Type:N",
            tooltip=[alt.Tooltip("yearmonth(Month):O", title="Month"),
                     alt.Tooltip("Type:N"),
                     alt.Tooltip("Amount:Q", title="Amount", format=",.0f")]
        ).properties(height=380)
    )
    st.altair_chart(chart_monthly, use_container_width=True)

with tab2:
    dept_tot = df_f.groupby("Department", as_index=False).agg(Allocated=("Budget_Allocated","sum"),
                                                              Spent=("Actual_Spent","sum"))
    dept_tot["Variance"] = dept_tot["Spent"] - dept_tot["Allocated"]
    order = dept_tot.sort_values("Variance", ascending=False)["Department"].tolist()
    dept_long = dept_tot.melt(id_vars="Department", value_vars=["Allocated","Spent"],
                              var_name="Type", value_name="Amount")
    chart_dept = (
        alt.Chart(dept_long)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Department:N", sort=order, axis=alt.Axis(labelAngle=-45), title="Department"),
            y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f"), title="Amount ($)"),
            color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"],
                                                                range=[pal["alloc"], pal["spent"]])),
            xOffset="Type:N",
            tooltip=[alt.Tooltip("Department:N"), alt.Tooltip("Type:N"),
                     alt.Tooltip("Amount:Q", title="Amount", format=",.0f")]
        ).properties(height=420)
    )
    st.altair_chart(chart_dept, use_container_width=True)

with tab3:
    cat_tot = df_f.groupby("Category", as_index=False).agg(Allocated=("Budget_Allocated","sum"),
                                                           Spent=("Actual_Spent","sum"))
    cat_tot["Variance"] = cat_tot["Spent"] - cat_tot["Allocated"]
    order_cat = cat_tot.sort_values("Variance", ascending=False)["Category"].tolist()
    cat_long = cat_tot.melt(id_vars="Category", value_vars=["Allocated","Spent"],
                            var_name="Type", value_name="Amount")
    chart_cat = (
        alt.Chart(cat_long)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Category:N", sort=order_cat, axis=alt.Axis(labelAngle=-45), title="Category"),
            y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f"), title="Amount ($)"),
            color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"],
                                                                range=[pal["alloc"], pal["spent"]])),
            xOffset="Type:N",
            tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Type:N"),
                     alt.Tooltip("Amount:Q", title="Amount", format=",.0f")]
        ).properties(height=420)
    )
    st.altair_chart(chart_cat, use_container_width=True)

# =============================
# AI-Powered Insights
# =============================
st.markdown("---")
st.markdown(f"""
<div class="ai-box">
  <h2>🤖 AI-Powered Insights</h2>
  <p>Get intelligent analysis and answers about your budget data</p>
</div>
""", unsafe_allow_html=True)

cols = st.columns([1, 2])

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
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {pal['ok']}, #059669); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
              <strong>✅ Analysis Complete!</strong>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="ai-result">{ans}</div>
            """, unsafe_allow_html=True)

with cols[1]:
    st.markdown("### 💬 Ask Questions")
    q = st.text_input(
        "🔍 Ask about your budget data:",
        placeholder="e.g., 'Actual Spent in 2023-06 for HR', 'Budget for Marketing 2024-11', 'Forecast for 2025-06 IT Software'"
    )
    if q:
        wants_forecast = any(k in q.lower() for k in ["forecast","predicted","prediction"])
        wants_actual = ("actual" in q.lower()) or ("spent" in q.lower())
        wants_budget = ("budget" in q.lower()) and not wants_forecast
        _ = "variance" in q.lower()

        def df_slice_actuals():
            d = df_f.copy()
            y, m = parse_month_year(q)
            if y and m:
                d = d[d["Month"].dt.strftime("%Y-%m") == f"{y}-{m:02d}"]
            dept_match = extract_match(q, df["Department"].unique())
            cat_match  = extract_match(q, df["Category"].unique())
            if dept_match: d = d[d["Department"] == dept_match]
            if cat_match:  d = d[d["Category"] == cat_match]
            return d

        def df_slice_forecast():
            if f_f is None: return pd.DataFrame()
            d = f_f.copy()
            y, m = parse_month_year(q)
            if y and m:
                d = d[d["Month"].dt.strftime("%Y-%m") == f"{y}-{m:02d}"]
            dept_match = extract_match(q, df["Department"].unique())
            cat_match  = extract_match(q, df["Category"].unique())
            if dept_match: d = d[d["Department"] == dept_match]
            if cat_match:  d = d[d["Category"] == cat_match]
            return d

        if wants_forecast:
            primary, df_q = "forecast", df_slice_forecast()
        elif wants_actual or wants_budget:
            primary, df_q = "actuals", df_slice_actuals()
        else:
            primary, df_q = "auto", pd.DataFrame()

        if primary != "auto" and df_q.empty:
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, {pal['warn']}, #B45309); color: white; padding: 1rem; border-radius: 8px;">
              <strong>⚠️ No Data Found:</strong> Try adjusting filters or rephrasing your question.
            </div>
            """, unsafe_allow_html=True)
        else:
            MAX_ROWS = 40
            if not df_q.empty:
                send_cols = ["Month","Department","Category","Predicted_Spent"] if primary=="forecast" else \
                            ["Month","Department","Category","Budget_Allocated","Actual_Spent","Variance"]
                slim = df_q[send_cols].copy()
                slim["Month"] = pd.to_datetime(slim["Month"]).dt.strftime("%Y-%m")
                slim = slim.head(MAX_ROWS)
                context_table = slim.to_string(index=False)
            else:
                context_table = "(no direct rows selected; using summaries)"

            compact = build_compact_summary(df_f, f_f if (show_forecast and f_f is not None) else None)
            final_prompt = f"""User question: {q}

Primary context ({primary}):
{context_table}

Additional summaries:
{compact}

Instructions:
- Prefer primary context if available; otherwise use summaries.
- Use $ and % with brief calculations as needed.
- Keep answers concise and practical, scoped to the current filters.
"""
            with st.spinner("🤔 AI thinking..."):
                ans = call_openai(
                    system_msg="You are a precise financial analyst. Be concise and numeric.",
                    user_msg=final_prompt,
                    temperature=0.2,
                    max_tokens=500
                )
            st.markdown(f"""<div class="ai-result">{ans}</div>""", unsafe_allow_html=True)

# Tips & Download
st.markdown("""
<div class="caption-style">
  <strong>Pro Tips:</strong> Charts & insights respect your filters. Try "Which department overspent most in 2024?" or "Forecast outlook for IT in 2025-06".
</div>
""", unsafe_allow_html=True)

csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered data (CSV)",
    data=csv,
    file_name=f"budget_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)
