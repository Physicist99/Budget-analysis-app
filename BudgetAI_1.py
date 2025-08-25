# app.py
import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import altair as alt
from datetime import datetime

# =============================
# Page
# =============================
st.set_page_config(
    page_title="AI Budget Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# Themes (Light/Dark/Brand) + Altair theme
# =============================
THEMES = {
    "Light (Slate/Indigo)": {
        "bg": "#f8fafc", "panel": "#ffffff", "card": "#ffffff",
        "text": "#0f172a", "muted": "#475569", "grid": "#e2e8f0",
        "accent": "#1d4ed8", "accent2": "#6366f1", "accent3": "#10b981"
    },
    "Dark (Charcoal/Teal)": {
        "bg": "#0b1220", "panel": "#0f172a", "card": "#111827",
        "text": "#e5e7eb", "muted": "#94a3b8", "grid": "#1f2937",
        "accent": "#06b6d4", "accent2": "#0ea5e9", "accent3": "#22c55e"
    },
    "Brand (Blue/Gold)": {
        "bg": "#f8fafc", "panel": "#ffffff", "card": "#ffffff",
        "text": "#0f172a", "muted": "#475569", "grid": "#e2e8f0",
        "accent": "#003a70", "accent2": "#f59e0b", "accent3": "#16a34a"
    }
}

with st.sidebar:
    st.markdown("**🎨 Appearance**")
    theme_name = st.selectbox("Theme", list(THEMES.keys()), index=0)
pal = THEMES[theme_name]

def _alt_theme(p):
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {"labelColor": p["text"], "titleColor": p["text"], "gridColor": p["grid"]},
            "legend": {"labelColor": p["text"], "titleColor": p["text"]},
            "title": {"color": p["text"]},
            "range": {
                # Allocated / Spent / Forecast
                "category": [p["accent"], p["accent2"], p["accent3"], "#9CA3AF", "#EF4444", "#10B981"]
            }
        }
    }

alt.themes.register("budget_theme", lambda: _alt_theme(pal))
alt.themes.enable("budget_theme")

# =============================
# Global CSS using variables
# =============================
st.markdown(f"""
<style>
:root {{
  --bg: {pal['bg']}; --panel: {pal['panel']}; --card: {pal['card']};
  --text: {pal['text']}; --muted: {pal['muted']}; --grid: {pal['grid']};
  --accent: {pal['accent']}; --accent2: {pal['accent2']};
}}
body, div, p, span {{ font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji"; }}
.stApp {{ background: var(--bg); }}
.block-container {{ padding-top: 1rem; }}

.main-header {{
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  padding: 1.5rem 2rem; border-radius: 14px; margin-bottom: 1.5rem; color: white;
  box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}}
.filter-header {{
  background: var(--panel); color: var(--text); padding: 0.6rem 0.9rem; border-radius: 10px;
  margin: 1rem 0 0.6rem 0; font-weight: 600; border: 1px solid var(--grid);
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
.stTabs [data-baseweb="tab"] {{
  background-color: var(--panel); border-radius: 10px; color: var(--muted); font-weight: 600; border: 1px solid var(--grid);
}}
.stTabs [aria-selected="true"] {{ background: var(--accent); color: #fff; border: 1px solid var(--accent); }}

.stButton > button {{
  background: var(--accent); color: white; border: none; border-radius: 10px; padding: 0.6rem 1rem;
  font-weight: 700; transition: all 0.2s ease; box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}}
.stButton > button:hover {{ transform: translateY(-1px); filter: brightness(1.03); }}

div[data-testid="stMetricValue"] {{ color: var(--text); }}
div[data-testid="stMetricDelta"] {{ color: var(--muted); }}

.ai-box {{
  border-radius: 12px; padding: 1rem 1.25rem;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  color: white; margin-top: 1rem;
}}
.ai-result {{
  border-radius: 12px; padding: 1rem 1.25rem; background: var(--card); color: var(--text);
  border: 1px solid var(--grid); box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
}}
.dataframe, .stDataFrame {{ border-radius: 10px; border: 1px solid var(--grid); background: var(--card); }}
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
        ⚠️ <strong>OpenAI API Key not found:</strong> Add OPENAI_API_KEY to .env or Streamlit secrets to enable GPT features.
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

@st.cache_data(ttl=600)
def load_actuals(path: str) -> pd.DataFrame:
    """Load and validate actuals/allocated CSV."""
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        if "Budget_Allcated" in df.columns:
            df = df.rename(columns={"Budget_Allcated": "Budget_Allocated"})

        req = ["Month", "Department", "Category", "Budget_Allocated", "Actual_Spent"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            st.error(f"Missing required columns in actuals data: {miss}")
            return pd.DataFrame()

        if "Variance" not in df.columns:
            df["Variance"] = df["Actual_Spent"] - df["Budget_Allocated"]

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

@st.cache_data(ttl=600)
def load_forecast(path: str) -> pd.DataFrame | None:
    """Load optional forecast file."""
    try:
        if not os.path.exists(path):
            return None
        f = pd.read_csv(path)
        f.columns = [c.strip().replace(" ", "_") for c in f.columns]
        if "Predicted_Spent" not in f.columns:
            for alt_name in ["Forecast", "Forecasted", "yhat", "y_pred", "Prediction"]:
                if alt_name in f.columns:
                    f = f.rename(columns={alt_name: "Predicted_Spent"})
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

def build_ai_context(df_f: pd.DataFrame) -> dict:
    """Compact, structured context for the AI—robust across pandas versions."""
    df_f = df_f.copy()
    df_f["Month"] = pd.to_datetime(df_f["Month"], errors="coerce")

    totals = {
        "total_budget": float(df_f["Budget_Allocated"].sum()),
        "total_spent": float(df_f["Actual_Spent"].sum()),
        "total_variance": float((df_f["Actual_Spent"] - df_f["Budget_Allocated"]).sum()),
        "avg_variance_pct": float(df_f["Variance_Percent"].mean()),
        "date_min": df_f["Month"].min().strftime("%Y-%m"),
        "date_max": df_f["Month"].max().strftime("%Y-%m"),
        "records": int(len(df_f)),
        "departments": sorted(df_f["Department"].unique().tolist()),
        "categories": sorted(df_f["Category"].unique().tolist()),
    }

    m = df_f.resample("MS", on="Month")[["Budget_Allocated", "Actual_Spent"]].sum().reset_index()
    m["Variance"] = m["Actual_Spent"] - m["Budget_Allocated"]
    monthly = [
        {"Month": row["Month"].strftime("%Y-%m"),
         "Allocated": float(row["Budget_Allocated"]),
         "Spent": float(row["Actual_Spent"]),
         "Variance": float(row["Variance"])}
        for _, row in m.iterrows()
    ]

    dept = (df_f.groupby("Department", as_index=False)
            .agg(Allocated=("Budget_Allocated", "sum"),
                 Spent=("Actual_Spent", "sum")))
    dept["Variance"] = dept["Spent"] - dept["Allocated"]
    dept_over = dept.sort_values("Variance", ascending=False).head(8).to_dict(orient="records")
    dept_under = dept.sort_values("Variance", ascending=True).head(8).to_dict(orient="records")

    cat = (df_f.groupby("Category", as_index=False)
           .agg(Allocated=("Budget_Allocated", "sum"),
                Spent=("Actual_Spent", "sum")))
    cat["Variance"] = cat["Spent"] - cat["Allocated"]
    cat_top = (cat.assign(absV=cat["Variance"].abs())
                   .sort_values("absV", ascending=False)
                   .drop(columns="absV")
                   .head(8)
                   .to_dict(orient="records"))

    return {
        "totals": totals,
        "monthly": monthly,
        "department_top_over": dept_over,
        "department_top_under": dept_under,
        "category_top": cat_top,
    }

def call_openai(system_msg: str, user_msg: str, temperature: float = 0.2) -> str:
    """Simple wrapper for OpenAI chat completions."""
    if not client:
        return "OpenAI API key not configured."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            max_tokens=900,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

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
    dept_sel = st.multiselect("🏢 Department(s)", options=sorted(df["Department"].unique()), default=sorted(df["Department"].unique()))
    cat_sel = st.multiselect("📁 Category(ies)", options=sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))

    min_m = pd.to_datetime(df["Month"].min()).to_pydatetime()
    max_m = pd.to_datetime(df["Month"].max()).to_pydatetime()
    date_range = st.slider("📅 Month Range", value=(min_m, max_m), min_value=min_m, max_value=max_m, format="YYYY-MM")

    st.markdown('<div class="filter-header">⚙️ Advanced Filters</div>', unsafe_allow_html=True)
    year_opts = sorted(df["Year"].unique())
    years_sel = st.multiselect("📅 Years", options=year_opts, default=year_opts)

    vmin, vmax = float(df["Variance_Percent"].min()), float(df["Variance_Percent"].max())
    variance_range = st.slider("📊 Variance % Range", min_value=vmin, max_value=vmax, value=(vmin, vmax), step=1.0)

    amin, amax = float(df["Actual_Spent"].min()), float(df["Actual_Spent"].max())
    amount_range = st.slider("💰 Actual Spent Range", min_value=amin, max_value=amax, value=(amin, amax), step=1000.0, format="$%.0f")

    perf = st.selectbox("🎯 Budget Performance",
                        ["All", "Over Budget (>0%)", "Under Budget (<0%)", "On Target (±5%)", "Significant Variance (>±10%)"])

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
  <h1>AI Budget Forecast & Analysis</h1>
  <p style="font-size: 1.05rem; margin: 0.4rem 0;">Advanced Financial Intelligence Platform</p>
  <p style="font-size: 0.9rem; opacity: 0.9;">Actuals: historical to present · Forecast: 2025 (optional)</p>
</div>
""", unsafe_allow_html=True)

if df_f.empty:
    st.warning("No data matches your current filters. Try expanding the date range or selections.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
total_budget = df_f["Budget_Allocated"].sum()
total_actual = df_f["Actual_Spent"].sum()
total_var = (df_f["Actual_Spent"] - df_f["Budget_Allocated"]).sum()
var_pct = (total_var / total_budget * 100) if total_budget else 0.0

with col1:
    st.metric("Total Budget", money_fmt(total_budget), delta=f"{len(df_f):,} records")
with col2:
    delta_color = "normal" if abs(var_pct) < 5 else "inverse"
    st.metric("Actual Spent", money_fmt(total_actual), delta=f"vs Budget: {var_pct:+.1f}%", delta_color=delta_color)
with col3:
    variance_color = "inverse" if total_var > 0 else "normal"
    st.metric("Net Variance", f"{money_fmt(total_var)}", delta=f"{var_pct:+.2f}%", delta_color=variance_color)
with col4:
    efficiency = max(0.0, 100 - abs(var_pct))
    tag = "Excellent" if efficiency > 95 else "Good" if efficiency > 85 else "Needs Review"
    eff_color = "normal" if efficiency > 90 else "inverse"
    st.metric("Budget Efficiency", f"{efficiency:.1f}%", delta=tag, delta_color=eff_color)

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
# Visual Analytics
# =============================
st.markdown("### 📊 Visual Analytics")

# Monthly trend (Allocated vs Spent) + optional forecast
monthly = (df_f.groupby("Month", as_index=False)[["Budget_Allocated", "Actual_Spent"]].sum().sort_values("Month"))
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

# Department & Category prep
dept = df_f.groupby("Department", as_index=False)[["Budget_Allocated", "Actual_Spent"]].sum()
dept_long = dept.melt("Department", ["Budget_Allocated", "Actual_Spent"], var_name="Type", value_name="Amount")
dept_long["Type"] = dept_long["Type"].map(type_map)

cat = df_f.groupby("Category", as_index=False)[["Budget_Allocated", "Actual_Spent"]].sum()
cat_long = cat.melt("Category", ["Budget_Allocated", "Actual_Spent"], var_name="Type", value_name="Amount")
cat_long["Type"] = cat_long["Type"].map(type_map)

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
        ).properties(height=360)
    )
    st.altair_chart(chart_monthly, use_container_width=True)

with tab2:
    st.caption("Allocated vs Spent by Department (sorted by variance).")
    dept_tot = df_f.groupby("Department", as_index=False).agg(
        Allocated=("Budget_Allocated", "sum"), Spent=("Actual_Spent", "sum")
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
        ).properties(height=420)
    )
    st.altair_chart(chart_dept, use_container_width=True)

with tab3:
    st.caption("Allocated vs Spent by Category (sorted by variance).")
    cat_tot = df_f.groupby("Category", as_index=False).agg(
        Allocated=("Budget_Allocated", "sum"), Spent=("Actual_Spent", "sum")
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
        ).properties(height=420)
    )
    st.altair_chart(chart_cat, use_container_width=True)

# =============================
# 🤖 AI-Powered Insights
# =============================
st.markdown("### 🤖 AI-Powered Insights")
st.markdown('<div class="ai-box">Get intelligent analysis and answers about your budget data.</div>', unsafe_allow_html=True)

if client:
    ai_mode = st.radio("Choose an option:", ["Ask Question", "Quick Analysis"], horizontal=True)
    context = build_ai_context(df_f)

    if ai_mode == "Ask Question":
        with st.form("ask_form", clear_on_submit=False):
            q = st.text_area(
                "Ask about the CURRENTLY FILTERED data (e.g., “Which departments are most over budget in FY2025 Q3?”)",
                placeholder="Type your question..."
            )
            temp = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
            submitted = st.form_submit_button("Ask")
        if submitted:
            sys = (
                "You are a precise financial analyst. Answer ONLY using the provided structured context. "
                "If the answer is not derivable from context, say so briefly. Use clear bullets, include $-formatted values, "
                "and reference the active filter window when relevant."
            )
            user = f"Question: {q}\n\nContext JSON:\n{context}"
            ans = call_openai(sys, user, temperature=temp)
            st.markdown("#### Answer")
            st.markdown(f"<div class='ai-result'>{ans}</div>", unsafe_allow_html=True)
    else:
        with st.form("quick_form"):
            style = st.selectbox(
                "Summary style",
                ["Executive (bullets + short narrative)", "Risks & Opportunities", "Action Items"],
                index=0
            )
            submitted2 = st.form_submit_button("Generate Summary")
        if submitted2:
            sys = (
                "You are a senior FP&A analyst. Based on the provided structured data, produce a concise, decision-focused summary. "
                "Include 5–7 bullet points and a 3–5 sentence narrative. Call out variances, trends, and material drivers by department/category. "
                "Use $ and % formatting as shown and keep it scoped to the filter window."
            )
            prompt_style = {
                "Executive (bullets + short narrative)": "Provide bullets then a short narrative.",
                "Risks & Opportunities": "Focus on top risks and opportunities and mitigation ideas.",
                "Action Items": "List prioritized actions with owners and time horizons."
            }[style]
            user = f"Write a '{style}' summary for the filtered data. {prompt_style}\n\nContext JSON:\n{context}"
            ans = call_openai(sys, user, temperature=0.2)
            st.markdown("#### Summary")
            st.markdown(f"<div class='ai-result'>{ans}</div>", unsafe_allow_html=True)
else:
    st.info("Add your OPENAI_API_KEY to enable Ask Question and Quick Analysis.")

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
