import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import altair as alt
from datetime import datetime
import re

st.set_page_config(page_title="AI Budget Assistant", layout="wide")

# === OpenAI client ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None
if not api_key:
    st.warning("OPENAI_API_KEY not found. Add it to .env or Streamlit secrets for GPT features.")

# === Loaders ===
@st.cache_data
def load_actuals(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # fix common typo
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
        for altcol in ["Forecast", "Forecasted", "yhat", "y_pred"]:
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

ACTUALS_PATH = "rich_dummy_budget_data.csv"
FORECAST_PATH = "forecasted_budget_2025.csv"

df = load_actuals(ACTUALS_PATH)
forecast = load_forecast(FORECAST_PATH)

# === Sidebar filters ===
st.sidebar.header("📂 Filters")
dept_sel = st.sidebar.multiselect(
    "Department(s)", sorted(df["Department"].unique()), default=sorted(df["Department"].unique())
)
cat_sel = st.sidebar.multiselect(
    "Category(ies)", sorted(df["Category"].unique()), default=sorted(df["Category"].unique())
)

min_month_py = pd.to_datetime(df["Month"].min()).to_pydatetime()
max_month_py = pd.to_datetime(df["Month"].max()).to_pydatetime()
date_range = st.sidebar.slider(
    "Actuals Month Range",
    value=(min_month_py, max_month_py),
    min_value=min_month_py,
    max_value=max_month_py,
    format="YYYY-MM",
)

show_forecast = st.sidebar.checkbox("Show 2025 forecast", value=(forecast is not None))

# === Filtered data (for charts/tables/KPIs) ===
mask = (
    df["Department"].isin(dept_sel)
    & df["Category"].isin(cat_sel)
    & (df["Month"] >= pd.to_datetime(date_range[0]))
    & (df["Month"] <= pd.to_datetime(date_range[1]))
)
df_f = df.loc[mask].copy()

if forecast is not None:
    f_mask = forecast["Department"].isin(dept_sel) & forecast["Category"].isin(cat_sel)
    f_f = forecast.loc[f_mask].copy()
else:
    f_f = None

# === Header & KPIs ===
st.title("💼 AI Budget Forecast & Analysis")
st.caption("Actuals: 2021–present | Forecast: 2025 (from ML model)")

col1, col2, col3, col4 = st.columns(4)
total_budget = df_f["Budget_Allocated"].sum()
total_actual = df_f["Actual_Spent"].sum()
total_var = df_f["Variance"].sum()
var_pct = (total_var / total_budget * 100) if total_budget else 0.0
col1.metric("Total Budget Allocated", f"${total_budget:,.0f}")
col2.metric("Total Actual Spent", f"${total_actual:,.0f}")
col3.metric("Total Variance", f"${total_var:,.0f}")
col4.metric("Variance % of Budget", f"{var_pct:,.2f}%")

# === Tables ===
st.subheader("📊 Filtered Actuals")
st.dataframe(df_f, use_container_width=True)

if show_forecast and f_f is not None:
    st.subheader("🔮 Filtered Forecast (2025)")
    st.dataframe(f_f, use_container_width=True)

# === Charts ===
st.subheader("📈 Charts")
tab1, tab2, tab3 = st.tabs(["Monthly Trend", "By Department", "By Category"])

with tab1:
    monthly_actuals = df_f.groupby("Month", as_index=False)[["Actual_Spent", "Budget_Allocated"]].sum()
    actual_chart = (
        alt.Chart(monthly_actuals)
        .transform_fold(["Actual_Spent", "Budget_Allocated"], as_=["Type", "Amount"])
        .mark_line(point=True)
        .encode(
            x=alt.X("Month:T", title="Month"),
            y=alt.Y("Amount:Q", title="Amount"),
            color=alt.Color("Type:N"),
            tooltip=["Month:T", "Type:N", alt.Tooltip("Amount:Q", format=",.0f")],
        )
    )
    if show_forecast and f_f is not None:
        monthly_fc = f_f.groupby("Month", as_index=False)["Predicted_Spent"].sum()
        fc_chart = (
            alt.Chart(monthly_fc)
            .mark_line(point=True)
            .encode(
                x="Month:T",
                y=alt.Y("Predicted_Spent:Q", title="Predicted Spent"),
                color=alt.value("#8a2be2"),
                tooltip=["Month:T", alt.Tooltip("Predicted_Spent:Q", format=",.0f")],
            )
        )
        st.altair_chart(alt.layer(actual_chart, fc_chart).resolve_scale(y="independent").interactive(), use_container_width=True)
    else:
        st.altair_chart(actual_chart.interactive(), use_container_width=True)

with tab2:
    by_dept = df_f.groupby("Department", as_index=False)[["Actual_Spent", "Budget_Allocated"]].sum()
    by_dept_melt = by_dept.melt("Department", var_name="Type", value_name="Amount")
    dept_chart = (
        alt.Chart(by_dept_melt)
        .mark_bar()
        .encode(
            x=alt.X("Department:N", sort="-y"),
            y="Amount:Q",
            color="Type:N",
            tooltip=["Department", "Type", alt.Tooltip("Amount:Q", format=",.0f")],
        )
        .properties(height=350)
    )
    st.altair_chart(dept_chart, use_container_width=True)

with tab3:
    by_cat = df_f.groupby("Category", as_index=False)[["Actual_Spent", "Budget_Allocated"]].sum()
    by_cat_melt = by_cat.melt("Category", var_name="Type", value_name="Amount")
    cat_chart = (
        alt.Chart(by_cat_melt)
        .mark_bar()
        .encode(
            x=alt.X("Category:N", sort="-y"),
            y="Amount:Q",
            color="Type:N",
            tooltip=["Category", "Type", alt.Tooltip("Amount:Q", format=",.0f")],
        )
        .properties(height=350)
    )
    st.altair_chart(cat_chart, use_container_width=True)

# === Helpers ===
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

# === GPT Insights ===
st.markdown("---")
st.header("🤖 GPT Insights")

cols = st.columns([1, 2])

# ---- Summary button (compact forecast+actuals) ----
with cols[0]:
    if st.button("Generate Summary from Current View"):
        if not client:
            st.error("OpenAI key not set.")
        else:
            compact = build_compact_summary(df_f, f_f if (show_forecast and f_f is not None) else None)
            user_prompt = compact + "\nPlease summarize key trends and give 2–3 actionable recommendations."
            with st.spinner("Asking GPT..."):
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role":"system","content":"You are a budget analyst. Be concise, numeric, and practical."},
                        {"role":"user","content":user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
            st.success("Summary ready!")
            st.write(resp.choices[0].message.content)

# ---- Q&A (smart filtering before GPT) ----
with cols[1]:
    q = st.text_input(
        "Ask about ACTUALS and/or FORECAST (e.g., 'Actual Spent in June 2022 for HR', 'Budget for Marketing 2023-11', 'Forecast for June 2025 IT Software')"
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
            if dept_in_q: d = d[d["Department"] == dept_in_q]
            if cat_in_q: d = d[d["Category"] == cat_in_q]
            return d

        def df_slice_forecast():
            if f_f is None: return pd.DataFrame()
            d = f_f.copy()
            if year and month:
                ms = f"{year}-{month:02d}"
                d["Month_str"] = d["Month"].dt.strftime("%Y-%m")
                d = d[d["Month_str"] == ms]
            if dept_in_q: d = d[d["Department"] == dept_in_q]
            if cat_in_q: d = d[d["Category"] == cat_in_q]
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
            if year and month: detail.append(f"month={year}-{month:02d}")
            if dept_in_q: detail.append(f"dept={dept_in_q}")
            if cat_in_q: detail.append(f"cat={cat_in_q}")
            detail_text = ", ".join(detail) if detail else "current filters"
            st.info(f"No {target} rows found for {detail_text}. Try changing filters or the query.")
        else:
            if not client:
                st.error("OpenAI key not set.")
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

                with st.spinner("Thinking..."):
                    resp = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role":"system","content":"You are a precise financial analyst. Be concise and numeric."},
                            {"role":"user","content":final_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=450
                    )
                st.write(resp.choices[0].message.content)

st.caption("Tip: Filters limit charts/tables/summaries. The Q&A sends only the most relevant rows + compact summaries to OpenAI.")
