# RAG_APP.py — AI Budget Assistant with LangChain RAG
# - Keeps your fiscal calendarization + filters + KPIs + charts
# - Adds RAG Q&A: builds a semantic index over the filtered context (top variance rows/aggregates)
# - Robust to missing OpenAI key (disables AI features gracefully)
# - Fixes prior syntax errors (balanced brackets, complete expressions)

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv

# ========= Optional AI / LangChain stack =========
# These imports are inside try blocks so the app can still run without AI libs installed
client = None
llm_model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_has_langchain = False
_has_openai = False

try:
    from openai import OpenAI
    _has_openai = True
except Exception:
    _has_openai = False

try:
    # LangChain core
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    # OpenAI bindings
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    # Vector store
    from langchain_community.vectorstores import FAISS
    _has_langchain = True
except Exception:
    _has_langchain = False

# =============================
# Page
# =============================
st.set_page_config(page_title="AI Budget RAG Assistant", layout="wide", initial_sidebar_state="expanded")

# =============================
# Themes (Executive Light/Dark)
# =============================
THEMES = {
    "Executive Light": {
        "bg": "#F7FAFC", "sidebar": "#0B1E3E", "panel": "#FFFFFF", "card": "#FFFFFF",
        "text": "#0F172A", "muted": "#475569", "grid": "#E2E8F0",
        "brand1": "#0B1E3E", "brand2": "#2F6BFF",
        "alloc": "#2151B8", "spent": "#53A2FF",
        "ok": "#10B981", "warn": "#D97706"
    },
    "Executive Dark": {
        "bg": "#0B1E3E", "sidebar": "#071A34", "panel": "#0E2044", "card": "#0F234B",
        "text": "#FFFFFF", "muted": "#CBD5E1", "grid": "#14315F",
        "brand1": "#0B1E3E", "brand2": "#2F6BFF",
        "alloc": "#8FB7FF", "spent": "#5FB0FF",
        "ok": "#10B981", "warn": "#F59E0B"
    }
}
with st.sidebar:
    st.markdown("**🎨 Theme**")
    theme_name = st.selectbox("Select", list(THEMES.keys()), index=1)  # default Dark
pal = THEMES[theme_name]
alloc_col, spent_col = pal["alloc"], pal["spent"]

# Altair theme bound to palette
def _alt_theme(p):
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {"labelColor": p["text"], "titleColor": p["text"], "gridColor": p["grid"]},
            "legend": {"labelColor": p["text"], "titleColor": p["text"]},
            "title": {"color": p["text"]},
            "range": {"category": [p["alloc"], p["spent"], "#9CA3AF", "#A78BFA"]}
        }
    }
alt.themes.register("fin_theme", lambda: _alt_theme(pal))
alt.themes.enable("fin_theme")

# =============================
# CSS (palette-aware + readable sidebar inputs)
# =============================
st.markdown(f"""
<style>
  .stApp {{ background:{pal['bg']}; color:{pal['text']}; }}
  .block-container {{ padding-top: 1rem; }}
  [data-testid="stSidebar"] > div:first-child {{ background:{pal['sidebar']}; }}
  [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] p {{ color:#E5E7EB !important; }}

  [data-testid="stSidebar"] .stSelectbox > div > div,
  [data-testid="stSidebar"] .stMultiSelect > div > div,
  [data-testid="stSidebar"] .stSlider > div > div {{
    background:#FFFFFF !important; color:#111827 !important; border-radius:10px; border:1px solid #D1D5DB;
  }}
  [data-testid="stSidebar"] [role="combobox"] * {{ color:#111827 !important; }}
  [role="listbox"] {{ background:#FFFFFF !important; color:#111827 !important; border:1px solid #D1D5DB; }}
  [role="option"] * {{ color:#111827 !important; }}

  .main-header {{
    background: linear-gradient(135deg, {pal['brand1']} 0%, {pal['brand2']} 100%);
    padding: 2rem; border-radius: 15px; margin-bottom: 1.25rem; text-align: center; color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,.15);
  }}
  .filter-header {{
    background:{pal['panel']}; color:{pal['text']}; padding:.6rem; border-radius:10px;
    border:1px solid {pal['grid']}; font-weight:700; text-align:center; margin: 0.5rem 0;
  }}
  .ai-box {{ border-radius:12px; padding:1rem 1.25rem;
            background:linear-gradient(135deg, {pal['brand1']} 0%, {pal['brand2']} 100%); color:white; }}
  .ai-result {{ border-radius:12px; padding:1rem 1.25rem; background:{pal['card']}; color:{pal['text']};
               border:1px solid {pal['grid']}; }}
</style>
""", unsafe_allow_html=True)

# =============================
# OpenAI (optional)
# =============================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")

if _has_openai and api_key:
    try:
        client = OpenAI(api_key=api_key)
    except Exception:
        client = None
else:
    client = None

if not api_key:
    st.sidebar.warning("OpenAI API key not set. AI & RAG features will be disabled.", icon="⚠️")

def call_openai(system_msg: str, user_msg: str, temperature=0.2, max_tokens=900) -> str:
    if not client:
        return "OpenAI API key not configured."
    try:
        resp = client.chat.completions.create(
            model=llm_model_name,
            messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
            temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

# =============================
# Safe formatters (avoid KPI crashes)
# =============================
def nfloat(x) -> float:
    try:
        xf = float(x)
        return 0.0 if not np.isfinite(xf) else xf
    except Exception:
        try:
            return float(np.nansum(pd.to_numeric(x, errors="coerce")))
        except Exception:
            return 0.0

def money_fmt(x) -> str:
    try:
        xf = float(x)
        return "—" if not np.isfinite(xf) else f"${xf:,.0f}"
    except Exception:
        return "—"

def pct_fmt(x, digits=1, signed=True) -> str:
    try:
        xf = float(x)
        if not np.isfinite(xf): return "—"
        return f"{xf:{'+' if signed else ''}.{digits}f}%"
    except Exception:
        return "—"

# =============================
# Column mapping & parsers
# =============================
def _n(x: str) -> str:
    return re.sub(r"\s+", " ", str(x)).strip().upper()

ALIAS = {
    "BUDGET YEAR": ["FISCAL YEAR","FY","YEAR","BUDGET_YEAR"],
    "ACCOUNTING PERIOD": ["PERIOD","PERIOD NUMBER","ACCOUNTING_PERIOD","PERIOD NO","MONTH","ACCOUNTING MONTH","PERIOD NAME"],
    "BUDGET AMOUNT": ["BUDGET","ADOPTED BUDGET","AMENDED BUDGET","BUDGET TOTAL","APPROPRIATION","APPROPRIATED AMOUNT","BUDGET_AMT"],
    "EXPENSE AMOUNT": ["EXPENDITURE AMOUNT","ACTUALS","ACTUAL EXPENSE","ACTUAL EXPENDITURE","YTD EXPENSE","AMOUNT EXPENDED","EXPENSE","ACTUAL_AMOUNT"],

    "DEPARTMENT ID DESCRIPTION": ["DEPARTMENT NAME","DEPARTMENT","DEPARTMENT DESC"],
    "ACCOUNT TYPE": ["ACCT TYPE"],
    "ACCOUNT DESCRIPTION": ["ACCOUNT DESC"],

    "FUND CODE DESCRIPTION": ["FUND DESCRIPTION","FUND DESC"],
    "PROGRAM DESCRIPTION": ["PROGRAM DESC"],
    "LEDGER GROUP": ["LEDGER","LEDGER GROUP NAME"],

    "ENCUMBERED AMOUNT": ["ENCUMBRANCE","ENCUMBERED"],
    "PRE ENCUMBERED AMOUNT": ["PRE ENCUMBRANCE","PRE-ENCUMBRANCE"],
    "REVENUE AMOUNT": ["REVENUE","REV AMOUNT","REVENUE_TOTAL"],
}
CANON = {
    "BUDGET YEAR": "Budget_Year",
    "ACCOUNTING PERIOD": "Accounting_Period",
    "BUDGET AMOUNT": "Budget_Allocated",
    "EXPENSE AMOUNT": "Actual_Spent",
    "DEPARTMENT ID DESCRIPTION": "Department",
    "ACCOUNT TYPE": "Account_Type",
    "ACCOUNT DESCRIPTION": "Account_Desc",
    "FUND CODE DESCRIPTION": "Fund_Desc",
    "PROGRAM DESCRIPTION": "Program_Desc",
    "LEDGER GROUP": "Ledger_Group",
    "ENCUMBERED AMOUNT": "Encumbered",
    "PRE ENCUMBERED AMOUNT": "Pre_Encumbered",
    "REVENUE AMOUNT": "Revenue_Amount",
}

def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx",".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

def _parse_year(val):
    if pd.isna(val): return pd.NA
    s = str(val).strip()
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        y = int(m.group(0))
        return pd.NA if y < 1900 or y > 2100 else y
    try:
        y = int(float(s))
        return pd.NA if y < 1900 or y > 2100 else y
    except Exception:
        return pd.NA

def _parse_period(val):
    # Supports digits or names; fiscal mapping done later
    if pd.isna(val): return pd.NA
    s = str(val).strip()
    m = re.match(r"^(20\d{2})[-/](0?[1-9]|1[0-2])$", s)  # YYYY-MM
    if m: return int(m.group(2))
    mon_map = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"SEPT":9,"OCT":10,"NOV":11,"DEC":12}
    up = s.upper()
    if up.startswith("SEPT"): return 9
    if up[:3] in mon_map: return mon_map[up[:3]]
    return pd.to_numeric(s, errors="coerce")

# =============================
# Loader (returns processed df only)
# =============================
@st.cache_data(show_spinner=True)
def load_budget_pull(path: str) -> pd.DataFrame:
    raw = _read_any(path)

    # Build rename map
    lookup = {}
    for k, syns in ALIAS.items():
        lookup[_n(k)] = k
        for s in syns: lookup[_n(s)] = k

    rename_map = {}
    for c in raw.columns:
        if _n(c) in lookup:
            rename_map[c] = CANON[lookup[_n(c)]]

    df = raw.rename(columns=rename_map).copy()

    # Required columns
    for req in ["Budget_Year","Accounting_Period","Budget_Allocated","Actual_Spent"]:
        if req not in df.columns:
            raise ValueError(f"Missing required column: {req}")

    # Optional defaults
    for k in ["Department","Account_Desc","Account_Type","Fund_Desc","Program_Desc","Ledger_Group",
              "Encumbered","Pre_Encumbered","Revenue_Amount"]:
        if k not in df.columns:
            df[k] = 0.0 if k in ["Encumbered","Pre_Encumbered","Revenue_Amount"] else np.nan

    # Normalize string dims
    for dim in ["Department","Account_Desc","Account_Type","Fund_Desc","Program_Desc","Ledger_Group"]:
        if dim in df.columns:
            df[dim] = df[dim].astype(str).str.strip()

    # Types & cleaning
    df["Budget_Year"] = df["Budget_Year"].map(_parse_year)
    df["Accounting_Period"] = df["Accounting_Period"].map(_parse_period)
    df = df[(df["Accounting_Period"] >= 1) & (df["Accounting_Period"] <= 12)].copy()

    for c in ["Budget_Allocated","Actual_Spent","Encumbered","Pre_Encumbered","Revenue_Amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Fiscal (1=Oct … 12=Sep) → Calendar
    F2C = {1:10,2:11,3:12,4:1,5:2,6:3,7:4,8:5,9:6,10:7,11:8,12:9}
    df["Cal_Month"] = df["Accounting_Period"].map(F2C)

    try:
        df["Cal_Month"] = df["Cal_Month"].astype(pd.Int64Dtype())
        df["Budget_Year"] = df["Budget_Year"].astype(pd.Int64Dtype())
        df["Cal_Year"] = np.where(df["Cal_Month"] >= 10, df["Budget_Year"] - 1, df["Budget_Year"])
        df["Cal_Year"] = pd.Series(df["Cal_Year"]).astype(pd.Int64Dtype())
    except Exception:
        df = df[df["Cal_Month"].notna() & df["Budget_Year"].notna()].copy()
        df["Cal_Month"] = df["Cal_Month"].astype("int64")
        df["Budget_Year"] = df["Budget_Year"].astype("int64")
        df["Cal_Year"] = np.where(df["Cal_Month"] >= 10, df["Budget_Year"] - 1, df["Budget_Year"]).astype("int64")

    df["Month"] = pd.to_datetime(
        df["Cal_Year"].astype(str) + "-" + pd.Series(df["Cal_Month"]).astype(str).str.zfill(2) + "-01",
        errors="coerce"
    )
    df = df.dropna(subset=["Month"]).copy()

    # Helper time fields
    df["Year"] = df["Month"].dt.year
    df["Quarter"] = df["Month"].dt.quarter
    df["Fiscal_Year"] = df["Budget_Year"]
    df["Fiscal_Quarter"] = (((pd.Series(df["Cal_Month"]) - 10) % 12) // 3 + 1)

    # Effective measures (commitment toggle later)
    df["Actual_Effective"] = df["Actual_Spent"]
    df["Variance_Effective"] = df["Actual_Effective"] - df["Budget_Allocated"]
    den = df["Budget_Allocated"].replace({0: np.nan})
    df["Variance_Percent_Effective"] = (df["Variance_Effective"] / den) * 100

    return df.sort_values("Month").reset_index(drop=True)

# =============================
# Load Data (fixed path with CSV fallback)
# =============================
ACTUALS_PATH = "FY 2021 Budget Pull.xlsx"  # put file at repo root

try:
    df = load_budget_pull(ACTUALS_PATH)
except ImportError as e:
    base, _ = os.path.splitext(ACTUALS_PATH)
    csv_path = base + ".csv"
    if os.path.exists(csv_path):
        df = load_budget_pull(csv_path)
    else:
        st.error("❌ Excel engine missing. Install `openpyxl` for .xlsx (and `xlrd` for .xls), "
                 "or provide a CSV with the same base name.\n\n"
                 f"Details: {e}")
        st.stop()
except Exception as e:
    st.error(f"❌ Could not load data at {ACTUALS_PATH}: {e}")
    st.stop()

# =============================
# Header
# =============================
st.markdown(f"""
<div class="main-header">
  <h1>AI Budget Forecast & RAG Analysis</h1>
  <p style="opacity:.9">Professional analytics with fiscal-year calendarization (1=Oct … 12=Sep)</p>
</div>
""", unsafe_allow_html=True)

# =============================
# Sidebar Filters
# =============================
with st.sidebar:
    st.markdown('<div class="filter-header">🔍 Primary Filters</div>', unsafe_allow_html=True)

    dept_options = sorted(pd.Series(df["Department"]).dropna().astype(str).unique().tolist()) if "Department" in df.columns else []
    accd_options = sorted(pd.Series(df["Account_Desc"]).dropna().astype(str).unique().tolist()) if "Account_Desc" in df.columns else []
    atype_options = sorted(pd.Series(df["Account_Type"]).dropna().astype(str).unique().tolist()) if "Account_Type" in df.columns else []

    dept_sel = st.multiselect("Department", dept_options, default=dept_options)
    accd_sel = st.multiselect("Account Description", accd_options, default=accd_options)
    atype_sel = st.multiselect("Account Type", atype_options, default=atype_options)

    st.markdown('<div class="filter-header">🧰 Options</div>', unsafe_allow_html=True)
    include_commitments = st.checkbox("Include Encumbrances in Spent", value=False)

    # Month range slider (handles single-month span)
    min_m = pd.to_datetime(df["Month"].min()).to_pydatetime()
    max_m = pd.to_datetime(df["Month"].max()).to_pydatetime()
    if min_m >= max_m:
        from pandas.tseries.offsets import MonthBegin
        faux_max = (pd.Timestamp(min_m) + MonthBegin(1)).to_pydatetime()
        date_range = st.slider("📅 Month Range", value=(min_m, faux_max), min_value=min_m, max_value=faux_max, format="YYYY-MM")
    else:
        date_range = st.slider("📅 Month Range", value=(min_m, max_m), min_value=min_m, max_value=max_m, format="YYYY-MM")

    st.markdown('<div class="filter-header">⚙️ Advanced</div>', unsafe_allow_html=True)
    fy_opts = sorted(pd.Series(df["Fiscal_Year"]).dropna().unique().tolist())
    fy_sel = st.multiselect("Fiscal Year", fy_opts, default=fy_opts)

    # For slider bounds (if commitments included)
    df_tmp = df.copy()
    if include_commitments:
        df_tmp["Actual_Effective"] = df_tmp["Actual_Spent"] + df_tmp.get("Encumbered", 0) + df_tmp.get("Pre_Encumbered", 0)
        df_tmp["Variance_Effective"] = df_tmp["Actual_Effective"] - df_tmp["Budget_Allocated"]
        den_t = df_tmp["Budget_Allocated"].replace({0: np.nan})
        df_tmp["Variance_Percent_Effective"] = (df_tmp["Variance_Effective"] / den_t) * 100

    v = pd.Series(df_tmp["Variance_Percent_Effective"]).dropna()
    vmin, vmax = (float(v.min()), float(v.max())) if len(v) else (-100.0, 100.0)
    variance_range = st.slider("Variance % (effective)", min_value=vmin, max_value=vmax, value=(vmin, vmax), step=1.0)

    a = pd.Series(df_tmp["Actual_Effective"])
    amin, amax = float(a.min()), float(a.max())
    amount_range = st.slider("Spent Range ($, effective)", min_value=amin, max_value=amax, value=(amin, amax), step=1000.0, format="$%.0f")

    budget_perf = st.selectbox(
        "🎯 Budget Performance",
        ["All","Over Budget (>0%)","Under Budget (<0%)","On Target (±5%)","Significant Variance (>±10%)"]
    )

    st.markdown('<div class="filter-header">🧠 AI Settings</div>', unsafe_allow_html=True)
    ai_rows_cap = st.slider(
        "AI context row cap (after aggregation)",
        min_value=100, max_value=3000, value=600, step=100,
        help="How many grouped rows to include in the AI prompt (top by absolute variance)."
    )
    rag_top_k = st.slider("RAG: results per query (k)", 2, 10, 5)

# =============================
# Apply filters (balanced & readable)
# =============================
df_work = df.copy()
if include_commitments:
    df_work["Actual_Effective"] = df_work["Actual_Spent"] + df_work.get("Encumbered", 0) + df_work.get("Pre_Encumbered", 0)
    df_work["Variance_Effective"] = df_work["Actual_Effective"] - df_work["Budget_Allocated"]
    den_work = df_work["Budget_Allocated"].replace({0: np.nan})
    df_work["Variance_Percent_Effective"] = (df_work["Variance_Effective"] / den_work) * 100

c1 = (df_work["Department"].isin(dept_sel)) if dept_sel else True
c2 = (df_work["Account_Desc"].isin(accd_sel)) if accd_sel else True
c3 = (df_work["Account_Type"].isin(atype_sel)) if atype_sel else True
c4 = (df_work["Month"] >= pd.to_datetime(date_range[0]))
c5 = (df_work["Month"] <= pd.to_datetime(date_range[1]))
c6 = (pd.Series(df_work["Fiscal_Year"]).isin(fy_sel))
c7 = pd.Series(df_work["Variance_Percent_Effective"]).between(variance_range[0], variance_range[1])
c8 = pd.Series(df_work["Actual_Effective"]).between(amount_range[0], amount_range[1])

mask = c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8

if budget_perf == "Over Budget (>0%)":
    mask = mask & (pd.Series(df_work["Variance_Percent_Effective"]) > 0)
elif budget_perf == "Under Budget (<0%)":
    mask = mask & (pd.Series(df_work["Variance_Percent_Effective"]) < 0)
elif budget_perf == "On Target (±5%)":
    mask = mask & (pd.Series(df_work["Variance_Percent_Effective"]).between(-5, 5))
elif budget_perf == "Significant Variance (>±10%)":
    vpe = pd.Series(df_work["Variance_Percent_Effective"])
    mask = mask & ((vpe > 10) | (vpe < -10))

df_f = df_work.loc[mask].copy()

# =============================
# KPIs (NaN-safe)
# =============================
st.markdown("### 🔎 Overview")
if df_f.empty:
    st.markdown(
        f"""
        <div style="padding:1rem;border:1px solid {pal['warn']};border-radius:10px;background:rgba(245,158,11,.1);">
          <b>⚠️ No data matches your current filters.</b><br>
          Try resetting one or more filters or expand the date range.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    k1, k2, k3, k4 = st.columns(4)
    tot_alloc = nfloat(df_f["Budget_Allocated"].sum())
    tot_spent = nfloat(df_f["Actual_Effective"].sum())
    tot_var = nfloat(tot_spent - tot_alloc)
    pct = (tot_var / tot_alloc) * 100.0 if tot_alloc != 0.0 else 0.0

    with k1: st.metric("Total Budget", money_fmt(tot_alloc), delta=f"{len(df_f):,} rows")
    with k2:
        inc_note = " (Incl. Enc.)" if include_commitments else ""
        st.metric("Total Spent" + inc_note, money_fmt(tot_spent), delta=f"vs Budget {pct_fmt(pct, 1)}")
    with k3: st.metric("Net Variance", money_fmt(tot_var), delta=pct_fmt(pct, 2))
    with k4:
        efficiency = max(0.0, 100 - abs(nfloat(pct)))
        tag = "Excellent" if efficiency > 95 else "Good" if efficiency > 85 else "Needs Review"
        st.metric("Budget Efficiency", f"{efficiency:.1f}%", delta=tag)

# =============================
# Details table
# =============================
st.markdown("---")
st.subheader("📄 Details (filtered)")
if not df_f.empty:
    show_cols = [
        "Month","Fiscal_Year","Department","Account_Type","Account_Desc",
        "Fund_Desc","Program_Desc","Ledger_Group",
        "Budget_Allocated","Actual_Effective","Variance_Effective","Variance_Percent_Effective"
    ]
    show_cols = [c for c in show_cols if c in df_f.columns]
    df_disp = df_f[show_cols].copy()
    df_disp["Month"] = pd.to_datetime(df_disp["Month"]).dt.strftime("%Y-%m")
    st.dataframe(df_disp, use_container_width=True, hide_index=True, height=380)
else:
    st.info("Adjust filters to see the detailed table.")

# =============================
# Visual Analytics
# =============================
st.markdown("---")
st.subheader("📈 Visual Analytics")
t1, t2, t3 = st.tabs(["📊 Monthly Trend (clustered bars)", "🏢 By Department", "🧾 By Account Description"])

if not df_f.empty:
    with t1:
        monthly = (df_f.groupby("Month", as_index=False)[["Budget_Allocated","Actual_Effective"]].sum()
                      .sort_values("Month"))
        mlong = monthly.melt("Month", ["Budget_Allocated","Actual_Effective"], var_name="Type", value_name="Amount")
        mlong["Type"] = mlong["Type"].map({"Budget_Allocated":"Allocated","Actual_Effective":"Spent"})
        chart = (
            alt.Chart(mlong)
            .mark_bar()
            .encode(
                x=alt.X("yearmonth(Month):O", title="Month"),
                y=alt.Y("Amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[alloc_col, spent_col])),
                xOffset="Type:N",
                tooltip=[alt.Tooltip("yearmonth(Month):O", title="Month"), "Type:N", alt.Tooltip("Amount:Q", format=",.0f")]
            ).properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

    with t2:
        dept = df_f.groupby("Department", as_index=False).agg(
            Allocated=("Budget_Allocated","sum"),
            Spent=("Actual_Effective","sum")
        )
        if not dept.empty:
            dept["Variance"] = dept["Spent"] - dept["Allocated"]
            order = dept.sort_values("Variance", ascending=False)["Department"].astype(str).tolist()
            dlong = dept.melt("Department", ["Allocated","Spent"], var_name="Type", value_name="Amount")
            chart = (
                alt.Chart(dlong)
                .mark_bar(cornerRadius=3)
                .encode(
                    x=alt.X("Department:N", sort=order, axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f")),
                    color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[alloc_col, spent_col])),
                    xOffset="Type:N",
                    tooltip=["Department","Type",alt.Tooltip("Amount:Q", format=",.0f")]
                ).properties(height=420)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No Department data in current selection.")

    with t3:
        accd = df_f.groupby("Account_Desc", as_index=False).agg(
            Allocated=("Budget_Allocated","sum"),
            Spent=("Actual_Effective","sum")
        )
        if not accd.empty:
            accd["Variance"] = accd["Spent"] - accd["Allocated"]
            order = accd.sort_values("Variance", ascending=False)["Account_Desc"].astype(str).tolist()
            along = accd.melt("Account_Desc", ["Allocated","Spent"], var_name="Type", value_name="Amount")
            chart = (
                alt.Chart(along)
                .mark_bar(cornerRadius=3)
                .encode(
                    x=alt.X("Account_Desc:N", sort=order, axis=alt.Axis(labelAngle=-45), title="Account Description"),
                    y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f")),
                    color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[alloc_col, spent_col])),
                    xOffset="Type:N",
                    tooltip=["Account_Desc","Type",alt.Tooltip("Amount:Q", format=",.0f")]
                ).properties(height=420)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No Account Description data in current selection.")
else:
    st.info("Adjust filters to render charts.")

# =============================
# Utilities for AI context
# =============================
def tbl(df_):
    return "(none)" if df_.empty else df_.to_string(index=False)

def build_compact_summary(actuals_df: pd.DataFrame) -> str:
    if actuals_df.empty:
        return "No rows."
    a_month = actuals_df.groupby("Month", as_index=False)[["Actual_Effective","Budget_Allocated"]].sum()
    a_month["Month"] = a_month["Month"].dt.strftime("%Y-%m")
    a_dept = (actuals_df.groupby("Department", as_index=False)[["Actual_Effective","Budget_Allocated"]]
              .sum().sort_values("Actual_Effective", ascending=False).head(15)) if "Department" in actuals_df.columns else pd.DataFrame()
    a_desc = (actuals_df.groupby("Account_Desc", as_index=False)[["Actual_Effective","Budget_Allocated"]]
              .sum().sort_values("Actual_Effective", ascending=False).head(15)) if "Account_Desc" in actuals_df.columns else pd.DataFrame()
    return f"""
ACTUALS — monthly (Allocated vs Spent, last 24):
{tbl(a_month.tail(24))}

Top Departments:
{tbl(a_dept)}

Top Account Descriptions:
{tbl(a_desc)}
"""

# =============================
# RAG: Build vector index over filtered context
# =============================
@st.cache_resource(show_spinner=False)
def _get_embedder(_api_key: str):
    if not (_has_langchain and _api_key):
        return None
    try:
        return OpenAIEmbeddings(api_key=_api_key)
    except Exception:
        return None

def _rows_to_docs(df_ctx: pd.DataFrame, source_label: str = "budget_filtered") -> list:
    docs = []
    if df_ctx.empty:
        return docs

    # 1) Grouped by Department and Account_Desc (big drivers)
    if "Department" in df_ctx.columns:
        g_dept = df_ctx.groupby("Department", as_index=False).agg(
            Allocated=("Budget_Allocated","sum"),
            Spent=("Actual_Effective","sum"),
            Var=("Variance_Effective","sum")
        )
        g_dept["VarPct"] = (g_dept["Var"] / g_dept["Allocated"].replace({0:np.nan})) * 100
        g_dept["AbsVar"] = g_dept["Var"].abs()
        g_dept = g_dept.sort_values("AbsVar", ascending=False)
        for _, r in g_dept.iterrows():
            text = (f"[Department Summary]\n"
                    f"Department: {r['Department']}\n"
                    f"Allocated: {r['Allocated']:.2f}\n"
                    f"Spent: {r['Spent']:.2f}\n"
                    f"Variance: {r['Var']:.2f} ({(0 if pd.isna(r['VarPct']) else r['VarPct']):.2f}%)\n")
            docs.append(Document(page_content=text, metadata={"type":"dept", "source":source_label}))

    if "Account_Desc" in df_ctx.columns:
        g_acc = df_ctx.groupby("Account_Desc", as_index=False).agg(
            Allocated=("Budget_Allocated","sum"),
            Spent=("Actual_Effective","sum"),
            Var=("Variance_Effective","sum")
        )
        g_acc["VarPct"] = (g_acc["Var"] / g_acc["Allocated"].replace({0:np.nan})) * 100
        g_acc["AbsVar"] = g_acc["Var"].abs()
        g_acc = g_acc.sort_values("AbsVar", ascending=False)
        for _, r in g_acc.iterrows():
            text = (f"[Account Description Summary]\n"
                    f"Account_Desc: {r['Account_Desc']}\n"
                    f"Allocated: {r['Allocated']:.2f}\n"
                    f"Spent: {r['Spent']:.2f}\n"
                    f"Variance: {r['Var']:.2f} ({(0 if pd.isna(r['VarPct']) else r['VarPct']):.2f}%)\n")
            docs.append(Document(page_content=text, metadata={"type":"accdesc", "source":source_label}))

    # 2) Monthly aggregates (trend)
    g_mon = df_ctx.groupby("Month", as_index=False)[["Budget_Allocated","Actual_Effective"]].sum().sort_values("Month")
    for _, r in g_mon.iterrows():
        text = (f"[Monthly]\n"
                f"Month: {pd.to_datetime(r['Month']).strftime('%Y-%m')}\n"
                f"Allocated: {r['Budget_Allocated']:.2f}\n"
                f"Spent: {r['Actual_Effective']:.2f}\n"
                f"Variance: {(r['Actual_Effective']-r['Budget_Allocated']):.2f}\n")
        docs.append(Document(page_content=text, metadata={"type":"month", "source":source_label}))

    # 3) Top-N rows by absolute variance for grounding
    df_tmp = df_ctx.copy()
    df_tmp["_abs"] = df_tmp["Variance_Effective"].abs()
    df_tmp = df_tmp.sort_values("_abs", ascending=False).head(100).drop(columns="_abs")
    for _, r in df_tmp.iterrows():
        parts = []
        for col in ["Month","Fiscal_Year","Department","Account_Type","Account_Desc","Fund_Desc","Program_Desc","Ledger_Group"]:
            if col in df_tmp.columns and pd.notna(r.get(col, None)):
                if col == "Month":
                    parts.append(f"Month={pd.to_datetime(r['Month']).strftime('%Y-%m')}")
                else:
                    parts.append(f"{col}={r[col]}")
        parts.append(f"Allocated={r['Budget_Allocated']:.2f}")
        parts.append(f"Spent={r['Actual_Effective']:.2f}")
        parts.append(f"Variance={r['Variance_Effective']:.2f}")
        if pd.notna(r.get("Variance_Percent_Effective", np.nan)):
            parts.append(f"VariancePct={float(r['Variance_Percent_Effective']):.2f}%")
        text = "[Row] " + " | ".join(parts)
        docs.append(Document(page_content=text, metadata={"type":"row", "source":source_label}))
    return docs

def _build_index(docs: list, _api_key: str):
    if not docs or not (_has_langchain and _api_key):
        return None
    try:
        embeddings = _get_embedder(_api_key)
        if embeddings is None:
            return None
        vs = FAISS.from_documents(docs, embeddings)
        return vs
    except Exception as e:
        st.toast(f"Vector index error: {e}", icon="⚠️")
        return None

def _make_retriever_chain(vs, model_name: str, _api_key: str, k: int = 5):
    if vs is None or not _has_langchain or not _api_key:
        return None
    retriever = vs.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model_name, temperature=0.2, api_key=_api_key)

    system_template = (
        "You are a precise FP&A analyst. Use ONLY the provided context to answer.\n"
        "If the answer is not in the context, say you don't have enough data and suggest which filters to adjust.\n"
        "Return concise numeric bullets with $ and %, then a brief explanation. Call out largest drivers by absolute variance."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "Question:\n{question}\n\nContext:\n{context}")
    ])

    def format_docs(dlist):
        return "\n\n".join([d.page_content for d in dlist])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain

# =============================
# 🤖 AI — Quick Analysis & RAG Q&A
# =============================
st.markdown("---")
st.markdown(f"""<div class="ai-box"><h3>🤖 AI-Powered Insights + RAG</h3><p>Ask questions or generate a quick summary grounded in the filtered data.</p></div>""", unsafe_allow_html=True)

left, right = st.columns([1,2])

with left:
    if st.button("📈 Quick Analysis (no RAG)", use_container_width=True):
        if not client:
            st.error("OpenAI key not configured.")
        elif df_f.empty:
            st.warning("No rows in the current filters.")
        else:
            prompt = build_compact_summary(df_f) + "\nProvide 5–7 numeric bullets and a short narrative with 2–3 actions."
            ans = call_openai(
                "You are a senior FP&A analyst. Be concise, numeric, and practical. Use $ and %.",
                prompt, temperature=0.25, max_tokens=700
            )
            st.markdown(f"""<div class="ai-result">{ans}</div>""", unsafe_allow_html=True)

with right:
    q = st.text_input("💬 Ask a budget question (RAG): e.g., 'Which departments overspent most in FY2021?'")
    if q:
        if not api_key or not _has_langchain:
            st.error("RAG dependencies not available. Ensure OPENAI_API_KEY is set and langchain packages installed.")
        elif df_f.empty:
            st.warning("No rows in the current filters. Adjust filters, then ask again.")
        else:
            # Build RAG index over current filtered context
            docs = _rows_to_docs(df_f, source_label="filtered_selection")
            vs = _build_index(docs, api_key)
            chain = _make_retriever_chain(vs, llm_model_name, api_key, k=rag_top_k)
            if chain is None:
                st.error("Could not initialize the RAG chain.")
            else:
                try:
                    ans = chain.invoke(q)
                    # LangChain returns an AIMessage. Convert to string safely.
                    answer_text = getattr(ans, "content", str(ans))
                    st.markdown(f"""<div class="ai-result">{answer_text}</div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"RAG error: {e}")

# =============================
# Download
# =============================
st.markdown("---")
csv = df_f.to_csv(index=False).encode("utf-8") if not df_f.empty else b""
st.download_button("⬇️ Download filtered data (CSV)", data=csv,
                   file_name=f"budget_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                   mime="text/csv", disabled=df_f.empty)

# =============================
# Stable context signature (fixes your prior syntax error)
# =============================
# Useful if you later want to cache vector indexes per-filter:
try:
    num_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        head_sig = f"{len(df_f)}|NO_NUM_COLS"
    else:
        sums = [f"{float(pd.to_numeric(df_f[c], errors='coerce').sum()):.6f}" for c in num_cols]
        head_sig = f"{len(df_f)}|" + "|".join(sums)
    # st.caption(f"Context signature: {head_sig[:64]}...")
except Exception:
    pass
