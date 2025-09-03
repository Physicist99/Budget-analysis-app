# app.py — AI Budget Assistant + LangChain RAG (KB + Data index)
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ==== LangChain / RAG ====
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Vector stores (FAISS preferred, Chroma fallback)
try:
    from langchain_community.vectorstores import FAISS
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
from langchain_community.vectorstores import Chroma

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="AI Budget Assistant (RAG)", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Themes (Executive Light/Dark)
# -----------------------------
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
    theme_name = st.selectbox("Select", list(THEMES.keys()), index=1)
pal = THEMES[theme_name]
alloc_col, spent_col = pal["alloc"], pal["spent"]

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

# -----------------------------
# CSS
# -----------------------------
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

# -----------------------------
# OpenAI client (optional – used by “Quick Analysis”)
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=api_key) if api_key else None
if not api_key:
    st.sidebar.warning("OpenAI API key not set. AI features will be disabled.")

def call_openai(system_msg: str, user_msg: str, temperature=0.2, max_tokens=900) -> str:
    if not client:
        return "OpenAI API key not configured."
    try:
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
            temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

# -----------------------------
# Safe formatters
# -----------------------------
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

# -----------------------------
# Column mapping & parsers (same as your original)
# -----------------------------
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
    if pd.isna(val): return pd.NA
    s = str(val).strip()
    m = re.match(r"^(20\d{2})[-/](0?[1-9]|1[0-2])$", s)
    if m: return int(m.group(2))
    mon_map = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"SEPT":9,"OCT":10,"NOV":11,"DEC":12}
    up = s.upper()
    if up.startswith("SEPT"): return 9
    if up[:3] in mon_map: return mon_map[up[:3]]
    return pd.to_numeric(s, errors="coerce")

# -----------------------------
# Loader
# -----------------------------
@st.cache_data(show_spinner=True)
def load_budget_pull(path: str) -> pd.DataFrame:
    raw = _read_any(path)

    lookup = {}
    for k, syns in ALIAS.items():
        lookup[_n(k)] = k
        for s in syns: lookup[_n(s)] = k

    rename_map = {}
    for c in raw.columns:
        if _n(c) in lookup:
            rename_map[c] = CANON[lookup[_n(c)]]

    df = raw.rename(columns=rename_map).copy()

    for req in ["Budget_Year","Accounting_Period","Budget_Allocated","Actual_Spent"]:
        if req not in df.columns:
            raise ValueError(f"Missing required column: {req}")

    for k in ["Department","Account_Desc","Account_Type","Fund_Desc","Program_Desc","Ledger_Group",
              "Encumbered","Pre_Encumbered","Revenue_Amount"]:
        if k not in df.columns:
            df[k] = 0.0 if k in ["Encumbered","Pre_Encumbered","Revenue_Amount"] else np.nan

    for dim in ["Department","Account_Desc","Account_Type","Fund_Desc","Program_Desc","Ledger_Group"]:
        if dim in df.columns:
            df[dim] = df[dim].astype(str).str.strip()

    df["Budget_Year"] = df["Budget_Year"].map(_parse_year)
    df["Accounting_Period"] = df["Accounting_Period"].map(_parse_period)
    df = df[(df["Accounting_Period"] >= 1) & (df["Accounting_Period"] <= 12)].copy()

    for c in ["Budget_Allocated","Actual_Spent","Encumbered","Pre_Encumbered","Revenue_Amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

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

    df["Year"] = df["Month"].dt.year
    df["Quarter"] = df["Month"].dt.quarter
    df["Fiscal_Year"] = df["Budget_Year"]
    df["Fiscal_Quarter"] = (((pd.Series(df["Cal_Month"]) - 10) % 12) // 3 + 1)

    df["Actual_Effective"] = df["Actual_Spent"]
    df["Variance_Effective"] = df["Actual_Effective"] - df["Budget_Allocated"]
    den = df["Budget_Allocated"].replace({0: np.nan})
    df["Variance_Percent_Effective"] = (df["Variance_Effective"] / den) * 100

    return df.sort_values("Month").reset_index(drop=True)

# -----------------------------
# Load Data (Excel with CSV fallback)
# -----------------------------
ACTUALS_PATH = "FY 2021 Budget Pull.xlsx"
try:
    df = load_budget_pull(ACTUALS_PATH)
except ImportError as e:
    base, _ = os.path.splitext(ACTUALS_PATH)
    csv_path = base + ".csv"
    if os.path.exists(csv_path):
        df = load_budget_pull(csv_path)
    else:
        st.error("❌ Excel engine missing. Install `openpyxl` (xlsx) / `xlrd` (xls), or provide a CSV of same base name.\n\n"
                 f"Details: {e}")
        st.stop()
except Exception as e:
    st.error(f"❌ Could not load data at {ACTUALS_PATH}: {e}")
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.markdown(f"""
<div class="main-header">
  <h1>AI Budget Forecast & Analysis — with RAG</h1>
  <p style="opacity:.9">Professional analytics with fiscal-year calendarization (1=Oct … 12=Sep) + knowledge-aware Q&A</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Filters
# -----------------------------
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
    ai_rows_cap = st.slider("AI context row cap (after aggregation)", min_value=100, max_value=3000, value=600, step=100)

    st.markdown('<div class="filter-header">📚 Knowledge Base</div>', unsafe_allow_html=True)
    kb_dir = st.text_input("KB folder (PDF/TXT/MD/CSV/XLSX)", value="knowledge_base")
    kb_k = st.slider("Retriever k (KB)", 2, 15, 5)
    include_df_in_rag = st.checkbox("Also index current filtered data in RAG", value=True)
    df_k = st.slider("Retriever k (Data)", 2, 15, 5, disabled=not include_df_in_rag)
    if st.button("🔁 Reindex KB"):
        st.cache_resource.clear()

# -----------------------------
# Apply filters
# -----------------------------
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

# -----------------------------
# KPIs
# -----------------------------
st.markdown("### 🔎 Overview")
if df_f.empty:
    st.markdown(
        f"""
        <div style="padding:1rem;border:1px solid {pal['warn']};border-radius:10px;background:rgba(245,158,11,.1);">
          <b>⚠️ No data matches your current filters.</b><br>
          Try resetting one or more filters or expand the date range.
        </div>
        """, unsafe_allow_html=True
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

# -----------------------------
# Details table
# -----------------------------
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

# -----------------------------
# Visual Analytics
# -----------------------------
st.markdown("---")
st.subheader("📈 Visual Analytics")
t1, t2, t3 = st.tabs(["📊 Monthly Trend", "🏢 By Department", "🧾 By Account Description"])

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

# -----------------------------
# 🔎 Utility for AI section
# -----------------------------
def tbl(df_):
    return "(none)" if df_.empty else df_.to_string(index=False)

def build_compact_summary(actuals_df: pd.DataFrame) -> str:
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

# -----------------------------
# 🤖 AI — Quick Analysis (unchanged)
# -----------------------------
st.markdown("---")
st.markdown(f"""<div class="ai-box"><h3>🤖 AI-Powered Insights</h3><p>Ask questions or generate a quick summary.</p></div>""", unsafe_allow_html=True)
cA, cB = st.columns([1,2])
with cA:
    if st.button("📈 Quick Analysis", use_container_width=True):
        if not client:
            st.error("OpenAI key not configured.")
        elif df_f.empty:
            st.warning("No rows in the current filters. Adjust filters, then try again.")
        else:
            prompt = build_compact_summary(df_f) + "\nProvide 5–7 numeric bullets and a short narrative with 2–3 actions."
            ans = call_openai(
                "You are a senior FP&A analyst. Be concise, numeric, and practical. Use $ and %.",
                prompt, temperature=0.25, max_tokens=700
            )
            st.markdown(f"""<div class="ai-result">{ans}</div>""", unsafe_allow_html=True)

# =========================================================
#               RAG: Knowledge Base + Data
# =========================================================
st.markdown("---")
st.subheader("🧠 RAG Knowledge & Data Q&A (LangChain)")

# --- helpers to build vector stores ---
def dir_signature(path: str) -> Tuple[str, int]:
    """Return a stable signature (concat of names+mtimes+sizes) and file count for caching."""
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return ("", 0)
    sig_parts = []
    n = 0
    for f in sorted(p.rglob("*")):
        if f.is_file():
            try:
                stat = f.stat()
                sig_parts.append(f"{f.name}:{int(stat.st_mtime)}:{stat.st_size}")
                n += 1
            except Exception:
                continue
    return ("|".join(sig_parts), n)

def read_text_file(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return fp.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""

def pandas_preview_text(fp: Path, limit_rows: int = 2000) -> str:
    try:
        if fp.suffix.lower() in [".xlsx", ".xls"]:
            dfc = pd.read_excel(fp)
        else:
            dfc = pd.read_csv(fp)
        # limit to a reasonable number of rows for embeddings
        if len(dfc) > limit_rows:
            dfc = dfc.head(limit_rows).copy()
        # string preview with schema at top
        schema = ", ".join([f"{c}({str(dfc[c].dtype)})" for c in dfc.columns])
        return f"FILE: {fp.name}\nSCHEMA: {schema}\nROWS:\n{dfc.to_csv(index=False)}"
    except Exception:
        # fallback: raw text read
        return read_text_file(fp)

def collect_documents_from_folder(folder: str) -> List[Document]:
    docs: List[Document] = []
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return docs
    for f in sorted(p.rglob("*")):
        if not f.is_file(): 
            continue
        ext = f.suffix.lower()
        if ext in [".pdf"]:
            try:
                from langchain_community.document_loaders import PyPDFLoader
                for page in PyPDFLoader(str(f)).load():
                    page.metadata["source"] = str(f)
                    docs.append(page)
            except Exception:
                # fallback: read raw text if PDF loader fails
                content = read_text_file(f)
                if content:
                    docs.append(Document(page_content=content, metadata={"source": str(f)}))
        elif ext in [".txt", ".md"]:
            content = read_text_file(f)
            if content:
                docs.append(Document(page_content=content, metadata={"source": str(f)}))
        elif ext in [".csv", ".xlsx", ".xls"]:
            content = pandas_preview_text(f)
            if content:
                docs.append(Document(page_content=content, metadata={"source": str(f)}))
        # ignore others by default to keep dependencies light
    return docs

def chunk_docs(docs: List[Document], chunk_size=1200, chunk_overlap=150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

@st.cache_resource(show_spinner=True)
def build_kb_vectorstore(folder: str, sig: str):
    if not api_key:
        raise RuntimeError("OpenAI API key not configured.")
    base_docs = collect_documents_from_folder(folder)
    if not base_docs:
        return None, 0
    chunks = chunk_docs(base_docs)
    embed = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    if _HAS_FAISS:
        vs = FAISS.from_documents(chunks, embed)
    else:
        vs = Chroma.from_documents(chunks, embed, collection_name="budget_kb")
    return vs, len(chunks)

def df_to_docs(df_: pd.DataFrame, cap_rows=600) -> List[Document]:
    """Convert filtered dataframe to a few rich textual docs for retrieval."""
    if df_.empty:
        return []
    # Prioritize rows by absolute variance
    cols = ["Month","Department","Account_Desc","Budget_Allocated","Actual_Effective","Variance_Effective","Variance_Percent_Effective"]
    cols = [c for c in cols if c in df_.columns]
    slim = df_[cols].copy()
    if "Variance_Effective" in slim.columns:
        slim["_abs"] = slim["Variance_Effective"].abs()
        slim = slim.sort_values("_abs", ascending=False).drop(columns="_abs")
    slim = slim.head(cap_rows).copy()
    # summary blocks
    txt = "FILTERED BUDGET ROWS (top by |variance|):\n" + slim.to_csv(index=False)
    return [Document(page_content=txt, metadata={"source": "filtered_budget_rows"})]

@st.cache_resource(show_spinner=True)
def build_df_vectorstore(df_sig: str, as_docs: List[Document]):
    if not api_key:
        raise RuntimeError("OpenAI API key not configured.")
    if not as_docs:
        return None
    embed = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    if _HAS_FAISS:
        return FAISS.from_documents(as_docs, embed)
    else:
        return Chroma.from_documents(as_docs, embed, collection_name="budget_data")

def get_kb_retriever(folder: str, k: int):
    sig, count = dir_signature(folder)
    if count == 0:
        return None, 0
    vs, n_chunks = build_kb_vectorstore(folder, sig)
    if vs is None:
        return None, 0
    return vs.as_retriever(search_kwargs={"k": k}), n_chunks

def get_df_retriever(df_ctx: pd.DataFrame, k: int, cap_rows=600):
    # build a simple signature of data for cache: size + head hash
    if df_ctx.empty:
        return None
    head_sig = str(len(df_ctx)) + "|" + "|".join([str(df_ctx[c].sum()) for c in df_ctx.select_d
