# app.py — AI Budget Assistant (fixed path + CSV fallback, robust dtypes, clean mask)
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =============================
# Page
# =============================
st.set_page_config(page_title="AI Budget Assistant", layout="wide", initial_sidebar_state="expanded")

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
    "Executive Dark": {  # navy background + white text
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

  /* Sidebar controls: white background + dark text for legibility */
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
  .stTabs [data-baseweb="tab-list"] {{ gap:8px; }}
  .stTabs [data-baseweb="tab"] {{ background:{pal['panel']}; color:{pal['muted']}; border-radius:10px; border:1px solid {pal['grid']}; }}
  .stTabs [aria-selected="true"] {{ background:{pal['brand2']}; color:#fff; border:1px solid {pal['brand2']}; }}
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
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=api_key) if api_key else None
if not api_key:
    st.sidebar.warning("OpenAI API key not set. AI features will be disabled.")

def call_openai(system_msg: str, user_msg: str, temperature=0.2, max_tokens=900) -> str:
    if not client: return "OpenAI API key not configured."
    try:
        resp = client.chat.completions.create(
            model=openai_model,
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

    # primary dims
    "DEPARTMENT ID DESCRIPTION": ["DEPARTMENT NAME","DEPARTMENT","DEPARTMENT DESC"],
    "ACCOUNT TYPE": ["ACCT TYPE"],
    "ACCOUNT DESCRIPTION": ["ACCOUNT DESC"],

    # optional dims
    "FUND CODE DESCRIPTION": ["FUND DESCRIPTION","FUND DESC"],
    "PROGRAM DESCRIPTION": ["PROGRAM DESC"],
    "LEDGER GROUP": ["LEDGER","LEDGER GROUP NAME"],

    # optional financials
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
    # Let ImportError bubble up for .xlsx/.xls so outer block can do CSV fallback
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

    # Build rename map from ALIAS/CANON
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

    # Normalize string dims for accurate unique counts
    for dim in ["Department","Account_Desc","Account_Type","Fund_Desc","Program_Desc","Ledger_Group"]:
        if dim in df.columns:
            df[dim] = df[dim].astype(str).str.strip()

    # Types & cleaning
    df["Budget_Year"] = df["Budget_Year"].map(_parse_year)
    df["Accounting_Period"] = df["Accounting_Period"].map(_parse_period)
    df = df[(df["Accounting_Period"] >= 1) & (df["Accounting_Period"] <= 12)].copy()

    for c in ["Budget_Allocated","Actual_Spent","Encumbered","Pre_Encumbered","Revenue_Amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Fiscal (1=Oct … 12=Sep) → Calendar (YYYY-MM)
    F2C = {1:10,2:11,3:12,4:1,5:2,6:3,7:4,8:5,9:6,10:7,11:8,12:9}
    df["Cal_Month"] = df["Accounting_Period"].map(F2C)

    # prefer pandas nullable ints; fallback to plain int64
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

    # Effective measures (commitment toggle handled later)
    df["Actual_Effective"] = df["Actual_Spent"]
    df["Variance_Effective"] = df["Actual_Effective"] - df["Budget_Allocated"]
    den = df["Budget_Allocated"].replace({0: np.nan})
    df["Variance_Percent_Effective"] = (df["Variance_Effective"] / den) * 100

    return df.sort_values("Month").reset_index(drop=True)

# =============================
# Load Data (single fixed path with CSV fallback if Excel engine missing)
# =============================
ACTUALS_PATH = "FY 2021 Budget Pull.xlsx"  # put this exact file at repo root

try:
    df = load_budget_pull(ACTUALS_PATH)
except ImportError as e:
    # Fallback to CSV of the same base name if openpyxl/xlrd is missing
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
  <h1>AI Budget Forecast & Analysis</h1>
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

# =============================
# Apply filters (readable, no dangling parentheses)
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
# Visual Analytics (Monthly, Department, Account Description)
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
# 🤖 AI — Quick Analysis & Q&A (optional)
# =============================
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

with cB:
    q = st.text_input("💬 Ask a question (e.g., 'Which departments overspent most in FY2021?')")
    if q:
        if not client:
            st.error("OpenAI key not configured.")
        elif df_f.empty:
            st.warning("No rows in the current filters. Adjust filters, then ask again.")
        else:
            ql = q.lower()
            prefer = "Department" if ("department" in ql or "dept" in ql) else (
                     "Account_Desc" if ("account description" in ql or "account" in ql) else None)

            df_q = df_f.copy()

            # Try to match a specific Department or Account_Desc mentioned in the question
            def extract_match(text, options):
                tl = str(text).lower()
                for opt in options:
                    if str(opt).lower() in tl: return opt
                return None

            dept_options = sorted(pd.Series(df["Department"]).dropna().astype(str).unique().tolist()) if "Department" in df.columns else []
            accd_options = sorted(pd.Series(df["Account_Desc"]).dropna().astype(str).unique().tolist()) if "Account_Desc" in df.columns else []

            hit_dept = extract_match(q, dept_options)
            hit_accd = extract_match(q, accd_options)
            if hit_dept is not None and "Department" in df_q.columns:
                df_q = df_q[df_q["Department"] == hit_dept]
                if prefer == "Department": prefer = None
            if hit_accd is not None and "Account_Desc" in df_q.columns:
                df_q = df_q[df_q["Account_Desc"] == hit_accd]
                if prefer == "Account_Desc": prefer = None

            # Aggregate-first if a preferred dimension exists
            if prefer and prefer in df_q.columns and not df_q.empty:
                agg = df_q.groupby(prefer, as_index=False).agg(
                    Allocated=("Budget_Allocated","sum"),
                    Spent=("Actual_Effective","sum")
                )
                agg["Variance"] = agg["Spent"] - agg["Allocated"]
                agg["AbsVar"] = agg["Variance"].abs()
                agg = agg.sort_values("AbsVar", ascending=False)
                k = min(ai_rows_cap, len(agg))
                send = agg.head(k).drop(columns=["AbsVar"]).rename(columns={prefer:"Group"})
                primary = send.to_string(index=False)
                coverage = f"(Top {k} {prefer} by |variance|; {len(agg)} groups total)"
            else:
                cols = ["Month","Department","Account_Desc","Budget_Allocated","Actual_Effective","Variance_Effective","Variance_Percent_Effective"]
                cols = [c for c in cols if c in df_q.columns]
                slim = df_q[cols].copy() if not df_q.empty else pd.DataFrame(columns=cols)
                if not slim.empty and "Variance_Effective" in slim.columns:
                    slim["_abs"] = slim["Variance_Effective"].abs()
                    slim = slim.sort_values("_abs", ascending=False).drop(columns="_abs")
                primary = slim.head(min(ai_rows_cap, 800)).to_string(index=False) if not slim.empty else "(no matching rows)"
                coverage = ""

            compact = build_compact_summary(df_f)
            final = f"""User question: {q}

Primary context:
{primary}
{coverage}

Additional summaries:
{compact}

Instructions:
- Use the primary context first; otherwise use the summaries.
- Provide concise, numeric bullets and call out the biggest drivers by absolute variance.
- Scope to current filters; if more precision needs a narrower filter, say exactly what to filter.
"""
            ans = call_openai("You are a precise financial analyst. Be concise and numeric.", final, temperature=0.2, max_tokens=900)
            st.markdown(f"""<div class="ai-result">{ans}</div>""", unsafe_allow_html=True)

# =============================
# Download
# =============================
st.markdown("---")
csv = df_f.to_csv(index=False).encode("utf-8") if not df_f.empty else b""
st.download_button("⬇️ Download filtered data (CSV)", data=csv,
                   file_name=f"budget_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                   mime="text/csv", disabled=df_f.empty)
