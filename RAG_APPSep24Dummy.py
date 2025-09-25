# -*- coding: utf-8 -*-
# AI Budget Analysis (Streamlit)

import os
import io
import re
import math
import json
import hashlib
from datetime import datetime, date
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from pandas.tseries.offsets import MonthBegin

# OpenAI (new v1+ SDK)
from openai import OpenAI, APIError, AuthenticationError
from openai import APIConnectionError, RateLimitError, APIStatusError

# --- Try LangChain imports (optional) ---
try:
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    _has_langchain = True
except Exception:
    Document = ChatPromptTemplate = RunnablePassthrough = None
    ChatOpenAI = OpenAIEmbeddings = FAISS = RetrievalQA = None
    _has_langchain = False

# =============================
# Page / Theme
# =============================
st.set_page_config(page_title="Budget RAG Assistant", layout="wide", initial_sidebar_state="expanded")

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

# =============================
# Reset / Session keys
# =============================
RESET_KEYS = [
    "dept_sel", "accd_sel", "atype_sel", "fy_sel",
    "include_commitments", "date_range", "variance_range", "amount_range",
    "ai_rows_cap", "rag_top_k", "chain_style", "show_ctx", "_data_sig"
]

def reset_filters(keys=None):
    keys = keys or RESET_KEYS
    for k in keys:
        st.session_state.pop(k, None)
    st.rerun()

# =============================
# OpenAI setup (robust)
# =============================
load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
llm_model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str) -> Optional[OpenAI]:
    if not api_key:
        st.warning("⚠️ No OPENAI_API_KEY set. AI features disabled.")
        return None
    try:
        return OpenAI(api_key=api_key)
    except AuthenticationError:
        st.error("❌ Invalid OpenAI API key.")
    except (APIConnectionError, RateLimitError, APIStatusError, APIError) as e:
        st.error(f"❌ OpenAI init error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected OpenAI init error: {e}")
    return None

_oai_client = get_openai_client(_api_key)

def call_openai(system_msg: str, user_msg: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
    if not _oai_client:
        return "⚠️ OpenAI API not available."
    try:
        resp = _oai_client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except (APIConnectionError, RateLimitError, APIStatusError, APIError) as e:
        return f"⚠️ OpenAI API error: {e}"
    except Exception as e:
        return f"⚠️ Unexpected error: {e}"

# =============================
# Helpers / Parsers
# =============================

# Nullable-int compatibility shim (works on old/new pandas)
try:
    _NULLABLE_INT = pd.Int64Dtype()
except Exception:
    _NULLABLE_INT = None

def to_nullable_int(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if _NULLABLE_INT is not None:
        try:
            return s.astype(_NULLABLE_INT)
        except Exception:
            pass
    return s

def money_fmt(x: Any, digits: int = 0) -> str:
    try:
        xf = float(str(x).replace(",", "").replace("$", ""))
        if not np.isfinite(xf): return "—"
        return f"${xf:,.{digits}f}"
    except Exception:
        return "—"

def pct_fmt(x: Any, digits: int = 1, signed: bool = True) -> str:
    try:
        xf = float(x)
        if not np.isfinite(xf): return "—"
        sign = "+" if signed else ""
        return f"{xf:{sign}.{digits}f}%"
    except Exception:
        return "—"

def _n(x: Any) -> str:
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

def _read_any(path: str, **read_kwargs) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith((".xlsx",".xls")):
        kw = {"engine": "openpyxl"} | read_kwargs
        return pd.read_excel(path, **kw)
    kw = {"on_bad_lines": "skip"} | read_kwargs
    return pd.read_csv(path, **kw)

NAType = type(pd.NA)

def _parse_year(val: Any) -> int | NAType:
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        y = int(m.group(0))
        return y if 1900 <= y <= 2100 else pd.NA
    m = re.search(r"FY[’']?\s*(\d{2})", s, flags=re.IGNORECASE)
    if m:
        two = int(m.group(1))
        y = 2000 + two if two <= 30 else 1900 + two
        return y if 1900 <= y <= 2100 else pd.NA
    try:
        y = int(float(s))
        return y if 1900 <= y <= 2100 else pd.NA
    except Exception:
        return pd.NA

def _parse_period(val: Any) -> int | NAType:
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    m = re.search(r"(?<!\d)(20\d{2})[-/](0?[1-9]|1[0-2])(?!\d)", s)
    if m:
        return int(m.group(2))
    m = re.search(r"(?<!\d)(0?[1-9]|1[0-2])[-/](20\d{2})(?!\d)", s)
    if m:
        return int(m.group(1))
    mon_map = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"SEPT":9,"OCT":10,"NOV":11,"DEC":12}
    up = s.upper()
    if up.startswith("SEPT"): return 9
    if up[:3] in mon_map: return mon_map[up[:3]]
    x = pd.to_numeric(s, errors="coerce")
    if pd.isna(x): return pd.NA
    mnum = int(x)
    return mnum if 1 <= mnum <= 12 else pd.NA

def _coerce_money(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    out = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return out

def _build_alias_lookup() -> Dict[str, str]:
    lookup = {}
    for k, syns in ALIAS.items():
        lookup[_n(k)] = k
        for s in syns:
            lookup[_n(s)] = k
    return lookup

def _fiscal_maps(fiscal_start_month: int = 10) -> Tuple[Dict[int,int], Dict[int,int]]:
    f2c = {}
    for f in range(1, 13):
        cm = ((f - 1 + (fiscal_start_month - 1)) % 12) + 1
        f2c[f] = cm
    f2q = {m: ((m - 1) // 3) + 1 for m in range(1, 13)}
    return f2c, f2q

@st.cache_data(show_spinner=True)
def load_budget_pull(path: str,
                     fiscal_start_month: int = 10,
                     read_kwargs: Optional[Dict[str, Any]] = None
                     ) -> pd.DataFrame:
    read_kwargs = read_kwargs or {}
    raw = _read_any(path, **read_kwargs)

    # Rename via alias map
    lookup = _build_alias_lookup()
    rename_map = {}
    for c in raw.columns:
        key = _n(c)
        if key in lookup:
            human = lookup[key]
            rename_map[c] = CANON[human]
    df = raw.rename(columns=rename_map).copy()

    # Require core fields
    required = ["Budget_Year","Accounting_Period","Budget_Allocated","Actual_Spent"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        sample = list(raw.columns)[:15]
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Known headers in file (sample): {sample}. "
            f"Check aliases for typos."
        )

    # Optional columns with defaults
    optional_dims = ["Department","Account_Desc","Account_Type","Fund_Desc","Program_Desc","Ledger_Group"]
    optional_meas = ["Encumbered","Pre_Encumbered","Revenue_Amount"]
    for k in optional_dims:
        if k not in df.columns:
            df[k] = pd.NA
    for k in optional_meas:
        if k not in df.columns:
            df[k] = 0.0

    # Normalize strings
    for dim in optional_dims:
        if dim in df.columns:
            df[dim] = df[dim].astype(str).str.strip().astype("category")

    # Year/period parsing and filter 1..12
    df["Budget_Year"] = df["Budget_Year"].map(_parse_year).pipe(to_nullable_int)
    df["Accounting_Period"] = df["Accounting_Period"].map(_parse_period).pipe(to_nullable_int)
    df = df[df["Accounting_Period"].between(1, 12, inclusive="both")].copy()

    # Money coercion
    for c in ["Budget_Allocated","Actual_Spent","Encumbered","Pre_Encumbered","Revenue_Amount"]:
        df[c] = _coerce_money(df[c])

    # Fiscal → Calendar month
    F2C, _ = _fiscal_maps(fiscal_start_month)
    df["Cal_Month"] = df["Accounting_Period"].map(F2C).pipe(to_nullable_int)

    # Calendar year: months >= pivot belong to previous calendar year
    pivot = fiscal_start_month
    df["Cal_Year"] = to_nullable_int(
        df["Budget_Year"].where(
            df["Cal_Month"].lt(pivot) | df["Cal_Month"].isna(),
            df["Budget_Year"] - 1
        )
    )

    # Month date and drop invalids
    df["Month"] = pd.to_datetime(
        df["Cal_Year"].astype(str) + "-" + df["Cal_Month"].astype(str).str.zfill(2) + "-01",
        errors="coerce"
    )
    df = df.dropna(subset=["Month"]).copy()

    # Time helpers
    df["Year"] = df["Month"].dt.year
    df["Quarter"] = df["Month"].dt.quarter
    df["Fiscal_Year"] = df["Budget_Year"]
    df["Fiscal_Quarter"] = to_nullable_int(((pd.to_numeric(df["Cal_Month"], errors="coerce") - pivot) % 12) // 3 + 1)

    # Effective measures
    df["Actual_Effective"] = df["Actual_Spent"]
    df["Variance_Effective"] = df["Actual_Effective"] - df["Budget_Allocated"]
    den = df["Budget_Allocated"].replace({0: np.nan})
    df["Variance_Percent_Effective"] = (df["Variance_Effective"] / den) * 100

    return df.sort_values("Month").reset_index(drop=True)

# =============================
# Load Data (fixed path + CSV fallback)
# =============================
ACTUALS_PATH = "FY 2021 Budget Pull -DUMMY.xlsx"  # change as needed
try:
    df = load_budget_pull(ACTUALS_PATH)
except ImportError as e:
    base, _ = os.path.splitext(ACTUALS_PATH)
    csv_path = base + ".csv"
    if os.path.exists(csv_path):
        df = load_budget_pull(csv_path)
    else:
        st.error("❌ Excel engine missing. Install `openpyxl` (xlsx) / `xlrd` (xls), or place a CSV with same base name.\n\n"
                 f"Details: {e}")
        st.stop()
except Exception as e:
    st.error(f"❌ Could not load data at {ACTUALS_PATH}: {e}")
    st.stop()

# =============================
# State init (run before widgets)
# =============================
def _safe_percentile(s: pd.Series, lo: float, hi: float, fallback: Tuple[float, float]) -> Tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return fallback
    lo_v = float(np.nanpercentile(s, lo))
    hi_v = float(np.nanpercentile(s, hi))
    if not np.isfinite(lo_v) or not np.isfinite(hi_v):
        return fallback
    if lo_v == hi_v:
        pad = abs(lo_v) * 0.05 + 1.0
        return lo_v - pad, hi_v + pad
    return lo_v, hi_v

def _nice_bounds(lo: float, hi: float, step_hint: float) -> Tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return lo, hi
    if lo > hi:
        lo, hi = hi, lo
    lo_n = math.floor(lo / step_hint) * step_hint
    hi_n = math.ceil(hi / step_hint) * step_hint
    return lo_n, hi_n

def _amount_step_hint(max_abs: float) -> float:
    if max_abs <= 0:
        return 100.0
    exp = max(0, int(math.floor(math.log10(max_abs))) - 2)
    base = 10 ** exp
    for m in (1, 2, 5, 10):
        if max_abs / (base * m) <= 200:
            return base * m
    return base * 10

def _dataset_signature(_df: pd.DataFrame) -> Tuple[str, str, int]:
    mmin = pd.to_datetime(_df["Month"].min(), errors="coerce")
    mmax = pd.to_datetime(_df["Month"].max(), errors="coerce")
    return (str(mmin.date()) if pd.notna(mmin) else "NaT",
            str(mmax.date()) if pd.notna(mmax) else "NaT",
            int(len(_df)))

def _clamp_range(cur: Tuple, lo, hi, is_date=False) -> Tuple:
    if cur is None: return (lo, hi)
    a, b = cur
    if is_date:
        a = max(min(a, hi), lo)
        b = max(min(b, hi), lo)
    else:
        a = float(np.clip(a, lo, hi))
        b = float(np.clip(b, lo, hi))
    if a > b: a, b = b, a
    return (a, b)

def init_filter_state(_df: pd.DataFrame, force: bool = False) -> None:
    required = ["Month", "Variance_Percent_Effective", "Actual_Effective"]
    missing = [c for c in required if c not in _df.columns]
    if missing:
        raise ValueError(f"init_filter_state: required columns missing: {missing}")

    mmin = pd.to_datetime(_df["Month"].min(), errors="coerce")
    mmax = pd.to_datetime(_df["Month"].max(), errors="coerce")
    if pd.isna(mmin) or pd.isna(mmax):
        today = pd.Timestamp("today").normalize().date()
        date_default = (today, today)
    else:
        date_default = (mmin.date(), mmax.date())

    v_lo, v_hi = _safe_percentile(_df["Variance_Percent_Effective"], 1, 99, (-100.0, 100.0))
    v_lo, v_hi = _nice_bounds(v_lo, v_hi, step_hint=5.0)

    a_lo, a_hi = _safe_percentile(_df["Actual_Effective"], 1, 99, (0.0, 0.0))
    step = _amount_step_hint(max(abs(a_lo), abs(a_hi)))
    a_lo, a_hi = _nice_bounds(a_lo, a_hi, step_hint=step)

    defaults = {
        "dept_sel": [],
        "accd_sel": [],
        "atype_sel": [],
        "fy_sel": [],
        "include_commitments": False,
        "date_range": date_default,
        "variance_range": (v_lo, v_hi),
        "amount_range": (a_lo, a_hi),
        "ai_rows_cap": 400,
        "rag_top_k": 5,
        "chain_style": "LCEL (modern)",
        "show_ctx": False,
    }

    if force:
        st.session_state.update(defaults)
    else:
        for k, v in defaults.items():
            st.session_state.setdefault(k, v)
        st.session_state["date_range"] = _clamp_range(
            st.session_state.get("date_range"), defaults["date_range"][0], defaults["date_range"][1], is_date=True
        )
        st.session_state["variance_range"] = _clamp_range(
            st.session_state.get("variance_range"), v_lo, v_hi
        )
        st.session_state["amount_range"] = _clamp_range(
            st.session_state.get("amount_range"), a_lo, a_hi
        )

_data_sig = _dataset_signature(df)
if st.session_state.get("_data_sig") != _data_sig:
    init_filter_state(df, force=True)
    st.session_state["_data_sig"] = _data_sig
else:
    init_filter_state(df, force=False)

# =============================
# Header
# =============================
st.markdown(f"""
<div class="main-header">
  <h1>AI Budget Analysis</h1>
  <p style="font-size: 1.5rem; margin: 0.5rem 0;">Advanced Financial Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# =============================
# Sidebar (Reset button FIRST, then widgets)
# =============================
with st.sidebar:
    if st.button("🔄 Reset filters to full range"):
        reset_filters()

    st.markdown("**🎨 Theme**")
    theme_name = st.selectbox("Select", list(THEMES.keys()), index=1)  # default Dark

pal = THEMES[theme_name]
alloc_col, spent_col = pal["alloc"], pal["spent"]

# Altair theme based on palette
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
# Global / Theme CSS (single source of truth)
# =============================
# Global CSS using current palette
st.markdown(f"""
<style>
  .stApp {{ background:{pal['bg']}; color:{pal['text']}; }}
  .block-container {{ padding-top: 0.75rem; max-width: 100% !important; padding-left: 1.25rem; padding-right: 1.25rem; }}
  [data-testid="stSidebar"] > div:first-child {{ background:{pal['sidebar']}; }}
  [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] p {{ color:#E5E7EB !important; }}

  .main-header {{
    width: 100%;
    margin: 0 auto 1.5rem auto;
    text-align: center;
    background: linear-gradient(135deg, {pal['brand1']} 0%, {pal['brand2']} 100%);
    padding: 4rem 1.5rem 2.5rem 1.5rem;
    border-radius: 14px;
    color: #fff;
  }}
  .main-header h1 {{
    margin: 0; line-height: 1.3; font-size: clamp(40px, 4.5vw, 64px);
    font-weight:700; color: #fff;
  }}
  .main-header p {{ margin: 1rem 0 0 0; font-size: clamp(18px, 2vw, 24px); color: #fff; }}
</style>
""", unsafe_allow_html=True)

# Executive Dark: consolidated tweaks (buttons, inputs, metrics, tabs, tables, links, charts)
if theme_name == "Executive Dark":
    st.markdown(
        f"""
        <style>
        /* ===== Buttons (main & sidebar, incl. download) ===== */
        .stButton > button, .stDownloadButton > button {{
          background: {pal['brand2']} !important;
          color: #FFFFFF !important;
          border: 1px solid rgba(255,255,255,0.30) !important;
          border-radius: 12px !important;
          font-weight: 600 !important;
          box-shadow: 0 2px 10px rgba(0,0,0,0.25) !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
          filter: brightness(1.08) !important;
          transform: translateY(-1px);
        }}
        .stButton > button:focus, .stDownloadButton > button:focus {{
          outline: 2px solid #FFFFFF !important;
          outline-offset: 2px !important;
        }}
        .stButton > button:disabled, .stDownloadButton > button:disabled {{
          background: rgba(255,255,255,0.18) !important;
          color: rgba(255,255,255,0.85) !important;
          border-color: rgba(255,255,255,0.28) !important;
          opacity: 1 !important;
        }}
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] .stDownloadButton > button {{
          width: 100% !important;
        }}

        /* ===== Metric cards ===== */
        [data-testid="stMetric"] {{
          background: rgba(255,255,255,0.05);
          border-radius: 12px;
          padding: .75rem .9rem;
        }}
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] p {{
          color: #FFFFFF !important;
        }}

        /* ===== Tabs ===== */
        [data-testid="stTabs"] [role="tab"] {{
          color: #FFFFFF !important;
        }}
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
          color: #FFFFFF !important;
          border-bottom: 2px solid {pal['brand2']} !important;
        }}

        /* ===== Labels and general text ===== */
        label, .stMarkdown, .stCaption, .stText {{
          color: #FFFFFF !important;
        }}

        /* ===== Main text input (RAG): dark text on light input ===== */
        .block-container div[data-testid="stTextInput"] input,
        .block-container div[data-baseweb="input"] input {{
          color: #0F172A !important;               /* dark typing color */
          background: #FFFFFF !important;          /* light background for contrast */
          border: 1px solid rgba(0,0,0,0.35) !important;
          border-radius: 10px !important;
          caret-color: #0F172A !important;
        }}
        .block-container div[data-testid="stTextInput"] input::placeholder,
        .block-container div[data-baseweb="input"] input::placeholder {{
          color: rgba(15,23,42,0.60) !important;   /* darker placeholder */
        }}

        /* ===== Sidebar selects/multiselects: act like dropdowns ===== */
        [data-testid="stSidebar"] .stMultiSelect [role="combobox"] input,
        [data-testid="stSidebar"] .stSelectbox  [role="combobox"] input {{
          color: transparent !important;
          caret-color: transparent !important;
          width: 0 !important;
          min-width: 0 !important;
          opacity: 0 !important;
        }}
        [data-testid="stSidebar"] .stMultiSelect [role="combobox"],
        [data-testid="stSidebar"] .stSelectbox  [role="combobox"] {{
          background: rgba(255,255,255,0.10) !important;
          border: 1px solid rgba(255,255,255,0.35) !important;
          border-radius: 10px !important;
        }}

        /* ===== Dataframe/table text ===== */
        .stDataFrame, .stDataFrame table, .stDataFrame th, .stDataFrame td {{
          color: #FFFFFF !important;
        }}

        /* ===== Links ===== */
        a {{ color: #C7D2FE !important; }}

        /* ===== Altair/Vega fallback: chart text white ===== */
        .vega-embed svg text {{ fill: #FFFFFF !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
if theme_name == "Executive Dark":
    st.markdown("""
    <style>
      /* Force white text for the AI & RAG checkbox (sidebar + main) */
      [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
      [data-testid="stSidebar"] [data-testid="stCheckbox"] label > div,
      [data-testid="stSidebar"] [data-testid="stCheckbox"] span,
      [data-testid="stSidebar"] [data-testid="stCheckbox"] p,
      [data-testid="stCheckbox"] label,
      [data-testid="stCheckbox"] label > div,
      [data-testid="stCheckbox"] span,
      [data-testid="stCheckbox"] p {
        color: #FFFFFF !important;
      }

      /* Optional: improve checkbox outline in dark mode */
      [data-testid="stCheckbox"] div[role="checkbox"] {
        border-color: rgba(255,255,255,0.6) !important;
      }
    </style>
    """, unsafe_allow_html=True)

# Sidebar text size tweaks
st.markdown("""
<style>
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] p {
      font-size: 1.05rem !important;
  }
  [data-testid="stSidebar"] .stSelectbox div,
  [data-testid="stSidebar"] .stMultiSelect div,
  [data-testid="stSidebar"] .stSlider div {
      font-size: 1.05rem !important;
  }
  .filter-header {
      font-size: 1.15rem !important;
      font-weight: 700 !important;
  }
</style>
""", unsafe_allow_html=True)

# =============================
# Primary Filters
with st.sidebar:
    st.markdown('<div class="filter-header">🔍 Primary Filters</div>', unsafe_allow_html=True)

    # -- Ensure defaults exist BEFORE creating widgets
    st.session_state.setdefault("dept_sel", [])
    st.session_state.setdefault("accd_sel", [])
    st.session_state.setdefault("atype_sel", [])
    st.session_state.setdefault("fy_sel", [])
    st.session_state.setdefault("include_commitments", False)

    # Options
    dept_options = sorted(pd.Series(df["Department"]).dropna().astype(str).unique().tolist()) if "Department" in df.columns else []
    accd_options = sorted(pd.Series(df["Account_Desc"]).dropna().astype(str).unique().tolist()) if "Account_Desc" in df.columns else []
    atype_options = sorted(pd.Series(df["Account_Type"]).dropna().astype(str).unique().tolist()) if "Account_Type" in df.columns else []

    # -- Remove `default=`; rely on `key=...`
    dept_sel = st.multiselect("Department (leave empty = all)", dept_options, key="dept_sel")
    accd_sel = st.multiselect("Account Description (leave empty = all)", accd_options, key="accd_sel")
    atype_sel = st.multiselect("Account Type (leave empty = all)", atype_options, key="atype_sel")
    st.caption("Tip: leave a list empty to include all values.")

    st.markdown('<div class="filter-header">🧰 Options</div>', unsafe_allow_html=True)
    # -- Remove `value=...`; rely on `key=...`
    include_commitments = st.checkbox("Include Encumbrances in Spent", key="include_commitments")
# Prepare df_tmp for slider statistics (must exist before the sliders)
    df_tmp = df.copy()
    
    # Ensure numeric bases
    df_tmp["Actual_Spent"]      = pd.to_numeric(df_tmp.get("Actual_Spent", 0), errors="coerce").fillna(0.0)
    df_tmp["Budget_Allocated"]  = pd.to_numeric(df_tmp.get("Budget_Allocated", 0), errors="coerce").fillna(0.0)
    enc = pd.to_numeric(df_tmp.get("Encumbered", 0), errors="coerce").fillna(0.0)
    pre = pd.to_numeric(df_tmp.get("Pre_Encumbered", 0), errors="coerce").fillna(0.0)
    
    # Actual_Effective possibly includes commitments
    if include_commitments:
        df_tmp["Actual_Effective"] = df_tmp["Actual_Spent"] + enc + pre
    else:
        df_tmp["Actual_Effective"] = df_tmp["Actual_Spent"]
    
    # Variance (absolute and %)
    df_tmp["Variance_Effective"] = df_tmp["Actual_Effective"] - df_tmp["Budget_Allocated"]
    _den = df_tmp["Budget_Allocated"].replace({0: np.nan})
    df_tmp["Variance_Percent_Effective"] = (df_tmp["Variance_Effective"] / _den) * 100


    # ----- Month slider (dates) -----
    min_m = pd.to_datetime(df["Month"].min()).date()
    max_m = pd.to_datetime(df["Month"].max()).date()

    # make sure date_range exists and is clamped BEFORE creating the slider
    st.session_state.setdefault("date_range", (min_m, max_m))
    lo, hi = st.session_state["date_range"]
    lo = max(min(lo, max_m), min_m)
    hi = max(min(hi, max_m), min_m)
    if lo > hi:
        lo, hi = hi, lo
    st.session_state["date_range"] = (lo, hi)

    from pandas.tseries.offsets import MonthBegin
    if min_m >= max_m:
        faux_max = (pd.Timestamp(min_m) + MonthBegin(1)).date()
        lo, hi = st.session_state["date_range"]
        st.session_state["date_range"] = (lo, min(hi, faux_max))
        st.slider("📅 Month Range", min_value=min_m, max_value=faux_max, format="YYYY-MM", key="date_range")
    else:
        st.slider("📅 Month Range", min_value=min_m, max_value=max_m, format="YYYY-MM", key="date_range")

    st.markdown('<div class="filter-header">⚙️ Advanced</div>', unsafe_allow_html=True)

    # FY multiselect — remove default=
    fy_opts = sorted(pd.Series(df["Fiscal_Year"]).dropna().unique().tolist())
    fy_sel = st.multiselect("Fiscal Year (leave empty = all)", fy_opts, key="fy_sel")

    # ----- Variance % slider -----
    v = pd.to_numeric(df_tmp["Variance_Percent_Effective"], errors="coerce")
    v = v[np.isfinite(v)]
    vmin, vmax = (-100.0, 100.0) if v.empty else (float(v.min()), float(v.max()))
    if abs(vmax - vmin) < 1e-6:
        vmin -= 1.0; vmax += 1.0

    # prepare session value BEFORE the slider, then no `value=` on the widget
    st.session_state.setdefault("variance_range", (vmin, vmax))
    vlo, vhi = st.session_state["variance_range"]
    vlo = float(np.clip(vlo, vmin, vmax)); vhi = float(np.clip(vhi, vmin, vmax))
    if vlo > vhi: vlo, vhi = vhi, vlo
    st.session_state["variance_range"] = (vlo, vhi)

    var_step = max(0.1, round((vmax - vmin) / 200, 3))
    st.slider("Variance % (effective)", min_value=float(vmin), max_value=float(vmax),
              step=var_step, key="variance_range")

    # ----- Amount slider -----
    a = pd.to_numeric(df_tmp["Actual_Effective"], errors="coerce")
    a = a[np.isfinite(a)]
    amin, amax = (0.0, 0.0) if a.empty else (float(a.min()), float(a.max()))
    if abs(amax - amin) < 1e-9:
        pad = max(100.0, abs(amin) * 0.05 + 1.0); amin -= pad; amax += pad

    st.session_state.setdefault("amount_range", (amin, amax))
    alo, ahi = st.session_state["amount_range"]
    alo = float(np.clip(alo, amin, amax)); ahi = float(np.clip(ahi, amin, amax))
    if alo > ahi: alo, ahi = ahi, alo
    st.session_state["amount_range"] = (alo, ahi)

    amt_step = float(_amount_step_hint(max(abs(amin), abs(amax))))
    st.slider("Spent Range ($, effective)", min_value=float(amin), max_value=float(amax),
              step=amt_step, format="$%.0f", key="amount_range")
    budget_perf = st.selectbox(
    "🎯 Budget Performance",
    ["All", "Over Budget (>0%)", "Under Budget (<0%)",
     "On Target (±5%)", "Significant Variance (>±10%)"],
    index=0,
    key="budget_perf",   # <-- add this
)

    
# --- AI & RAG controls (simple-by-default) ---
SHOW_ADVANCED_AI = os.getenv("SHOW_ADVANCED_AI", "").strip().lower() in ("1", "true", "yes", "on")

# Set defaults BEFORE creating widgets (ok to use setdefault here)
st.session_state.setdefault("ai_rows_cap", min(400, max(50, len(df)//2)))
st.session_state.setdefault("rag_top_k", 5)
st.session_state.setdefault("chain_style", "LCEL (modern)")
st.session_state.setdefault("show_ctx", False)

st.markdown('<div class="filter-header">🧠 AI & RAG</div>', unsafe_allow_html=True)

# Do NOT assign back into session_state here—just read the return value
show_ctx = st.checkbox(
    "Explain answer: show retrieved context",
    value=st.session_state["show_ctx"],
    key="show_ctx",
)

# Advanced panel (only if enabled)
if SHOW_ADVANCED_AI:
    adv_open = st.checkbox("Show advanced AI controls", value=False, key="show_adv_ai")
    if adv_open:
        ai_rows_cap = st.slider(
            "Context cap from data (rows)", 50, 1500,
            st.session_state["ai_rows_cap"], step=50, key="ai_rows_cap"
        )
        rag_top_k = st.slider(
            "RAG: results per query (k)", 2, 10,
            st.session_state["rag_top_k"], step=1, key="rag_top_k"
        )
        _chain_default = st.session_state["chain_style"]
        _chain_index = 0 if str(_chain_default).startswith("LCEL") else 1
        chain_style = st.selectbox(
            "Chain style", ["LCEL (modern)", "RetrievalQA (classic)"],
            index=_chain_index, key="chain_style"
        )

# =============================
# Apply filters (robust)
# =============================
budget_perf = st.session_state.get("budget_perf", "All")

df_work = df.copy()

# Ensure numeric columns are numeric
for col in ["Actual_Spent", "Encumbered", "Pre_Encumbered", "Budget_Allocated"]:
    if col in df_work:
        df_work[col] = pd.to_numeric(df_work[col], errors="coerce").fillna(0.0)

# Effective measures (recompute if encumbrances included)
if include_commitments:
    enc = df_work.get("Encumbered", 0.0)
    pre = df_work.get("Pre_Encumbered", 0.0)
    df_work["Actual_Effective"] = df_work["Actual_Spent"] + enc + pre
    df_work["Variance_Effective"] = df_work["Actual_Effective"] - df_work["Budget_Allocated"]
    den_work = df_work["Budget_Allocated"].replace({0: np.nan})
    df_work["Variance_Percent_Effective"] = (df_work["Variance_Effective"] / den_work) * 100
else:
    # Make sure the columns exist even if upstream changed
    if "Actual_Effective" not in df_work:
        df_work["Actual_Effective"] = df_work["Actual_Spent"]
    if "Variance_Effective" not in df_work:
        df_work["Variance_Effective"] = df_work["Actual_Effective"] - df_work["Budget_Allocated"]
    if "Variance_Percent_Effective" not in df_work:
        den_work = df_work["Budget_Allocated"].replace({0: np.nan})
        df_work["Variance_Percent_Effective"] = (df_work["Variance_Effective"] / den_work) * 100

# ----- Primary list filters -----
all_true = pd.Series(True, index=df_work.index)
dept_ok = df_work["Department"].isin(dept_sel) if ("Department" in df_work and dept_sel) else all_true
accd_ok = df_work["Account_Desc"].isin(accd_sel) if ("Account_Desc" in df_work and accd_sel) else all_true
atype_ok = df_work["Account_Type"].isin(atype_sel) if ("Account_Type" in df_work and atype_sel) else all_true
fy_ok   = df_work["Fiscal_Year"].isin(fy_sel)   if ("Fiscal_Year"   in df_work and fy_sel)   else all_true

# ----- Date range (pull safely from session, normalize to dates) -----
_min_date = pd.to_datetime(df_work["Month"].min()).date()
_max_date = pd.to_datetime(df_work["Month"].max()).date()
_dr = st.session_state.get("date_range", (_min_date, _max_date))

def _to_date(x):
    return x.date() if hasattr(x, "date") else x

date_range = (_to_date(_dr[0]), _to_date(_dr[1]))  # keep for later use if needed
start = pd.to_datetime(date_range[0]).normalize()
end   = pd.to_datetime(date_range[1]).normalize()
date_ok = df_work["Month"].ge(start) & df_work["Month"].le(end)

# ----- Variance % range -----
vpe = pd.to_numeric(df_work["Variance_Percent_Effective"], errors="coerce")
has_v = np.isfinite(vpe)
vmin = float(np.nanmin(vpe[has_v])) if has_v.any() else -100.0
vmax = float(np.nanmax(vpe[has_v])) if has_v.any() else  100.0
variance_range = tuple(map(float, st.session_state.get("variance_range", (vmin, vmax))))
variance_ok = vpe.between(variance_range[0], variance_range[1]) | vpe.isna()

# ----- Amount range (Actual_Effective) -----
ae = pd.to_numeric(df_work["Actual_Effective"], errors="coerce")
has_ae = np.isfinite(ae)
amin = float(np.nanmin(ae[has_ae])) if has_ae.any() else 0.0
amax = float(np.nanmax(ae[has_ae])) if has_ae.any() else 0.0
amount_range = tuple(map(float, st.session_state.get("amount_range", (amin, amax))))
amt_ok = ae.between(amount_range[0], amount_range[1])

# ----- Combine mask -----
mask = dept_ok & accd_ok & atype_ok & fy_ok & date_ok & variance_ok & amt_ok

# Optional performance bucket
perf_filters = {
    "Over Budget (>0%)":            vpe > 0,
    "Under Budget (<0%)":           vpe < 0,
    "On Target (±5%)":              vpe.between(-5, 5),
    "Significant Variance (>±10%)": (vpe > 10) | (vpe < -10),
}
if budget_perf in perf_filters:
    mask &= perf_filters[budget_perf]

df_f = df_work.loc[mask].copy()

# =============================
# KPIs
# =============================
st.markdown("### 🔎 Overview")

if df_f.empty:
    st.warning("No data matches your current filters. Try resetting one or more filters or expanding the date range.")
else:
    totals = df_f.agg({"Budget_Allocated":"sum", "Actual_Effective":"sum"})
    tot_alloc = float(totals.get("Budget_Allocated", 0.0)) if np.isfinite(totals.get("Budget_Allocated", 0.0)) else 0.0
    tot_spent = float(totals.get("Actual_Effective", 0.0))  if np.isfinite(totals.get("Actual_Effective", 0.0))  else 0.0
    tot_var = tot_spent - tot_alloc
    pct = (tot_var / tot_alloc) * 100.0 if tot_alloc > 0 else np.nan

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Budget", money_fmt(tot_alloc), delta=f"{len(df_f):,} rows")
    with k2:
        spent_label = "Total Spent" + (" (Incl. Enc.)" if include_commitments else "")
        delta_txt = f"vs Budget {pct_fmt(pct, 1)}" if np.isfinite(pct) else "vs Budget —"
        st.metric(spent_label, money_fmt(tot_spent), delta=delta_txt, delta_color="inverse")
    with k3:
        st.metric("Net Variance", money_fmt(tot_var),
                  delta=pct_fmt(pct, 2) if np.isfinite(pct) else "—",
                  delta_color="inverse")
    with k4:
        eff_base = float(pct) if np.isfinite(pct) else 0.0
        efficiency = max(0.0, 100.0 - abs(eff_base))
        tag = "Excellent" if efficiency > 95 else ("Good" if efficiency > 85 else "Needs Review")
        st.metric("Budget Efficiency", f"{efficiency:.1f}%", delta=tag)

# =============================
# Details
# =============================
st.markdown("---")
st.subheader("📄 Details (filtered)")

if df_f.empty:
    st.info("Adjust filters to see the detailed table.")
else:
    show_cols = [
        "Month","Fiscal_Year","Department","Account_Type","Account_Desc",
        "Fund_Desc","Program_Desc","Ledger_Group",
        "Budget_Allocated","Actual_Effective","Variance_Effective","Variance_Percent_Effective",
    ]
    show_cols = [c for c in show_cols if c in df_f.columns]
    df_disp = df_f.loc[:, show_cols].copy()
    if "Month" in df_disp.columns:
        df_disp["Month"] = pd.to_datetime(df_disp["Month"], errors="coerce")

    sort_cols = [c for c in ["Month","Department","Account_Type","Account_Desc"] if c in df_disp.columns]
    if sort_cols:
        df_disp = df_disp.sort_values(sort_cols).reset_index(drop=True)

    col_cfg = {}
    if "Month" in df_disp.columns:
        col_cfg["Month"] = st.column_config.DateColumn(label="Month", format="YYYY-MM")
    for money_col in ["Budget_Allocated","Actual_Effective","Variance_Effective"]:
        if money_col in df_disp.columns:
            col_cfg[money_col] = st.column_config.NumberColumn(
                label=money_col.replace("_"," "), format="$%.0f"
            )
    if "Variance_Percent_Effective" in df_disp.columns:
        col_cfg["Variance_Percent_Effective"] = st.column_config.NumberColumn(
            label="Variance % (Effective)", format="%.1f%%"
        )

    st.dataframe(df_disp, use_container_width=True, hide_index=True, height=380, column_config=col_cfg)

    # Downloads
    csv_bytes = df_disp.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="details_filtered.csv",
        mime="text/csv",
        key="dl_details_csv"
    )

# =============================
# Visuals
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
                color=alt.Color("Type:N", title=""),
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
                    color=alt.Color("Type:N", title=""),
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
                    color=alt.Color("Type:N", title=""),
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
# AI context builders (DataFrame → Documents)
# =============================
def make_doc(text: str, meta: dict):
    try:
        return Document(page_content=text, metadata=meta)  # if LangChain present
    except Exception:
        return {"page_content": text, "metadata": meta}    # fallback dict

def tbl(df_: pd.DataFrame) -> str:
    if df_.empty:
        return "(none)"
    fmts = {}
    for c in ["Budget_Allocated","Actual_Effective","Variance_Effective"]:
        if c in df_.columns:
            fmts[c] = lambda x: f"{float(x):,.2f}"
    return df_.to_string(index=False, formatters=fmts or None)

def build_compact_summary(actuals_df: pd.DataFrame) -> str:
    if actuals_df.empty:
        return "No rows."
    dfm = actuals_df.copy()
    dfm["Month"] = pd.to_datetime(dfm["Month"], errors="coerce")
    a_month = (dfm.groupby("Month", as_index=False)[["Actual_Effective","Budget_Allocated"]]
                  .sum().sort_values("Month"))
    a_month["Month"] = a_month["Month"].dt.strftime("%Y-%m")
    a_dept = (dfm.groupby("Department", as_index=False)[["Actual_Effective","Budget_Allocated"]]
                .sum().nlargest(15, "Actual_Effective")) if "Department" in dfm.columns else pd.DataFrame()
    a_desc = (dfm.groupby("Account_Desc", as_index=False)[["Actual_Effective","Budget_Allocated"]]
                .sum().nlargest(15, "Actual_Effective")) if "Account_Desc" in dfm.columns else pd.DataFrame()
    return f"""ACTUALS — monthly (Allocated vs Spent, last 24):
{tbl(a_month.tail(24))}

Top Departments:
{tbl(a_dept)}

Top Account Descriptions:
{tbl(a_desc)}
"""

def df_to_docs(df_ctx: pd.DataFrame, source_label: str = "budget_filtered", cap_rows: int = 400) -> list:
    docs = []
    if df_ctx.empty:
        return docs
    dfc = df_ctx.copy()
    dfc["Month"] = pd.to_datetime(dfc["Month"], errors="coerce")
    for col in ["Budget_Allocated","Actual_Effective","Variance_Effective","Variance_Percent_Effective"]:
        if col in dfc:
            dfc[col] = pd.to_numeric(dfc[col], errors="coerce")

    overview = build_compact_summary(dfc)
    docs.append(make_doc(overview, {"type": "overview", "source": source_label}))
    remaining = max(0, cap_rows - 1)

    if "Department" in dfc.columns and remaining > 0:
        g = (dfc.groupby("Department", as_index=False)
                .agg(Allocated=("Budget_Allocated","sum"),
                     Spent=("Actual_Effective","sum"),
                     Var=("Variance_Effective","sum")))
        g["VarPct"] = (g["Var"] / g["Allocated"].replace({0: np.nan})) * 100
        g["AbsVar"] = g["Var"].abs()
        g = g.nlargest(min(200, remaining), "AbsVar")
        g["text"] = (
            "[Department Summary]\n"
            "Department: " + g["Department"].astype(str) + "\n"
            "Allocated: " + g["Allocated"].map(lambda x: f"{x:,.2f}") + "\n"
            "Spent: "     + g["Spent"].map(lambda x: f"{x:,.2f}") + "\n"
            "Variance: "  + g["Var"].map(lambda x: f"{x:,.2f}") +
            " (" + g["VarPct"].fillna(0.0).map(lambda x: f"{x:.2f}%") + ")"
        )
        docs.extend(make_doc(t, {"type":"dept","source":source_label,"dept":d}) 
                    for t, d in zip(g["text"], g["Department"]))
        remaining -= len(g)

    if "Account_Desc" in dfc.columns and remaining > 0:
        g = (dfc.groupby("Account_Desc", as_index=False)
                .agg(Allocated=("Budget_Allocated","sum"),
                     Spent=("Actual_Effective","sum"),
                     Var=("Variance_Effective","sum")))
        g["VarPct"] = (g["Var"] / g["Allocated"].replace({0: np.nan})) * 100
        g["AbsVar"] = g["Var"].abs()
        g = g.nlargest(min(200, remaining), "AbsVar")
        g["text"] = (
            "[Account Description Summary]\n"
            "Account_Desc: " + g["Account_Desc"].astype(str) + "\n"
            "Allocated: " + g["Allocated"].map(lambda x: f"{x:,.2f}") + "\n"
            "Spent: "     + g["Spent"].map(lambda x: f"{x:,.2f}") + "\n"
            "Variance: "  + g["Var"].map(lambda x: f"{x:,.2f}") +
            " (" + g["VarPct"].fillna(0.0).map(lambda x: f"{x:.2f}%") + ")"
        )
        docs.extend(make_doc(t, {"type":"accdesc","source":source_label,"account_desc":a})
                    for t, a in zip(g["text"], g["Account_Desc"]))
        remaining -= len(g)

    if remaining > 0:
        g = (dfc.groupby("Month", as_index=False)[["Budget_Allocated","Actual_Effective"]].sum()
               .sort_values("Month"))
        g["MonthStr"] = g["Month"].dt.strftime("%Y-%m")
        g["Var"] = g["Actual_Effective"] - g["Budget_Allocated"]
        g = g.tail(min(len(g), remaining))
        g["text"] = (
            "[Monthly]\n"
            "Month: " + g["MonthStr"] + "\n"
            "Allocated: " + g["Budget_Allocated"].map(lambda x: f"{x:,.2f}") + "\n"
            "Spent: "     + g["Actual_Effective"].map(lambda x: f"{x:,.2f}") + "\n"
            "Variance: "  + g["Var"].map(lambda x: f"{x:,.2f}")
        )
        docs.extend(make_doc(t, {"type":"month","source":source_label,"month":m})
                    for t, m in zip(g["text"], g["MonthStr"]))
        remaining -= len(g)

    if remaining > 0 and "Variance_Effective" in dfc.columns:
        take = min(100, remaining)
        top = dfc.copy()
        top["_abs"] = top["Variance_Effective"].abs()
        top = top.nlargest(take, "_abs").drop(columns="_abs")
        top["MonthStr"] = top["Month"].dt.strftime("%Y-%m")
        def row_text(r):
            parts = []
            for col in ["MonthStr","Fiscal_Year","Department","Account_Type","Account_Desc","Fund_Desc","Program_Desc","Ledger_Group"]:
                if col in top.columns and pd.notna(r.get(col, None)):
                    key = "Month" if col == "MonthStr" else col
                    parts.append(f"{key}={r[col]}")
            parts.append(f"Allocated={r['Budget_Allocated']:.2f}")
            parts.append(f"Spent={r['Actual_Effective']:.2f}")
            parts.append(f"Variance={r['Variance_Effective']:.2f}")
            if "Variance_Percent_Effective" in top.columns and pd.notna(r["Variance_Percent_Effective"]):
                parts.append(f"VariancePct={float(r['Variance_Percent_Effective']):.2f}%")
            return "[Row] " + " | ".join(parts)
        docs.extend(make_doc(row_text(r), {"type":"row","source":source_label}) for _, r in top.iterrows())

    return docs

# =============================
# Embeddings / Vector store / Chains
# =============================
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

def _docs_fingerprint(docs: list) -> str:
    m = hashlib.md5()
    for d in docs:
        content = getattr(d, "page_content", None) or d.get("page_content", "")
        meta = getattr(d, "metadata", None) or d.get("metadata", {})
        m.update(content.encode("utf-8"))
        m.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    return m.hexdigest()

@st.cache_resource(show_spinner=False)
def get_embedder(_key: str, model: str = DEFAULT_EMBED_MODEL):
    if not (_has_langchain and _key):
        return None
    try:
        return OpenAIEmbeddings(api_key=_key, model=model)
    except Exception:
        st.warning("Embedding init failed. Verify OPENAI key, network, and model name.")
        return None

def build_vectorstore(docs: list, _key: str, embed_model: str = DEFAULT_EMBED_MODEL):
    if not (_has_langchain and _key and docs):
        return None
    fp = f"{_docs_fingerprint(docs)}::{embed_model}"
    cache_key = f"_vs_{fp}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        embeddings = get_embedder(_key, embed_model)
        if embeddings is None:
            return None
        vs = FAISS.from_documents(docs, embeddings)
        st.session_state[cache_key] = vs
        return vs
    except Exception as e:
        st.warning(f"Vector index error: {e}")
        return None

def make_lcel_chain(vs, model_name: str, _key: str, k: int = 5, *, search_type: str = "mmr"):
    if vs is None or not _has_langchain or not _key:
        return None
    k = max(1, int(k))
    if search_type == "mmr":
        retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": max(20, k*4), "lambda_mult": 0.7})
    else:
        retriever = vs.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model_name, temperature=0.2, api_key=_key)
    system_template = (
        "You are a precise FP&A analyst. Use ONLY the provided context to answer.\n"
        "If the answer is not in the context, say you don't have enough data and suggest which filters to adjust.\n"
        "Return concise numeric bullets with $ and %, then a brief explanation. Call out largest drivers by absolute variance.\n"
        "Only use numbers present in the context."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "Question:\n{question}\n\nContext:\n{context}")
    ])
    def format_docs(dlist, max_chars: int = 4000):
        out, count = [], 0
        for d in dlist:
            t = d.page_content
            if count + len(t) > max_chars: break
            out.append(t); count += len(t)
        return "\n\n".join(out) if out else "(no relevant context retrieved)"
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain

def make_retrievalqa(vs, model_name: str, _key: str, k: int = 5, *, search_type: str = "similarity"):
    if vs is None or not _has_langchain or not _key:
        return None
    k = max(1, int(k))
    if search_type == "mmr":
        retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": max(20, k*4), "lambda_mult": 0.7})
    else:
        retriever = vs.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model_name, temperature=0.2, api_key=_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": ChatPromptTemplate.from_messages([
                ("system",
                 "You are a precise FP&A analyst. Use ONLY the provided context to answer. "
                 "If not in context, say you don't have enough data and suggest filters to adjust. "
                 "Return concise numeric bullets with $ and %, then a brief explanation. "
                 "Only use numbers present in the context."),
                ("human", "Question:\n{question}\n\nContext:\n{context}")
            ])
        },
        return_source_documents=True
    )
    return qa

# =============================
# 🤖 AI — Quick Analysis & RAG Q&A + Provenance
# =============================
st.markdown("---")
st.markdown(
    '<div class="ai-box"><h3>🤖 AI Insights + RAG (DataFrame only)</h3><p>Ask grounded questions using the current filters.</p></div>',
    unsafe_allow_html=True
)

left, right = st.columns([1,2])

with left:
    if st.button("📈 Quick Analysis (no RAG)", use_container_width=True):
        if not _oai_client:
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
        if (not _api_key) or (not _has_langchain):
            st.error("RAG deps not available. Ensure OPENAI_API_KEY is set and langchain packages installed.")
        elif df_f.empty:
            st.warning("No rows in the current filters. Adjust filters, then ask again.")
        else:
            docs = df_to_docs(df_f, source_label="filtered_selection", cap_rows=int(st.session_state.get("ai_rows_cap", 400)))
            vs = build_vectorstore(docs, _api_key)
            if vs is None:
                st.error("Could not initialize vector store (FAISS or embeddings).")
            else:
                try:
                    if st.session_state.get("chain_style","LCEL (modern)").startswith("LCEL"):
                        chain = make_lcel_chain(vs, llm_model_name, _api_key, int(st.session_state.get("rag_top_k",5)))
                        if chain is None:
                            st.error("Could not initialize LCEL chain.")
                        else:
                            ans = chain.invoke(q)
                            answer_text = getattr(ans, "content", str(ans))
                            st.markdown(f"""<div class="ai-result">{answer_text}</div>""", unsafe_allow_html=True)

                            hits = vs.similarity_search_with_score(q, k=int(st.session_state.get("rag_top_k",5)))
                            if hits:
                                types = [d.metadata.get("type","?") for (d, _) in hits]
                                months, depts, accds = [], [], []
                                row_count = 0
                                for d, sc in hits:
                                    txt = d.page_content
                                    if "[Row]" in txt: row_count += 1
                                    for line in txt.splitlines():
                                        if line.startswith("Month:"):
                                            months.append(line.split(":",1)[1].strip())
                                        if "Department:" in line:
                                            depts.append(line.split(":",1)[1].strip())
                                        if "Account_Desc:" in line:
                                            accds.append(line.split(":",1)[1].strip())
                                        if "Month=" in line:
                                            for p in [p for p in line.split("|") if "Month=" in p]:
                                                months.append(p.strip().split("=",1)[1])
                                type_counts = {t: types.count(t) for t in sorted(set(types))}
                                prov = f"Source context • types: {type_counts} • months: {sorted(set(months))[:6]}{' …' if len(set(months))>6 else ''} • depts: {sorted(set([d for d in depts if d]))[:6]} • rows: ~{row_count}"
                                st.markdown(f"""<div class="provenance">{prov}</div>""", unsafe_allow_html=True)

                            if st.session_state.get("show_ctx", False) and hits:
                                st.markdown("**Retrieved context (top-k) with similarity scores):**")
                                for i, (doc, sc) in enumerate(hits, start=1):
                                    with st.expander(f"#{i}  score={sc:.4f}  •  {doc.metadata.get('type','?')}"):
                                        st.code(doc.page_content)

                    else:
                        qa = make_retrievalqa(vs, llm_model_name, _api_key, int(st.session_state.get("rag_top_k",5)))
                        if qa is None:
                            st.error("Could not initialize RetrievalQA chain.")
                        else:
                            res = qa.invoke({"query": q})
                            answer_text = res.get("result", "")
                            st.markdown(f"""<div class="ai-result">{answer_text}</div>""", unsafe_allow_html=True)

                            sources = res.get("source_documents", []) or []
                            if sources:
                                types = [d.metadata.get("type","?") for d in sources]
                                months, depts, accds = [], [], []
                                row_count = 0
                                for d in sources:
                                    txt = d.page_content
                                    if "[Row]" in txt: row_count += 1
                                    for line in txt.splitlines():
                                        if line.startswith("Month:"):
                                            months.append(line.split(":",1)[1].strip())
                                        if "Department:" in line:
                                            depts.append(line.split(":",1)[1].strip())
                                        if "Account_Desc:" in line:
                                            accds.append(line.split(":",1)[1].strip())
                                        if "Month=" in line:
                                            for p in [p for p in line.split("|") if "Month=" in p]:
                                                months.append(p.strip().split("=",1)[1])
                                type_counts = {t: types.count(t) for t in sorted(set(types))}
                                prov = f"Source context • types: {type_counts} • months: {sorted(set(months))[:6]}{' …' if len(set(months))>6 else ''} • depts: {sorted(set([d for d in depts if d]))[:6]} • rows: ~{row_count}"
                                st.markdown(f"""<div class="provenance">{prov}</div>""", unsafe_allow_html=True)

                            if st.session_state.get("show_ctx", False) and sources:
                                st.markdown("**Retrieved context (top-k):**")
                                for i, d in enumerate(sources, 1):
                                    with st.expander(f"#{i}  •  {d.metadata.get('type','?')}"):
                                        st.code(d.page_content)
                except Exception as e:
                    st.error(f"RAG error: {e}")

# =============================
# Downloads (CSV + Excel multi-sheet)
# =============================
st.markdown("---")
if not df_f.empty:
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data (CSV)", data=csv,
                       file_name=f"budget_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                       mime="text/csv")

    try:
        monthly = (df_f.groupby("Month", as_index=False)[["Budget_Allocated","Actual_Effective"]].sum()
                   .sort_values("Month").copy())
        dept = df_f.groupby("Department", as_index=False).agg(
            Allocated=("Budget_Allocated","sum"),
            Spent=("Actual_Effective","sum"),
            Variance=("Variance_Effective","sum")
        )
        accd = df_f.groupby("Account_Desc", as_index=False).agg(
            Allocated=("Budget_Allocated","sum"),
            Spent=("Actual_Effective","sum"),
            Variance=("Variance_Effective","sum")
        )
        monthly_disp = monthly.copy()
        monthly_disp["Month"] = pd.to_datetime(monthly_disp["Month"]).dt.strftime("%Y-%m")

        show_cols = [
            "Month","Fiscal_Year","Department","Account_Type","Account_Desc",
            "Fund_Desc","Program_Desc","Ledger_Group",
            "Budget_Allocated","Actual_Effective","Variance_Effective","Variance_Percent_Effective"
        ]
        show_cols = [c for c in show_cols if c in df_f.columns]
        details_xlsx = df_f[show_cols].copy()
        details_xlsx["Month"] = pd.to_datetime(details_xlsx["Month"]).dt.strftime("%Y-%m")

        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            details_xlsx.to_excel(writer, index=False, sheet_name="Details")
            monthly_disp.to_excel(writer, index=False, sheet_name="Monthly")
            dept.to_excel(writer, index=False, sheet_name="By Department")
            accd.to_excel(writer, index=False, sheet_name="By Account Desc")
        bio.seek(0)

        st.download_button(
            "⬇️ Download Excel (Details + Monthly + Dept + AccDesc)",
            data=bio,
            file_name=f"budget_export_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.warning(f"Excel export unavailable: {e}")
else:
    st.info("Adjust filters to enable downloads.")

# =============================
# Optional: context signature
# =============================
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
