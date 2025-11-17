# -*- coding: utf-8 -*-
# Budget RAG Assistant — v4 (single clean file)

import os, re, json, hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv

# -------------------------------------------------
# OPTIONAL RAG imports
# -------------------------------------------------
try:
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    _HAS_LANGCHAIN = True
except Exception:
    Document = OpenAIEmbeddings = FAISS = None
    _HAS_LANGCHAIN = False

# ============================
# THEME + ALT CONFIG + CSS (all-in-one)
# ============================
st.set_page_config(page_title="Budget AI Assistant", layout="wide", initial_sidebar_state="expanded")

THEMES = {
    "Executive Light": {
        "bg": "#F7FAFC", "sidebar": "#0B1E3E", "panel": "#FFFFFF", "card": "#FFFFFF",
        "text": "#0F172A", "muted": "#475569", "grid": "#E2E8F0",
        "brand1": "#0B1E3E", "brand2": "#2F6BFF",
        "alloc": "#2151B8", "spent": "#53A2FF",
        "ok": "#10B981", "warn": "#D97706",
    },
    "Executive Dark": {
        "bg": "#0B1E3E", "sidebar": "#071A34", "panel": "#0E2044", "card": "#0F234B",
        "text": "#FFFFFF", "muted": "#CBD5E1", "grid": "#14315F",
        "brand1": "#0B1E3E", "brand2": "#2F6BFF",
        "alloc": "#8FB7FF", "spent": "#5FB0FF",
        "ok": "#10B981", "warn": "#F59E0B",
    },
}

with st.sidebar:
    st.markdown("**🎨 Theme**")
    theme_name = st.selectbox("Select theme", list(THEMES.keys()), index=1)

pal = THEMES[theme_name]

def _alt_theme(p):
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {"labelColor": p["text"], "titleColor": p["text"], "gridColor": p["grid"]},
            "legend": {"labelColor": p["text"], "titleColor": p["text"]},
            "title": {"color": p["text"]},
            "range": {"category": [p["alloc"], p["spent"], "#9CA3AF", "#A78BFA"]},
        }
    }

alt.themes.register("fin_theme_v4", lambda: _alt_theme(pal))
alt.themes.enable("fin_theme_v4")

# ---------- derived style vars ----------
light_mode        = (theme_name == "Executive Light")
kpi_border        = "#CBD5E1" if light_mode else "rgba(255,255,255,0.12)"
card_border       = kpi_border
df_head_bg        = "#F1F5F9" if light_mode else "rgba(7,26,52,0.35)"
df_head_text      = "#0F172A" if light_mode else "#FFFFFF"
idle_tab          = "#334155" if light_mode else pal["muted"]
q_placeholder     = "rgba(15,23,42,0.60)" if light_mode else "rgba(255,255,255,0.55)"
kpi_shadow        = "0 2px 6px rgba(2, 6, 23, 0.06)" if light_mode else "0 2px 8px rgba(0,0,0,0.25)"
ai_bg             = "rgba(255,255,255,0.50)" if light_mode else "rgba(255,255,255,0.04)"
ai_body_bg        = "rgba(255,255,255,0.65)" if light_mode else "rgba(255,255,255,0.03)"
btn_border        = "#E2E8F0" if light_mode else "rgba(255,255,255,0.30)"
kpi_value_color   = "#0F172A" if light_mode else "#FFFFFF"
delta_opacity     = "0.85" if light_mode else "1"
kpi_card_bg       = pal["card"] if light_mode else "#0F234B"

# ---------- set data-theme attr (optional) ----------
st.markdown(
    f"<script>document.body.setAttribute('data-theme','{'exec-dark' if not light_mode else 'exec-light'}');</script>",
    unsafe_allow_html=True
)

# ---------- BASE + KPI CSS ----------
st.markdown(
    f"""
<style>
  .stApp {{
    background:{pal['bg']};
    color:{pal['text']};
    font-size: 18px;
  }}

  /* sidebar */
  [data-testid="stSidebar"] > div:first-child {{
    background:{pal['sidebar']};
  }}
  [data-testid="stSidebar"] * {{
    color:#E5E7EB !important;
    font-size: 18px;
  }}
  :root [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] span,
  :root [data-testid="stSidebar"] .stMultiSelect input::placeholder {{
    color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 600 !important;
  }}

  /* header */
  .main-header {{
    text-align:center;
    padding:2.25rem 1rem;
    border-radius:14px;
    background: linear-gradient(135deg, {pal['brand1']} 0%, {pal['brand2']} 100%);
    color:#fff;
    margin-bottom:1rem;
  }}
  .main-header h1 {{
    margin:0;
    font-size:clamp(40px,4.4vw,64px);
    letter-spacing:-0.01em;
  }}

  /* section card */
  .section-card {{
    background:{pal['card']};
    border: 1px solid {card_border};
    border-radius: 1.1rem;
    padding: 1.1rem 1.2rem 1.2rem;
    margin-bottom: 1.05rem;
    box-shadow: {kpi_shadow};
  }}

  /* tabs */
  :root .stTabs [role="tab"] {{
    color: {idle_tab} !important;
    font-weight: 600 !important;
  }}
  :root .stTabs [role="tab"][aria-selected="true"] {{
    color: {pal['text']} !important;
    border-bottom: 2px solid {pal['brand2']} !important;
  }}

  /* dataframe header */
  :root .stDataFrame thead th {{
    background: {df_head_bg} !important;
    color: {df_head_text} !important;
    border-bottom: 1px solid {kpi_border} !important;
  }}

  /* Altair text/tooltips */
  .vega-embed, .vega-embed * {{
    color: {pal['text']} !important;
  }}

  /* AI boxes */
  .ai-box {{
    padding:.75rem 1rem;
    border-radius:10px;
    background:{ai_bg};
    border:1px solid {kpi_border};
  }}
  .ai-result, .provenance {{
    padding:.75rem 1rem;
    border-radius:10px;
    margin-top:.5rem;
    background:{ai_body_bg};
    border:1px solid {kpi_border};
    color: {pal['text']};
  }}

  /* question label + placeholder */
  label, .stTextInput label {{
    color: {pal['text']} !important;
    font-weight: 700;
    font-size:1.25rem;
  }}
  .stTextInput input::placeholder {{
    color: {q_placeholder};
    font-size: 1.00rem;
  }}

  /* buttons */
  .stButton > button,
  .stDownloadButton > button {{
    background: {pal['brand2']} !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: 1px solid {btn_border} !important;
    font-size: 15px !important;
  }}

  /* ===== KPI tiles (scoped to .kpi-force wrapper) ===== */
  :root [data-testid="stAppViewContainer"] .kpi-force [data-testid="stMetric"] {{
    background: {kpi_card_bg} !important;
    border: 1px solid {kpi_border} !important;
    box-shadow: {kpi_shadow} !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.2rem !important;
    min-height: 140px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
  }}
  :root [data-testid="stAppViewContainer"] .kpi-force [data-testid="stMetric"] div,
  :root [data-testid="stAppViewContainer"] .kpi-force [data-testid="stMetric"] label {{
    color: {pal['text']} !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    line-height: 1.15 !important;
    margin-bottom: .25rem !important;
  }}
  :root [data-testid="stAppViewContainer"] .kpi-force [data-testid="stMetricValue"] {{
    color: {kpi_value_color} !important;
    -webkit-text-fill-color: {kpi_value_color} !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -.01em !important;
    line-height: 1.05 !important;
    opacity: 1 !important;
  }}
  :root [data-testid="stAppViewContainer"] .kpi-force [data-testid="stMetricDelta"] {{
    color: {pal['text']} !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    opacity: {delta_opacity} !important;
  }}

  /* Results typography */
  .result-title {{
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 0.90rem;
    margin-bottom: 0.70rem;
    color: {pal['text']};
  }}
  .result-body {{
    font-size: 1.15rem;
    line-height: 1.35;
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
  }}
  .result-body strong {{
    font-size: 1.22rem;
    font-weight: 700;
  }}
</style>
""",
    unsafe_allow_html=True,
)
# ---------- Title ----------
st.markdown(
    """
<div class="main-header">
  <h1>Budget AI Assistant</h1>
  <p></p>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# OpenAI client (optional for quick analysis box)
# -------------------------------------------------
load_dotenv()
_API_KEY = os.getenv("OPENAI_API_KEY", "") or getattr(st, "secrets", {}).get("OPENAI_API_KEY", "")
_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

_oai_client = get_openai_client(_API_KEY)

def call_openai(system_msg: str, user_msg: str, temperature: float = 0.25, max_tokens: int = 500) -> str:
    if not _oai_client:
        return "⚠️ OpenAI API not available."
    try:
        resp = _oai_client.chat.completions.create(
            model=_MODEL_NAME,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

# -------------------------------------------------
# basic helpers
# -------------------------------------------------
def _read_any(path: str, **kw) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        kw = {"engine": "openpyxl"} | kw
        return pd.read_excel(path, **kw)
    kw = {"on_bad_lines": "skip"} | kw
    return pd.read_csv(path, **kw)

def _coerce_money(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def money_fmt(x: Any, digits: int = 0) -> str:
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return "—"
        return f"${xf:,.{digits}f}"
    except Exception:
        return "—"

# -------------------------------------------------
# column normalization
# -------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _find_col(df: pd.DataFrame, targets: List[str]) -> Optional[str]:
    for t in targets:
        if t in df.columns:
            return t
    lower_map = {c.lower(): c for c in df.columns}
    for t in targets:
        tl = t.lower()
        if tl in lower_map:
            return lower_map[tl]
    for c in df.columns:
        for t in targets:
            if c.lower().startswith(t.lower()):
                return c
    return None

# -------------------------------------------------
# explicit phrases for ALL columns
# -------------------------------------------------
COLUMN_TO_PATTERNS = {
    "Budget Year": [r"\bbudget\s*year\b", r"\byear\b"],
    "Account Type ID": [r"\baccount\s*type\s*id\b"],
    "Account Type": [r"\baccount\s*type\b"],
    "Fund Number": [r"\bfund\s*(?:number|no\.?|#)\b"],
    "Fund Description": [r"\bfund\s*description\b"],
    "Department Number": [r"\bdepartment\s*(?:number|no\.?|#)\b", r"\bdept\s*(?:number|no\.?|#)\b"],
    "Department Description": [r"\bdepartment\s*description\b", r"\bdept\b"],
    "Bureau Code": [r"\bbureau\s*code\b"],
    "Bureau Description": [r"\bbureau\s*description\b"],
    "Command Code": [r"\bcommand\s*code\b"],
    "Command Description": [r"\bcommand\s*description\b"],
    "CFW Division Code": [r"\bcfw\s*division\s*code\b", r"\bdivision\s*code\b"],
    "CFW Division Description": [r"\bcfw\s*division\s*description\b", r"\bdivision\s*description\b"],
    "Cost Center Number": [r"\bcost\s*center\s*(?:number|no\.?|#)\b"],
    "Cost Center Name": [r"\bcost\s*center\s*name\b"],
    "Fund Category Code": [r"\bfund\s*category\s*code\b"],
    "Fund Category Description": [r"\bfund\s*category\s*description\b"],
    "Reporting Type Code": [r"\breporting\s*type\s*code\b"],
    "Reporting Type Description": [r"\breporting\s*type\s*description\b"],
    "Fund Group Code": [r"\bfund\s*group\s*code\b"],
    "Fund Group Description": [r"\bfund\s*group\s*description\b"],
    "Budget Code Number": [r"\bbudget\s*code\s*(?:number|no\.?|#)\b"],
    "Budget Code Description": [r"\bbudget\s*code\s*description\b"],
    "Categories Code": [r"\bcategories\s*code\b", r"\bcategory\s*code\b"],
    "License & Permits Revenue": [
        r"\blicense\s*&\s*permits\s*revenue\b",
        r"\blicense\s*and\s*permits\s*revenue\b",
    ],
    "Account Number": [r"\baccount\s*(?:number|no\.?|#)\b"],
    "Account Name": [r"\baccount\s*name\b"],
    "WKAccountName": [r"\bwk\s*account\s*name\b"],
    "WKAccountNumber": [r"\bwk\s*account\s*(?:number|no\.?|#)\b"],
    "Adopted Budget": [r"\badopted\s*budget\b"],
    "Adjusted Budget": [r"\badjusted\s*budget\b", r"\bamended\s*budget\b"],
    "Current Year Budget": [r"\bcurrent\s*year\s*budget\b", r"\bcurrent\s*budget\b"],
}

EXPLICIT_PATTERNS: List[Tuple[re.Pattern, str]] = []
for col_name, patterns in COLUMN_TO_PATTERNS.items():
    for p in patterns:
        EXPLICIT_PATTERNS.append((re.compile(p, flags=re.IGNORECASE), col_name))

def extract_explicit_column(user_q: str) -> Optional[str]:
    for pat, col_name in EXPLICIT_PATTERNS:
        if pat.search(user_q):
            return col_name
    return None

def extract_value_after_colname(user_q: str, col_name: str) -> Optional[str]:
    pat = re.compile(re.escape(col_name) + r"\s*[:=]?\s*(.+)$", flags=re.IGNORECASE)
    m = pat.search(user_q.strip())
    if m:
        return m.group(1).strip()
    if col_name == "Department Number":
        m2 = re.search(r"\bdept(?:artment)?\s*(?:number)?\s*[:=]?\s*(\S+)", user_q, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).strip()
    return None

def parse_metric_from_question(q: str) -> str:
    ql = q.lower()
    if "adjusted" in ql or "amended" in ql:
        return "Adjusted Budget"
    if "adopted" in ql:
        return "Adopted Budget"
    if "current year" in ql or "current-year" in ql or "current budget" in ql or "current" in ql:
        return "Current Year Budget"
    return "Current Year Budget"

# -------------------------------------------------
# LOAD EXCEL
# -------------------------------------------------
DATA_PATH = "PY Adopted and Adjusted Budget 2026.xlsx"

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    raw = _read_any(path)
    raw = _normalize_columns(raw)

    adopted_col = _find_col(raw, ["Adopted Budget", "Adopted", "Adopted_Budget"])
    adjusted_col = _find_col(raw, ["Adjusted Budget", "Amended Budget", "Adjusted_Budget"])
    current_col  = _find_col(raw, ["Current Year Budget", "Current Budget", "Current_Year_Budget"])

    if adopted_col is None:
        raise ValueError(f"Could not find an Adopted Budget column. Found: {list(raw.columns)}")

    if adjusted_col is None:
        raw["Adjusted Budget"] = 0.0
        adjusted_col = "Adjusted Budget"
    else:
        raw[adjusted_col] = _coerce_money(raw[adjusted_col])

    if current_col is None:
        raise ValueError(f"Could not find a Current Year Budget column. Found: {list(raw.columns)}")

    raw[adopted_col] = _coerce_money(raw[adopted_col])
    raw[current_col] = _coerce_money(raw[current_col])

    if adopted_col != "Adopted Budget":
        raw = raw.rename(columns={adopted_col: "Adopted Budget"})
    if adjusted_col != "Adjusted Budget":
        raw = raw.rename(columns={adjusted_col: "Adjusted Budget"})
    if current_col != "Current Year Budget":
        raw = raw.rename(columns={current_col: "Current Year Budget"})

    raw["Δ CY vs Adjusted"] = raw["Current Year Budget"] - raw["Adjusted Budget"]
    base = raw["Adjusted Budget"].replace({0: np.nan})
    raw["Δ % (vs Adjusted)"] = (raw["Δ CY vs Adjusted"] / base) * 100.0

    return raw

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"❌ Could not load data at {DATA_PATH}: {e}")
    st.stop()

# -------------------------------------------------
# FILTERS
# -------------------------------------------------
st.sidebar.markdown("### 🔍 Filters")

FILTER_FIELDS = {
    "Budget Year": "Budget Year",
    "Account Type": "Account Type",
    "Fund Number": "Fund Number",
    "Fund Description": "Fund Description",
    "Department Number": "Department Number",
    "Department Description": "Department Description",
    "Bureau Description": "Bureau Description",
    "Command Description": "Command Description",
    "CFW Division Description": "CFW Division Description",
    "Cost Center Number": "Cost Center Number",
    "Cost Center Name": "Cost Center Name",
    "Fund Category Description": "Fund Category Description",
    "Reporting Type Description": "Reporting Type Description",
    "Fund Group Description": "Fund Group Description",
    "Budget Code Description": "Budget Code Description",
    "Categories Description": "Categories Description",
    "Account Number": "Account Number",
    "Account Name": "Account Name",
    "WKAccountNumber": "WKAccountNumber",
}

user_sel: Dict[str, list] = {}
for label, col in FILTER_FIELDS.items():
    if col in df.columns:
        opts = sorted(df[col].dropna().astype(str).unique().tolist())
    else:
        opts = []
    user_sel[col] = st.sidebar.multiselect(label, opts, key=f"f_{col}")

st.sidebar.markdown("### 💵 Amount ranges")
ranges = {}
for c in ["Adopted Budget", "Adjusted Budget", "Current Year Budget"]:
    s = pd.to_numeric(df[c], errors="coerce")
    lo, hi = float(s.min()), float(s.max())
    if abs(hi - lo) < 1e-9:
        pad = max(100.0, abs(lo) * 0.05 + 1.0)
        lo -= pad; hi += pad
    ranges[c] = st.sidebar.slider(
        c,
        min_value=float(lo),
        max_value=float(hi),
        value=(float(lo), float(hi)),
        step=float(max(1.0, (hi - lo) / 200)),
        format="$%.0f",
        key=f"rng_{c}",
    )

mask = pd.Series(True, index=df.index)
for col, chosen in user_sel.items():
    if chosen and col in df.columns:
        mask &= df[col].astype(str).isin(chosen)
for c, (lo, hi) in ranges.items():
    mask &= df[c].between(lo, hi)

df_f = df.loc[mask].copy()

# -------------------------------------------------
# KPIs (WITH WRAPPER)
# -------------------------------------------------
# ============================
st.markdown("### 🔎 Overview")
# Pick the scope (filtered if any)
scope_for_kpi = df_f if (isinstance(df_f, pd.DataFrame) and not df_f.empty) else df
if not isinstance(scope_for_kpi, pd.DataFrame) or scope_for_kpi.empty:
    st.info("No data to summarize.")
else:
    # --- totals ---
    tot_adopted  = float(pd.to_numeric(scope_for_kpi.get("Adopted Budget", pd.Series(dtype=float)), errors="coerce").sum())
    tot_adjusted = float(pd.to_numeric(scope_for_kpi.get("Adjusted Budget", pd.Series(dtype=float)), errors="coerce").sum())
    tot_current  = float(pd.to_numeric(scope_for_kpi.get("Current Year Budget", pd.Series(dtype=float)), errors="coerce").sum())
    dlt = tot_current - tot_adjusted
    if tot_adjusted > 0:
        pct = (dlt / tot_adjusted) * 100.0
        delta_txt = f"{pct:+.1f}%"
    else:
        delta_txt = "—"   # no percent if baseline is <= 0
    # primary % calc
    pct = (dlt / tot_adjusted * 100.0) if tot_adjusted > 1e-12 else np.nan

# robust fallback so we NEVER show just a dash
    if not np.isfinite(pct):
        # try current as denominator; if that's also ~0, fall back to |dlt| to show 0.0%
        denom = abs(tot_current) if abs(tot_current) > 1e-12 else (abs(dlt) if abs(dlt) > 1e-12 else 1.0)
        pct = (dlt / denom) * 100.0
    
    # always show + / - sign and a % symbol
    delta_txt = f"{pct:+.1f}%"

       # --- theme-aware colors (no dim text) ---
    is_dark          = (theme_name == "Executive Dark")
    kpi_bg           = "#0F234B" if is_dark else pal["card"]
    kpi_border       = "rgba(255,255,255,0.18)" if is_dark else "#CBD5E1"
    kpi_label_color  = "#FFFFFF" if is_dark else pal["text"]
    kpi_value_color  = "#FFFFFF" if is_dark else "#0F172A"
    kpi_shadow       = "0 2px 8px rgba(0,0,0,0.25)" if is_dark else "0 2px 6px rgba(2, 6, 23, 0.06)"

    # delta styling: green if >=0, red if <0
    if np.isfinite(pct):
        delta_txt   = f"{pct:+.1f}%"
        delta_color = "#10B981" if pct >= 0 else "#EF4444"
    else:
        delta_txt   = "—"
        delta_color = "#FFFFFF" if is_dark else pal["text"]

    # --- CSS (scoped) ---
    st.markdown(f"""
    <style>
      .kpi-wrap {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 10px 0 6px;
      }}
      @media (max-width: 1200px) {{
        .kpi-wrap {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      }}
      @media (max-width: 700px) {{
        .kpi-wrap {{ grid-template-columns: 1fr; }}
      }}

      .kpi-card {{
        background: {kpi_bg};
        border: 1px solid {kpi_border};
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 136px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;         /* center horizontally */
        text-align: center;          /* center text */
        box-shadow: {kpi_shadow};
      }}

      .kpi-label {{
        color: {kpi_label_color};
        font-size: 1.28rem;
        font-weight: 700;
        line-height: 1.15;
        margin: 0 0 .35rem 0;
      }}

      .kpi-value {{
        color: {kpi_value_color};
        -webkit-text-fill-color: {kpi_value_color};  /* avoid dimming in dark */
        font-size: 2.12rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.01em;
        margin: 0;
      }}

      .kpi-delta {{
        color: {delta_color};
        font-size: 1.06rem;
        font-weight: 700;
        margin-top: .42rem;
      }}
    </style>
    """, unsafe_allow_html=True)

    # --- HTML ---
    kpi_html = f"""
    <div class="kpi-wrap">
      <div class="kpi-card">
        <div class="kpi-label">Total Adopted Budget</div>
        <div class="kpi-value">{money_fmt(tot_adopted, 0)}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Total Adjusted Budget</div>
        <div class="kpi-value">{money_fmt(tot_adjusted, 0)}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Total Current Year Budget</div>
        <div class="kpi-value">{money_fmt(tot_current, 0)}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Δ CY vs Adjusted</div>
        <div class="kpi-value">{money_fmt(dlt, 0)}</div>
        <div class="kpi-delta">{delta_txt}</div>
      </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)


# -------------------------------------------------
# DETAILS
# -------------------------------------------------
st.markdown("---")
st.subheader("📄 Details (first 200 rows)")
if df_f.empty:
    st.info("Adjust filters to see the detailed table.")
else:
    df_disp = df_f.copy()
    money_cols = [c for c in ["Adopted Budget","Adjusted Budget","Current Year Budget","Δ CY vs Adjusted"] if c in df_disp.columns]
    col_cfg = {mc: st.column_config.NumberColumn(label=mc, format="$%,.0f") for mc in money_cols}
    if "Δ % (vs Adjusted)" in df_disp.columns:
        col_cfg["Δ % (vs Adjusted)"] = st.column_config.NumberColumn(label="Δ % (vs Adjusted)", format="%.1f%%")
    st.dataframe(df_disp.head(200), use_container_width=True, hide_index=True, height=420, column_config=col_cfg)
    st.download_button(
        "⬇️ Download CSV (all filtered rows)",
        data=df_disp.to_csv(index=False).encode("utf-8"),
        file_name="details_all_columns.csv",
        mime="text/csv",
    )

# -------------------------------------------------
# VISUALS
# -------------------------------------------------
st.markdown("---")
st.subheader("📈 Visuals")
t1, t2, t3 = st.tabs(["📊 By Budget Year", "🏢 By Department", "🧾 By Account"])

viz_df = df_f if not df_f.empty else df
if not viz_df.empty:
    for c in ["Adopted Budget","Adjusted Budget","Current Year Budget"]:
        viz_df[c] = pd.to_numeric(viz_df[c], errors="coerce").fillna(0.0)

    with t1:
        if "Budget Year" in viz_df.columns:
            p = (
                viz_df.groupby("Budget Year", as_index=False)
                .agg({"Adopted Budget":"sum","Adjusted Budget":"sum","Current Year Budget":"sum"})
                .sort_values("Budget Year")
            )
            long = p.melt("Budget Year",
                          ["Adopted Budget","Adjusted Budget","Current Year Budget"],
                          var_name="Type", value_name="Amount")
            chart = (
                alt.Chart(long).mark_bar().encode(
                    x=alt.X("Budget Year:O", title="Budget Year"),
                    y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f")),
                    color=alt.Color("Type:N", title=""),
                    xOffset="Type:N",
                    tooltip=["Budget Year","Type",alt.Tooltip("Amount:Q", format=",.0f")]
                ).properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No 'Budget Year' column.")

    with t2:
        key = "Department Description"
        if key in viz_df.columns:
            p = (
                viz_df.groupby(key, as_index=False)
                .agg({"Adopted Budget":"sum","Adjusted Budget":"sum","Current Year Budget":"sum"})
            )
            p["Δ"] = p["Current Year Budget"] - p["Adjusted Budget"]
            order = p.sort_values("Δ", ascending=False)[key].astype(str).tolist()
            long = p.melt(key,
                          ["Adopted Budget","Adjusted Budget","Current Year Budget"],
                          var_name="Type", value_name="Amount")
            chart = (
                alt.Chart(long).mark_bar().encode(
                    x=alt.X(f"{key}:N", sort=order, axis=alt.Axis(labelAngle=-45), title="Department"),
                    y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f")),
                    color=alt.Color("Type:N", title=""),
                    xOffset="Type:N",
                    tooltip=[key, "Type", alt.Tooltip("Amount:Q", format=",.0f")],
                ).properties(height=420)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No Department column.")

    with t3:
        key = "Account Name"
        if key in viz_df.columns:
            p = (
                viz_df.groupby(key, as_index=False)
                .agg({"Adopted Budget":"sum","Adjusted Budget":"sum","Current Year Budget":"sum"})
            )
            p["Δ"] = p["Current Year Budget"] - p["Adjusted Budget"]
            order = p.sort_values("Δ", ascending=False)[key].astype(str).tolist()
            long = p.melt(key,
                          ["Adopted Budget","Adjusted Budget","Current Year Budget"],
                          var_name="Type", value_name="Amount")
            chart = (
                alt.Chart(long).mark_bar().encode(
                    x=alt.X(f"{key}:N", sort=order, axis=alt.Axis(labelAngle=-45), title="Account"),
                    y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f")),
                    color=alt.Color("Type:N", title=""),
                    xOffset="Type:N",
                    tooltip=[key, "Type", alt.Tooltip("Amount:Q", format=",.0f")],
                ).properties(height=420)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No Account column.")
else:
    st.info("Adjust filters to render charts.")

# ======================================================
# RAG (optional)
# ======================================================
CACHE_DIR = Path(".faiss_cache_v4")
CACHE_DIR.mkdir(exist_ok=True)
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

def df_to_docs_for_rag(df_ctx: pd.DataFrame, cap_rows: int = 600) -> list:
    if df_ctx.empty:
        return []
    docs = []

    def _doc(txt, meta):
        if Document is not None:
            return Document(page_content=txt, metadata=meta)
        return {"page_content": txt, "metadata": meta}

    d = df_ctx.copy()
    for c in ["Adopted Budget","Adjusted Budget","Current Year Budget"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    if "Budget Year" in d.columns:
        by_year = (
            d.groupby("Budget Year", as_index=False)
             .agg({"Adopted Budget":"sum","Adjusted Budget":"sum","Current Year Budget":"sum"})
        )
        docs.append(_doc("[Overview]\n" + by_year.to_string(index=False), {"type":"overview"}))

    if "Department Description" in d.columns:
        g = (d.groupby("Department Description", as_index=False)
               .agg(Adjusted=("Adjusted Budget","sum"),
                    Current=("Current Year Budget","sum"),
                    Delta=("Δ CY vs Adjusted","sum")))
        g["absd"] = g["Delta"].abs()
        g = g.nlargest(200, "absd")
        for _, r in g.iterrows():
            txt = (
                f"[Department]\nName: {r['Department Description']}\n"
                f"Adjusted: {r['Adjusted']:.2f}\nCurrent: {r['Current']:.2f}\nDelta: {r['Delta']:.2f}"
            )
            docs.append(_doc(txt, {"type":"dept","dept":r["Department Description"]}))

    if "Cost Center Name" in d.columns:
        g = (d.groupby("Cost Center Name", as_index=False)
               .agg(Adjusted=("Adjusted Budget","sum"),
                    Current=("Current Year Budget","sum")))
        g = g.nlargest(150, "Current")
        for _, r in g.iterrows():
            txt = (
                f"[CostCenter]\nName: {r['Cost Center Name']}\n"
                f"Adjusted: {r['Adjusted']:.2f}\nCurrent: {r['Current']:.2f}"
            )
            docs.append(_doc(txt, {"type":"cc","cost_center":r["Cost Center Name"]}))
    return docs[:cap_rows]

def _docs_fingerprint(docs: list, embed_model: str) -> str:
    m = hashlib.md5(); m.update(embed_model.encode("utf-8"))
    for d in docs:
        content = getattr(d, "page_content", None) or d.get("page_content", "")
        meta = getattr(d, "metadata", None) or d.get("metadata", {})
        m.update(content.encode("utf-8"))
        m.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    return m.hexdigest()[:16]

def _index_dir(fp: str) -> Path:
    return CACHE_DIR / f"faiss_{fp}"

@st.cache_resource(show_spinner=False)
def build_vectorstore(full_df: pd.DataFrame, api_key: str):
    if (not _HAS_LANGCHAIN) or (not api_key):
        return None, "LangChain or OpenAI key missing"
    docs = df_to_docs_for_rag(full_df, 600)
    if not docs:
        return None, "No docs to index"

    try:
        embeddings = OpenAIEmbeddings(api_key=api_key, model=DEFAULT_EMBED_MODEL)
    except Exception as e:
        return None, f"Embedding init failed: {e}"

    fp = _docs_fingerprint(docs, DEFAULT_EMBED_MODEL)
    cache_path = _index_dir(fp)
    if cache_path.exists():
        try:
            vs = FAISS.load_local(str(cache_path), embeddings=embeddings, allow_dangerous_deserialization=True)
            return vs, f"loaded:{fp}"
        except Exception:
            pass

    try:
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(str(cache_path))
        return vs, f"built:{fp}"
    except Exception as e:
        return None, f"FAISS build failed: {e}"

def extract_candidates_from_docs(docs: list) -> Dict[str, set]:
    depts, ccs = set(), set()
    for d in docs:
        text = getattr(d, "page_content", "") if hasattr(d, "page_content") else d.get("page_content","")
        m = re.search(r"(?:Dept|Department)\s*:\s*(.+)", text)
        if m:
            depts.add(m.group(1).strip())
        m2 = re.search(r"(?:Name|CostCenter)\s*:\s*(.+)", text)
        if m2 and "CostCenter" in text:
            ccs.add(m2.group(1).strip())
    return {"departments": depts, "cost_centers": ccs}

# -------------------------------------------------
# numeric-ish matching helpers (ONE)
# -------------------------------------------------
NUMERICISH_COLUMNS = {
    "Fund Number",
    "Department Number",
    "Cost Center Number",
    "Account Number",
    "Categories Code",
    "CFW Division Code",
    "Bureau Code",
    "Command Code",
    "Fund Category Code",
    "Reporting Type Code",
    "Fund Group Code",
    "Budget Code Number",
    "WKAccountNumber",
}

def normalize_numericish(val: Any) -> str:
    return re.sub(r"[^0-9]", "", str(val)).strip()

def match_df_by_column_value(df_scope: pd.DataFrame, col: str, raw_val: str) -> pd.DataFrame:
    if col not in df_scope.columns:
        return df_scope.iloc[0:0]

    ser = df_scope[col].astype(str).str.strip()
    val = str(raw_val).strip()

    if col in NUMERICISH_COLUMNS:
        target_norm = normalize_numericish(val)
        ser_norm = ser.map(normalize_numericish)
        exact = df_scope.loc[ser_norm == target_norm]
        if not exact.empty:
            return exact
        return df_scope.iloc[0:0]

    mask = ser.str.casefold() == val.casefold()
    if (not mask.any()) and col.lower() == "department number" and val.isdigit():
        zp = val.zfill(3)
        mask = ser.str.casefold() == zp.casefold()
    exact_df = df_scope.loc[mask]
    if not exact_df.empty:
        return exact_df
    contains = df_scope.loc[ser.str.casefold().str.contains(re.escape(val.casefold()), na=False)]
    return contains

# -------------------------------------------------
# free-text extractor
# -------------------------------------------------
def extract_free_text(q: str) -> str:
    q = q.strip()
    m = re.search(r'"([^"]+)"', q)
    if m: return m.group(1).strip()
    m = re.search(r"'([^']+)'", q)
    if m: return m.group(1).strip()
    m = re.search(r"\bfor\b(.+)$", q, flags=re.IGNORECASE)
    if m: return m.group(1).strip().rstrip(" ?.")
    return q

# -------------------------------------------------
# AI / Q&A SECTION
# -------------------------------------------------
st.markdown("---")
st.markdown('<div class="ai-box"><h3>AI Budget Query</h3></div>', unsafe_allow_html=True)

left, right = st.columns([1, 2])

def build_compact_summary(df_in: pd.DataFrame) -> str:
    if df_in.empty:
        return "No rows."
    d = df_in.copy()
    for c in ["Adopted Budget","Adjusted Budget","Current Year Budget"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    if "Budget Year" in d.columns:
        by_year = (
            d.groupby("Budget Year", as_index=False)
             .agg({"Adopted Budget":"sum","Adjusted Budget":"sum","Current Year Budget":"sum"})
        )
        return "By Budget Year:\n" + by_year.to_string(index=False)
    return "No grouped summary."

with left:
    if st.button("📈 Quick Analysis", use_container_width=True):
        scope = df_f if not df_f.empty else df
        if scope.empty:
            st.warning("No rows in current data.")
        else:
            prompt = build_compact_summary(scope) + "\nProvide 5–7 numeric bullets and 2–3 actions."
            ans = call_openai("You are a senior FP&A analyst. Be concise, numeric, and practical.", prompt)
            st.markdown(f"""<div class="ai-result">{ans}</div>""", unsafe_allow_html=True)

# ID-first patterns
ID_FIRST_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bfund\s+number\s*(?:the\s+)?(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Fund Number"),
    (re.compile(r"\bfund\s+description\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Fund Description"),
    (re.compile(r"\bdepartment\s+number\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Department Number"),
    (re.compile(r"\bdept\s+number\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Department Number"),
    (re.compile(r"\bdepartment\s+description\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Department Description"),
    (re.compile(r"\bdept\s+description\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Department Description"),
    (re.compile(r"\bbureau\s+description\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Bureau Description"),
    (re.compile(r"\bbureau\s+desc\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Bureau Description"),
    (re.compile(r"\bcommand\s+description\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Command Description"),
    (re.compile(r"\bcommand\s+desc\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Command Description"),
    (re.compile(r"\bcfw\s+division\s+description\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "CFW Division Description"),
    (re.compile(r"\bdivision\s+description\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "CFW Division Description"),
    (re.compile(r"\bcost\s*center\s*name\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Cost Center Name"),
    (re.compile(r"\baccount\s+number\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Account Number"),
    (re.compile(r"\bcost\s*center\s*number\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Cost Center Number"),
    (re.compile(r"\baccount\s+name\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Account Name"),

    (re.compile(r"\blicense\s*(?:&|and)\s*permits\s*revenue\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "License & Permits Revenue"),
    (re.compile(r"\blicense\s*permits\s*revenue\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "License & Permits Revenue"),

    (re.compile(r"\bwk\s*account\s*name\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "WKAccountName"),
    (re.compile(r"\bwk\s*account\s*number\s*(?:is|=|:)?\s*([0-9A-Za-z\-\_]+)", re.IGNORECASE), "WKAccountNumber"),

    # code-ish
    (re.compile(r"\bcommand\s+code\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Command Code"),
    (re.compile(r"\bbureau\s+code\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Bureau Code"),
    (re.compile(r"\bcfw\s+division\s+code\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "CFW Division Code"),
    (re.compile(r"\bfund\s+category\s+code\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Fund Category Code"),
    (re.compile(r"\breporting\s+type\s+code\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Reporting Type Code"),
    (re.compile(r"\bfund\s+group\s+code\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Fund Group Code"),
    (re.compile(r"\bbudget\s+code\s+number\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Budget Code Number"),

    (re.compile(r"\bcategories\s+code\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Categories Code"),
    (re.compile(r"\bcategory\s+code\s*(?:is|=|:)?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "Categories Code"),

    (re.compile(r"\bfund\s+category\s+description\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Fund Category Description"),
    (re.compile(r"\bfund\s+group\s+description\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Fund Group Description"),
    (re.compile(r"\bbudget\s+code\s+description\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "Budget Code Description"),
    (re.compile(r"\blicense\s*&\s*permits\s*revenue\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "License & Permits Revenue"),
    (re.compile(r"\blicense\s+and\s+permits\s+revenue\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE), "License & Permits Revenue"),
]

def try_id_first(user_q: str, df_scope: pd.DataFrame) -> Optional[Tuple[str, pd.DataFrame]]:
    for pat, col in ID_FIRST_PATTERNS:
        m = pat.search(user_q)
        if m and col in df_scope.columns:
            raw_val = m.group(1).strip()
            sub = match_df_by_column_value(df_scope, col, raw_val)
            if not sub.empty:
                return col, sub
    return None

with right:
    st.markdown(
        "<p style='font-size:1.15rem; font-weight:600; margin-bottom:0.35rem;'>"
        "💬 Ask a question (example: 'What is the current year budget for fund number 20102')"
        "</p>",
        unsafe_allow_html=True,
    )
    user_q = st.text_input(
        "",
        value="What is the current year budget for fund number 20102",
        key="q_v4_fixed",
        label_visibility="collapsed",
    )
    if user_q:
        metric_col = parse_metric_from_question(user_q)
        scope_df = df_f if not df_f.empty else df

        answer_lines: List[str] = []
        detail_df_list: List[pd.DataFrame] = []

        # 1) ID-FIRST
        id_hit = try_id_first(user_q, scope_df)
        if id_hit is not None:
            colname, sub = id_hit
            total_val = float(sub[metric_col].sum())
            shown_val = sub[colname].astype(str).iloc[0]
            answer_lines.append(f"{metric_col} for {colname} = {shown_val}:   {money_fmt(total_val, 0)}")
            detail_df_list.append(sub.assign(_matched_column=colname))
        else:
            # 2) explicit column wording
            explicit_col = extract_explicit_column(user_q)
            explicit_val = extract_value_after_colname(user_q, explicit_col) if explicit_col else None
            if explicit_col and explicit_val and explicit_col in scope_df.columns:
                sub = match_df_by_column_value(scope_df, explicit_col, explicit_val)
                if sub.empty:
                    ser = scope_df[explicit_col].astype(str).str.casefold()
                    sub = scope_df.loc[ser.str.contains(re.escape(explicit_val.casefold()), na=False)]
                if not sub.empty:
                    total_val = float(sub[metric_col].sum())
                    answer_lines.append(f"{metric_col} for {explicit_col} = {explicit_val}:    {money_fmt(total_val, 0)}")
                    detail_df_list.append(sub.assign(_matched_column=explicit_col, _matched_value=explicit_val))

            # 3) free-text search
            if not answer_lines:
                tail = extract_free_text(user_q)
                tail_cf = tail.casefold()
                text_cols = [
                    "Department Description",
                    "Cost Center Name",
                    "Account Name",
                    "Fund Description",
                    "Budget Code Description",
                ]
                mask_any = pd.Series(False, index=scope_df.index)
                for col in text_cols:
                    if col in scope_df.columns:
                        ser = scope_df[col].astype(str).str.casefold()
                        mask_any = mask_any | ser.str.contains(re.escape(tail_cf), na=False)
                sub = scope_df.loc[mask_any]
                if not sub.empty:
                    total_val = float(sub[metric_col].sum())
                    answer_lines.append(f"{metric_col} for anything containing “{tail}”:    {money_fmt(total_val, 0)}")
                    detail_df_list.append(sub.assign(_matched_freetext=tail))

            # 4) RAG if still nothing
            if not answer_lines:
                vs_msg = ""
                vs = None
                if _API_KEY and _HAS_LANGCHAIN:
                    vs, vs_msg = build_vectorstore(df, _API_KEY)
                if vs is not None:
                    retriever = vs.as_retriever(search_kwargs={"k": 6})
                    docs = retriever.get_relevant_documents(user_q)
                    from_rag = extract_candidates_from_docs(docs)

                    for dept in from_rag.get("departments", set()):
                        sub = match_df_by_column_value(scope_df, "Department Description", dept)
                        if not sub.empty:
                            total_val = float(sub[metric_col].sum())
                            answer_lines.append(f"{metric_col} for Department Description = {dept}:   {money_fmt(total_val, 0)}")
                            detail_df_list.append(sub.assign(_matched_department=dept))

                    if not answer_lines:
                        for cc in from_rag.get("cost_centers", set()):
                            sub = match_df_by_column_value(scope_df, "Cost Center Name", cc)
                            if not sub.empty:
                                total_val = float(sub[metric_col].sum())
                                answer_lines.append(f"{metric_col} for Cost Center Name = {cc}:   {money_fmt(total_val, 0)}")
                                detail_df_list.append(sub.assign(_matched_cost_center=cc))

                    if vs_msg and not answer_lines:
                        st.caption(vs_msg)

        # 5) fallback — totals
        if not answer_lines:
            total_val = float(scope_df[metric_col].sum())
            answer_lines.append(
                f"Could not match a specific column/value; showing total {metric_col} in scope:   {money_fmt(total_val, 0)}"
            )

        st.markdown("<div class='result-title'>Results</div>", unsafe_allow_html=True)
        html_lines = [f"<div class='result-line'>{line}</div>" for line in answer_lines]
        st.markdown("<div class='result-body'>" + "".join(html_lines) + "</div>", unsafe_allow_html=True)

        if detail_df_list:
            st.dataframe(pd.concat(detail_df_list, ignore_index=True), use_container_width=True)

# -------------------------------------------------
# extra download
# -------------------------------------------------
st.markdown("---")
if not df_f.empty:
    csv2 = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=csv2,
        file_name=f"budget_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
else:
    st.info("Adjust filters to enable downloads.")
