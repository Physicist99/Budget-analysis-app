# app.py — AI Budget Assistant (raw vs used coverage; scalable AI; exec dark)
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import os, re
from dotenv import load_dotenv
from openai import OpenAI
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
    "Executive Dark": {
        "bg": "#0B1E3E",        # navy
        "sidebar": "#071A34",
        "panel": "#0E2044",
        "card": "#0F234B",
        "text": "#FFFFFF",      # white text
        "muted": "#CBD5E1",
        "grid": "#14315F",
        "brand1": "#0B1E3E",
        "brand2": "#2F6BFF",
        "alloc": "#8FB7FF",
        "spent": "#5FB0FF",
        "forecast": "#22C55E",
        "warn": "#F59E0B",
        "ok": "#10B981"
    }
}

with st.sidebar:
    st.markdown("**🎨 Theme**")
    theme_names = list(THEMES.keys())
    default_idx = theme_names.index("Executive Dark") if "Executive Dark" in theme_names else 0
    theme_name = st.selectbox("Select", theme_names, index=default_idx)
pal = THEMES[theme_name]

# Altair theme
def _alt_theme(p):
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {"labelColor": p["text"], "titleColor": p["text"], "gridColor": p["grid"]},
            "legend": {"labelColor": p["text"], "titleColor": p["text"]},
            "title": {"color": p["text"]},
            "range": {"category": [p["alloc"], p["spent"], p["forecast"], "#9CA3AF"]}
        }
    }
alt.themes.register("fin_theme", lambda: _alt_theme(pal))
alt.themes.enable("fin_theme")

# =============================
# CSS (palette-aware) + readable sidebar controls
# =============================
st.markdown(f"""
<style>
  .stApp {{ background: {pal['bg']}; color: {pal['text']}; }}
  .block-container {{ padding-top: 1rem; color: {pal['text']}; }}
  a, p, span, label, .markdown-text-container, .stMarkdown {{ color: {pal['text']}; }}

  [data-testid="stSidebar"] > div:first-child {{ background: {pal['sidebar']}; }}
  [data-testid="stSidebar"] .stMarkdown, 
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] h1, 
  [data-testid="stSidebar"] h2, 
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] p {{ color: #E5E7EB !important; }}

  [data-testid="stSidebar"] .stSelectbox > div > div,
  [data-testid="stSidebar"] .stMultiSelect > div > div,
  [data-testid="stSidebar"] .stSlider > div > div {{
    background: #FFFFFF !important;
    color: #111827 !important;
    border-radius: 10px;
    border: 1px solid #D1D5DB;
  }}
  [data-testid="stSidebar"] .stSelectbox [role="combobox"] *,
  [data-testid="stSidebar"] .stMultiSelect [role="combobox"] * {{ color: #111827 !important; }}
  [role="listbox"] {{ background: #FFFFFF !important; color: #111827 !important; border: 1px solid #D1D5DB; }}
  [role="option"] *, [role="option"] {{ color: #111827 !important; }}

  .main-header {{
    background: linear-gradient(135deg, {pal['brand1']} 0%, {pal['brand2']} 100%);
    padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
    text-align: center; color: white; box-shadow: 0 4px 24px rgba(0,0,0,0.10);
  }}
  .filter-header {{
    background: {pal['panel']}; color: {pal['text']};
    padding: 0.75rem; border-radius: 10px; margin: 1rem 0 0.5rem 0;
    font-weight: 700; text-align: center; border: 1px solid {pal['grid']};
  }}
  .stButton > button {{
    background: {pal['brand2']}; color: white; border: none;
    border-radius: 10px; padding: 0.6rem 1rem; font-weight: 700;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08); transition: all .2s ease;
  }}
  .stButton > button:hover {{ transform: translateY(-1px); filter: brightness(1.03); }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
  .stTabs [data-baseweb="tab"] {{
    background-color: {pal['panel']}; border-radius: 10px; color: {pal['muted']};
    font-weight: 700; border: 1px solid {pal['grid']};
  }}
  .stTabs [aria-selected="true"] {{
    background: {pal['brand2']}; color: #fff; border: 1px solid {pal['brand2']};
  }}
  div[data-testid="stMetricValue"] {{ color: {pal['text']}; }}
  div[data-testid="stMetricDelta"] {{ color: {pal['muted']}; }}
  .dataframe, .stDataFrame {{ border-radius: 10px; border: 1px solid {pal['grid']}; background: {pal['card']}; color: {pal['text']}; }}
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
    background: linear-gradient(90deg, #0f172a11, #e2e8f011);
    padding: 0.75rem; border-radius: 8px; border-left: 3px solid {pal['brand2']};
    font-style: italic; color: {pal['muted']}; margin: 1rem 0;
  }}
</style>
""", unsafe_allow_html=True)

# =============================
# OpenAI client (optional)
# =============================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
            model=openai_model, temperature=temperature,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

def _n(col: str) -> str:
    return re.sub(r"\s+", " ", str(col)).strip().upper()

def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

# Column aliasing
ALIAS = {
    # REQUIRED
    "BUDGET YEAR": ["FISCAL YEAR","FY","YEAR","BUDGET_YEAR"],
    "ACCOUNTING PERIOD": ["PERIOD","PERIOD NUMBER","ACCOUNTING_PERIOD","PERIOD NO","MONTH","ACCOUNTING MONTH","PERIOD NAME"],
    "BUDGET AMOUNT": ["BUDGET","ADOPTED BUDGET","AMENDED BUDGET","BUDGET TOTAL","APPROPRIATION","APPROPRIATED AMOUNT","BUDGET_AMT"],
    "EXPENSE AMOUNT": ["EXPENDITURE AMOUNT","ACTUALS","ACTUAL EXPENSE","ACTUAL EXPENDITURE","YTD EXPENSE","AMOUNT EXPENDED","EXPENSE","ACTUAL_AMOUNT"],
    # OPTIONAL
    "DEPARTMENT ID DESCRIPTION": ["DEPARTMENT NAME","DEPARTMENT","DEPARTMENT DESC"],
    "FUND CODE DESCRIPTION": ["FUND DESCRIPTION","FUND DESC"],
    "PROGRAM DESCRIPTION": ["PROGRAM DESC"],
    "ACCOUNT TYPE": ["ACCT TYPE"],
    "ACCOUNT DESCRIPTION": ["ACCOUNT DESC"],
    "ENCUMBERED AMOUNT": ["ENCUMBRANCE","ENCUMBERED"],
    "PRE ENCUMBERED AMOUNT": ["PRE ENCUMBRANCE","PRE-ENCUMBRANCE"],
    "REVENUE AMOUNT": ["REVENUE","REV AMOUNT","REVENUE_TOTAL"],
    "DEPARTMENT ID": ["DEPT ID","DEPARTMENT_ID"],
    "FUND CODE": ["FUND","FUND_ID"],
    "PROGRAM CODE": ["PROGRAM","PROGRAM_ID"],
    "LEDGER GROUP": ["LEDGER","LEDGER GROUP NAME"],
    "ACCOUNT": ["ACCOUNT CODE","ACCT"]
}
CANON = {
    "BUDGET YEAR": "Budget_Year",
    "ACCOUNTING PERIOD": "Accounting_Period",
    "BUDGET AMOUNT": "Budget_Allocated",
    "EXPENSE AMOUNT": "Actual_Spent",
    "DEPARTMENT ID DESCRIPTION": "Department",
    "FUND CODE DESCRIPTION": "Fund_Desc",
    "PROGRAM DESCRIPTION": "Program_Desc",
    "ACCOUNT TYPE": "Account_Type",
    "ACCOUNT DESCRIPTION": "Account_Desc",
    "ENCUMBERED AMOUNT": "Encumbered",
    "PRE ENCUMBERED AMOUNT": "Pre_Encumbered",
    "REVENUE AMOUNT": "Revenue_Amount",
    "DEPARTMENT ID": "Department_ID",
    "FUND CODE": "Fund_Code",
    "PROGRAM CODE": "Program_Code",
    "LEDGER GROUP": "Ledger_Group",
    "ACCOUNT": "Account"
}

def find_raw_col(raw: pd.DataFrame, logical: str) -> str | None:
    """Return the raw column name for a logical header (e.g., 'DEPARTMENT ID DESCRIPTION')."""
    lookup = {}
    for key, syns in ALIAS.items():
        lookup[_n(key)] = key
        for s in syns: lookup[_n(s)] = key
    want = _n(logical)
    for c in raw.columns:
        if lookup.get(_n(c)) == want:
            return c
    return None

def coerce_period(val):
    if pd.isna(val): return pd.NA
    s = str(val).strip()
    m = re.match(r"^(20\d{2})[-/](0?[1-9]|1[0-2])$", s)  # YYYY-MM
    if m: return int(m.group(2))
    mon_map = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"SEPT":9,"OCT":10,"NOV":11,"DEC":12}
    up = s.upper()
    if up.startswith("SEPT"): return 9
    if up[:3] in mon_map: return mon_map[up[:3]]
    return pd.to_numeric(s, errors="coerce")

@st.cache_data
def load_budget_pull_with_raw(path: str):
    """Load raw + processed; return (df_processed, raw, coverage_stats, raw_counts)."""
    raw = _read_any(path)

    # Map headers -> canonical
    lookup = {}
    for key, syns in ALIAS.items():
        lookup[_n(key)] = key
        for s in syns: lookup[_n(s)] = key

    norm_cols = {_n(c): c for c in raw.columns}
    rename_map = {}
    for norm, orig in norm_cols.items():
        if norm in lookup:
            rename_map[orig] = CANON[lookup[norm]]
    df = raw.rename(columns=rename_map).copy()

    # Required present?
    required = ["Budget_Year","Accounting_Period","Budget_Allocated","Actual_Spent"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Budget Pull loader: missing required columns {miss}")

    # Optional defaults
    for k, v in {
        "Fund_Code": None, "Fund_Desc": None, "Department_ID": None, "Department": None,
        "Account": None, "Account_Type": None, "Account_Desc": None, "Program_Code": None,
        "Program_Desc": None, "Ledger_Group": None, "Encumbered": 0.0, "Pre_Encumbered": 0.0,
        "Revenue_Amount": 0.0
    }.items():
        if k not in df.columns: df[k] = v

    # Raw unique counts (before any drops)
    raw_dept_col = find_raw_col(raw, "DEPARTMENT ID DESCRIPTION")
    raw_accd_col = find_raw_col(raw, "ACCOUNT DESCRIPTION")
    raw_dept_n = raw[raw_dept_col].nunique(dropna=True) if raw_dept_col else 0
    raw_accd_n = raw[raw_accd_col].nunique(dropna=True) if raw_accd_col else 0

    # Coverage diagnostics (on raw)
    rp = find_raw_col(raw, "ACCOUNTING PERIOD")
    ry = find_raw_col(raw, "BUDGET YEAR")
    raw_period = pd.to_numeric(raw[rp].map(coerce_period), errors="coerce") if rp else pd.Series(dtype="float64")
    raw_year = pd.to_numeric(raw[ry], errors="coerce") if ry else pd.Series(dtype="float64")

    bad_period = (raw_period.isna()) | (raw_period < 1) | (raw_period > 12)
    bad_year = raw_year.isna()
    # Note: "no month" is a downstream artifact; we approximate: rows where either bad_period or bad_year
    total_raw = len(raw)
    rows_bad_period = int(bad_period.sum()) if not raw_period.empty else 0
    rows_bad_year = int(bad_year.sum()) if not raw_year.empty else 0

    # Parse numerics
    for c in ["Budget_Year","Department_ID","Account","Program_Code","Fund_Code"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fiscal mapping
    F2C = {1:10, 2:11, 3:12, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7, 11:8, 12:9}
    df["Accounting_Period"] = pd.to_numeric(df["Accounting_Period"].map(coerce_period), errors="coerce")
    df = df[(df["Accounting_Period"] >= 1) & (df["Accounting_Period"] <= 12)].copy()

    # Money
    for c in ["Budget_Allocated","Actual_Spent","Encumbered","Pre_Encumbered","Revenue_Amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Calendar month/year
    df["Cal_Month"] = df["Accounting_Period"].map(F2C).astype("Int64")
    df["Cal_Year"] = np.where(
        df["Cal_Month"] >= 10,
        df["Budget_Year"].astype("Int64") - 1,
        df["Budget_Year"].astype("Int64")
    )
    df["Month"] = pd.to_datetime(
        df["Cal_Year"].astype(str) + "-" + df["Cal_Month"].astype(str).str.zfill(2) + "-01",
        errors="coerce"
    )

    # Variance
    df["Variance"] = df["Actual_Spent"] - df["Budget_Allocated"]
    den = df["Budget_Allocated"].replace({0: np.nan})
    df["Variance_Percent"] = ((df["Variance"] / den) * 100).round(2)

    # Drop rows without Month
    before_drop = len(df)
    df = df.dropna(subset=["Month"]).copy()
    after_drop = len(df)
    rows_no_month = before_drop - after_drop

    # Helpers
    df["Year"] = df["Month"].dt.year
    df["Quarter"] = df["Month"].dt.quarter
    df["Fiscal_Year"] = df["Budget_Year"].astype("Int64")
    df["Fiscal_Quarter"] = (((df["Cal_Month"] - 10) % 12) // 3 + 1).astype("Int64")
    df = df.sort_values("Month").reset_index(drop=True)

    # Processed unique counts
    used_dept_n = df["Department"].nunique(dropna=True) if "Department" in df.columns else 0
    used_accd_n = df["Account_Desc"].nunique(dropna=True) if "Account_Desc" in df.columns else 0

    coverage = {
        "total_raw_rows": int(total_raw),
        "rows_used": int(len(df)),
        "rows_dropped_bad_period_or_year": int(max(rows_bad_period, rows_bad_year)),  # approximate overlap
        "rows_dropped_no_month": int(rows_no_month),
    }
    raw_counts = {
        "raw_dept_n": int(raw_dept_n),
        "raw_accdesc_n": int(raw_accd_n),
        "used_dept_n": int(used_dept_n),
        "used_accdesc_n": int(used_accd_n),
    }
    return df, raw, coverage, raw_counts

# =============================
# Load Data (fixed path, CSV fallback)
# =============================
ACTUALS_PATH = "FY 2021 Budget Pull.xlsx"

try:
    df, raw_df, coverage_stats, raw_counts = load_budget_pull_with_raw(ACTUALS_PATH)
except ImportError as e:
    base, _ = os.path.splitext(ACTUALS_PATH)
    csv_path = base + ".csv"
    if os.path.exists(csv_path):
        df, raw_df, coverage_stats, raw_counts = load_budget_pull_with_raw(csv_path)
    else:
        st.error("❌ Excel engine missing. Install `openpyxl` for .xlsx (and `xlrd` for .xls), "
                 "or provide a CSV with the same base name.\n\n"
                 f"Details: {e}")
        st.stop()
except Exception as e:
    st.error(f"❌ Could not load data at {ACTUALS_PATH}: {e}")
    st.stop()

# =============================
# Sidebar Filters
# =============================
with st.sidebar:
    st.markdown('<div class="filter-header">🎛️ Control Panel</div>', unsafe_allow_html=True)

    st.markdown("**📊 Dataset Overview**")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("📋 Rows (raw → used)", f"{coverage_stats['total_raw_rows']:,} → {coverage_stats['rows_used']:,}")
        st.metric("🏢 Departments (raw → used)", f"{raw_counts['raw_dept_n']:,} → {raw_counts['used_dept_n']:,}")
    with c2:
        st.metric("🧾 Account Descriptions (raw → used)", f"{raw_counts['raw_accdesc_n']:,} → {raw_counts['used_accdesc_n']:,}")
        fy_min = int(df["Fiscal_Year"].min()) if "Fiscal_Year" in df.columns else int(df["Year"].min())
        fy_max = int(df["Fiscal_Year"].max()) if "Fiscal_Year" in df.columns else int(df["Year"].max())
        st.metric("📅 Span (Fiscal)", f"{fy_min}–{fy_max}")

    st.caption(
        f"Dropped rows due to period/year/month issues: "
        f"{coverage_stats['rows_dropped_bad_period_or_year']:,} + {coverage_stats['rows_dropped_no_month']:,}."
    )

    st.markdown("---")
    st.markdown('<div class="filter-header">🔍 Primary Filters</div>', unsafe_allow_html=True)

    # Primary: Department, Account Description, Account Type (from processed df)
    dept_opts = sorted(df["Department"].dropna().unique()) if "Department" in df.columns else []
    acct_desc_opts = sorted(df["Account_Desc"].dropna().unique()) if "Account_Desc" in df.columns else []
    acct_type_opts = sorted(df["Account_Type"].dropna().unique()) if "Account_Type" in df.columns else []

    dept_sel = st.multiselect("🏢 Department(s)", dept_opts, default=dept_opts) if dept_opts else []
    acct_desc_sel = st.multiselect("🧾 Account Description(s)", acct_desc_opts, default=acct_desc_opts) if acct_desc_opts else []
    acct_type_sel = st.multiselect("🏷️ Account Type(s)", acct_type_opts, default=acct_type_opts) if acct_type_opts else []

    # Month slider (handles one-month edge-case)
    min_ts = pd.to_datetime(df["Month"].min()); max_ts = pd.to_datetime(df["Month"].max())
    if pd.isna(min_ts) or pd.isna(max_ts):
        st.error("No valid dates found in Month column."); st.stop()
    min_m = min_ts.to_pydatetime(); max_m = max_ts.to_pydatetime()
    if min_m >= max_m:
        from pandas.tseries.offsets import MonthBegin
        faux_max = (pd.Timestamp(min_m) + MonthBegin(1)).to_pydatetime()
        st.caption("ℹ️ Only one month available; showing a 1-month slider window.")
        date_range = st.slider("📅 Month Range", value=(min_m, faux_max),
                               min_value=min_m, max_value=faux_max, format="YYYY-MM")
    else:
        date_range = st.slider("📅 Month Range", value=(min_m, max_m),
                               min_value=min_m, max_value=max_m, format="YYYY-MM")

    st.markdown('<div class="filter-header">🔧 Options</div>', unsafe_allow_html=True)
    include_commitments = st.checkbox("Include Encumbrances in 'Spent' (Actual + Encumbered + Pre-Enc.)", value=False)

    # Effective columns preview
    df_preview = df.copy()
    if include_commitments:
        df_preview["Actual_Effective"] = df_preview["Actual_Spent"] + df_preview.get("Encumbered", 0) + df_preview.get("Pre_Encumbered", 0)
    else:
        df_preview["Actual_Effective"] = df_preview["Actual_Spent"]
    df_preview["Variance_Effective"] = df_preview["Actual_Effective"] - df_preview["Budget_Allocated"]
    den_prev = df_preview["Budget_Allocated"].replace({0: np.nan})
    df_preview["Variance_Percent_Effective"] = (df_preview["Variance_Effective"] / den_prev) * 100

    st.markdown('<div class="filter-header">⚙️ Advanced Filters</div>', unsafe_allow_html=True)
    year_options = sorted(df["Fiscal_Year"].dropna().unique()) if "Fiscal_Year" in df.columns else sorted(df["Year"].dropna().unique())
    selected_years = st.multiselect("📅 Filter by Fiscal Year(s)", year_options, default=year_options)

    v_series = df_preview["Variance_Percent_Effective"]
    vmin = float(v_series.dropna().min()) if v_series.notna().any() else -100.0
    vmax = float(v_series.dropna().max()) if v_series.notna().any() else 100.0
    variance_range = st.slider("📊 Variance % Range (effective)",
                               min_value=vmin, max_value=vmax,
                               value=(vmin, vmax), step=1.0)

    a_series = df_preview["Actual_Effective"]
    amin, amax = float(a_series.min()), float(a_series.max())
    amount_range = st.slider("💳 Spent Range ($, effective)",
                             min_value=amin, max_value=amax, value=(amin, amax),
                             step=1000.0, format="$%.0f")

    budget_performance = st.selectbox(
        "🎯 Budget Performance",
        ["All","Over Budget (>0%)","Under Budget (<0%)","On Target (±5%)","Significant Variance (>±10%)"]
    )

    with st.expander("Additional Filters (Fund / Program / Ledger)"):
        sel_fund = st.multiselect("🏦 Fund Description",
                                  sorted(df["Fund_Desc"].dropna().unique())) if "Fund_Desc" in df.columns else []
        sel_prog = st.multiselect("📘 Program Description",
                                  sorted(df["Program_Desc"].dropna().unique())) if "Program_Desc" in df.columns else []
        sel_ledger = st.multiselect("📚 Ledger Group",
                                    sorted(df["Ledger_Group"].dropna().unique())) if "Ledger_Group" in df.columns else []

    st.markdown('<div class="filter-header">🧠 AI Settings</div>', unsafe_allow_html=True)
    ai_rows_cap = st.slider(
        "AI context row cap (after aggregation)",
        min_value=50, max_value=2000, value=400, step=50,
        help="How many grouped rows to include in the AI prompt (top by absolute variance)."
    )

# =============================
# Apply filters
# =============================
df_work = df.copy()
if include_commitments:
    df_work["Actual_Effective"] = df_work["Actual_Spent"] + df_work.get("Encumbered", 0) + df_work.get("Pre_Encumbered", 0)
else:
    df_work["Actual_Effective"] = df_work["Actual_Spent"]
df_work["Variance_Effective"] = df_work["Actual_Effective"] - df_work["Budget_Allocated"]
den_work = df_work["Budget_Allocated"].replace({0: np.nan})
df_work["Variance_Percent_Effective"] = (df_work["Variance_Effective"] / den_work) * 100

mask = (
    (df_work["Department"].isin(dept_sel) if dept_sel else True) &
    (df_work["Account_Desc"].isin(acct_desc_sel) if acct_desc_sel else True) &
    (df_work["Account_Type"].isin(acct_type_sel) if acct_type_sel else True) &
    (df_work["Month"] >= pd.to_datetime(date_range[0])) &
    (df_work["Month"] <= pd.to_datetime(date_range[1])) &
    (df_work["Fiscal_Year"].isin(selected_years) if "Fiscal_Year" in df_work.columns else df_work["Year"].isin(selected_years)) &
    df_work["Variance_Percent_Effective"].between(variance_range[0], variance_range[1]) &
    df_work["Actual_Effective"].between(amount_range[0], amount_range[1])
)

if "Fund_Desc" in df_work.columns and sel_fund:
    mask &= df_work["Fund_Desc"].isin(sel_fund)
if "Program_Desc" in df_work.columns and sel_prog:
    mask &= df_work["Program_Desc"].isin(sel_prog)
if "Ledger_Group" in df_work.columns and sel_ledger:
    mask &= df_work["Ledger_Group"].isin(sel_ledger)

if budget_performance == "Over Budget (>0%)":
    mask &= df_work["Variance_Percent_Effective"] > 0
elif budget_performance == "Under Budget (<0%)":
    mask &= df_work["Variance_Percent_Effective"] < 0
elif budget_performance == "On Target (±5%)":
    mask &= df_work["Variance_Percent_Effective"].between(-5, 5)
elif budget_performance == "Significant Variance (>±10%)":
    mask &= (df_work["Variance_Percent_Effective"] > 10) | (df_work["Variance_Percent_Effective"] < -10)

df_f = df_work.loc[mask].copy()
if df_f.empty:
    st.markdown(f"""
    <div class="main-header" style="margin-top:0;">
      <h1>AI Budget Forecast & Analysis</h1>
      <p style="font-size:.95rem;opacity:.9;">No data matches your current filters. Adjust filters to see results.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =============================
# Header & KPIs
# =============================
st.markdown(f"""
<div class="main-header">
  <h1>AI Budget Forecast & Analysis</h1>
  <p style="font-size:1.05rem;margin:.4rem 0;">Professional Visual Analytics (Fiscal-Year Aware)</p>
  <p style="font-size:.9rem;opacity:.9;">Overview metrics show raw → used to explain differences.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
total_budget = df_f["Budget_Allocated"].sum()
total_spent_eff = df_f["Actual_Effective"].sum()
total_var_eff = df_f["Variance_Effective"].sum()
var_pct_eff = (total_var_eff / total_budget * 100) if total_budget else 0.0

with col1:
    st.metric("Total Budget", money_fmt(total_budget), delta=f"{len(df_f):,} records")
with col2:
    st.metric("Total Spent" + (" (Incl. Enc.)" if include_commitments else ""), money_fmt(total_spent_eff),
              delta=f"vs Budget: {var_pct_eff:+.1f}%",
              delta_color="normal" if abs(var_pct_eff) < 5 else "inverse")
with col3:
    st.metric("Net Variance (Effective)", money_fmt(total_var_eff), delta=f"{var_pct_eff:+.2f}%",
              delta_color="inverse" if total_var_eff > 0 else "normal")
with col4:
    efficiency = max(0.0, 100 - abs(var_pct_eff))
    tag = "Excellent" if efficiency > 95 else "Good" if efficiency > 85 else "Needs Review"
    st.metric("Budget Efficiency", f"{efficiency:.1f}%", delta=tag,
              delta_color="normal" if efficiency > 90 else "inverse")

# Quick stats (effective)
c1, c2, c3, c4 = st.columns(4)
over_b = (df_f["Variance_Percent_Effective"] > 0).sum()
under_b = (df_f["Variance_Percent_Effective"] < 0).sum()
on_tgt = df_f["Variance_Percent_Effective"].between(-2, 2).sum()
avg_var = df_f["Variance_Percent_Effective"].mean()
with c1: st.metric("Over Budget", f"{over_b}", f"{over_b/len(df_f)*100:.1f}%")
with c2: st.metric("Under Budget", f"{under_b}", f"{under_b/len(df_f)*100:.1f}%")
with c3: st.metric("On Target (±2%)", f"{on_tgt}", f"{on_tgt/len(df_f)*100:.1f}%")
with c4: st.metric("Avg Variance (Eff.)", f"{avg_var:+.1f}%", "Overall")

# =============================
# Detailed Table
# =============================
st.markdown("---")
st.subheader("📊 Detailed Analysis")

display_df = df_f.copy()
display_df["Month_Display"] = display_df["Month"].dt.strftime("%Y-%m")
display_df["Budget_Display"] = display_df["Budget_Allocated"].apply(lambda x: f"${x:,.0f}")
display_df["Actual_Display"] = display_df["Actual_Effective"].apply(lambda x: f"${x:,.0f}")
display_df["Variance_Display"] = display_df["Variance_Effective"].apply(lambda x: f"${x:+,.0f}")
display_df["Variance_Pct_Display"] = display_df["Variance_Percent_Effective"].apply(
    lambda x: "—" if pd.isna(x) else f"{x:+.1f}%"
)

cols_show = [
    "Month_Display","Fiscal_Year","Fiscal_Quarter","Accounting_Period","Department",
    "Account_Type","Account_Desc","Fund_Desc","Program_Desc","Ledger_Group",
    "Budget_Display","Actual_Display","Variance_Display","Variance_Pct_Display"
]
cols_show = [c for c in cols_show if c in display_df.columns]

st.dataframe(
    display_df[cols_show].rename(columns={
        "Month_Display":"Month", "Accounting_Period":"Fiscal Period",
        "Fund_Desc":"Fund", "Program_Desc":"Program", "Ledger_Group":"Ledger Group",
        "Budget_Display":"Budget", "Actual_Display":"Actual (Eff.)",
        "Variance_Display":"Variance ($)", "Variance_Pct_Display":"Variance (%)"
    }),
    use_container_width=True, height=420
)

# =============================
# Visual Analytics
# =============================
st.markdown("---")
st.subheader("📈 Visual Analytics")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Monthly Trends",
    "🏢 By Department",
    "🏷️ By Account Type",
    "🧾 By Account Description",
    "🔎 By Any Column"
])

with tab1:
    monthly = df_f.groupby("Month", as_index=False)[["Budget_Allocated","Actual_Effective"]].sum().sort_values("Month")
    m_long = monthly.melt(id_vars="Month", value_vars=["Budget_Allocated","Actual_Effective"],
                          var_name="Type", value_name="Amount")
    m_long["Type"] = m_long["Type"].map({"Budget_Allocated":"Allocated","Actual_Effective":"Spent"})
    chart_monthly = (
        alt.Chart(m_long)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(Month):O", title="Month"),
            y=alt.Y("Amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[pal["alloc"], pal["spent"]])),
            xOffset="Type:N",
            tooltip=[
                alt.Tooltip("yearmonth(Month):O", title="Month"),
                alt.Tooltip("Type:N"),
                alt.Tooltip("Amount:Q", title="Amount", format=",.0f")
            ]
        ).properties(height=380)
    )
    st.altair_chart(chart_monthly, use_container_width=True)

with tab2:
    if "Department" in df_f.columns:
        dept_tot = df_f.groupby("Department", as_index=False).agg(
            Allocated=("Budget_Allocated","sum"),
            Spent=("Actual_Effective","sum")
        )
        dept_tot["Variance"] = dept_tot["Spent"] - dept_tot["Allocated"]
        order = dept_tot.sort_values("Variance", ascending=False)["Department"].astype(str).tolist()
        dept_long = dept_tot.melt(id_vars="Department", value_vars=["Allocated","Spent"],
                                  var_name="Type", value_name="Amount")
        chart_dept = (
            alt.Chart(dept_long)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("Department:N", sort=order, axis=alt.Axis(labelAngle=-45), title="Department"),
                y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f"), title="Amount ($)"),
                color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[pal["alloc"], pal["spent"]])),
                xOffset="Type:N",
                tooltip=[alt.Tooltip("Department:N"), alt.Tooltip("Type:N"),
                         alt.Tooltip("Amount:Q", title="Amount", format=",.0f")]
            ).properties(height=420)
        )
        st.altair_chart(chart_dept, use_container_width=True)
    else:
        st.info("No Department column found.")

with tab3:
    if "Account_Type" in df_f.columns and df_f["Account_Type"].notna().any():
        type_tot = df_f.groupby("Account_Type", as_index=False).agg(
            Allocated=("Budget_Allocated","sum"),
            Spent=("Actual_Effective","sum")
        )
        type_tot["Variance"] = type_tot["Spent"] - type_tot["Allocated"]
        order_type = type_tot.sort_values("Variance", ascending=False)["Account_Type"].astype(str).tolist()
        type_long = type_tot.melt(id_vars="Account_Type", value_vars=["Allocated","Spent"],
                                  var_name="Type", value_name="Amount")
        chart_type = (
            alt.Chart(type_long)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("Account_Type:N", sort=order_type, axis=alt.Axis(labelAngle=-45), title="Account Type"),
                y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f"), title="Amount ($)"),
                color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[pal["alloc"], pal["spent"]])),
                xOffset="Type:N",
                tooltip=[alt.Tooltip("Account_Type:N"), alt.Tooltip("Type:N"),
                         alt.Tooltip("Amount:Q", title="Amount", format=",.0f")]
            ).properties(height=420)
        )
        st.altair_chart(chart_type, use_container_width=True)
    else:
        st.info("No Account Type data available.")

with tab4:
    if "Account_Desc" in df_f.columns and df_f["Account_Desc"].notna().any():
        desc_tot = df_f.groupby("Account_Desc", as_index=False).agg(
            Allocated=("Budget_Allocated","sum"),
            Spent=("Actual_Effective","sum")
        )
        desc_tot["Variance"] = desc_tot["Spent"] - desc_tot["Allocated"]
        order_desc = desc_tot.sort_values("Variance", ascending=False)["Account_Desc"].astype(str).tolist()
        desc_long = desc_tot.melt(id_vars="Account_Desc", value_vars=["Allocated","Spent"],
                                  var_name="Type", value_name="Amount")
        chart_desc = (
            alt.Chart(desc_long)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("Account_Desc:N", sort=order_desc, axis=alt.Axis(labelAngle=-45), title="Account Description"),
                y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f"), title="Amount ($)"),
                color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[pal["alloc"], pal["spent"]])),
                xOffset="Type:N",
                tooltip=[alt.Tooltip("Account_Desc:N"), alt.Tooltip("Type:N"),
                         alt.Tooltip("Amount:Q", title="Amount", format=",.0f")]
            ).properties(height=420)
        )
        st.altair_chart(chart_desc, use_container_width=True)
    else:
        st.info("No Account Description data available.")

with tab5:
    dims = [
        "Department","Fund_Desc","Program_Desc","Ledger_Group",
        "Account_Type","Account_Desc",
        "Fund_Code","Program_Code","Account","Department_ID","Fiscal_Year","Fiscal_Quarter","Accounting_Period"
    ]
    dims = [d for d in dims if d in df_f.columns]
    if dims:
        col = st.selectbox("Group by any column", dims, index=0, key="any_col")
        any_tot = df_f.groupby(col, as_index=False).agg(
            Allocated=("Budget_Allocated","sum"),
            Spent=("Actual_Effective","sum")
        )
        any_tot["Variance"] = any_tot["Spent"] - any_tot["Allocated"]
        order_any = any_tot.sort_values("Variance", ascending=False)[col].astype(str).tolist()
        any_long = any_tot.melt(id_vars=col, value_vars=["Allocated","Spent"],
                                var_name="Type", value_name="Amount")
        chart_any = (
            alt.Chart(any_long)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X(f"{col}:N", sort=order_any, axis=alt.Axis(labelAngle=-45), title=str(col)),
                y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f"), title="Amount ($)"),
                color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[pal["alloc"], pal["spent"]])),
                xOffset="Type:N",
                tooltip=[alt.Tooltip(f"{col}:N"), alt.Tooltip("Type:N"),
                         alt.Tooltip("Amount:Q", title="Amount", format=",.0f")]
            ).properties(height=420)
        )
        st.altair_chart(chart_any, use_container_width=True)
    else:
        st.info("No additional dimensions available.")

# =============================
# 🤖 AI-Powered Insights
# =============================
def tbl(df_):
    return "(none)" if df_.empty else df_.to_string(index=False)

def build_compact_summary(actuals_df: pd.DataFrame):
    a_month = actuals_df.groupby("Month", as_index=False)[["Actual_Effective","Budget_Allocated"]].sum()
    a_month["Month"] = a_month["Month"].dt.strftime("%Y-%m")
    a_dept = (actuals_df.groupby("Department", as_index=False)[["Actual_Effective","Budget_Allocated"]]
              .sum().sort_values("Actual_Effective", ascending=False).head(10)) if "Department" in actuals_df.columns else pd.DataFrame()
    a_type = (actuals_df.groupby("Account_Type", as_index=False)[["Actual_Effective","Budget_Allocated"]]
              .sum().sort_values("Actual_Effective", ascending=False).head(10)) if "Account_Type" in actuals_df.columns else pd.DataFrame()
    a_desc = (actuals_df.groupby("Account_Desc", as_index=False)[["Actual_Effective","Budget_Allocated"]]
              .sum().sort_values("Actual_Effective", ascending=False).head(10)) if "Account_Desc" in actuals_df.columns else pd.DataFrame()

    prompt = f"""
Use these compact summaries to answer succinctly.

ACTUALS — monthly (Allocated vs Spent):
{tbl(a_month.tail(24))}

ACTUALS — top departments:
{tbl(a_dept)}

ACTUALS — top by Account Type:
{tbl(a_type)}

ACTUALS — top by Account Description:
{tbl(a_desc)}
"""
    return prompt

def parse_month_year(text: str):
    month_map = {m.lower(): i for i, m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"], 1)}
    m1 = re.search(r'\b([A-Za-z]{3,9})\s+(\d{4})\b', text or "")
    if m1:
        mname = m1.group(1).lower()
        yr = int(m1.group(2)); mon = None
        for full, idx in month_map.items():
            if full.startswith(mname): mon = idx; break
        if mon: return yr, mon
    m2 = re.search(r'\b(20\d{2})[-/](0?[1-9]|1[0-2])\b', text or "")
    if m2: return int(m2.group(1)), int(m2.group(2))
    m3 = re.search(r'\b(0?[1-9]|1[0-2])[-/](20\d{2})\b', text or "")
    if m3: return int(m3.group(2)), int(m3.group(1))
    return None, None

def extract_match(text, options):
    text_l = str(text).lower()
    for opt in options:
        if str(opt).lower() in text_l:
            return opt
    return None

st.markdown("---")
st.markdown(f"""
<div class="ai-box">
  <h2>🤖 AI-Powered Insights</h2>
  <p>Get intelligent analysis and answers about your budget data</p>
</div>
""", unsafe_allow_html=True)

cols = st.columns([1, 2])

# ---- Quick Analysis ----
with cols[0]:
    st.markdown("### 📈 Quick Analysis")
    if st.button("🔍 Generate Smart Summary", use_container_width=True):
        if not client:
            st.error("❌ OpenAI API key not configured")
        else:
            compact = build_compact_summary(df_f)
            user_prompt = compact + "\nSummarize key trends and give 2–3 actionable recommendations, scoped to the current filters."
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
            st.markdown(f"""<div class="ai-result">{ans}</div>""", unsafe_allow_html=True)

# ---- Ask Questions (aggregate-first, top-K, coverage) ----
with cols[1]:
    st.markdown("### 💬 Ask Questions")
    q = st.text_input(
        "🔍 Ask about your budget data:",
        placeholder="e.g., 'Which departments overspent most in FY2021?', 'Top account descriptions by variance in 2021-11'"
    )

    if q:
        y, m = parse_month_year(q)
        df_q = df_f.copy()
        if y and m:
            df_q = df_q[df_q["Month"].dt.strftime("%Y-%m") == f"{y}-{m:02d}"]

        ql = q.lower()
        prefer_dim = None
        if "department" in ql or "dept" in ql:
            prefer_dim = "Department" if "Department" in df_q.columns else None
        elif "account type" in ql or "acct type" in ql:
            prefer_dim = "Account_Type" if "Account_Type" in df_q.columns else None
        elif "account description" in ql or "account desc" in ql or "account" in ql:
            prefer_dim = "Account_Desc" if "Account_Desc" in df_q.columns else None

        for dim in ["Department","Account_Type","Account_Desc","Fund_Desc","Program_Desc","Ledger_Group"]:
            if dim in df_q.columns:
                match = extract_match(q, df[dim].dropna().unique())
                if match:
                    df_q = df_q[df_q[dim] == match]
                    if prefer_dim == dim:
                        prefer_dim = None

        def to_pretty(df_small: pd.DataFrame) -> str:
            if df_small.empty:
                return "(no matching rows)"
            out = df_small.copy()
            if "Month" in out.columns:
                out["Month"] = pd.to_datetime(out["Month"]).dt.strftime("%Y-%m")
            for cc in ["Allocated","Spent","Variance","Budget_Allocated","Actual_Effective","Variance_Effective"]:
                if cc in out.columns:
                    out[cc] = pd.to_numeric(out[cc], errors="coerce").fillna(0).round(0)
            return out.to_string(index=False)

        coverage_note = ""
        if prefer_dim and prefer_dim in df_q.columns and not df_q.empty:
            agg = (df_q.groupby(prefer_dim, as_index=False)
                        .agg(Allocated=("Budget_Allocated","sum"),
                             Spent=("Actual_Effective","sum")))
            agg["Variance"] = agg["Spent"] - agg["Allocated"]
            agg["AbsVar"] = agg["Variance"].abs()
            agg = agg.sort_values("AbsVar", ascending=False)

            total_spent = float(agg["Spent"].sum())
            total_alloc = float(agg["Allocated"].sum())
            k = min(ai_rows_cap, len(agg))
            send = agg.head(k).drop(columns=["AbsVar"])
            covered_spent = float(send["Spent"].sum())
            covered_alloc = float(send["Allocated"].sum())
            cov_spent = (covered_spent / total_spent * 100.0) if total_spent else 0.0
            cov_alloc = (covered_alloc / total_alloc * 100.0) if total_alloc else 0.0
            coverage_note = (
                f"(Top {k} {prefer_dim} by |variance| — "
                f"covers {cov_spent:.1f}% of Spent, {cov_alloc:.1f}% of Allocated; "
                f"{len(agg)} total groups)"
            )

            send = send.rename(columns={prefer_dim: "Group"})
            primary_context = to_pretty(send[["Group","Allocated","Spent","Variance"]])
        else:
            cols = ["Month","Department","Account_Type","Account_Desc",
                    "Budget_Allocated","Actual_Effective","Variance_Effective","Variance_Percent_Effective"]
            cols = [c for c in cols if c in df_q.columns]
            slim = df_q[cols].copy() if not df_q.empty else pd.DataFrame(columns=cols)
            if not slim.empty and "Variance_Effective" in slim.columns:
                slim["_abs"] = slim["Variance_Effective"].abs()
                slim = slim.sort_values("_abs", ascending=False).drop(columns="_abs")
            slim = slim.head(ai_rows_cap)
            primary_context = to_pretty(slim)

        compact = build_compact_summary(df_f)

        final_prompt = f"""User question: {q}

Primary context:
{primary_context}
{coverage_note}

Additional summaries:
{compact}

Instructions:
- Use the primary context first (aggregated if provided); if insufficient, use the summaries.
- Provide concise, numeric bullets and brief rankings by absolute variance when relevant.
- Keep the scope to the current filters/time window.
- If more precision requires a narrower filter, state exactly what to filter by.
"""

        with st.spinner("🤔 AI thinking..."):
            ans = call_openai(
                system_msg="You are a precise financial analyst. Be concise and numeric.",
                user_msg=final_prompt,
                temperature=0.2,
                max_tokens=900
            )
        st.markdown(f"""<div class="ai-result">{ans}</div>""", unsafe_allow_html=True)

# =============================
# Tips & Download
# =============================
st.markdown("""
<div class="caption-style">
  <strong>Why raw ≠ used?</strong> We compute calendar months from fiscal period (1=Oct … 12=Sep) and the budget year.
  Rows with bad/empty period/year cannot be placed on a timeline and are excluded from analytics, which lowers the
  “used” counts. The panel above shows raw → used for transparency.
</div>
""", unsafe_allow_html=True)

csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered data (CSV)",
    data=csv,
    file_name=f"budget_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)
