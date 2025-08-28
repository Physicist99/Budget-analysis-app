# app.py
import streamlit as st
import pandas as pd
import altair as alt
import os, re, glob
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
    "Executive Dark": {  # navy background + white text
        "bg": "#0B1E3E",
        "sidebar": "#071A34",
        "panel": "#0E2044",
        "card": "#0F234B",
        "text": "#FFFFFF",
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

LOGO_LIGHT_PATH = "logo_light.png"  # dark/navy logo for light background
LOGO_DARK_PATH  = "logo_dark.png"   # white/knockout logo for dark background

# ========= Sidebar: Theme =========
with st.sidebar:
    st.markdown("**🎨 Theme**")
    theme_name = st.selectbox("Select", list(THEMES.keys()), index=0)
pal = THEMES[theme_name]

# Altair theme to keep charts consistent with UI
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
# Professional CSS (palette-aware) + readable sidebar selects
# =============================
st.markdown(f"""
<style>
  .stApp {{ background: {pal['bg']}; color: {pal['text']}; }}
  .block-container {{ padding-top: 1rem; color: {pal['text']}; }}
  a, p, span, label, .markdown-text-container, .stMarkdown {{ color: {pal['text']}; }}

  /* Sidebar container */
  [data-testid="stSidebar"] > div:first-child {{ background: {pal['sidebar']}; }}

  /* Sidebar base text stays light for headings/labels */
  [data-testid="stSidebar"] .stMarkdown, 
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] h1, 
  [data-testid="stSidebar"] h2, 
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] p {{ color: #E5E7EB !important; }}

  /* === Make sidebar selects readable (dark text on white) === */
  [data-testid="stSidebar"] .stSelectbox > div > div,
  [data-testid="stSidebar"] .stMultiSelect > div > div {{
    background: #FFFFFF !important;
    color: #111827 !important;
    border-radius: 10px;
    border: 1px solid #D1D5DB;
  }}
  [data-testid="stSidebar"] .stSelectbox [role="combobox"] *,
  [data-testid="stSidebar"] .stMultiSelect [role="combobox"] * {{ color: #111827 !important; }}
  [role="listbox"] {{ background: #FFFFFF !important; color: #111827 !important; border: 1px solid #D1D5DB; }}
  [role="option"] *, [role="option"] {{ color: #111827 !important; }}

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
  .dataframe, .stDataFrame {{ border-radius: 10px; border: 1px solid {pal['grid']}; background: {pal['card']}; color: {pal['text']}; }}

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
    background: linear-gradient(90deg, #0f172a11, #e2e8f011);
    padding: 0.75rem; border-radius: 8px; border-left: 3px solid {pal['brand2']};
    font-style: italic; color: {pal['muted']}; margin: 1rem 0;
  }}
</style>
""", unsafe_allow_html=True)

# =============================
# Sidebar brand area + logo
# =============================
with st.sidebar:
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="text-align:center; padding: 8px 0;">
            <div style="font-size:12px; opacity:.8;">Powered by</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    logo_path = LOGO_DARK_PATH if theme_name == "Executive Dark" else LOGO_LIGHT_PATH
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown(
            f"""
            <div style="text-align:center; font-weight:700; letter-spacing:.5px; margin-top:4px;">
                <span style="color:#FFFFFF;">Your</span><span style="color:#FFD166;">&nbsp;Brand</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# =============================
# OpenAI client (optional)
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
    try:
        return f"${x:,.0f}"
    except Exception:
        return "-"

def call_openai(system_msg: str, user_msg: str, temperature: float = 0.2, max_tokens: int = 900) -> str:
    if not client:
        return "OpenAI API key not configured."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

def _normalize_header(col: str) -> str:
    return re.sub(r"\s+", " ", str(col)).strip().upper()

def discover_default_data():
    """
    Find a budget file in the working dir (CSV/XLSX) using flexible patterns.
    """
    patterns = [
        "FY 2021 Budget Pull.csv",
        "FY 2021 Budget Pull.xlsx",
        "FY 2021 Budget Pull.xls",
        "FY*2021*Budget*Pull*.csv",
        "FY*2021*Budget*Pull*.xlsx",
        "FY*2021*Budget*Pull*.xls",
        "FY_2021_Budget_Pull.*",
        "FY2021*Budget*Pull*.*",
    ]
    search_dirs = list({os.getcwd(), os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()})
    for d in search_dirs:
        for pat in patterns:
            matches = glob.glob(os.path.join(d, pat))
            if matches:
                return matches[0]
    return None

@st.cache_data
def load_budget_pull(uploaded_file_or_path):
    """
    Load 'FY 2021 Budget Pull' (CSV/XLSX) with robust header synonyms & period parsing.
    Standardized columns returned (subset shown):
    Month, Year, Quarter, Budget_Year, Accounting_Period,
    Budget_Allocated, Actual_Spent, Variance, Variance_Percent,
    Encumbered, Pre_Encumbered, Revenue_Amount,
    Department, Department_ID, Fund_Code, Fund_Desc, Account, Account_Type,
    Account_Desc, Program_Code, Program_Desc, Ledger_Group, ...
    """
    # Read raw
    if uploaded_file_or_path is None:
        raise ValueError("No file provided.")
    path = uploaded_file_or_path
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")
    if path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    # Aliases
    alias = {
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
    canonical = {
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
    lookup = {}
    for key, syns in alias.items():
        lookup[_normalize_header(key)] = key
        for s in syns:
            lookup[_normalize_header(s)] = key

    # Rename to canonical app names
    norm_cols = {_normalize_header(c): c for c in raw.columns}
    rename_map = {}
    for norm, orig in norm_cols.items():
        if norm in lookup:
            canonical_key = lookup[norm]
            rename_map[orig] = canonical[canonical_key]
    df = raw.rename(columns=rename_map).copy()

    # Validate required
    required = ["Budget_Year", "Accounting_Period", "Budget_Allocated", "Actual_Spent"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns after header normalization. "
            f"Not found: {missing}. Rename your headers or extend the alias list."
        )

    # Ensure optional presence
    optional_defaults = {
        "Fund_Code": None, "Fund_Desc": None, "Department_ID": None, "Department": None,
        "Account": None, "Account_Type": None, "Account_Desc": None, "Program_Code": None,
        "Program_Desc": None, "Ledger_Group": None, "Encumbered": 0.0,
        "Pre_Encumbered": 0.0, "Revenue_Amount": 0.0
    }
    for k, v in optional_defaults.items():
        if k not in df.columns:
            df[k] = v

    # Types
    for c in ["Budget_Year","Department_ID","Account","Program_Code","Fund_Code"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Accounting_Period: accept 1..12, "Jan"/"FEB"/"March", or "YYYY-MM"
    def _coerce_period(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        m = re.match(r"^(20\d{2})[-/](0?[1-9]|1[0-2])$", s)  # YYYY-MM
        if m:
            return int(m.group(2))
        mon_map = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"SEPT":9,"OCT":10,"NOV":11,"DEC":12}
        up = s.upper()
        if up.startswith("SEPT"): return 9
        if up[:3] in mon_map: return mon_map[up[:3]]
        return pd.to_numeric(s, errors="coerce")

    df["Accounting_Period"] = df["Accounting_Period"].apply(_coerce_period)
    df["Accounting_Period"] = pd.to_numeric(df["Accounting_Period"], errors="coerce")

    # Money columns
    for c in ["Budget_Allocated","Actual_Spent","Encumbered","Pre_Encumbered","Revenue_Amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Valid months only
    df = df[(df["Accounting_Period"] >= 1) & (df["Accounting_Period"] <= 12)].copy()

    # Build Month
    df["Month"] = pd.to_datetime(
        df["Budget_Year"].astype("Int64").astype(str) + "-" +
        df["Accounting_Period"].astype("Int64").astype(str).str.zfill(2) + "-01",
        errors="coerce"
    )

    # Fallback for Department
    if df["Department"].isna().all() and "Department_ID" in df.columns:
        df["Department"] = df["Department_ID"].astype("Int64").astype(str)

    # Variance baseline (Actual only)
    df["Variance"] = df["Actual_Spent"] - df["Budget_Allocated"]
    df["Variance_Percent"] = (
        df["Variance"] / df["Budget_Allocated"].replace({0: pd.NA})
    ).astype(float) * 100
    df["Variance_Percent"] = df["Variance_Percent"].round(2)

    df = df.dropna(subset=["Month"]).copy()
    df["Year"] = df["Month"].dt.year
    df["Quarter"] = df["Month"].dt.quarter

    pref_cols = [
        "Month","Year","Quarter","Budget_Year","Accounting_Period",
        "Budget_Allocated","Actual_Spent","Variance","Variance_Percent",
        "Encumbered","Pre_Encumbered","Revenue_Amount",
        "Department","Department_ID","Fund_Code","Fund_Desc","Account","Account_Type",
        "Account_Desc","Program_Code","Program_Desc","Ledger_Group"
    ]
    other_cols = [c for c in df.columns if c not in pref_cols]
    df = df[pref_cols + other_cols]

    return df.sort_values("Month").reset_index(drop=True)

# =============================
# Load Data (no upload UI)
# =============================
DATA_PATH = os.getenv("BUDGET_DATA_PATH")  # optional env var
df = None
load_error = None

if DATA_PATH:
    try:
        df = load_budget_pull(DATA_PATH)
    except Exception as e:
        load_error = f"{e}"

if df is None:
    discovered = discover_default_data()
    if discovered:
        try:
            df = load_budget_pull(discovered)
            st.sidebar.info(f"Loaded: {os.path.basename(discovered)}")
        except Exception as e:
            load_error = f"{e}"

if df is None:
    if load_error:
        st.error(f"❌ Could not load data: {load_error}")
    st.markdown("""
    <div style="padding:1.25rem;border:1px dashed #94a3b8;border-radius:12px;background:#ffffff0d">
      <h3 style="margin:0 0 .5rem 0;">No data loaded</h3>
      <p style="margin:.25rem 0 .5rem 0;">
      Place your file in the working directory as <code>FY 2021 Budget Pull.xlsx</code> (or .csv / .xls),
      or set an environment variable <code>BUDGET_DATA_PATH=/abs/path/to/file.xlsx</code>.
      </p>
      <ul style="margin:.5rem 0 0 1rem;">
        <li>Supported: CSV / XLSX / XLS</li>
        <li>Required headers (any synonyms supported): <em>BUDGET YEAR, ACCOUNTING PERIOD, BUDGET AMOUNT, EXPENSE AMOUNT</em></li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =============================
# Sidebar Filters
# =============================
with st.sidebar:
    st.markdown('<div class="filter-header">🎛️ Control Panel</div>', unsafe_allow_html=True)

    st.markdown("**📊 Dataset Overview**")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("📋 Records", f"{len(df):,}")
        st.metric("🏢 Departments", f"{df['Department'].nunique():,}")
    with c2:
        st.metric("📁 Account Types", f"{df['Account_Type'].nunique():,}")
        st.metric("📅 Span", f"{df['Year'].min()}–{df['Year'].max()}")

    st.markdown("---")
    st.markdown('<div class="filter-header">🔍 Primary Filters</div>', unsafe_allow_html=True)

    cat_field_display_map = {
        "Account Type": "Account_Type",
        "Account Description": "Account_Desc",
        "Program Description": "Program_Desc",
        "Fund Description": "Fund_Desc",
        "Ledger Group": "Ledger_Group"
    }
    category_choice = st.selectbox("Category Dimension", list(cat_field_display_map.keys()), index=0)
    CAT_FIELD = cat_field_display_map[category_choice]

    dept_sel = st.multiselect("🏢 Department(s)", sorted(df["Department"].dropna().unique()),
                              default=sorted(df["Department"].dropna().unique()))
    cat_sel = st.multiselect(f"📁 {category_choice}(s)", sorted(df[CAT_FIELD].dropna().unique()),
                             default=sorted(df[CAT_FIELD].dropna().unique()))

    min_m = pd.to_datetime(df["Month"].min()).to_pydatetime()
    max_m = pd.to_datetime(df["Month"].max()).to_pydatetime()
    date_range = st.slider("📅 Month Range", value=(min_m, max_m),
                           min_value=min_m, max_value=max_m, format="YYYY-MM")

    st.markdown('<div class="filter-header">🔧 Options</div>', unsafe_allow_html=True)
    include_commitments = st.checkbox("Include Encumbrances in 'Spent' (Actual + Encumbered + Pre-Enc.)", value=False)

    # Effective preview columns (for slider ranges)
    df_preview = df.copy()
    if include_commitments:
        df_preview["Actual_Effective"] = df_preview["Actual_Spent"] + df_preview["Encumbered"] + df_preview["Pre_Encumbered"]
    else:
        df_preview["Actual_Effective"] = df_preview["Actual_Spent"]
    df_preview["Variance_Effective"] = df_preview["Actual_Effective"] - df_preview["Budget_Allocated"]
    df_preview["Variance_Percent_Effective"] = (
        df_preview["Variance_Effective"] / df_preview["Budget_Allocated"].replace({0: pd.NA})
    ).astype(float) * 100

    st.markdown('<div class="filter-header">⚙️ Advanced Filters</div>', unsafe_allow_html=True)
    year_options = sorted(df["Year"].dropna().unique())
    selected_years = st.multiselect("📅 Filter by Year(s)", year_options, default=year_options)

    vmin = float(df_preview["Variance_Percent_Effective"].min()) if df_preview["Variance_Percent_Effective"].notna().any() else -100.0
    vmax = float(df_preview["Variance_Percent_Effective"].max()) if df_preview["Variance_Percent_Effective"].notna().any() else 100.0
    variance_range = st.slider("📊 Variance % Range (effective)",
                               min_value=vmin, max_value=vmax,
                               value=(vmin, vmax), step=1.0)

    amin, amax = float(df_preview["Actual_Effective"].min()), float(df_preview["Actual_Effective"].max())
    amount_range = st.slider("💳 Spent Range ($, effective)",
                             min_value=amin, max_value=amax, value=(amin, amax),
                             step=1000.0, format="$%.0f")

    budget_performance = st.selectbox("🎯 Budget Performance",
                                      ["All","Over Budget (>0%)","Under Budget (<0%)","On Target (±5%)","Significant Variance (>±10%)"])

    with st.expander("Additional Filters (Fund / Program / Account / Ledger)"):
        fund_sel = st.multiselect("🏦 Fund Description", sorted(df["Fund_Desc"].dropna().unique()),
                                  default=sorted(df["Fund_Desc"].dropna().unique()))
        prog_sel = st.multiselect("📘 Program Description", sorted(df["Program_Desc"].dropna().unique()),
                                  default=sorted(df["Program_Desc"].dropna().unique()))
        acc_type_sel = st.multiselect("🧾 Account Type", sorted(df["Account_Type"].dropna().unique()),
                                      default=sorted(df["Account_Type"].dropna().unique()))
        ledger_sel = st.multiselect("📚 Ledger Group", sorted(df["Ledger_Group"].dropna().unique()),
                                    default=sorted(df["Ledger_Group"].dropna().unique()))

# Apply filters on working copy
df_work = df.copy()
if include_commitments:
    df_work["Actual_Effective"] = df_work["Actual_Spent"] + df_work["Encumbered"] + df_work["Pre_Encumbered"]
else:
    df_work["Actual_Effective"] = df_work["Actual_Spent"]
df_work["Variance_Effective"] = df_work["Actual_Effective"] - df_work["Budget_Allocated"]
df_work["Variance_Percent_Effective"] = (
    df_work["Variance_Effective"] / df_work["Budget_Allocated"].replace({0: pd.NA})
).astype(float) * 100

mask = (
    df_work["Department"].isin(dept_sel) &
    df_work[CAT_FIELD].isin(cat_sel) &
    (df_work["Month"] >= pd.to_datetime(date_range[0])) &
    (df_work["Month"] <= pd.to_datetime(date_range[1])) &
    df_work["Year"].isin(selected_years) &
    df_work["Variance_Percent_Effective"].between(variance_range[0], variance_range[1]) &
    df_work["Actual_Effective"].between(amount_range[0], amount_range[1]) &
    df_work["Fund_Desc"].isin(fund_sel) &
    df_work["Program_Desc"].isin(prog_sel) &
    df_work["Account_Type"].isin(acc_type_sel) &
    df_work["Ledger_Group"].isin(ledger_sel)
)

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
  <p style="font-size:1.05rem;margin:.4rem 0;">‘FY 2021 Budget Pull’ — Professional Visual Analytics</p>
  <p style="font-size:.9rem;opacity:.9;">Actuals vs Budget with flexible dimensions & encumbrance option</p>
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
display_df["Variance_Pct_Display"] = display_df["Variance_Percent_Effective"].apply(lambda x: f"{x:+.1f}%")

st.dataframe(
    display_df[[
        "Month_Display","Budget_Year","Accounting_Period","Department",
        "Fund_Desc","Program_Desc","Ledger_Group",
        "Budget_Display","Actual_Display","Variance_Display","Variance_Pct_Display"
    ]].rename(columns={
        "Month_Display":"Month", "Budget_Year":"Budget Year", "Accounting_Period":"Period",
        "Fund_Desc":"Fund", "Program_Desc":"Program", "Ledger_Group":"Ledger Group",
        "Budget_Display":"Budget", "Actual_Display":"Actual (Eff.)",
        "Variance_Display":"Variance ($)", "Variance_Pct_Display":"Variance (%)"
    }),
    use_container_width=True, height=420
)

# =============================
# Visual Analytics (clustered bars + flexible dims)
# =============================
st.markdown("---")
st.subheader("📈 Visual Analytics")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Monthly Trends", "🏢 By Department", "🗂️ By Category", "🔎 By Any Column"])

with tab1:
    monthly = df_f.groupby("Month", as_index=False)[["Budget_Allocated","Actual_Effective"]].sum().sort_values("Month")
    m_long = monthly.melt(id_vars="Month", value_vars=["Budget_Allocated","Actual_Effective"],
                          var_name="Type", value_name="Amount")
    m_long["Type"] = m_long["Type"].map({"Budget_Allocated":"Allocated","Actual_Effective":"Spent"})
    color_scale = alt.Scale(domain=["Allocated","Spent"], range=[pal["alloc"], pal["spent"]])
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
    dept_tot = df_f.groupby("Department", as_index=False).agg(
        Allocated=("Budget_Allocated","sum"),
        Spent=("Actual_Effective","sum")
    )
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
                                                                range=[pal["alloc"], pal["spent']])),
            xOffset="Type:N",
            tooltip=[alt.Tooltip("Department:N"), alt.Tooltip("Type:N"),
                     alt.Tooltip("Amount:Q", title="Amount", format=",.0f")]
        ).properties(height=420)
    )
    st.altair_chart(chart_dept, use_container_width=True)

with tab3:
    # Category = the select above (CAT_FIELD)
    cat_tot = df_f.groupby(CAT_FIELD, as_index=False).agg(
        Allocated=("Budget_Allocated","sum"),
        Spent=("Actual_Effective","sum")
    )
    cat_tot["Variance"] = cat_tot["Spent"] - cat_tot["Allocated"]
    order_cat = cat_tot.sort_values("Variance", ascending=False)[CAT_FIELD].astype(str).tolist()
    cat_long = cat_tot.melt(id_vars=CAT_FIELD, value_vars=["Allocated","Spent"],
                            var_name="Type", value_name="Amount")
    chart_cat = (
        alt.Chart(cat_long)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X(f"{CAT_FIELD}:N", sort=order_cat, axis=alt.Axis(labelAngle=-45), title="Category"),
            y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f"), title="Amount ($)"),
            color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"],
                                                                range=[pal["alloc"], pal["spent']])),
            xOffset="Type:N",
            tooltip=[alt.Tooltip(f"{CAT_FIELD}:N", title="Category"), alt.Tooltip("Type:N"),
                     alt.Tooltip("Amount:Q", title="Amount", format=",.0f")]
        ).properties(height=420)
    )
    st.altair_chart(chart_cat, use_container_width=True)

with tab4:
    # Pick any column to group by and compare Allocated vs Spent
    dims = [
        "Department","Fund_Desc","Program_Desc","Account_Type","Account_Desc","Ledger_Group",
        "Fund_Code","Program_Code","Account","Department_ID","Year","Accounting_Period"
    ]
    dims = [d for d in dims if d in df_f.columns]
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
            color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"],
                                                                range=[pal["alloc"], pal["spent']])),
            xOffset="Type:N",
            tooltip=[alt.Tooltip(f"{col}:N"), alt.Tooltip("Type:N"),
                     alt.Tooltip("Amount:Q", title="Amount", format=",.0f")]
        ).properties(height=420)
    )
    st.altair_chart(chart_any, use_container_width=True)

# =============================
# 🤖 AI-Powered Insights
# =============================
def tbl(df_):
    return "(none)" if df_.empty else df_.to_string(index=False)

def build_compact_summary(actuals_df: pd.DataFrame, cat_field: str):
    a_month = actuals_df.groupby("Month", as_index=False)[["Actual_Effective","Budget_Allocated"]].sum()
    a_month["Month"] = a_month["Month"].dt.strftime("%Y-%m")
    a_dept = (actuals_df.groupby("Department", as_index=False)[["Actual_Effective","Budget_Allocated"]]
              .sum().sort_values("Actual_Effective", ascending=False).head(10))
    a_cat = (actuals_df.groupby(cat_field, as_index=False)[["Actual_Effective","Budget_Allocated"]]
             .sum().sort_values("Actual_Effective", ascending=False).head(10))
    prompt = f"""
Use these compact summaries to answer succinctly.

ACTUALS — monthly (Allocated vs Spent):
{tbl(a_month.tail(24))}

ACTUALS — top departments:
{tbl(a_dept)}

ACTUALS — top by {cat_field}:
{tbl(a_cat)}
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

with cols[0]:
    st.markdown("### 📈 Quick Analysis")
    if st.button("🔍 Generate Smart Summary", use_container_width=True):
        if not client:
            st.error("❌ OpenAI API key not configured")
        else:
            compact = build_compact_summary(df_f, "Department" if "Department" in df_f.columns else "Account_Type")
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

with cols[1]:
    st.markdown("### 💬 Ask Questions")
    q = st.text_input(
        "🔍 Ask about your budget data:",
        placeholder="e.g., 'Actual spent in 2021-06 for HR', 'Budget for Program X 2021-11', 'Account Type Supplies variance 2021'"
    )
    if q:
        y, m = parse_month_year(q)
        df_q = df_f.copy()
        if y and m:
            df_q = df_q[df_q["Month"].dt.strftime("%Y-%m") == f"{y}-{m:02d}"]
        # Try match across common dimensions
        for dim in ["Department","Fund_Desc","Program_Desc","Account_Type","Account_Desc","Ledger_Group"]:
            if dim in df.columns:
                match = extract_match(q, df[dim].dropna().unique())
                if match:
                    df_q = df_q[df_q[dim] == match]

        MAX_ROWS = 40
        send_cols = ["Month","Department","Budget_Allocated","Actual_Effective","Variance_Effective","Variance_Percent_Effective"]
        if "Program_Desc" in df_q.columns: send_cols.append("Program_Desc")
        if "Account_Type" in df_q.columns: send_cols.append("Account_Type")
        slim = df_q[send_cols].copy() if not df_q.empty else pd.DataFrame(columns=send_cols)
        if not slim.empty:
            slim["Month"] = pd.to_datetime(slim["Month"]).dt.strftime("%Y-%m")
        context_table = slim.head(MAX_ROWS).to_string(index=False) if not slim.empty else "(no matching rows)"

        compact = build_compact_summary(df_f, "Department" if "Department" in df_f.columns else "Account_Type")
        final_prompt = f"""User question: {q}

Primary context (top rows):
{context_table}

Additional summaries:
{compact}

Instructions:
- Prefer primary context rows; otherwise use summaries.
- Use $ and % with brief calculations as needed.
- Keep answers concise and practical, scoped to the current filters.
- If insufficient to answer precisely, say so and suggest a filter refinement.
"""
        with st.spinner("🤔 AI thinking..."):
            ans = call_openai(
                system_msg="You are a precise financial analyst. Be concise and numeric.",
                user_msg=final_prompt,
                temperature=0.2,
                max_tokens=500
            )
        st.markdown(f"""<div class="ai-result">{ans}</div>""", unsafe_allow_html=True)

# =============================
# Tips & Download
# =============================
st.markdown("""
<div class="caption-style">
  <strong>Pro Tips:</strong> Switch the "Category Dimension" to analyze by Account, Program, Fund, or Ledger. Toggle "Include Encumbrances"
  to treat commitments as spend. All charts/insights respect your filters.
</div>
""", unsafe_allow_html=True)

csv = df_f.drop(columns=[], errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered data (CSV)",
    data=csv,
    file_name=f"budget_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)
