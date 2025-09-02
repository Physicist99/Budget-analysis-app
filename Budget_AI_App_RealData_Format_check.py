# app.py — AI Budget Assistant (raw vs used counts + robust parsing + raw-based filters)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re, os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# =============================
# Page
# =============================
st.set_page_config(page_title="AI Budget Assistant", layout="wide", initial_sidebar_state="expanded")

# =============================
# Themes (Executive Light / Dark)
# =============================
THEMES = {
    "Executive Light": {
        "bg": "#F7FAFC", "sidebar": "#0B1E3E", "panel": "#FFFFFF", "card": "#FFFFFF",
        "text": "#0F172A", "muted": "#475569", "grid": "#E2E8F0",
        "brand1": "#0B1E3E", "brand2": "#2F6BFF",
        "alloc": "#2151B8", "spent": "#53A2FF", "forecast": "#16A34A",
        "warn": "#D97706", "ok": "#10B981"
    },
    "Executive Dark": {  # navy background + white text
        "bg": "#0B1E3E", "sidebar": "#071A34", "panel": "#0E2044", "card": "#0F234B",
        "text": "#FFFFFF", "muted": "#CBD5E1", "grid": "#14315F",
        "brand1": "#0B1E3E", "brand2": "#2F6BFF",
        "alloc": "#8FB7FF", "spent": "#5FB0FF", "forecast": "#22C55E",
        "warn": "#F59E0B", "ok": "#10B981"
    }
}

with st.sidebar:
    st.markdown("**🎨 Theme**")
    theme_name = st.selectbox("Select", list(THEMES.keys()), index=1)  # default Dark
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
alt.themes.register("fin_theme", lambda: _alt_theme(pal)); alt.themes.enable("fin_theme")

# CSS (palette-aware + readable sidebar inputs)
st.markdown(f"""
<style>
  .stApp {{ background:{pal['bg']}; color:{pal['text']}; }}
  .block-container {{ padding-top: 1rem; }}
  [data-testid="stSidebar"] > div:first-child {{ background:{pal['sidebar']}; }}
  [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] p {{ color:#E5E7EB !important; }}
  /* Make sidebar selectors readable (dark text on white background) */
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
# Robust loader (raw + used)
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

def _find_raw_col(raw: pd.DataFrame, logical: str) -> str | None:
    lookup = {}
    for k, syns in ALIAS.items():
        lookup[_n(k)] = k
        for s in syns: lookup[_n(s)] = k
    want = _n(logical)
    for c in raw.columns:
        if lookup.get(_n(c)) == want:
            return c
    return None

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
    m = re.match(r"^(20\d{2})[-/](0?[1-9]|1[0-2])$", s)  # YYYY-MM
    if m: return int(m.group(2))
    mon_map = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"SEPT":9,"OCT":10,"NOV":11,"DEC":12}
    up = s.upper()
    if up.startswith("SEPT"): return 9
    if up[:3] in mon_map: return mon_map[up[:3]]
    return pd.to_numeric(s, errors="coerce")

@st.cache_data
def load_budget_pull_with_raw(path: str):
    raw = _read_any(path)

    # Rename to canonical
    lookup = {}
    for k, syns in ALIAS.items():
        lookup[_n(k)] = k
        for s in syns: lookup[_n(s)] = k

    rename_map = {}
    for c in raw.columns:
        if _n(c) in lookup:
            rename_map[c] = CANON[lookup[_n(c)]]

    df = raw.rename(columns=rename_map).copy()

    # Required
    for req in ["Budget_Year","Accounting_Period","Budget_Allocated","Actual_Spent"]:
        if req not in df.columns:
            raise ValueError(f"Missing required column: {req}")

    # Optional defaults
    for k in ["Department","Account_Desc","Account_Type","Fund_Desc","Program_Desc","Ledger_Group",
              "Encumbered","Pre_Encumbered","Revenue_Amount"]:
        if k not in df.columns: df[k] = np.nan if k not in ["Encumbered","Pre_Encumbered","Revenue_Amount"] else 0.0

    # ----- RAW unique counts (your 734 / 748 should show here) -----
    raw_dept_col = _find_raw_col(raw, "DEPARTMENT ID DESCRIPTION")
    raw_accd_col = _find_raw_col(raw, "ACCOUNT DESCRIPTION")
    raw_depts = (raw[raw_dept_col].dropna().astype(str).str.strip().unique().tolist()
                 if raw_dept_col else [])
    raw_accds = (raw[raw_accd_col].dropna().astype(str).str.strip().unique().tolist()
                 if raw_accd_col else [])

    raw_counts = {"raw_dept_n": len(raw_depts), "raw_accdesc_n": len(raw_accds)}

    # ----- Clean to “used” (calendarized) -----
    df["Budget_Year"] = df["Budget_Year"].map(_parse_year).astype("Int64")
    df["Accounting_Period"] = df["Accounting_Period"].map(_parse_period).astype("Int64")

    # keep only valid fiscal months 1..12
    df = df[(df["Accounting_Period"] >= 1) & (df["Accounting_Period"] <= 12)].copy()

    # Money
    for c in ["Budget_Allocated","Actual_Spent","Encumbered","Pre_Encumbered","Revenue_Amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Fiscal(1=Oct) → Calendar
    F2C = {1:10,2:11,3:12,4:1,5:2,6:3,7:4,8:5,9:6,10:7,11:8,12:9}
    df["Cal_Month"] = df["Accounting_Period"].map(F2C).astype("Int64")
    df["Cal_Year"] = np.where(df["Cal_Month"] >= 10, df["Budget_Year"] - 1, df["Budget_Year"]).astype("Int64")
    df["Month"] = pd.to_datetime(df["Cal_Year"].astype(str) + "-" + df["Cal_Month"].astype(str).str.zfill(2) + "-01",
                                 errors="coerce")

    # Effective + variance
    df["Actual_Effective"] = df["Actual_Spent"]  # commitments toggle later
    df["Variance_Effective"] = df["Actual_Effective"] - df["Budget_Allocated"]
    den = df["Budget_Allocated"].replace({0: np.nan})
    df["Variance_Percent_Effective"] = (df["Variance_Effective"] / den) * 100

    # Drop rows with no Month
    df = df.dropna(subset=["Month"]).copy()

    # Helpers
    df["Year"] = df["Month"].dt.year
    df["Quarter"] = df["Month"].dt.quarter
    df["Fiscal_Year"] = df["Budget_Year"]
    df["Fiscal_Quarter"] = (((df["Cal_Month"] - 10) % 12) // 3 + 1).astype("Int64")
    df = df.sort_values("Month").reset_index(drop=True)

    used_counts = {
        "used_dept_n": df["Department"].dropna().nunique(),
        "used_accdesc_n": df["Account_Desc"].dropna().nunique()
    }

    # Return processed, raw (for options), and counts
    return df, raw, raw_counts, used_counts, raw_depts, raw_accds

# =============================
# Load data
# =============================
DATA_PATH = "FY 2021 Budget Pull.xlsx"
try:
    df, raw_df, raw_counts, used_counts, raw_depts, raw_accds = load_budget_pull_with_raw(DATA_PATH)
except Exception as e:
    # CSV/XLS fallback with same basename
    base, _ = os.path.splitext(DATA_PATH)
    for alt in [base + ".csv", base + ".xls"]:
        if os.path.exists(alt):
            df, raw_df, raw_counts, used_counts, raw_depts, raw_accds = load_budget_pull_with_raw(alt)
            break
    else:
        st.error(f"❌ Could not load data: {e}")
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
    st.markdown('<div class="filter-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🏢 Departments (raw → used)", f"{raw_counts['raw_dept_n']:,} → {used_counts['used_dept_n']:,}")
    with c2:
        st.metric("🧾 Account Desc (raw → used)", f"{raw_counts['raw_accdesc_n']:,} → {used_counts['used_accdesc_n']:,}")

    st.markdown('<div class="filter-header">🔍 Primary Filters</div>', unsafe_allow_html=True)
    # IMPORTANT: build options from RAW, so you see full 734/748 sets
    dept_sel = st.multiselect("Department", sorted(raw_depts), default=sorted(raw_depts))
    accd_sel = st.multiselect("Account Description", sorted(raw_accds), default=sorted(raw_accds))
    acct_type_opts = sorted(df["Account_Type"].dropna().astype(str).unique()) if "Account_Type" in df.columns else []
    acct_type_sel = st.multiselect("Account Type", acct_type_opts, default=acct_type_opts)

    st.markdown('<div class="filter-header">🧰 Options</div>', unsafe_allow_html=True)
    include_commitments = st.checkbox("Include Encumbrances in Spent", value=False)

    # Month range
    min_m = pd.to_datetime(df["Month"].min()).to_pydatetime()
    max_m = pd.to_datetime(df["Month"].max()).to_pydatetime()
    if min_m >= max_m:
        from pandas.tseries.offsets import MonthBegin
        faux_max = (pd.Timestamp(min_m) + MonthBegin(1)).to_pydatetime()
        date_range = st.slider("📅 Month Range", value=(min_m, faux_max), min_value=min_m, max_value=faux_max, format="YYYY-MM")
    else:
        date_range = st.slider("📅 Month Range", value=(min_m, max_m), min_value=min_m, max_value=max_m, format="YYYY-MM")

    st.markdown('<div class="filter-header">⚙️ Advanced</div>', unsafe_allow_html=True)
    fy_opts = sorted(df["Fiscal_Year"].dropna().unique().tolist())
    fy_sel = st.multiselect("Fiscal Year", fy_opts, default=fy_opts)

    # Variance / amount sliders (effective)
    df_tmp = df.copy()
    if include_commitments:
        df_tmp["Actual_Effective"] = df_tmp["Actual_Spent"] + df_tmp.get("Encumbered", 0) + df_tmp.get("Pre_Encumbered", 0)
        df_tmp["Variance_Effective"] = df_tmp["Actual_Effective"] - df_tmp["Budget_Allocated"]
        den = df_tmp["Budget_Allocated"].replace({0: np.nan})
        df_tmp["Variance_Percent_Effective"] = (df_tmp["Variance_Effective"] / den) * 100
    v = df_tmp["Variance_Percent_Effective"].dropna()
    vmin, vmax = (float(v.min()), float(v.max())) if len(v) else (-100.0, 100.0)
    variance_range = st.slider("Variance % (effective)", min_value=vmin, max_value=vmax, value=(vmin, vmax), step=1.0)

    a = df_tmp["Actual_Effective"]
    amin, amax = float(a.min()), float(a.max())
    amount_range = st.slider("Spent Range ($, effective)", min_value=amin, max_value=amax, value=(amin, amax), step=1000.0, format="$%.0f")

# =============================
# Apply filters
# =============================
df_work = df.copy()
if include_commitments:
    df_work["Actual_Effective"] = df_work["Actual_Spent"] + df_work.get("Encumbered", 0) + df_work.get("Pre_Encumbered", 0)
    df_work["Variance_Effective"] = df_work["Actual_Effective"] - df_work["Budget_Allocated"]
    den = df_work["Budget_Allocated"].replace({0: np.nan})
    df_work["Variance_Percent_Effective"] = (df_work["Variance_Effective"] / den) * 100

mask = (
    (df_work["Department"].isin(dept_sel) if dept_sel else True) &
    (df_work["Account_Desc"].isin(accd_sel) if accd_sel else True) &
    (df_work["Account_Type"].isin(acct_type_sel) if acct_type_sel else True) &
    (df_work["Month"] >= pd.to_datetime(date_range[0])) &
    (df_work["Month"] <= pd.to_datetime(date_range[1])) &
    (df_work["Fiscal_Year"].isin(fy_sel))
    & df_work["Variance_Percent_Effective"].between(variance_range[0], variance_range[1])
    & df_work["Actual_Effective"].between(amount_range[0], amount_range[1])
)

df_f = df_work.loc[mask].copy()
if df_f.empty:
    st.warning("No data matches your filters. Adjust selections or date range.")
    st.stop()

# =============================
# KPIs
# =============================
st.markdown("### 🔎 Overview")
k1,k2,k3,k4 = st.columns(4)
tot_alloc = df_f["Budget_Allocated"].sum()
tot_spent = df_f["Actual_Effective"].sum()
tot_var = tot_spent - tot_alloc
pct = (tot_var / tot_alloc * 100) if tot_alloc else 0.0
with k1: st.metric("Total Budget", f"${tot_alloc:,.0f}", delta=f"{len(df_f):,} rows")
with k2: st.metric("Total Spent"+(" (Incl. Enc.)" if include_commitments else ""), f"${tot_spent:,.0f}", delta=f"vs Budget {pct:+.1f}%")
with k3: st.metric("Net Variance", f"${tot_var:,+.0f}", delta=f"{pct:+.2f}%")
with k4:
    eff = max(0.0, 100-abs(pct))
    tag = "Excellent" if eff>95 else "Good" if eff>85 else "Needs Review"
    st.metric("Budget Efficiency", f"{eff:.1f}%", delta=tag)

# =============================
# Details table
# =============================
st.markdown("---")
st.subheader("📄 Details (filtered)")
show_cols = ["Month","Fiscal_Year","Department","Account_Type","Account_Desc","Fund_Desc","Program_Desc","Ledger_Group",
             "Budget_Allocated","Actual_Effective","Variance_Effective","Variance_Percent_Effective"]
show_cols = [c for c in show_cols if c in df_f.columns]
df_disp = df_f[show_cols].copy()
df_disp["Month"] = pd.to_datetime(df_disp["Month"]).dt.strftime("%Y-%m")
st.dataframe(df_disp, use_container_width=True, hide_index=True, height=380)

# =============================
# Visual Analytics
# =============================
st.markdown("---")
st.subheader("📈 Visual Analytics")
t1,t2,t3 = st.tabs(["📊 Monthly Trend (clustered bars)","🏢 By Department","🧾 By Account Description"])

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
            color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[pal["alloc"], pal["spent"]])), 
            xOffset="Type:N",
            tooltip=[alt.Tooltip("yearmonth(Month):O","Month"),"Type:N",alt.Tooltip("Amount:Q", format=",.0f")]
        ).properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)

with t2:
    dept = df_f.groupby("Department", as_index=False).agg(Allocated=("Budget_Allocated","sum"),
                                                          Spent=("Actual_Effective","sum"))
    dept["Variance"] = dept["Spent"] - dept["Allocated"]
    order = dept.sort_values("Variance", ascending=False)["Department"].astype(str).tolist()
    dlong = dept.melt("Department", ["Allocated","Spent"], var_name="Type", value_name="Amount")
    chart = (
        alt.Chart(dlong)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Department:N", sort=order, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[pal["alloc"], pal["spent"]])),
            xOffset="Type:N",
            tooltip=["Department","Type",alt.Tooltip("Amount:Q", format=",.0f")]
        ).properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

with t3:
    accd = df_f.groupby("Account_Desc", as_index=False).agg(Allocated=("Budget_Allocated","sum"),
                                                            Spent=("Actual_Effective","sum"))
    accd["Variance"] = accd["Spent"] - accd["Allocated"]
    order = accd.sort_values("Variance", ascending=False)["Account_Desc"].astype(str).tolist()
    along = accd.melt("Account_Desc", ["Allocated","Spent"], var_name="Type", value_name="Amount")
    chart = (
        alt.Chart(along)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Account_Desc:N", sort=order, axis=alt.Axis(labelAngle=-45), title="Account Description"),
            y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Type:N", title="", scale=alt.Scale(domain=["Allocated","Spent"], range=[pal["alloc"], pal["spent"]])),
            xOffset="Type:N",
            tooltip=["Account_Desc","Type",alt.Tooltip("Amount:Q", format=",.0f")]
        ).properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

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
    if st.button("📈 Quick Analysis"):
        if not client: st.error("OpenAI key not configured.")
        else:
            prompt = build_compact_summary(df_f) + "\nProvide 5–7 numeric bullets and a short narrative with 2–3 actions."
            ans = call_openai(
                "You are a senior FP&A analyst. Be concise, numeric, and practical. Use $ and %.",
                prompt, temperature=0.25, max_tokens=700
            )
            st.markdown(f"""<div class="ai-result">{ans}</div>""", unsafe_allow_html=True)

with cB:
    q = st.text_input("💬 Ask a question (e.g., 'Which departments overspent most in FY2021?')")
    if q and client:
        # Aggregate-first, group by intent if present
        ql = q.lower()
        prefer = "Department" if "department" in ql or "dept" in ql else (
                 "Account_Desc" if "account description" in ql or "account" in ql else None)

        df_q = df_f.copy()
        # month filter (YYYY-MM or "Nov 2021" etc.)
        def parse_month_year(text):
            m1 = re.search(r'(20\d{2})[-/](0?[1-9]|1[0-2])', text or "")
            if m1: return int(m1.group(1)), int(m1.group(2))
            m2 = re.search(r'([A-Za-z]{3,9})\s+(20\d{2})', text or "")
            if m2:
                mon_map = {m.lower():i for i,m in enumerate(
                    ["January","February","March","April","May","June","July","August","September","October","November","December"],1)}
                key = m2.group(1).lower()
                for full, idx in mon_map.items():
                    if full.startswith(key): return int(m2.group(2)), idx
            return None, None
        y,m = parse_month_year(q)
        if y and m:
            df_q = df_q[df_q["Month"].dt.strftime("%Y-%m") == f"{y}-{m:02d}"]

        # Try to match a specific Department/Account_Desc mentioned in the text
        def extract_match(text, options):
            text_l = str(text).lower()
            for opt in options:
                if str(opt).lower() in text_l:
                    return opt
            return None
        hit_dept = extract_match(q, raw_depts)
        hit_accd = extract_match(q, raw_accds)
        if hit_dept is not None and "Department" in df_q.columns:
            df_q = df_q[df_q["Department"] == hit_dept]
            prefer = None if prefer == "Department" else prefer
        if hit_accd is not None and "Account_Desc" in df_q.columns:
            df_q = df_q[df_q["Account_Desc"] == hit_accd]
            prefer = None if prefer == "Account_Desc" else prefer

        if prefer and prefer in df_q.columns and not df_q.empty:
            agg = df_q.groupby(prefer, as_index=False).agg(Allocated=("Budget_Allocated","sum"),
                                                           Spent=("Actual_Effective","sum"))
            agg["Variance"] = agg["Spent"] - agg["Allocated"]
            agg["AbsVar"] = agg["Variance"].abs()
            agg = agg.sort_values("AbsVar", ascending=False)
            send = agg.head(1000).drop(columns=["AbsVar"]).rename(columns={prefer:"Group"})
            primary = send.to_string(index=False)
            coverage = f"(Top {min(1000, len(agg))} {prefer} by |variance|; {len(agg)} groups total)"
        else:
            cols = ["Month","Department","Account_Desc","Budget_Allocated","Actual_Effective","Variance_Effective","Variance_Percent_Effective"]
            cols = [c for c in cols if c in df_q.columns]
            slim = df_q[cols].copy() if not df_q.empty else pd.DataFrame(columns=cols)
            if not slim.empty and "Variance_Effective" in slim.columns:
                slim["_abs"] = slim["Variance_Effective"].abs()
                slim = slim.sort_values("_abs", ascending=False).drop(columns="_abs")
            primary = slim.head(800).to_string(index=False) if not slim.empty else "(no matching rows)"
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
csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered data (CSV)", data=csv,
                   file_name=f"budget_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
