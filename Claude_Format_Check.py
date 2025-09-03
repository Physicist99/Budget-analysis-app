# app.py — AI Budget Assistant (Fixed Version)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os, re
from datetime import datetime
from dotenv import load_dotenv

# Try to import OpenAI, but make it optional
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("OpenAI library not installed. Install with: pip install openai")

# =============================
# Page Configuration
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
    "Executive Dark": {
        "bg": "#0B1E3E", "sidebar": "#071A34", "panel": "#0E2044", "card": "#0F234B",
        "text": "#FFFFFF", "muted": "#CBD5E1", "grid": "#14315F",
        "brand1": "#0B1E3E", "brand2": "#2F6BFF",
        "alloc": "#8FB7FF", "spent": "#5FB0FF",
        "ok": "#10B981", "warn": "#F59E0B"
    }
}

# =============================
# Sample Data Generator (for demo purposes)
# =============================
@st.cache_data
def generate_sample_data():
    """Generate sample budget data if no file is found"""
    np.random.seed(42)
    
    departments = ["Finance", "HR", "IT", "Marketing", "Operations", "Legal", "Facilities"]
    account_types = ["Personnel", "Operating", "Capital", "Travel", "Supplies"]
    account_descs = [
        "Salaries & Benefits", "Office Supplies", "Software Licenses", "Travel Expenses",
        "Equipment Purchase", "Consulting Services", "Utilities", "Training", "Marketing Campaigns",
        "Legal Services", "Insurance", "Maintenance", "Communications", "Rent"
    ]
    
    data = []
    for month in pd.date_range('2021-10-01', '2024-09-30', freq='MS'):
        fiscal_period = ((month.month - 10) % 12) + 1
        budget_year = month.year + 1 if month.month >= 10 else month.year
        
        for dept in departments:
            for _ in range(np.random.randint(3, 8)):  # 3-7 records per dept per month
                budget_amt = np.random.uniform(5000, 100000)
                actual_amt = budget_amt * np.random.uniform(0.7, 1.3)  # 70%-130% of budget
                
                data.append({
                    'Budget_Year': budget_year,
                    'Accounting_Period': fiscal_period,
                    'Department': dept,
                    'Account_Type': np.random.choice(account_types),
                    'Account_Desc': np.random.choice(account_descs),
                    'Fund_Desc': f"Fund {np.random.randint(1, 6)}",
                    'Program_Desc': f"Program {np.random.choice(['A', 'B', 'C', 'D'])}",
                    'Ledger_Group': f"Group {np.random.randint(1, 4)}",
                    'Budget_Allocated': budget_amt,
                    'Actual_Spent': actual_amt,
                    'Encumbered': np.random.uniform(0, budget_amt * 0.1),
                    'Pre_Encumbered': np.random.uniform(0, budget_amt * 0.05),
                    'Revenue_Amount': 0
                })
    
    return pd.DataFrame(data)

# =============================
# Sidebar Theme Selection
# =============================
with st.sidebar:
    st.markdown("**🎨 Theme**")
    theme_name = st.selectbox("Select", list(THEMES.keys()), index=1)
pal = THEMES[theme_name]

# =============================
# Altair Theme Configuration
# =============================
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
# CSS Styling
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
# OpenAI Setup (Optional)
# =============================
client = None
if OPENAI_AVAILABLE:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key) if api_key else None
    if not api_key:
        st.sidebar.info("💡 Set OPENAI_API_KEY in .env file for AI features")

def call_openai(system_msg: str, user_msg: str, temperature=0.2, max_tokens=900) -> str:
    if not client: 
        return "OpenAI API key not configured. Set OPENAI_API_KEY in your .env file."
    try:
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
            temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {str(e)}"

# =============================
# Data Loading and Processing
# =============================
@st.cache_data
def load_and_process_data():
    """Load data from file or generate sample data"""
    
    # Try to load from various file formats
    possible_files = [
        "FY 2021 Budget Pull.xlsx",
        "FY 2021 Budget Pull.csv", 
        "FY 2021 Budget Pull.xls",
        "budget_data.xlsx",
        "budget_data.csv"
    ]
    
    df = None
    for filename in possible_files:
        if os.path.exists(filename):
            try:
                if filename.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(filename)
                else:
                    df = pd.read_csv(filename)
                st.sidebar.success(f"✅ Loaded data from {filename}")
                break
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {filename}: {str(e)}")
                continue
    
    # If no file found, use sample data
    if df is None:
        st.sidebar.info("📊 Using sample data (no data file found)")
        df = generate_sample_data()
    
    # Process the data
    return process_budget_data(df)

def process_budget_data(df):
    """Process and clean the budget data"""
    
    # Ensure required columns exist
    required_columns = {
        'Budget_Year': 'Budget_Year',
        'Accounting_Period': 'Accounting_Period', 
        'Budget_Allocated': 'Budget_Allocated',
        'Actual_Spent': 'Actual_Spent'
    }
    
    # Try to map common column name variations
    column_mapping = {}
    for required, canonical in required_columns.items():
        if required in df.columns:
            column_mapping[required] = canonical
        else:
            # Try common variations
            variations = {
                'Budget_Year': ['Budget Year', 'Fiscal Year', 'FY', 'Year'],
                'Accounting_Period': ['Period', 'Month', 'Accounting Month'],
                'Budget_Allocated': ['Budget', 'Budget Amount', 'Allocated', 'Appropriation'],
                'Actual_Spent': ['Actual', 'Spent', 'Expenditure', 'Expense']
            }
            
            for col in df.columns:
                if col in variations.get(required, []):
                    column_mapping[col] = canonical
                    break
    
    # Rename columns
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Add missing optional columns with defaults
    optional_columns = {
        'Department': 'Unknown',
        'Account_Type': 'General',
        'Account_Desc': 'General Account',
        'Fund_Desc': 'General Fund',
        'Program_Desc': 'General Program',
        'Ledger_Group': 'General',
        'Encumbered': 0.0,
        'Pre_Encumbered': 0.0,
        'Revenue_Amount': 0.0
    }
    
    for col, default_val in optional_columns.items():
        if col not in df.columns:
            df[col] = default_val
    
    # Ensure we have the required columns after mapping/creation
    required_final = ['Budget_Year', 'Accounting_Period', 'Budget_Allocated', 'Actual_Spent']
    for req_col in required_final:
        if req_col not in df.columns:
            raise ValueError(f"Critical error: {req_col} column is missing after processing")
    
    # Clean and convert data types
    df['Budget_Year'] = pd.to_numeric(df['Budget_Year'], errors='coerce').fillna(2021).astype(int)
    df['Accounting_Period'] = pd.to_numeric(df['Accounting_Period'], errors='coerce').fillna(1).astype(int)
    df['Budget_Allocated'] = pd.to_numeric(df['Budget_Allocated'], errors='coerce').fillna(0.0)
    df['Actual_Spent'] = pd.to_numeric(df['Actual_Spent'], errors='coerce').fillna(0.0)
    df['Encumbered'] = pd.to_numeric(df['Encumbered'], errors='coerce').fillna(0.0)
    df['Pre_Encumbered'] = pd.to_numeric(df['Pre_Encumbered'], errors='coerce').fillna(0.0)
    
    # Filter valid periods (1-12)
    df = df[(df['Accounting_Period'] >= 1) & (df['Accounting_Period'] <= 12)].copy()
    
    # Convert fiscal periods to calendar dates (assuming fiscal year starts in October)
    # Fiscal Period 1 = October, 2 = November, etc.
    fiscal_to_calendar = {1:10, 2:11, 3:12, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7, 11:8, 12:9}
    
    df['Cal_Month'] = df['Accounting_Period'].map(fiscal_to_calendar)
    df['Cal_Year'] = np.where(df['Cal_Month'] >= 10, df['Budget_Year'] - 1, df['Budget_Year'])
    
    # Create datetime column
    df['Month'] = pd.to_datetime(
        df['Cal_Year'].astype(str) + '-' + df['Cal_Month'].astype(str).str.zfill(2) + '-01',
        errors='coerce'
    )
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['Month']).copy()
    
    # Add helper columns
    df['Year'] = df['Month'].dt.year
    df['Quarter'] = df['Month'].dt.quarter
    df['Fiscal_Year'] = df['Budget_Year']
    
    # Calculate effective spending (with option to include encumbrances)
    df['Actual_Effective'] = df['Actual_Spent']
    df['Variance_Effective'] = df['Actual_Effective'] - df['Budget_Allocated']
    
    # Calculate variance percentages (handle division by zero)
    budget_nonzero = df['Budget_Allocated'].replace({0: np.nan})
    df['Variance_Percent_Effective'] = (df['Variance_Effective'] / budget_nonzero) * 100
    
    # Sort by date
    df = df.sort_values('Month').reset_index(drop=True)
    
    return df

# =============================
# Load Data
# =============================
try:
    df = load_and_process_data()
    data_loaded = True
except Exception as e:
    st.error(f"❌ Error processing data: {str(e)}")
    st.stop()

# =============================
# Header
# =============================
st.markdown(f"""
<div class="main-header">
  <h1>🤖 AI Budget Forecast & Analysis</h1>
  <p style="opacity:.9">Professional analytics with fiscal-year calendarization (1=Oct … 12=Sep)</p>
</div>
""", unsafe_allow_html=True)

# =============================
# Sidebar Filters
# =============================
with st.sidebar:
    st.markdown('<div class="filter-header">🔍 Primary Filters</div>', unsafe_allow_html=True)
    
    # Department filter
    dept_options = sorted(df['Department'].dropna().astype(str).unique().tolist())
    dept_selected = st.multiselect("Department", dept_options, default=dept_options)
    
    # Account Description filter  
    acct_desc_options = sorted(df['Account_Desc'].dropna().astype(str).unique().tolist())
    acct_desc_selected = st.multiselect("Account Description", acct_desc_options, default=acct_desc_options)
    
    # Account Type filter
    acct_type_options = sorted(df['Account_Type'].dropna().astype(str).unique().tolist())
    acct_type_selected = st.multiselect("Account Type", acct_type_options, default=acct_type_options)
    
    # Options
    st.markdown('<div class="filter-header">🧰 Options</div>', unsafe_allow_html=True)
    include_commitments = st.checkbox("Include Encumbrances in Spent", value=False)
    
    # Date range filter
    min_date = df['Month'].min().to_pydatetime()
    max_date = df['Month'].max().to_pydatetime()
    date_range = st.slider(
        "📅 Month Range", 
        min_value=min_date, 
        max_value=max_date, 
        value=(min_date, max_date),
        format="YYYY-MM"
    )
    
    # Advanced filters
    st.markdown('<div class="filter-header">⚙️ Advanced</div>', unsafe_allow_html=True)
    
    # Fiscal year filter
    fy_options = sorted(df['Fiscal_Year'].unique().tolist())
    fy_selected = st.multiselect("Fiscal Year", fy_options, default=fy_options)
    
    # Budget performance filter
    budget_perf = st.selectbox(
        "🎯 Budget Performance",
        ["All", "Over Budget (>0%)", "Under Budget (<0%)", "On Target (±5%)", "Significant Variance (>±10%)"]
    )

# =============================
# Apply Filters
# =============================
df_work = df.copy()

# Recalculate effective amounts if including commitments
if include_commitments:
    df_work['Actual_Effective'] = (df_work['Actual_Spent'] + 
                                  df_work['Encumbered'] + 
                                  df_work['Pre_Encumbered'])
    df_work['Variance_Effective'] = df_work['Actual_Effective'] - df_work['Budget_Allocated']
    budget_nonzero = df_work['Budget_Allocated'].replace({0: np.nan})
    df_work['Variance_Percent_Effective'] = (df_work['Variance_Effective'] / budget_nonzero) * 100

# Apply all filters
mask = (
    (df_work['Department'].isin(dept_selected) if dept_selected else True) &
    (df_work['Account_Desc'].isin(acct_desc_selected) if acct_desc_selected else True) &
    (df_work['Account_Type'].isin(acct_type_selected) if acct_type_selected else True) &
    (df_work['Month'] >= pd.to_datetime(date_range[0])) &
    (df_work['Month'] <= pd.to_datetime(date_range[1])) &
    (df_work['Fiscal_Year'].isin(fy_selected) if fy_selected else True)
)

# Apply budget performance filter
if budget_perf == "Over Budget (>0%)":
    mask &= df_work['Variance_Percent_Effective'] > 0
elif budget_perf == "Under Budget (<0%)":
    mask &= df_work['Variance_Percent_Effective'] < 0
elif budget_perf == "On Target (±5%)":
    mask &= df_work['Variance_Percent_Effective'].between(-5, 5)
elif budget_perf == "Significant Variance (>±10%)":
    mask &= (df_work['Variance_Percent_Effective'] > 10) | (df_work['Variance_Percent_Effective'] < -10)

df_filtered = df_work.loc[mask].copy()

# =============================
# KPIs
# =============================
st.markdown("### 🔎 Overview")
if df_filtered.empty:
    st.warning("⚠️ No data matches your current filters. Try adjusting the filter criteria.")
else:
    col1, col2, col3, col4 = st.columns(4)
    
    total_budget = df_filtered['Budget_Allocated'].sum()
    total_spent = df_filtered['Actual_Effective'].sum()
    total_variance = total_spent - total_budget
    variance_pct = (total_variance / total_budget * 100) if total_budget > 0 else 0
    
    with col1:
        st.metric("Total Budget", f"${total_budget:,.0f}", delta=f"{len(df_filtered):,} rows")
    with col2:
        suffix = " (Incl. Enc.)" if include_commitments else ""
        st.metric(f"Total Spent{suffix}", f"${total_spent:,.0f}", delta=f"{variance_pct:+.1f}%")
    with col3:
        st.metric("Net Variance", f"${total_variance:+,.0f}", delta=f"{variance_pct:+.1f}%")
    with col4:
        efficiency = max(0, 100 - abs(variance_pct))
        status = "Excellent" if efficiency > 95 else "Good" if efficiency > 85 else "Needs Review"
        st.metric("Budget Efficiency", f"{efficiency:.1f}%", delta=status)

# =============================
# Detailed Table
# =============================
st.markdown("---")
st.subheader("📄 Detailed Data (Filtered)")
if not df_filtered.empty:
    display_columns = [
        'Month', 'Fiscal_Year', 'Department', 'Account_Type', 'Account_Desc',
        'Budget_Allocated', 'Actual_Effective', 'Variance_Effective', 'Variance_Percent_Effective'
    ]
    
    df_display = df_filtered[display_columns].copy()
    df_display['Month'] = df_display['Month'].dt.strftime('%Y-%m')
    
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=400)
else:
    st.info("No data to display with current filters.")

# =============================
# Visualizations
# =============================
st.markdown("---")
st.subheader("📈 Visual Analytics")

if not df_filtered.empty:
    tab1, tab2, tab3 = st.tabs(["📊 Monthly Trend", "🏢 By Department", "🧾 By Account Description"])
    
    with tab1:
        # Monthly trend chart
        monthly_data = df_filtered.groupby('Month').agg({
            'Budget_Allocated': 'sum',
            'Actual_Effective': 'sum'
        }).reset_index().sort_values('Month')
        
        monthly_long = monthly_data.melt(
            id_vars=['Month'], 
            value_vars=['Budget_Allocated', 'Actual_Effective'],
            var_name='Type', 
            value_name='Amount'
        )
        monthly_long['Type'] = monthly_long['Type'].map({
            'Budget_Allocated': 'Allocated',
            'Actual_Effective': 'Spent'
        })
        
        chart = alt.Chart(monthly_long).mark_bar().encode(
            x=alt.X('yearmonth(Month):O', title='Month'),
            y=alt.Y('Amount:Q', title='Amount ($)', axis=alt.Axis(format='$,.0f')),
            color=alt.Color(
                'Type:N', 
                title='',
                scale=alt.Scale(
                    domain=['Allocated', 'Spent'], 
                    range=[pal['alloc'], pal['spent']]
                )
            ),
            xOffset='Type:N',
            tooltip=[
                alt.Tooltip('yearmonth(Month):O', title='Month'),
                'Type:N',
                alt.Tooltip('Amount:Q', format=',.0f')
            ]
        ).properties(height=400)
        
        st.altair_chart(chart, use_container_width=True)
    
    with tab2:
        # Department chart
        dept_data = df_filtered.groupby('Department').agg({
            'Budget_Allocated': 'sum',
            'Actual_Effective': 'sum'
        }).reset_index()
        
        if not dept_data.empty:
            dept_data['Variance'] = dept_data['Actual_Effective'] - dept_data['Budget_Allocated']
            dept_order = dept_data.sort_values('Variance', ascending=False)['Department'].tolist()
            
            dept_long = dept_data.melt(
                id_vars=['Department'],
                value_vars=['Budget_Allocated', 'Actual_Effective'],
                var_name='Type',
                value_name='Amount'
            )
            dept_long['Type'] = dept_long['Type'].map({
                'Budget_Allocated': 'Allocated',
                'Actual_Effective': 'Spent'
            })
            
            chart = alt.Chart(dept_long).mark_bar(cornerRadius=3).encode(
                x=alt.X('Department:N', sort=dept_order, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Amount:Q', axis=alt.Axis(format='$,.0f')),
                color=alt.Color(
                    'Type:N',
                    title='',
                    scale=alt.Scale(
                        domain=['Allocated', 'Spent'],
                        range=[pal['alloc'], pal['spent']]
                    )
                ),
                xOffset='Type:N',
                tooltip=['Department', 'Type', alt.Tooltip('Amount:Q', format=',.0f')]
            ).properties(height=420)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No department data available.")
    
    with tab3:
        # Account Description chart
        acct_data = df_filtered.groupby('Account_Desc').agg({
            'Budget_Allocated': 'sum',
            'Actual_Effective': 'sum'
        }).reset_index()
        
        if not acct_data.empty:
            acct_data['Variance'] = acct_data['Actual_Effective'] - acct_data['Budget_Allocated']
            acct_order = acct_data.sort_values('Variance', ascending=False)['Account_Desc'].tolist()
            
            acct_long = acct_data.melt(
                id_vars=['Account_Desc'],
                value_vars=['Budget_Allocated', 'Actual_Effective'],
                var_name='Type',
                value_name='Amount'
            )
            acct_long['Type'] = acct_long['Type'].map({
                'Budget_Allocated': 'Allocated',
                'Actual_Effective': 'Spent'
            })
            
            chart = alt.Chart(acct_long).mark_bar(cornerRadius=3).encode(
                x=alt.X('Account_Desc:N', sort=acct_order, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Amount:Q', axis=alt.Axis(format='$,.0f')),
                color=alt.Color(
                    'Type:N',
                    title='',
                    scale=alt.Scale(
                        domain=['Allocated', 'Spent'],
                        range=[pal['alloc'], pal['spent']]
                    )
                ),
                xOffset='Type:N',
                tooltip=['Account_Desc', 'Type', alt.Tooltip('Amount:Q', format=',.0f')]
            ).properties(height=420)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No account description data available.")

else:
    st.info("No data available for visualization with current filters.")

# =============================
# AI Analysis Section
# =============================
st.markdown("---")
st.markdown(f"""
<div class="ai-box">
    <h3>🤖 AI-Powered Insights</h3>
    <p>Get intelligent analysis of your budget data or ask specific questions.</p>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns([1, 2])

with col_a:
    if st.button("📈 Quick Analysis", use_container_width=True):
        if not client:
            st.error("OpenAI not configured. Set OPENAI_API_KEY in .env file.")
        elif df_filtered.empty:
            st.warning("No data available with current filters.")
        else:
            # Prepare summary data for AI
            monthly_summary = df_filtered.groupby('Month').agg({
                'Budget_Allocated': 'sum',
                'Actual_Effective': 'sum'
            }).reset_index().tail(12)  # Last 12 months
            
            dept_summary = df_filtered.groupby('Department').agg({
                'Budget_Allocated': 'sum',
                'Actual_Effective': 'sum'
            }).reset_index()
            dept_summary['Variance'] = dept_summary['Actual_Effective'] - dept_summary['Budget_Allocated']
            dept_summary = dept_summary.sort_values('Variance', key=abs, ascending=False).head(10)
            
            prompt = f"""
            Budget Analysis Summary:
            
            Monthly Data (Last 12 months):
            {monthly_summary.to_string(index=False)}
            
            Top Departments by Variance:
            {dept_summary.to_string(index=False)}
            
            Total Budget: ${df_filtered['Budget_Allocated'].sum():,.0f}
            Total Spent: ${df_filtered['Actual_Effective'].sum():,.0f}
            
            Provide 5-7 key insights with specific numbers and percentages, plus 2-3 actionable recommendations.
            """
            
            analysis = call_openai(
                "You are a senior financial analyst. Provide concise, numeric insights with specific dollar amounts and percentages.",
                prompt, temperature=0.2, max_tokens=700
            )
            
            st.markdown(f'<div class="ai-result">{analysis}</div>', unsafe_allow_html=True)

with col_b:
    question = st.text_input("💬 Ask a question about your budget data")
    if question and st.button("Ask", key="ask_question"):
        if not client:
            st.error("OpenAI not configured.")
        elif df_filtered.empty:
            st.warning("No data available with current filters.")
        else:
            # Prepare context based on the question
            context_data = df_filtered.groupby(['Department', 'Account_Desc']).agg({
                'Budget_Allocated': 'sum',
                'Actual_Effective': 'sum'
            }).reset_index()
            context_data['Variance'] = context_data['Actual_Effective'] - context_data['Budget_Allocated']
            context_data['Variance_Pct'] = (context_data['Variance'] / context_data['Budget_Allocated'] * 100).round(2)
            
            # Sort by absolute variance and take top 20
            context_data = context_data.sort_values('Variance', key=abs, ascending=False).head(20)
            
            prompt = f"""
            User Question: {question}
            
            Budget Data Context:
            {context_data.to_string(index=False)}
            
            Answer the user's question based on this data. Be specific with numbers and percentages.
            """
            
            answer = call_openai(
                "You are a financial analyst. Answer questions about budget data with specific insights and numbers.",
                prompt, temperature=0.2, max_tokens=800
            )
            
            st.markdown(f'<div class="ai-result">{answer}</div>', unsafe_allow_html=True)

# =============================
# Download Section
# =============================
st.markdown("---")
if not df_filtered.empty:
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        data=csv_data,
        file_name=f"budget_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
else:
    st.info("No data available to download.")

# =============================
# Footer Note
# =============================
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; font-size: 0.9em;">
    <p><strong>📝 Note:</strong> This app converts fiscal periods to calendar months (Oct=Period 1, Nov=Period 2, etc.)
    and filters out any records with invalid dates or periods for accurate time-series analysis.</p>
</div>
""", unsafe_allow_html=True)