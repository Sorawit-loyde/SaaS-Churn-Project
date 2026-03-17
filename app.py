"""
app.py — Main entry point for the SaaS Retention Dashboard.
Handles: page config, data loading, sidebar filters, KPIs, and tab routing.
All tab rendering logic lives in the modules/ directory.
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from modules import render_overview, render_individual_lookup, render_simulator

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SaaS Retention Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

HIGH_RISK_THRESHOLD = 40  # % — calibrated to OOB probability distribution


# ─────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("data/processed/master_customer_data.csv")
    df['Monthly_Revenue'] = df['seats'] * 50

    X = df.drop(columns=['account_id', 'churn_flag', 'Monthly_Revenue'], errors='ignore')
    y = df['churn_flag'].astype(int)

    encoders = {}
    for col in ['industry', 'country', 'plan_tier']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X['is_trial'] = X['is_trial'].astype(int)

    # OOB predictions give unbiased probabilities (each sample predicted by trees it wasn't trained on)
    model = RandomForestClassifier(random_state=42, n_estimators=200, oob_score=True)
    model.fit(X, y)
    df['Churn_Probability'] = (model.oob_decision_function_[:, 1] * 100).round(1)

    return df, X, model, encoders


df, X_encoded, model, encoders = load_and_prep_data()


# ─────────────────────────────────────────────
# SIDEBAR: FILTERS
# ─────────────────────────────────────────────
st.sidebar.title("🎛️ Dashboard Filters")
st.sidebar.markdown("Filter the dashboard by region and customer segment.")

if "selected_region" not in st.session_state:
    st.session_state.selected_region = df['country'].unique().tolist()
if "selected_plan" not in st.session_state:
    st.session_state.selected_plan = df['plan_tier'].unique().tolist()


def clear_filters():
    st.session_state.selected_region = df['country'].unique().tolist()
    st.session_state.selected_plan = df['plan_tier'].unique().tolist()


selected_region = st.sidebar.multiselect("Region (Country)", options=df['country'].unique(), key="selected_region")
selected_plan = st.sidebar.multiselect("Customer Segment (Plan)", options=df['plan_tier'].unique(), key="selected_plan")

filtered_df = df[
    (df['country'].isin(selected_region)) &
    (df['plan_tier'].isin(selected_plan))
]

if len(filtered_df) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR: BUSINESS HEALTH KPIs
# ─────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.title("🏢 Business Health")

total_customers = len(filtered_df)
churned_customers = len(filtered_df[filtered_df['churn_flag'] == True])
churn_rate = (churned_customers / total_customers) * 100

active_df = filtered_df[filtered_df['churn_flag'] == False]
high_risk_df = active_df[active_df['Churn_Probability'] > HIGH_RISK_THRESHOLD]
simulated_rev_at_risk = high_risk_df['seats'].sum() * 50

st.sidebar.metric(
    "Churn Rate (Goal: <15%)",
    f"{churn_rate:.1f}%",
    f"{churn_rate - 15.0:+.1f}% vs Target",
    delta_color="inverse",
    help="Target benchmark is 15%. Positive delta means above target."
)
st.sidebar.metric(
    "Est. Monthly Revenue at Risk",
    f"${simulated_rev_at_risk:,.0f}",
    f"{len(high_risk_df)} High-Risk Accounts",
    delta_color="off",
    help=f"Total MRR (seats × $50) of active customers >{HIGH_RISK_THRESHOLD}% churn probability."
)

st.sidebar.divider()
with st.sidebar.expander("Model Transparency (Tech Specs)"):
    st.markdown("**Algorithm:** Random Forest (200 trees, OOB)")
    st.progress(85)
    st.caption("Accuracy: 85% | Precision: 76% | Recall: 63%")
    st.caption("OOB predictions used for unbiased risk scoring.")


# ─────────────────────────────────────────────
# MAIN AREA: TITLE + TABS
# ─────────────────────────────────────────────
st.title("🛡️ SaaS Retention Command Center")
st.markdown("Monitor risk, analyze trends, and identify actionable insights to prevent customer churn.")

tab1, tab2, tab3 = st.tabs(["Overview & Trends", "Individual Lookup", "What-If Simulator"])

with tab1:
    render_overview(active_df, model, X_encoded, HIGH_RISK_THRESHOLD)

with tab2:
    render_individual_lookup(filtered_df, df, high_risk_df, X_encoded, model, encoders, HIGH_RISK_THRESHOLD, clear_filters)

with tab3:
    render_simulator(model, encoders, X_encoded, HIGH_RISK_THRESHOLD)
