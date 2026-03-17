import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np


FEATURE_NAMES = {
    'avg_sat_score': 'Avg Satisfaction Score',
    'total_tickets': 'Total Support Tickets',
    'avg_resolution_hours': 'Avg Resolution Time (Hrs)',
    'total_usage_events': 'Total Product Engagement',
    'beta_usage_events': 'Beta Feature Usage',
    'avg_usage_duration': 'Active Weekly Minutes',
    'is_trial': 'Is Trial User',
    'seats': 'Number of Seats',
    'industry': 'Industry',
    'country': 'Country',
    'plan_tier': 'Plan Tier'
}

HIGH_RISK_THRESHOLD = 40


def render_overview(active_df, model, X_encoded, HIGH_RISK_THRESHOLD):
    """Renders the Overview & Trends tab."""
    st.subheader("Global Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Churn Risk Distribution**")
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(0, 100, 21)
        sns.histplot(active_df[active_df['Churn_Probability'] <= 25]['Churn_Probability'], bins=bins, color='mediumseagreen', ax=ax, label="Safe (0-25%)")
        sns.histplot(active_df[(active_df['Churn_Probability'] > 25) & (active_df['Churn_Probability'] <= HIGH_RISK_THRESHOLD)]['Churn_Probability'], bins=bins, color='gold', ax=ax, label=f"Warning (25-{HIGH_RISK_THRESHOLD}%)")
        sns.histplot(active_df[active_df['Churn_Probability'] > HIGH_RISK_THRESHOLD]['Churn_Probability'], bins=bins, color='crimson', ax=ax, label=f"Danger (>{HIGH_RISK_THRESHOLD}%)")
        ax.set_xlabel("Predicted Churn Probability (%)")
        ax.set_ylabel("Number of Active Customers")
        ax.set_xlim(0, 100)
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.markdown("**Actionable Plan Analysis (Volume vs. Value)**")
        plan_risk_rev = active_df[active_df['Churn_Probability'] > HIGH_RISK_THRESHOLD].groupby('plan_tier')['Monthly_Revenue'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        if plan_risk_rev.empty:
            ax.text(0.5, 0.5, 'No at-risk accounts for selected filters', ha='center', va='center')
        else:
            sns.barplot(data=plan_risk_rev, x='plan_tier', y='Monthly_Revenue', ax=ax, palette='Blues_d')
        ax.set_title(f"Revenue at Risk by Plan Tier (>{HIGH_RISK_THRESHOLD}% Prob)")
        ax.set_ylabel("Total Est. Monthly Revenue ($)")
        ax.set_xlabel("Plan Tier")
        st.pyplot(fig)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Global Feature Importance**")
        st.caption("Which factors drive churn globally across all accounts?")
        importances = pd.Series(model.feature_importances_, index=[FEATURE_NAMES.get(c, c) for c in X_encoded.columns]).sort_values(ascending=False).head(6)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=importances.values, y=importances.index, palette='viridis', ax=ax)
        ax.set_xlabel("Relative Importance")
        st.pyplot(fig)
        st.info("💡 **Insight:** Platform engagement and resolution time are the strongest predictors of loyalty. Accounts with low active minutes or high ticket resolution times are significantly more likely to churn.")

    with col4:
        st.markdown("**Satisfaction vs. Risk Heatmap**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data=active_df, x="avg_sat_score", y="Churn_Probability", fill=True, cmap="Reds", ax=ax, thresh=0.05)
        ax.set_xlabel("Average Satisfaction Score")
        ax.set_ylabel("Predicted Churn Risk (%)")
        max_prob = active_df['Churn_Probability'].max()
        zone_y_start = max_prob * 0.7
        zone_height = max_prob - zone_y_start
        rect = patches.Rectangle((1.0, zone_y_start), 2.0, zone_height, linewidth=2, edgecolor='black', facecolor='none', linestyle='dashed')
        ax.add_patch(rect)
        ax.text(2.0, zone_y_start + zone_height / 2, "Priority\nIntervention\nZone", color='black', fontsize=9, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'))
        ax.set_xlim(1, 5.5)
        st.pyplot(fig)
