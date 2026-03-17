import shap
import streamlit as st
import pandas as pd

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


def render_individual_lookup(filtered_df, df, high_risk_df, X_encoded, model, encoders, HIGH_RISK_THRESHOLD, clear_filters):
    """Renders the Individual Account Lookup tab."""
    st.subheader("Deep Dive: Individual Account Risk")
    st.markdown("Select a specific highly-threatened account to see exactly *why* they might leave.")

    lookup_accounts = high_risk_df.sort_values('Churn_Probability', ascending=False)['account_id'].tolist()

    if len(lookup_accounts) > 0:
        selected_account = st.selectbox("Select Account ID (Sorted by Highest Risk):", lookup_accounts)

        if selected_account:
            acct_idx = filtered_df.index[filtered_df['account_id'] == selected_account][0]
            acct_data = filtered_df.loc[acct_idx]

            full_acct_pos = df.index.get_loc(df.index[df['account_id'] == selected_account][0])
            acct_features_encoded = X_encoded.iloc[[full_acct_pos]]

            prob = acct_data['Churn_Probability']

            col_m1, col_m2 = st.columns([1, 2])

            with col_m1:
                st.markdown(f"**Risk Score: {prob}%**")
                if prob > HIGH_RISK_THRESHOLD:
                    st.error("🚨 Critical Risk Level")
                elif prob > HIGH_RISK_THRESHOLD * 0.6:
                    st.warning("⚠️ Elevated Risk Level")
                else:
                    st.success("✅ Healthy Account")

                st.metric("Avg Satisfaction", acct_data['avg_sat_score'])
                st.metric("Total Support Tickets", acct_data['total_tickets'])
                st.metric("Avg Resolution (Hrs)", acct_data['avg_resolution_hours'])

                st.divider()
                st.markdown("**Quick Actions**")
                action_btn = st.button("Generate Retention Email ✉️", use_container_width=True)
                alert_btn = st.button("Flag for Success Manager 🚩", use_container_width=True)
                if action_btn:
                    st.success(f"Draft email generated for {selected_account}!")
                if alert_btn:
                    st.success(f"Account {selected_account} flagged in CRM.")

            with col_m2:
                st.markdown("**Top 3 Reasons for Risk (SHAP Explainer)**")
                st.caption("What specific metrics are pushing this user's risk up?")

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(acct_features_encoded)

                if isinstance(shap_values, list):
                    shap_val_target = shap_values[1][0]
                else:
                    shap_val_target = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]

                shap_df = pd.DataFrame({
                    'Feature': [FEATURE_NAMES.get(c, c) for c in X_encoded.columns],
                    'OriginalFeature': X_encoded.columns,
                    'Impact': shap_val_target
                })

                shap_df['Abs_Impact'] = shap_df['Impact'].abs()
                top_factors = shap_df.sort_values(by='Abs_Impact', ascending=False).head(3)

                for _, row in top_factors.iterrows():
                    direction = "📈 Increases Risk" if row['Impact'] > 0 else "📉 Decreases Risk"
                    val = acct_features_encoded[row['OriginalFeature']].values[0]
                    if row['OriginalFeature'] in encoders:
                        val = encoders[row['OriginalFeature']].inverse_transform([val])[0]
                    color = "red" if row['Impact'] > 0 else "green"
                    st.markdown(f"- **{row['Feature']}** ({val}): :{color}[{direction}]")
    else:
        st.info("No high-risk accounts found based on your current filters.")
        st.button("🔄 Clear Filters to Reset View", on_click=clear_filters)
