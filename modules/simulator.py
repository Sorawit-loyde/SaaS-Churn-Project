import streamlit as st
import pandas as pd


def render_simulator(model, encoders, X_encoded, HIGH_RISK_THRESHOLD):
    """Renders the What-If Retention Simulator tab."""
    st.subheader("Retention Simulator")
    st.markdown("Test how strategic changes could theoretically impact churn for a specific profile.")
    st.info("Imagine an 'Average' struggling user. Use the sliders to 'improve' their experience and watch the model instantly recalculate their risk.", icon="💡")

    col_w1, col_w2 = st.columns([1, 1])

    with col_w1:
        test_user = {
            'industry': 'HealthTech',
            'country': 'US',
            'plan_tier': 'Basic',
            'seats': st.slider("Seats", 1, 100, 5),
            'is_trial': 0,
            'total_tickets': 15,
            'avg_sat_score': st.slider("Average Satisfaction (1-5)", 1.0, 5.0, 2.5, 0.5),
            'avg_resolution_hours': st.slider("Support Ticket Resolution Time (Hrs)", 1, 100, 48),
            'total_usage_events': 150,
            'beta_usage_events': 5,
            'avg_usage_duration': st.slider("Weekly Platform Usage (Minutes)", 0, 5000, 500),
        }

    with col_w2:
        sim_df = pd.DataFrame([test_user])
        for col in ['industry', 'country', 'plan_tier']:
            sim_df[col] = encoders[col].transform(sim_df[col])
        sim_df = sim_df[X_encoded.columns]

        sim_prob = (model.predict_proba(sim_df)[:, 1][0] * 100).round(1)

        st.markdown("### Simulated Churn Risk")

        if sim_prob > HIGH_RISK_THRESHOLD:
            st.error(f"{sim_prob}%", icon="🚨")
            st.markdown(f"*This user is highly likely to leave. Try dropping the ticket resolution time below {test_user['avg_resolution_hours']} hours or increasing satisfaction!*")
        elif sim_prob > HIGH_RISK_THRESHOLD * 0.6:
            st.warning(f"{sim_prob}%", icon="⚠️")
            st.markdown("*Elevated risk. Keep improving the metrics to move them to the safe zone.*")
        else:
            st.success(f"{sim_prob}%", icon="✅")
            st.markdown("*Success! The risk is mitigated. The customer is currently in the safe zone.*")
