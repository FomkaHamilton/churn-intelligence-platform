from __future__ import annotations

import streamlit as st


def render_overview_page(settings) -> None:
    st.title("📊 Churn Intelligence Platform")
    st.markdown("*Transform subscription transaction data into actionable retention intelligence.*")

    if st.session_state.get("clean_df") is not None and st.session_state.get("filename"):
        st.success(f"✅ Data loaded: **{st.session_state['filename']}** — use the sidebar to navigate.")
    else:
        st.info("👈 Start by uploading your subscription data using **Upload Data** in the sidebar.", icon="📤")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Version", settings.app_version)
    col2.metric("Build Phase", "7 / 9")
    col3.metric("Churn Window", f"{st.session_state['churn_window_days']}d")
    col4.metric("AI Mode", "Live" if settings.has_ai_provider else "Template")

    st.divider()
    st.markdown("### Platform capabilities")
    capabilities = {
        "📤 Data Ingestion": ("CSV/XLSX upload, validation, quality profiling", "✅ Phase 2 — Live"),
        "📈 Cohort Analytics": ("Retention matrices, MRR, ARPU, churn rate", "✅ Phase 3 — Live"),
        "🤖 Churn Prediction": ("ML risk scoring with SHAP explainability", "✅ Phase 4 — Live"),
        "💰 CLV Modeling": ("Survival-analysis-based customer lifetime value", "✅ Phase 4 — Live"),
        "🔮 Revenue Forecasting": ("12-month subscriber and revenue forecasts", "✅ Phase 5 — Live"),
        "💡 AI Insights": ("Executive summaries and intervention recommendations", "✅ Phase 6 — Live"),
    }
    for feature, (desc, status) in capabilities.items():
        c1, c2, c3 = st.columns([2, 4, 2])
        c1.markdown(f"**{feature}**")
        c2.markdown(desc)
        c3.markdown(f"`{status}`")
