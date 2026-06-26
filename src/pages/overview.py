from __future__ import annotations

import streamlit as st


def render_overview_page(settings) -> None:
    st.title("📊 Churn Intelligence Platform")
    st.markdown("*Transform subscription transaction data into actionable retention intelligence.*")

    # ── Status banner ─────────────────────────────────────────────────────────
    if st.session_state.get("clean_df") is not None and st.session_state.get("filename"):
        st.success(f"✅ Data loaded: **{st.session_state['filename']}** — use the sidebar to navigate.")
    else:
        st.info(
            "No data loaded yet. Follow the steps below to get started.",
            icon="📤",
        )
        st.markdown("---")
        st.markdown("#### How to get started")
        steps = [
            ("📤", "**Upload Data**", "Upload a CSV or XLSX of your subscription transaction history."),
            ("🔍", "**Review Quality**", "Inspect the automated data quality report — fix warnings before proceeding."),
            ("📈", "**Explore Analytics**", "Cohort retention, MRR trends, RFM scores, and churn rates."),
            ("🤖", "**Train Predictions**", "One click trains the LR + RF ensemble and scores every customer."),
            ("🔮", "**Run Forecasting**", "12-month revenue and subscriber projections with confidence intervals."),
            ("💡", "**Generate Insights**", "Auto-generated executive briefing — download as PDF when ready."),
        ]
        for i, (icon, title, desc) in enumerate(steps, 1):
            with st.container(border=True):
                c_num, c_icon, c_text = st.columns([0.4, 0.4, 6])
                c_num.markdown(f"**{i}**")
                c_icon.markdown(icon)
                c_text.markdown(f"{title} — {desc}")
        st.markdown("---")

    # ── Headline metrics ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Version", settings.app_version)
    col2.metric("Build Phase", "7 / 9")
    col3.metric("Churn Window", f"{st.session_state['churn_window_days']}d")
    col4.metric("AI Mode", "Live" if settings.has_ai_provider else "Template")

    st.divider()

    # ── Capability cards ──────────────────────────────────────────────────────
    st.markdown("### Platform capabilities")
    capabilities = [
        ("📤 Data Ingestion",      "CSV/XLSX upload, validation, quality profiling",          "✅ Phase 2 — Live"),
        ("📈 Cohort Analytics",    "Retention matrices, MRR, ARPU, churn rate",               "✅ Phase 3 — Live"),
        ("🤖 Churn Prediction",    "ML risk scoring with SHAP explainability",                 "✅ Phase 4 — Live"),
        ("💰 CLV Modeling",        "Survival-analysis-based customer lifetime value",          "✅ Phase 4 — Live"),
        ("🔮 Revenue Forecasting", "12-month subscriber and revenue forecasts",                "✅ Phase 5 — Live"),
        ("💡 AI Insights",         "Executive summaries and intervention recommendations",     "✅ Phase 6 — Live"),
    ]
    for feature, desc, status in capabilities:
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 4, 2])
            c1.markdown(f"**{feature}**")
            c2.markdown(desc)
            c3.markdown(f"`{status}`")
