"""
Churn Intelligence Platform — Streamlit entry point.

This file wires together the page router and initialises
logging and settings on startup. Individual pages live in pages/.
"""
from __future__ import annotations

import streamlit as st

from src.config.settings import get_settings
from src.utils.log import configure_logging, get_logger

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_settings = get_settings()
configure_logging(_settings.log_level)
_logger = get_logger(__name__)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Churn Intelligence")
    st.caption(f"v{_settings.app_version}")
    st.divider()

    st.markdown("**Navigation**")
    page = st.radio(
        "page",
        options=[
            "🏠  Overview",
            "📤  Upload Data",
            "🔍  Data Quality",
            "📈  Analytics",
            "🤖  Predictions",
            "🔮  Forecasting",
            "💡  Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("⚙️ Settings"):
        churn_window = st.selectbox(
            "Churn window (days)",
            options=[30, 60, 90, 120],
            index=2,
            help="A customer is considered churned if they have no activity within this window.",
        )
        st.session_state["churn_window_days"] = churn_window

    st.divider()
    ai_status = "✅ AI insights active" if _settings.has_ai_provider else "⚠️ Template mode (no API key)"
    st.caption(ai_status)

# ── Main content ──────────────────────────────────────────────────────────────
if "🏠" in (page or ""):
    st.title("📊 Churn Intelligence Platform")
    st.markdown(
        "*Transform subscription transaction data into actionable retention intelligence.*"
    )

    st.info(
        "🚧 **Phase 1 complete — foundation layer is live.** "
        "Data upload and analytics are coming in Phase 2. "
        "Use **Upload Data** when it's ready to get started.",
        icon="🚧",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Version", _settings.app_version)
    col2.metric("Build Phase", "1 / 9")
    col3.metric("Churn Window", f"{_settings.churn_window_days}d")
    col4.metric("AI Mode", "Live" if _settings.has_ai_provider else "Template")

    st.divider()

    st.markdown("### Platform capabilities")

    capabilities = {
        "📤 Data Ingestion": ("Upload CSV/XLSX with automated quality checks", "Phase 2"),
        "📈 Cohort Analytics": ("Retention matrices, MRR, ARPU, churn rate", "Phase 3"),
        "🤖 Churn Prediction": ("ML-powered risk scoring with SHAP explainability", "Phase 4"),
        "💰 CLV Modeling": ("Survival-analysis-based customer lifetime value", "Phase 4"),
        "🔮 Revenue Forecasting": ("12-month subscriber and revenue forecasts", "Phase 5"),
        "💡 AI Insights": ("Executive summaries and intervention recommendations", "Phase 6"),
    }

    for feature, (description, phase) in capabilities.items():
        with st.container():
            c1, c2, c3 = st.columns([2, 4, 1])
            c1.markdown(f"**{feature}**")
            c2.markdown(description)
            c3.markdown(f"`{phase}`")

else:
    st.title(page or "")
    st.info("This section is under active development. Check back soon.", icon="🔄")

_logger.info("page_rendered", page=page, version=_settings.app_version)
