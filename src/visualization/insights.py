"""
Streamlit rendering for the Insights page.
"""
from __future__ import annotations

import streamlit as st

from src.insights.models import InsightReport

# Colour coding for the health summary panel based on keywords in the text
_HEALTH_GOOD = {"strong", "strong:", "strong.", "acceptable", "stable", "no customers have lapsed"}
_HEALTH_WARN = {"not alarming", "room to improve", "monitoring"}
_HEALTH_BAD = {"elevated", "high at", "urgent attention", "high enough"}


def _health_status(summary: str) -> str:
    lower = summary.lower()
    for kw in _HEALTH_BAD:
        if kw in lower:
            return "error"
    for kw in _HEALTH_WARN:
        if kw in lower:
            return "warning"
    return "success"


def render_insights_page(report: InsightReport) -> None:
    """Render a full InsightReport onto the current Streamlit page."""

    # ── Mode banner ───────────────────────────────────────────────────────────
    if report.client_type == "template":
        st.info(
            "Running in **template mode** — insights are generated directly from your data "
            "without an external AI. Add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to `.env` "
            "to enable AI-powered narrative summaries.",
            icon="⚙️",
        )

    # ── 1. Business Health ────────────────────────────────────────────────────
    status = _health_status(report.health_summary)
    st.markdown("### 🏥 Business Health")
    if status == "success":
        st.success(report.health_summary)
    elif status == "warning":
        st.warning(report.health_summary)
    else:
        st.error(report.health_summary)

    st.divider()

    # ── 2 & 3 — side by side ─────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📉 Churn Analysis")
        st.markdown(report.churn_analysis)

    with col_right:
        st.markdown("### 💰 Revenue Outlook")
        st.markdown(report.revenue_outlook)

    st.divider()

    # ── 4. Customer Segments ──────────────────────────────────────────────────
    st.markdown("### 👥 Customer Segments")
    st.markdown(report.customer_segments)

    st.divider()

    # ── 5. Recommendations ────────────────────────────────────────────────────
    st.markdown("### 🎯 Recommended Actions")
    st.markdown(report.recommendations)

    # ── 6. Model Confidence (optional) ───────────────────────────────────────
    if report.model_confidence is not None:
        st.divider()
        with st.expander("🤖 Model Confidence Details", expanded=False):
            st.markdown(report.model_confidence)
    else:
        st.divider()
        st.caption(
            "Train the churn model on the **Predictions** page to unlock model confidence "
            "details and SHAP-based churn driver analysis."
        )
