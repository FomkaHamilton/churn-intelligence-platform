from __future__ import annotations

import streamlit as st

from src.pages._cache import (
    compute_churn_labels,
    compute_cohort,
    compute_kpis,
    compute_rfm,
)
from src.visualization.analytics import (
    render_churn_trend,
    render_cohort_heatmap,
    render_kpi_strip,
    render_revenue_trend,
)


def render_analytics_page() -> None:
    st.title("📈 Analytics")

    df = st.session_state.get("clean_df")
    if df is None or len(df) == 0:
        st.info("Upload data first to see analytics.", icon="📤")
        return

    churn_window = int(st.session_state["churn_window_days"])
    st.caption(f"📊 Computed with a **{churn_window}-day** churn window — adjust in ⚙️ Settings")

    with st.spinner("Computing KPIs…"):
        kpi_ts = compute_kpis(df)
    with st.spinner("Building RFM features…"):
        rfm_result = compute_rfm(df)
    with st.spinner("Labelling churn…"):
        label_result = compute_churn_labels(df, churn_window)
    with st.spinner("Building cohort matrix…"):
        cohort_result = compute_cohort(df)

    render_kpi_strip(kpi_ts.snapshot)

    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Monthly Revenue")
        render_revenue_trend(kpi_ts)
    with col_right:
        st.markdown("#### Monthly Churn Rate")
        render_churn_trend(kpi_ts)

    st.divider()
    st.markdown("#### Cohort Retention Heatmap")
    if cohort_result is not None:
        render_cohort_heatmap(cohort_result)
    else:
        st.warning(
            "Not enough data to build cohort matrix — need at least 10 customers "
            "per cohort month.",
            icon="⚠️",
        )

    st.divider()
    col_churn, col_rfm = st.columns(2)

    with col_churn:
        st.markdown("#### Churn Summary")
        cc1, cc2 = st.columns(2)
        cc1.metric("Churned", f"{label_result.n_churned:,}",
                   delta=f"-{label_result.churn_rate:.1%}", delta_color="inverse")
        cc2.metric("Active", f"{label_result.n_active:,}",
                   delta=f"+{1 - label_result.churn_rate:.1%}", delta_color="normal")
        st.caption(
            f"Window: {churn_window} days · "
            f"Reference date: {label_result.reference_date.date()}"
        )

    with col_rfm:
        st.markdown("#### RFM Summary")
        if rfm_result is not None:
            feat = rfm_result.features
            rc1, rc2 = st.columns(2)
            rc1.metric("Avg Recency", f"{feat['recency_days'].mean():.0f} days")
            rc2.metric("Avg Frequency", f"{feat['frequency'].mean():.1f}×")
            rc1.metric("Avg Total Spend", f"${feat['monetary_total'].mean():,.0f}")
            rc2.metric("Avg AOV", f"${feat['aov'].mean():.2f}")
        else:
            st.info("Not enough customers to compute RFM features.", icon="ℹ️")
