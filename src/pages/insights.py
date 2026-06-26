from __future__ import annotations

import streamlit as st

from src.config.settings import get_yaml_config
from src.insights.factory import get_insight_client
from src.insights.models import InsightData
from src.pages._cache import (
    compute_churn_labels,
    compute_cohort,
    compute_forecast,
    compute_kpis,
    compute_rfm,
)
from src.visualization.insights import render_insights_page


def render_insights_tab(settings) -> None:
    st.title("💡 AI Insights")
    st.markdown("*Plain-language analysis of your subscription data — no jargon required.*")

    df = st.session_state.get("clean_df")
    if df is None or len(df) == 0:
        st.info("Upload data and run the Analytics page first.", icon="📤")
        return

    churn_window = int(st.session_state["churn_window_days"])
    model_results = st.session_state.get("model_results")

    with st.spinner("Assembling insights…"):
        kpi_ts = compute_kpis(df)
        rfm_result = compute_rfm(df)
        label_result = compute_churn_labels(df, churn_window)
        cohort_result = compute_cohort(df)

    if rfm_result is None:
        st.warning(
            "Not enough customers to generate insights. "
            "At least 50 unique customers are required.",
            icon="⚠️",
        )
        return

    insight_data = InsightData(
        kpi_snapshot=kpi_ts.snapshot,
        kpi_ts=kpi_ts,
        churn_label_result=label_result,
        rfm_result=rfm_result,
        churn_window_days=churn_window,
        cohort_result=cohort_result,
        model_metrics=model_results["metrics"] if model_results else None,
        shap_result=model_results["shap"] if model_results else None,
        segments=model_results["segments"] if model_results else None,
        clv_result=model_results["clv"] if model_results else None,
    )

    try:
        yaml_cfg = get_yaml_config()
        _fc_backend = yaml_cfg.get("forecasting", {}).get("backend", "statsmodels")
        _fc_horizon = int(yaml_cfg.get("forecasting", {}).get("horizon_months", 12))
        insight_data.forecast_bundle = compute_forecast(df, _fc_horizon, _fc_backend)
    except Exception:
        pass

    cache_key = "insights_report"
    prior_window = st.session_state.get("insights_churn_window")
    if prior_window is not None and prior_window != churn_window and cache_key in st.session_state:
        st.warning(
            f"⚠️ Churn window changed from **{prior_window} days** to **{churn_window} days**. "
            "Click **Regenerate Insights** to refresh.",
            icon="🔄",
        )

    regen = st.button("🔄 Regenerate Insights", type="secondary")
    if regen or cache_key not in st.session_state:
        with st.spinner("Generating insights…"):
            client = get_insight_client(
                anthropic_api_key=settings.anthropic_api_key,
                openai_api_key=settings.openai_api_key,
            )
            st.session_state[cache_key] = client.generate(insight_data)
            st.session_state["insights_churn_window"] = churn_window

    render_insights_page(st.session_state[cache_key])
