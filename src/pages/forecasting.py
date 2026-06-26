from __future__ import annotations

import streamlit as st

from src.config.settings import get_yaml_config
from src.pages._cache import compute_forecast
from src.visualization.forecasting import render_forecast_chart, render_forecast_metrics


def render_forecasting_page() -> None:
    st.title("🔮 Forecasting")
    st.markdown("*12-month revenue and subscriber projections with confidence intervals.*")

    df = st.session_state.get("clean_df")
    if df is None or len(df) == 0:
        st.info("Upload data first to run forecasts.", icon="📤")
        return

    yaml_cfg = get_yaml_config()
    backend = yaml_cfg.get("forecasting", {}).get("backend", "statsmodels")
    horizon = int(yaml_cfg.get("forecasting", {}).get("horizon_months", 12))

    try:
        with st.spinner("Running forecast…"):
            bundle = compute_forecast(df, horizon, backend)

        render_forecast_metrics(bundle.revenue, bundle.subscribers)

        st.divider()
        st.markdown("#### Revenue Forecast")
        render_forecast_chart(bundle.revenue, color="#4F8EF7", y_prefix="$")

        st.divider()
        st.markdown("#### Subscriber Forecast")
        render_forecast_chart(bundle.subscribers, color="#34D399", y_suffix=" subs")

        st.divider()
        st.caption(
            f"Backend: **{backend}** · Horizon: **{horizon} months** · "
            "Change backend in `config/settings.yaml` → `forecasting.backend`"
        )

    except Exception as exc:
        st.error(f"**Forecasting failed:** {exc}", icon="⚠️")
        st.info(
            "Forecasting requires at least 6 months of transaction history. "
            "Try the sample dataset (Load sample dataset button on the Upload page).",
            icon="💡",
        )
