from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.cohort import CohortAnalyzer, CohortResult
from src.analytics.kpis import KPICalculator, KPITimeSeries
from src.feature_engineering.churn_labels import ChurnLabelBuilder, ChurnLabelResult
from src.feature_engineering.rfm import RFMBuilder, RFMResult
from src.utils.log import get_logger

_logger = get_logger(__name__)


@st.cache_data
def compute_kpis(df: pd.DataFrame) -> KPITimeSeries:
    return KPICalculator().calculate(df)


@st.cache_data
def compute_rfm(df: pd.DataFrame) -> RFMResult | None:
    try:
        return RFMBuilder().build(df)
    except Exception:
        _logger.exception("compute_rfm_failed")
        return None


@st.cache_data
def compute_churn_labels(df: pd.DataFrame, churn_window_days: int) -> ChurnLabelResult:
    return ChurnLabelBuilder().build(df, churn_window_days=churn_window_days)


@st.cache_data
def compute_cohort(df: pd.DataFrame) -> CohortResult | None:
    try:
        return CohortAnalyzer().build(df)
    except Exception:
        _logger.exception("compute_cohort_failed")
        return None


@st.cache_data
def compute_forecast(df: pd.DataFrame, horizon: int, backend: str):
    from src.forecasting.pipeline import ForecastingPipeline
    return ForecastingPipeline().run(df, horizon_months=horizon, backend=backend)
