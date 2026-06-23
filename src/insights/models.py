"""
Data contracts for the insight layer.

InsightData  — all computed results bundled as input.
InsightReport — structured natural-language output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.analytics.cohort import CohortResult
from src.analytics.kpis import KPISnapshot, KPITimeSeries
from src.feature_engineering.churn_labels import ChurnLabelResult
from src.feature_engineering.rfm import RFMResult
from src.modeling.churn_model import ModelMetrics
from src.modeling.clv import CLVResult
from src.modeling.explainability import SHAPResult
from src.forecasting.pipeline import ForecastBundle


@dataclass
class InsightData:
    """All computed platform results, bundled for insight generation."""

    # Required — analytics layer must have run
    kpi_snapshot: KPISnapshot
    kpi_ts: KPITimeSeries
    churn_label_result: ChurnLabelResult
    rfm_result: RFMResult
    churn_window_days: int = 90

    # Optional — populated only if those pages have run
    cohort_result: CohortResult | None = None
    model_metrics: ModelMetrics | None = None
    shap_result: SHAPResult | None = None
    segments: pd.Series | None = None
    clv_result: CLVResult | None = None
    forecast_bundle: ForecastBundle | None = None


@dataclass
class InsightReport:
    """Structured natural-language output from an insight client."""

    health_summary: str
    churn_analysis: str
    revenue_outlook: str
    customer_segments: str
    recommendations: str
    model_confidence: str | None  # None when no model has been trained

    client_type: str = "template"
    sections_with_ml: bool = field(init=False)

    def __post_init__(self) -> None:
        self.sections_with_ml = self.model_confidence is not None
