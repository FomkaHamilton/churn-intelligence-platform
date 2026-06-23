"""
ForecastingPipeline — orchestrates revenue and subscriber forecasts.

Builds the monthly time series from raw transactions, selects the
configured backend (statsmodels or prophet), and returns a pair of
ForecastResult objects (one for revenue, one for active subscribers).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.analytics.kpis import KPICalculator
from src.forecasting.base import BaseForecaster, ForecastResult
from src.forecasting.holt_winters import HoltWintersForecaster
from src.utils.exceptions import InsufficientHistoryError
from src.utils.types import DataFrame


@dataclass
class ForecastBundle:
    revenue: ForecastResult
    subscribers: ForecastResult


def _get_forecaster(backend: str) -> BaseForecaster:
    if backend == "prophet":
        from src.forecasting.prophet_forecaster import ProphetForecaster
        return ProphetForecaster()
    return HoltWintersForecaster()


class ForecastingPipeline:
    """
    Build revenue and subscriber forecasts from a clean transaction DataFrame.

    Delegates to the configured backend (default: statsmodels Holt-Winters).
    """

    def run(
        self,
        df: DataFrame,
        *,
        horizon_months: int = 12,
        backend: str = "statsmodels",
    ) -> ForecastBundle:
        """
        Args:
            df: Clean transaction DataFrame.
            horizon_months: Number of months ahead to forecast.
            backend: "statsmodels" (default) or "prophet".
        """
        kpi_ts = KPICalculator().calculate(df)

        forecaster = _get_forecaster(backend)

        revenue_result = forecaster.fit_predict(
            kpi_ts.monthly_revenue, horizon_months=horizon_months
        )
        revenue_result.metric_name = "revenue"

        # Convert Period-indexed subscribers to string for consistency
        sub_series = kpi_ts.monthly_active.copy()

        subscribers_result = forecaster.fit_predict(
            sub_series, horizon_months=horizon_months
        )
        subscribers_result.metric_name = "subscribers"

        return ForecastBundle(revenue=revenue_result, subscribers=subscribers_result)
