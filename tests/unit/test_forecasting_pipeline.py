"""Unit tests for the ForecastingPipeline."""
from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.pipeline import ForecastBundle, ForecastingPipeline


def _make_tx_df(n_customers: int = 20, n_months: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n_customers):
        for m in range(n_months):
            rows.append({
                "customer_id": f"c{i}",
                "transaction_date": f"2023-{m + 1:02d}-15",
                "transaction_amount": 100.0 + i,
            })
    return pd.DataFrame(rows)


class TestPipelineOutput:
    def setup_method(self) -> None:
        self.pipeline = ForecastingPipeline()

    def test_returns_forecast_bundle(self) -> None:
        df = _make_tx_df(n_customers=20, n_months=12)
        result = self.pipeline.run(df, horizon_months=3)
        assert isinstance(result, ForecastBundle)

    def test_revenue_metric_name(self) -> None:
        df = _make_tx_df(n_customers=20, n_months=12)
        result = self.pipeline.run(df, horizon_months=3)
        assert result.revenue.metric_name == "revenue"

    def test_subscribers_metric_name(self) -> None:
        df = _make_tx_df(n_customers=20, n_months=12)
        result = self.pipeline.run(df, horizon_months=3)
        assert result.subscribers.metric_name == "subscribers"

    def test_forecast_horizon_respected(self) -> None:
        df = _make_tx_df(n_customers=20, n_months=12)
        result = self.pipeline.run(df, horizon_months=6)
        assert len(result.revenue.forecast) == 6
        assert len(result.subscribers.forecast) == 6

    def test_subscriber_forecast_non_negative(self) -> None:
        df = _make_tx_df(n_customers=20, n_months=12)
        result = self.pipeline.run(df, horizon_months=6)
        assert (result.subscribers.forecast.values >= 0).all()

    def test_revenue_forecast_non_negative(self) -> None:
        df = _make_tx_df(n_customers=20, n_months=12)
        result = self.pipeline.run(df, horizon_months=6)
        assert (result.revenue.forecast.values >= 0).all()

    def test_default_backend_is_statsmodels(self) -> None:
        df = _make_tx_df(n_customers=20, n_months=12)
        result = self.pipeline.run(df)
        assert result.revenue.backend == "statsmodels"
        assert result.subscribers.backend == "statsmodels"
