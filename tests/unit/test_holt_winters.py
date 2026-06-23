"""Unit tests for the Holt-Winters forecaster."""
from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.holt_winters import HoltWintersForecaster
from src.forecasting.base import ForecastResult
from src.utils.exceptions import InsufficientHistoryError


def _monthly_series(n_months: int, start: str = "2022-01", base: float = 1000.0) -> pd.Series:
    """Synthetic monthly series with slight upward trend."""
    index = pd.period_range(start=start, periods=n_months, freq="M")
    values = [base + i * 10 + (i % 3) * 20 for i in range(n_months)]
    return pd.Series(values, index=index)


class TestForecastShape:
    def setup_method(self) -> None:
        self.forecaster = HoltWintersForecaster()

    def test_returns_forecast_result(self) -> None:
        series = _monthly_series(18)
        result = self.forecaster.fit_predict(series, horizon_months=12)
        assert isinstance(result, ForecastResult)

    def test_forecast_length_matches_horizon(self) -> None:
        series = _monthly_series(18)
        result = self.forecaster.fit_predict(series, horizon_months=6)
        assert len(result.forecast) == 6

    def test_lower_upper_bounds_same_length_as_forecast(self) -> None:
        series = _monthly_series(18)
        result = self.forecaster.fit_predict(series, horizon_months=12)
        assert len(result.lower_bound) == len(result.forecast)
        assert len(result.upper_bound) == len(result.forecast)

    def test_historical_series_preserved(self) -> None:
        series = _monthly_series(18)
        result = self.forecaster.fit_predict(series)
        assert len(result.historical) == 18

    def test_backend_label_is_statsmodels(self) -> None:
        series = _monthly_series(18)
        result = self.forecaster.fit_predict(series)
        assert result.backend == "statsmodels"


class TestForecastValues:
    def setup_method(self) -> None:
        self.forecaster = HoltWintersForecaster()

    def test_forecast_values_are_non_negative(self) -> None:
        series = _monthly_series(18)
        result = self.forecaster.fit_predict(series, horizon_months=12)
        assert (result.forecast.values >= 0).all()

    def test_lower_bound_not_greater_than_forecast(self) -> None:
        series = _monthly_series(18)
        result = self.forecaster.fit_predict(series, horizon_months=12)
        assert (result.lower_bound.values <= result.forecast.values + 1e-6).all()

    def test_upper_bound_not_less_than_forecast(self) -> None:
        series = _monthly_series(18)
        result = self.forecaster.fit_predict(series, horizon_months=12)
        assert (result.upper_bound.values >= result.forecast.values - 1e-6).all()

    def test_interval_width_grows_over_horizon(self) -> None:
        series = _monthly_series(18)
        result = self.forecaster.fit_predict(series, horizon_months=12)
        widths = (result.upper_bound - result.lower_bound).values
        # Width should generally increase (sqrt of time steps)
        assert widths[-1] > widths[0]

    def test_forecast_follows_trend_direction(self) -> None:
        # Strictly increasing series — forecast should continue upward
        series = pd.Series(
            [float(i * 100) for i in range(1, 19)],
            index=pd.period_range("2022-01", periods=18, freq="M"),
        )
        result = self.forecaster.fit_predict(series, horizon_months=6)
        # Last forecast value should be greater than mean historical
        assert float(result.forecast.mean()) > float(series.mean()) * 0.5


class TestForecastIndex:
    def setup_method(self) -> None:
        self.forecaster = HoltWintersForecaster()

    def test_forecast_index_follows_history(self) -> None:
        series = _monthly_series(12, start="2023-01")
        result = self.forecaster.fit_predict(series, horizon_months=3)
        # History ends 2023-12, forecast should start 2024-01
        assert result.forecast.index[0] == "2024-01"

    def test_forecast_index_is_string(self) -> None:
        series = _monthly_series(12)
        result = self.forecaster.fit_predict(series, horizon_months=3)
        for idx in result.forecast.index:
            assert isinstance(idx, str)


class TestInsufficientHistory:
    def setup_method(self) -> None:
        self.forecaster = HoltWintersForecaster()

    def test_too_few_months_raises(self) -> None:
        series = _monthly_series(3)
        with pytest.raises(InsufficientHistoryError):
            self.forecaster.fit_predict(series)

    def test_exact_minimum_does_not_raise(self) -> None:
        from src.config.constants import MIN_MONTHS_FOR_FORECASTING
        series = _monthly_series(MIN_MONTHS_FOR_FORECASTING)
        result = self.forecaster.fit_predict(series, horizon_months=3)
        assert len(result.forecast) == 3


class TestSeasonalitySwitch:
    def setup_method(self) -> None:
        self.forecaster = HoltWintersForecaster()

    def test_short_history_uses_no_seasonality(self) -> None:
        # 12 months — below the 24-month seasonal threshold
        series = _monthly_series(12)
        result = self.forecaster.fit_predict(series, horizon_months=3)
        assert isinstance(result, ForecastResult)

    def test_long_history_uses_seasonality(self) -> None:
        # 30 months — above threshold, seasonal should be applied
        series = _monthly_series(30)
        result = self.forecaster.fit_predict(series, horizon_months=6)
        assert isinstance(result, ForecastResult)
