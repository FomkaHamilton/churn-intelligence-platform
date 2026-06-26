"""
Holt-Winters (Triple Exponential Smoothing) forecaster.

Default backend — requires only statsmodels which is already in
requirements.txt.  Handles trend and optional seasonality automatically.

Confidence intervals are computed from the model's prediction standard
errors using an 80 % coverage factor (±1.282 σ).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config.constants import MIN_MONTHS_FOR_FORECASTING
from src.forecasting.base import BaseForecaster, ForecastResult
from src.utils.exceptions import InsufficientHistoryError


class HoltWintersForecaster(BaseForecaster):
    """
    Triple Exponential Smoothing forecaster.

    Seasonality is enabled only when there are at least 24 months of history
    (two complete cycles needed for reliable seasonal parameter estimation).
    Falls back to additive trend + no seasonality for shorter series.
    """

    # 80 % prediction interval z-score
    _Z80 = 1.282

    def fit_predict(
        self,
        history: pd.Series,
        *,
        horizon_months: int = 12,
    ) -> ForecastResult:
        if len(history) < MIN_MONTHS_FOR_FORECASTING:
            raise InsufficientHistoryError(
                f"Holt-Winters requires at least {MIN_MONTHS_FOR_FORECASTING} months of "
                f"history, got {len(history)}."
            )

        series = history.sort_index().astype(float)
        n = len(series)

        # Choose seasonal configuration based on available data
        use_seasonal = n >= 24
        seasonal_periods = 12 if use_seasonal else None
        seasonal = "add" if use_seasonal else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                series.values,
                trend="add",
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True, use_brute=False)

        point_forecast = fit.forecast(horizon_months)
        point_forecast = np.maximum(point_forecast, 0)  # no negative revenue/subscribers

        # Approximate prediction intervals from in-sample residual std
        residual_std = float(np.std(fit.resid, ddof=1))
        interval_width = self._Z80 * residual_std * np.sqrt(
            np.arange(1, horizon_months + 1)
        )

        # Build month-period index for forecast
        last_month = self._last_period(series)
        next_start = last_month.to_timestamp() + pd.DateOffset(months=1)
        forecast_index = pd.period_range(start=next_start, periods=horizon_months, freq="M")
        str_index = [str(p) for p in forecast_index]

        forecast_s = pd.Series(point_forecast, index=str_index)
        lower_s = pd.Series(
            np.maximum(point_forecast - interval_width, 0),
            index=str_index,
        )
        upper_s = pd.Series(
            point_forecast + interval_width,
            index=str_index,
        )

        historical_s = pd.Series(series.values, index=[str(i) for i in series.index])

        return ForecastResult(
            forecast=forecast_s,
            lower_bound=lower_s,
            upper_bound=upper_s,
            historical=historical_s,
            metric_name="",  # caller sets this
            backend="statsmodels",
            horizon_months=horizon_months,
        )

    def _last_period(self, series: pd.Series) -> pd.Period:
        """Convert the last index value to a pd.Period(freq='M')."""
        last = series.index[-1]
        if isinstance(last, pd.Period):
            return last
        return pd.Period(str(last), freq="M")
