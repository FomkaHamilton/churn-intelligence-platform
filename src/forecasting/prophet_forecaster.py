"""
Prophet-based forecaster (optional backend).

Prophet is NOT listed in requirements.txt because it requires C++ build
tools (pystan) which fail silently on some Windows setups.  Install via:
    pip install -r requirements-optional.txt

This module is import-guarded so the rest of the platform works without it.
Switch the backend in config/settings.yaml:
    forecasting:
      backend: "prophet"
"""
from __future__ import annotations

import pandas as pd

from src.config.constants import MIN_MONTHS_FOR_FORECASTING
from src.forecasting.base import BaseForecaster, ForecastResult
from src.utils.exceptions import InsufficientHistoryError


class ProphetForecaster(BaseForecaster):
    """
    Meta Prophet forecaster with automatic trend changepoint detection.

    Better than Holt-Winters for series with:
      - Strong annual seasonality
      - Irregular changepoints (product launches, pricing events)
      - Missing months in the history
    """

    def fit_predict(
        self,
        history: pd.Series,
        *,
        horizon_months: int = 12,
    ) -> ForecastResult:
        try:
            from prophet import Prophet  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Prophet is not installed. Run: pip install -r requirements-optional.txt"
            ) from exc

        if len(history) < MIN_MONTHS_FOR_FORECASTING:
            raise InsufficientHistoryError(
                f"Prophet requires at least {MIN_MONTHS_FOR_FORECASTING} months of "
                f"history, got {len(history)}."
            )

        series = history.sort_index().astype(float)

        # Prophet expects a DataFrame with columns ds (datetime) and y (value)
        ds = pd.to_datetime([str(idx) for idx in series.index])
        df_train = pd.DataFrame({"ds": ds, "y": series.values})

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
            changepoint_prior_scale=0.05,
        )
        model.fit(df_train)

        future = model.make_future_dataframe(periods=horizon_months, freq="MS")
        forecast_df = model.predict(future)

        # Extract the horizon rows only
        horizon_rows = forecast_df.tail(horizon_months)
        forecast_index = [
            str(pd.Period(ts, freq="M")) for ts in horizon_rows["ds"]
        ]

        forecast_s = pd.Series(
            horizon_rows["yhat"].clip(lower=0).values, index=forecast_index
        )
        lower_s = pd.Series(
            horizon_rows["yhat_lower"].clip(lower=0).values, index=forecast_index
        )
        upper_s = pd.Series(horizon_rows["yhat_upper"].values, index=forecast_index)

        historical_s = pd.Series(series.values, index=[str(i) for i in series.index])

        return ForecastResult(
            forecast=forecast_s,
            lower_bound=lower_s,
            upper_bound=upper_s,
            historical=historical_s,
            metric_name="",
            backend="prophet",
            horizon_months=horizon_months,
        )
