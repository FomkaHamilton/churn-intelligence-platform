"""
BaseForecaster — common interface for all forecasting backends.

Both HoltWinters and Prophet implement this interface so the rest of
the codebase can swap backends via a single config flag without any
downstream changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from src.utils.types import DataFrame


@dataclass
class ForecastResult:
    forecast: pd.Series           # index=month (Period/str), values=predicted quantity
    lower_bound: pd.Series        # 80 % prediction interval lower edge
    upper_bound: pd.Series        # 80 % prediction interval upper edge
    historical: pd.Series         # the training series (for plotting continuity)
    metric_name: str              # "revenue" | "subscribers"
    backend: str                  # "statsmodels" | "prophet"
    horizon_months: int


class BaseForecaster(ABC):
    """Abstract base class for all time-series forecasting backends."""

    @abstractmethod
    def fit_predict(
        self,
        history: pd.Series,
        *,
        horizon_months: int = 12,
    ) -> ForecastResult:
        """
        Fit the model on history and return a ForecastResult.

        Args:
            history: Monthly time series — index must be sortable and represent
                     calendar months.  Values must be numeric (revenue / count).
            horizon_months: Number of months ahead to forecast.
        """
