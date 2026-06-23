"""
Customer Lifetime Value model using Kaplan-Meier survival analysis.

The KM fitter models "probability of remaining an active customer for at
least N days", treating churn as the event of interest and using tenure_days
as the observation duration.

Active customers are right-censored: we know they survived up to tenure_days
but don't know their eventual churn time.  Churned customers are uncensored:
their tenure_days is the observed lifetime.

CLV formula (per customer):
    expected_remaining_months = max(0, (median_lifetime_days - tenure_days) / 30.44)
    clv = monthly_spend × expected_remaining_months
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.exceptions import InsufficientDataError
from src.utils.types import DataFrame


@dataclass
class CLVResult:
    clv_per_customer: DataFrame    # customer_id | expected_clv | expected_remaining_months | monthly_spend
    median_lifetime_days: float
    population_survival_curve: DataFrame | None  # timeline | survival_prob (for charts)


class CLVModel:
    """Kaplan-Meier-based CLV estimator."""

    MIN_CUSTOMERS = 10

    def fit_and_predict(
        self,
        rfm_features: DataFrame,
        churn_label_df: DataFrame,
    ) -> CLVResult:
        """
        Fit the KM curve and compute per-customer CLV.

        Args:
            rfm_features: DataFrame with customer_id, tenure_days, aov, tx_per_month.
            churn_label_df: DataFrame with customer_id, is_churned.
        """
        try:
            from lifelines import KaplanMeierFitter
        except ImportError as exc:
            raise ImportError(
                "lifelines is required for CLV modeling. "
                "Run: pip install lifelines"
            ) from exc

        merged = rfm_features.merge(
            churn_label_df[["customer_id", "is_churned"]], on="customer_id"
        )

        if len(merged) < self.MIN_CUSTOMERS:
            raise InsufficientDataError("CLV", self.MIN_CUSTOMERS, len(merged))

        durations = merged["tenure_days"].clip(lower=1).values
        events = merged["is_churned"].astype(int).values

        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=events, label="Customer Lifetime")

        median_lifetime = float(kmf.median_survival_time_)
        if np.isnan(median_lifetime) or median_lifetime <= 0:
            # Very low churn rate — median can't be reached; use mean tenure × 2 as proxy
            median_lifetime = float(merged["tenure_days"].mean()) * 2

        # Per-customer expected remaining months.
        # Active customers always retain at least 30 days of expected future value —
        # clipping them to 0 when their tenure exceeds the median is wrong, because
        # reaching the median just means they're a survivor, not that they're about to leave.
        is_active = ~merged["is_churned"].astype(bool)
        remaining_days = median_lifetime - merged["tenure_days"]
        remaining_days_churned = remaining_days.clip(lower=0)
        remaining_days_active = remaining_days.clip(lower=30.0)
        remaining_days = remaining_days_active.where(is_active, remaining_days_churned)
        remaining_months = remaining_days / 30.44

        monthly_spend = (merged["aov"] * merged["tx_per_month"].clip(lower=0.01)).clip(lower=0)
        expected_clv = monthly_spend * remaining_months

        result_df = pd.DataFrame({
            "customer_id": merged["customer_id"].values,
            "expected_clv": expected_clv.round(2).values,
            "expected_remaining_months": remaining_months.round(1).values,
            "monthly_spend": monthly_spend.round(2).values,
        })

        # Survival curve for charting
        try:
            sf = kmf.survival_function_.reset_index()
            sf.columns = ["timeline_days", "survival_prob"]
            survival_curve: DataFrame | None = sf
        except Exception:
            survival_curve = None

        return CLVResult(
            clv_per_customer=result_df,
            median_lifetime_days=median_lifetime,
            population_survival_curve=survival_curve,
        )
