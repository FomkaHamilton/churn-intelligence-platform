"""
Rule-based customer segmentation.

Segments are assigned in priority order:
  1. Churned       — is_churned = True
  2. New           — tenure_days < new_threshold_days (default 90)
  3. At Risk       — churn_prob >= at_risk_threshold (default 0.60)
  4. High Value    — monetary_total >= 75th percentile of the population
  5. Healthy       — everything else

When no model probabilities are supplied, recency-based proxy scores
are used so segmentation works before any model is trained.
"""
from __future__ import annotations

import pandas as pd

from src.config.constants import (
    SEGMENT_AT_RISK,
    SEGMENT_CHURNED,
    SEGMENT_HEALTHY,
    SEGMENT_HIGH_VALUE,
    SEGMENT_NEW,
)
from src.utils.types import DataFrame


class CustomerSegmenter:
    """Assign a segment label to each customer."""

    def segment(
        self,
        rfm_features: DataFrame,
        churn_label_df: DataFrame,
        churn_probabilities: pd.Series | None = None,
        *,
        new_threshold_days: int = 90,
        at_risk_threshold: float = 0.60,
        high_value_percentile: float = 0.75,
    ) -> pd.Series:
        """
        Args:
            rfm_features: DataFrame with customer_id, tenure_days, monetary_total, recency_days.
            churn_label_df: DataFrame with customer_id, is_churned.
            churn_probabilities: Optional Series indexed by customer_id with model scores.
                                 Falls back to a recency-based proxy when None.

        Returns:
            pd.Series indexed by customer_id with segment label strings.
        """
        merged = rfm_features.merge(
            churn_label_df[["customer_id", "is_churned"]], on="customer_id"
        )

        if churn_probabilities is not None:
            prob_df = churn_probabilities.rename("churn_prob").reset_index()
            prob_df.columns = ["customer_id", "churn_prob"]
            merged = merged.merge(prob_df, on="customer_id", how="left")
            merged["churn_prob"] = merged["churn_prob"].fillna(0.5)
        else:
            # Recency proxy: normalised days-since-last-tx as rough churn signal
            max_recency = merged["recency_days"].max()
            merged["churn_prob"] = (merged["recency_days"] / max(max_recency, 1)).clip(0, 1)

        hv_threshold = float(merged["monetary_total"].quantile(high_value_percentile))

        def _label(row: pd.Series) -> str:
            if bool(row["is_churned"]):
                return SEGMENT_CHURNED
            if row["tenure_days"] < new_threshold_days:
                return SEGMENT_NEW
            if row["churn_prob"] >= at_risk_threshold:
                return SEGMENT_AT_RISK
            if row["monetary_total"] >= hv_threshold:
                return SEGMENT_HIGH_VALUE
            return SEGMENT_HEALTHY

        labels = merged.apply(_label, axis=1)
        labels.index = merged["customer_id"]
        labels.name = "segment"
        return labels
