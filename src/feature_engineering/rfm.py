"""
RFM (Recency, Frequency, Monetary) feature builder.

Computes per-customer behavioural features from raw transaction history.
reference_date defaults to max(transaction_date) to prevent using future
data that would not be available at prediction time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.constants import MIN_CUSTOMERS_FOR_MODELING
from src.utils.exceptions import InsufficientDataError
from src.utils.types import DataFrame


@dataclass
class RFMResult:
    features: DataFrame          # one row per customer
    reference_date: pd.Timestamp
    n_customers: int


class RFMBuilder:
    """Build per-customer RFM features from a clean transaction DataFrame."""

    FEATURE_COLS: tuple[str, ...] = (
        "recency_days",
        "frequency",
        "monetary_total",
        "aov",
        "tenure_days",
        "tx_per_month",
        "gap_std_days",
    )

    def build(
        self,
        df: DataFrame,
        *,
        reference_date: pd.Timestamp | None = None,
    ) -> RFMResult:
        """
        Compute RFM features.

        Args:
            df: Clean transactions with customer_id, transaction_date, transaction_amount.
            reference_date: Snapshot date for recency. Defaults to max(transaction_date)
                            so no future data leaks into training features.
        """
        if len(df) == 0:
            raise InsufficientDataError("RFM", MIN_CUSTOMERS_FOR_MODELING, 0)

        df = df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        if reference_date is None:
            reference_date = df["transaction_date"].max()

        n_unique = df["customer_id"].nunique()
        if n_unique < MIN_CUSTOMERS_FOR_MODELING:
            raise InsufficientDataError("RFM", MIN_CUSTOMERS_FOR_MODELING, n_unique)

        agg = (
            df.groupby("customer_id")
            .agg(
                last_tx=("transaction_date", "max"),
                first_tx=("transaction_date", "min"),
                frequency=("transaction_date", "count"),
                monetary_total=("transaction_amount", "sum"),
            )
            .reset_index()
        )

        agg["recency_days"] = (reference_date - agg["last_tx"]).dt.days
        agg["tenure_days"] = (agg["last_tx"] - agg["first_tx"]).dt.days.clip(lower=1)
        agg["aov"] = agg["monetary_total"] / agg["frequency"]
        # Normalise tx count to monthly rate; clip tenure at 1 day to avoid div/0
        agg["tx_per_month"] = agg["frequency"] / (agg["tenure_days"] / 30.44).clip(lower=1)

        gap_std = self._compute_gap_std(df)
        agg = agg.merge(gap_std, on="customer_id", how="left")
        agg["gap_std_days"] = agg["gap_std_days"].fillna(0.0)

        features = agg[["customer_id", *self.FEATURE_COLS]].copy()

        return RFMResult(
            features=features,
            reference_date=reference_date,
            n_customers=n_unique,
        )

    def _compute_gap_std(self, df: DataFrame) -> DataFrame:
        """Compute standard deviation of inter-transaction gap (days) per customer."""
        sorted_df = df.sort_values(["customer_id", "transaction_date"])
        sorted_df = sorted_df.copy()
        sorted_df["gap_days"] = (
            sorted_df.groupby("customer_id")["transaction_date"].diff().dt.days
        )
        gap_std = (
            sorted_df.groupby("customer_id")["gap_days"]
            .std()
            .fillna(0.0)
            .reset_index()
            .rename(columns={"gap_days": "gap_std_days"})
        )
        return gap_std
