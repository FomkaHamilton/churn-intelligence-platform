"""
Churn label builder.

Labels are derived EXCLUSIVELY from transaction date gaps.
subscription_status and similar columns are NEVER used as signals —
they directly encode the label and would cause trivially good training
metrics that collapse to random at deployment.

LABEL_LEAKAGE_COLUMNS is an explicit blocklist enforced at feature
matrix construction time via assert_no_leakage().
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.exceptions import TemporalLeakageError
from src.utils.types import DataFrame

# Columns that must NEVER appear in the ML feature matrix.
LABEL_LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "subscription_status",
        "is_active",
        "status",
        "churned",
        "churn_flag",
        "active",
    }
)


@dataclass
class ChurnLabelResult:
    labels: DataFrame          # customer_id | is_churned | last_tx_date | days_since_last_tx
    reference_date: pd.Timestamp
    churn_window_days: int
    n_churned: int
    n_active: int
    churn_rate: float          # 0.0–1.0


class ChurnLabelBuilder:
    """
    Construct binary churn labels from transaction history.

    Churn is defined as: no transaction within the last `churn_window_days`
    relative to `reference_date`.  A customer who transacted yesterday is
    active; one who last transacted 200 days ago (window=90) is churned.
    """

    def build(
        self,
        df: DataFrame,
        *,
        churn_window_days: int = 90,
        reference_date: pd.Timestamp | None = None,
    ) -> ChurnLabelResult:
        """
        Label customers as churned or active.

        reference_date defaults to max(transaction_date) so the model
        never sees information from after the latest event in the dataset.
        """
        df = df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        if reference_date is None:
            reference_date = df["transaction_date"].max()

        last_tx = (
            df.groupby("customer_id")["transaction_date"]
            .max()
            .reset_index()
            .rename(columns={"transaction_date": "last_tx_date"})
        )

        last_tx["days_since_last_tx"] = (reference_date - last_tx["last_tx_date"]).dt.days
        last_tx["is_churned"] = last_tx["days_since_last_tx"] > churn_window_days

        n_churned = int(last_tx["is_churned"].sum())
        n_active = int((~last_tx["is_churned"]).sum())
        n_total = n_churned + n_active

        return ChurnLabelResult(
            labels=last_tx[["customer_id", "is_churned", "last_tx_date", "days_since_last_tx"]],
            reference_date=reference_date,
            churn_window_days=churn_window_days,
            n_churned=n_churned,
            n_active=n_active,
            churn_rate=n_churned / n_total if n_total > 0 else 0.0,
        )

    def assert_no_leakage(self, feature_df: DataFrame) -> None:
        """
        Raise TemporalLeakageError if any LABEL_LEAKAGE_COLUMNS appear in feature_df.

        Call this immediately before any ML training step to enforce the invariant
        that the feature matrix contains no columns derived from the label.
        """
        leaky = LABEL_LEAKAGE_COLUMNS & set(feature_df.columns)
        if leaky:
            raise TemporalLeakageError(
                f"Label leakage detected: {sorted(leaky)} must never appear in the "
                "feature matrix. These columns directly encode churn state and would "
                "produce artificially high training metrics that collapse in production."
            )
