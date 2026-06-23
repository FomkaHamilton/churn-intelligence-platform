"""
Feature matrix builder.

Joins RFM features with churn labels to produce an (X, y) pair
ready for model training.  The leakage guard is applied here so
no caller can accidentally pass a leaky matrix to the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.feature_engineering.churn_labels import ChurnLabelBuilder
from src.feature_engineering.rfm import RFMBuilder
from src.utils.types import DataFrame


@dataclass
class FeatureMatrix:
    X: DataFrame              # features only — no label, no customer IDs
    y: pd.Series              # binary churn labels (int 0/1)
    customer_ids: pd.Series   # customer IDs aligned with X rows
    first_tx_dates: pd.Series # first transaction date per row (for temporal split)
    feature_names: list[str]  # X column names in order
    rfm_features: DataFrame   # full RFM frame with customer_id (for CLV / segmentation)
    churn_label_df: DataFrame # churn label frame with customer_id (for segmentation)


class FeatureMatrixBuilder:
    """Build a labelled feature matrix from a clean transaction DataFrame."""

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
        churn_window_days: int = 90,
        reference_date: pd.Timestamp | None = None,
    ) -> FeatureMatrix:
        """
        Build features + labels from a transaction DataFrame.

        reference_date defaults to max(transaction_date) in RFMBuilder and
        ChurnLabelBuilder so both use the same snapshot.
        """
        df_copy = df.copy()
        df_copy["transaction_date"] = pd.to_datetime(df_copy["transaction_date"])

        rfm_result = RFMBuilder().build(df_copy, reference_date=reference_date)
        label_result = ChurnLabelBuilder().build(
            df_copy, churn_window_days=churn_window_days, reference_date=reference_date
        )

        # First transaction date per customer (needed for temporal split)
        first_tx = (
            df_copy.groupby("customer_id")["transaction_date"]
            .min()
            .rename("first_tx_date")
            .reset_index()
        )

        merged = (
            rfm_result.features
            .merge(label_result.labels[["customer_id", "is_churned"]], on="customer_id")
            .merge(first_tx, on="customer_id")
        )

        feature_cols = [c for c in self.FEATURE_COLS if c in merged.columns]
        X = merged[feature_cols].fillna(0.0).reset_index(drop=True)

        # Enforce leakage guard — will raise if any banned column sneaks in
        ChurnLabelBuilder().assert_no_leakage(X)

        return FeatureMatrix(
            X=X,
            y=merged["is_churned"].astype(int).reset_index(drop=True),
            customer_ids=merged["customer_id"].reset_index(drop=True),
            first_tx_dates=merged["first_tx_date"].reset_index(drop=True),
            feature_names=feature_cols,
            rfm_features=rfm_result.features,
            churn_label_df=label_result.labels,
        )
