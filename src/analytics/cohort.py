"""
Monthly cohort retention analysis.

A cohort is defined by the calendar month of a customer's first transaction.
Retention for period N = (customers from that cohort still active in month N)
                        / (cohort size at month 0) * 100.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.constants import MIN_CUSTOMERS_FOR_COHORT
from src.utils.exceptions import InsufficientDataError
from src.utils.types import DataFrame


@dataclass
class CohortResult:
    retention_matrix: DataFrame   # index=cohort_month str, cols="Month 0"…"Month N"
    cohort_sizes: pd.Series       # cohort_month → initial customer count
    n_cohorts: int


class CohortAnalyzer:
    """Build a monthly cohort retention matrix from transaction data."""

    def build(self, df: DataFrame) -> CohortResult:
        """
        Compute retention percentages for every cohort with at least
        MIN_CUSTOMERS_FOR_COHORT members.
        """
        df = df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        # Cohort month = month of customer's first transaction
        first_tx = (
            df.groupby("customer_id")["transaction_date"]
            .min()
            .reset_index()
            .rename(columns={"transaction_date": "first_tx_date"})
        )
        first_tx["cohort_month"] = first_tx["first_tx_date"].dt.to_period("M")

        df = df.merge(first_tx[["customer_id", "cohort_month"]], on="customer_id")
        df["tx_month"] = df["transaction_date"].dt.to_period("M")
        df["period_number"] = (df["tx_month"] - df["cohort_month"]).apply(lambda x: x.n)

        # Keep only non-negative periods (shouldn't arise but guard defensively)
        df = df[df["period_number"] >= 0]

        cohort_data = (
            df.groupby(["cohort_month", "period_number"])["customer_id"]
            .nunique()
            .reset_index(name="customers")
        )

        cohort_sizes = (
            cohort_data[cohort_data["period_number"] == 0]
            .set_index("cohort_month")["customers"]
        )

        valid_cohorts = cohort_sizes[cohort_sizes >= MIN_CUSTOMERS_FOR_COHORT].index
        if len(valid_cohorts) == 0:
            raise InsufficientDataError(
                "cohort analysis", MIN_CUSTOMERS_FOR_COHORT, int(cohort_sizes.max()) if len(cohort_sizes) > 0 else 0
            )

        cohort_data = cohort_data[cohort_data["cohort_month"].isin(valid_cohorts)]

        pivot = cohort_data.pivot_table(
            index="cohort_month",
            columns="period_number",
            values="customers",
            fill_value=0,
        )

        cohort_sizes_valid = cohort_sizes[valid_cohorts]
        retention = pivot.divide(cohort_sizes_valid, axis=0) * 100
        retention.columns = [f"Month {c}" for c in retention.columns]
        retention.index = retention.index.astype(str)
        cohort_sizes_valid.index = cohort_sizes_valid.index.astype(str)

        return CohortResult(
            retention_matrix=retention,
            cohort_sizes=cohort_sizes_valid,
            n_cohorts=len(valid_cohorts),
        )
