"""
Core subscription KPI calculator.

Computes monthly time-series and point-in-time snapshots for:
  MRR        Monthly Recurring Revenue (sum of transaction_amount per month)
  Active     Unique customers who transacted in the period
  ARPU       MRR / Active subscribers
  Churn Rate % of last month's active customers absent this month
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.types import DataFrame


@dataclass
class KPISnapshot:
    """Point-in-time values for the most recent full month in the dataset."""

    mrr: float
    active_subscribers: int
    arpu: float
    monthly_churn_rate: float    # percentage, e.g. 5.2 means 5.2 %
    total_revenue: float
    avg_transaction_value: float


@dataclass
class KPITimeSeries:
    """Monthly KPI trends over the full date range of the dataset."""

    monthly_revenue: pd.Series       # Period index → float
    monthly_active: pd.Series        # Period index → int
    monthly_arpu: pd.Series          # Period index → float
    monthly_churn_rate: pd.Series    # Period index → float (%)
    snapshot: KPISnapshot


class KPICalculator:
    """Compute subscription KPIs from a clean transaction DataFrame."""

    def calculate(self, df: DataFrame) -> KPITimeSeries:
        df = df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["month"] = df["transaction_date"].dt.to_period("M")

        monthly_revenue = df.groupby("month")["transaction_amount"].sum()
        monthly_active = df.groupby("month")["customer_id"].nunique()
        monthly_arpu = monthly_revenue / monthly_active.replace(0, pd.NA)

        monthly_churn_rate = self._compute_churn_rate(df)

        latest_month = monthly_revenue.index.max()
        snapshot = KPISnapshot(
            mrr=float(monthly_revenue.get(latest_month, 0.0)),
            active_subscribers=int(monthly_active.get(latest_month, 0)),
            arpu=float(monthly_arpu.get(latest_month, 0.0) or 0.0),
            monthly_churn_rate=float(monthly_churn_rate.get(latest_month, 0.0)),
            total_revenue=float(df["transaction_amount"].sum()),
            avg_transaction_value=float(df["transaction_amount"].mean()),
        )

        return KPITimeSeries(
            monthly_revenue=monthly_revenue,
            monthly_active=monthly_active,
            monthly_arpu=monthly_arpu,
            monthly_churn_rate=monthly_churn_rate,
            snapshot=snapshot,
        )

    def _compute_churn_rate(self, df: DataFrame) -> pd.Series:
        """
        Monthly churn rate: % of prior-month active customers absent this month.

        Defined as:
            churned_M = customers active in M-1 but not in M
            rate_M    = len(churned_M) / len(active_{M-1}) * 100
        """
        months = sorted(df["month"].unique())
        rates: dict[object, float] = {}

        for i in range(1, len(months)):
            prev_month = months[i - 1]
            curr_month = months[i]

            prev_customers = set(df[df["month"] == prev_month]["customer_id"])
            curr_customers = set(df[df["month"] == curr_month]["customer_id"])

            if not prev_customers:
                rates[curr_month] = 0.0
                continue

            churned = prev_customers - curr_customers
            rates[curr_month] = len(churned) / len(prev_customers) * 100

        return pd.Series(rates)
