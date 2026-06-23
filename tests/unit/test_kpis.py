"""Unit tests for the KPI calculator."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.kpis import KPICalculator


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _two_month_df() -> pd.DataFrame:
    """10 customers in Jan, same 10 in Feb. No churn."""
    rows = []
    for i in range(10):
        rows.append({"customer_id": f"c{i}", "transaction_date": "2024-01-15",
                     "transaction_amount": 100.0})
        rows.append({"customer_id": f"c{i}", "transaction_date": "2024-02-15",
                     "transaction_amount": 120.0})
    return pd.DataFrame(rows)


class TestMonthlyRevenue:
    def setup_method(self) -> None:
        self.calc = KPICalculator()

    def test_revenue_sums_correctly_per_month(self) -> None:
        df = _two_month_df()
        result = self.calc.calculate(df)
        months = [str(m) for m in result.monthly_revenue.index]
        assert "2024-01" in months
        assert "2024-02" in months
        assert result.monthly_revenue.iloc[0] == pytest.approx(1000.0)   # 10 × 100
        assert result.monthly_revenue.iloc[1] == pytest.approx(1200.0)   # 10 × 120

    def test_total_revenue_in_snapshot(self) -> None:
        df = _two_month_df()
        result = self.calc.calculate(df)
        assert result.snapshot.total_revenue == pytest.approx(2200.0)


class TestActiveSubscribers:
    def setup_method(self) -> None:
        self.calc = KPICalculator()

    def test_active_counts_unique_customers_per_month(self) -> None:
        df = _two_month_df()
        result = self.calc.calculate(df)
        assert result.monthly_active.iloc[0] == 10
        assert result.monthly_active.iloc[1] == 10

    def test_active_subscriber_snapshot_is_latest_month(self) -> None:
        df = _two_month_df()
        result = self.calc.calculate(df)
        assert result.snapshot.active_subscribers == 10


class TestARPU:
    def setup_method(self) -> None:
        self.calc = KPICalculator()

    def test_arpu_equals_revenue_divided_by_active(self) -> None:
        df = _two_month_df()
        result = self.calc.calculate(df)
        assert result.monthly_arpu.iloc[0] == pytest.approx(100.0)
        assert result.monthly_arpu.iloc[1] == pytest.approx(120.0)

    def test_arpu_snapshot_matches_latest_month(self) -> None:
        df = _two_month_df()
        result = self.calc.calculate(df)
        assert result.snapshot.arpu == pytest.approx(120.0)


class TestChurnRate:
    def setup_method(self) -> None:
        self.calc = KPICalculator()

    def test_zero_churn_when_all_customers_return(self) -> None:
        df = _two_month_df()
        result = self.calc.calculate(df)
        # All 10 Jan customers also appear in Feb → 0 % churn
        feb = pd.Period("2024-02", "M")
        assert result.monthly_churn_rate.get(feb, None) == pytest.approx(0.0)

    def test_full_churn_when_no_customers_return(self) -> None:
        rows = [
            {"customer_id": "c1", "transaction_date": "2024-01-15",
             "transaction_amount": 100.0},
            {"customer_id": "c2", "transaction_date": "2024-02-15",
             "transaction_amount": 100.0},
        ]
        df = pd.DataFrame(rows)
        result = self.calc.calculate(df)
        feb = pd.Period("2024-02", "M")
        # c1 was in Jan but not Feb → 100 % churn
        assert result.monthly_churn_rate.get(feb, None) == pytest.approx(100.0)

    def test_partial_churn_rate(self) -> None:
        rows = []
        for i in range(4):
            rows.append({"customer_id": f"c{i}", "transaction_date": "2024-01-15",
                         "transaction_amount": 100.0})
        # Only c0 and c1 return in Feb → 50 % churn
        for i in range(2):
            rows.append({"customer_id": f"c{i}", "transaction_date": "2024-02-15",
                         "transaction_amount": 100.0})
        df = pd.DataFrame(rows)
        result = self.calc.calculate(df)
        feb = pd.Period("2024-02", "M")
        assert result.monthly_churn_rate.get(feb, None) == pytest.approx(50.0)

    def test_single_month_has_no_churn_rate(self) -> None:
        df = pd.DataFrame({
            "customer_id": ["c1", "c2"],
            "transaction_date": ["2024-01-01", "2024-01-15"],
            "transaction_amount": [100.0, 100.0],
        })
        result = self.calc.calculate(df)
        assert len(result.monthly_churn_rate) == 0


class TestSnapshot:
    def setup_method(self) -> None:
        self.calc = KPICalculator()

    def test_snapshot_reflects_latest_month(self) -> None:
        df = _two_month_df()
        result = self.calc.calculate(df)
        # Latest month is Feb; MRR = 1200
        assert result.snapshot.mrr == pytest.approx(1200.0)

    def test_snapshot_avg_transaction_value(self) -> None:
        df = _two_month_df()
        result = self.calc.calculate(df)
        # 10×100 + 10×120 = 2200 / 20 = 110.0
        assert result.snapshot.avg_transaction_value == pytest.approx(110.0)
