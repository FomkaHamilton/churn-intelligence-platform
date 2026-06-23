"""Unit tests for the cohort retention analyzer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.cohort import CohortAnalyzer
from src.utils.exceptions import InsufficientDataError


def _regular_dataset(n_customers: int = 50, n_months: int = 6) -> pd.DataFrame:
    """All customers join in month 1 and transact every subsequent month."""
    rows = []
    for i in range(n_customers):
        for m in range(n_months):
            rows.append({
                "customer_id": f"c{i}",
                "transaction_date": f"2024-{m + 1:02d}-15",
                "transaction_amount": 100.0,
            })
    return pd.DataFrame(rows)


class TestRetentionMatrix:
    def setup_method(self) -> None:
        self.analyzer = CohortAnalyzer()

    def test_month_zero_retention_is_100_percent(self) -> None:
        df = _regular_dataset(n_customers=50, n_months=4)
        result = self.analyzer.build(df)
        assert "Month 0" in result.retention_matrix.columns
        vals = result.retention_matrix["Month 0"].values.astype(float)
        assert (np.abs(vals - 100.0) < 1e-6).all()

    def test_full_retention_gives_100_across_all_periods(self) -> None:
        df = _regular_dataset(n_customers=50, n_months=3)
        result = self.analyzer.build(df)
        for col in result.retention_matrix.columns:
            vals = result.retention_matrix[col].values.astype(float)
            assert (np.abs(vals - 100.0) < 1.0).all(), f"Column {col} not ~100%"

    def test_partial_churn_reduces_retention(self) -> None:
        # 50 customers join in Jan, only 25 return in Feb
        rows = []
        for i in range(50):
            rows.append({"customer_id": f"c{i}", "transaction_date": "2024-01-10",
                         "transaction_amount": 100.0})
        for i in range(25):
            rows.append({"customer_id": f"c{i}", "transaction_date": "2024-02-10",
                         "transaction_amount": 100.0})
        df = pd.DataFrame(rows)
        result = self.analyzer.build(df)
        cohort_row = result.retention_matrix.loc["2024-01"]
        assert float(cohort_row["Month 1"]) == pytest.approx(50.0)

    def test_period_numbers_are_sequential(self) -> None:
        df = _regular_dataset(n_customers=50, n_months=5)
        result = self.analyzer.build(df)
        cols = result.retention_matrix.columns.tolist()
        expected = [f"Month {i}" for i in range(len(cols))]
        assert cols == expected


class TestCohortSizes:
    def setup_method(self) -> None:
        self.analyzer = CohortAnalyzer()

    def test_cohort_sizes_match_initial_customer_count(self) -> None:
        df = _regular_dataset(n_customers=50, n_months=3)
        result = self.analyzer.build(df)
        assert (result.cohort_sizes.values == 50).all()

    def test_n_cohorts_correct(self) -> None:
        # 50 customers per month, 3 distinct cohort months
        rows = []
        for m in range(3):
            for i in range(50):
                rows.append({
                    "customer_id": f"m{m}_c{i}",
                    "transaction_date": f"2024-{m + 1:02d}-01",
                    "transaction_amount": 100.0,
                })
        df = pd.DataFrame(rows)
        result = self.analyzer.build(df)
        assert result.n_cohorts == 3


class TestSmallCohortFiltering:
    def setup_method(self) -> None:
        self.analyzer = CohortAnalyzer()

    def test_tiny_cohort_excluded_from_results(self) -> None:
        # 1 customer in Jan, 50 in Feb
        rows = [{"customer_id": "lone", "transaction_date": "2024-01-15",
                 "transaction_amount": 100.0}]
        for i in range(50):
            rows.append({"customer_id": f"c{i}", "transaction_date": "2024-02-15",
                         "transaction_amount": 100.0})
        df = pd.DataFrame(rows)
        result = self.analyzer.build(df)
        assert "2024-01" not in result.retention_matrix.index

    def test_all_cohorts_too_small_raises(self) -> None:
        df = pd.DataFrame({
            "customer_id": ["c1", "c2"],
            "transaction_date": ["2024-01-01", "2024-02-01"],
            "transaction_amount": [100.0, 100.0],
        })
        with pytest.raises(InsufficientDataError):
            self.analyzer.build(df)


class TestIndexFormat:
    def setup_method(self) -> None:
        self.analyzer = CohortAnalyzer()

    def test_retention_index_is_string(self) -> None:
        df = _regular_dataset(n_customers=50, n_months=2)
        result = self.analyzer.build(df)
        for idx in result.retention_matrix.index:
            assert isinstance(idx, str)

    def test_cohort_sizes_index_is_string(self) -> None:
        df = _regular_dataset(n_customers=50, n_months=2)
        result = self.analyzer.build(df)
        for idx in result.cohort_sizes.index:
            assert isinstance(idx, str)
