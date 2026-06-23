"""Unit tests for TimeSeriesChurnSplit."""
from __future__ import annotations

import pandas as pd
import pytest

from src.modeling.validation import TimeSeriesChurnSplit
from src.utils.exceptions import InsufficientDataError, TemporalLeakageError


def _make_split_inputs(n_customers: int = 100, date_range_days: int = 365):
    """Create aligned X, y, customer_ids, first_tx_dates for n_customers."""
    dates = pd.date_range("2023-01-01", periods=n_customers, freq="3D")
    X = pd.DataFrame({"recency_days": range(n_customers), "frequency": [3] * n_customers})
    y = pd.Series([i % 2 for i in range(n_customers)])
    customer_ids = pd.Series([f"c{i}" for i in range(n_customers)])
    first_tx_dates = pd.Series(dates)
    return X, y, customer_ids, first_tx_dates


class TestSplitRatios:
    def setup_method(self) -> None:
        self.splitter = TimeSeriesChurnSplit()

    def test_default_80_20_split(self) -> None:
        X, y, ids, dates = _make_split_inputs(100)
        result = self.splitter.split(X, y, ids, dates)
        total = len(result.X_train) + len(result.X_test)
        assert total == 100
        assert len(result.X_train) == pytest.approx(80, abs=2)
        assert len(result.X_test) == pytest.approx(20, abs=2)

    def test_custom_ratio_respected(self) -> None:
        X, y, ids, dates = _make_split_inputs(100)
        result = self.splitter.split(X, y, ids, dates, train_ratio=0.70)
        assert len(result.X_train) == pytest.approx(70, abs=2)

    def test_aligned_lengths(self) -> None:
        X, y, ids, dates = _make_split_inputs(100)
        result = self.splitter.split(X, y, ids, dates)
        assert len(result.X_train) == len(result.y_train) == len(result.customer_ids_train)
        assert len(result.X_test) == len(result.y_test) == len(result.customer_ids_test)


class TestTemporalOrdering:
    def setup_method(self) -> None:
        self.splitter = TimeSeriesChurnSplit()

    def test_test_customers_joined_after_cutoff(self) -> None:
        X, y, ids, dates = _make_split_inputs(100)
        result = self.splitter.split(X, y, ids, dates)
        # Reconstruct test first_tx_dates from test customer_ids
        id_to_date = dict(zip(ids, dates))
        for cid in result.customer_ids_test:
            assert id_to_date[cid] > result.cutoff_date

    def test_train_customers_joined_on_or_before_cutoff(self) -> None:
        X, y, ids, dates = _make_split_inputs(100)
        result = self.splitter.split(X, y, ids, dates)
        id_to_date = dict(zip(ids, dates))
        for cid in result.customer_ids_train:
            assert id_to_date[cid] <= result.cutoff_date

    def test_cutoff_date_stored_in_result(self) -> None:
        X, y, ids, dates = _make_split_inputs(100)
        result = self.splitter.split(X, y, ids, dates)
        assert isinstance(result.cutoff_date, pd.Timestamp)

    def test_no_customer_in_both_sets(self) -> None:
        X, y, ids, dates = _make_split_inputs(100)
        result = self.splitter.split(X, y, ids, dates)
        train_ids = set(result.customer_ids_train)
        test_ids = set(result.customer_ids_test)
        assert train_ids.isdisjoint(test_ids)


class TestEdgeCases:
    def setup_method(self) -> None:
        self.splitter = TimeSeriesChurnSplit()

    def test_two_customers_produces_one_in_each_set(self) -> None:
        X = pd.DataFrame({"f": [1, 2]})
        y = pd.Series([0, 1])
        ids = pd.Series(["c0", "c1"])
        dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-01"]))
        result = self.splitter.split(X, y, ids, dates)
        assert len(result.X_train) >= 1
        assert len(result.X_test) >= 1

    def test_single_customer_raises(self) -> None:
        X = pd.DataFrame({"f": [1]})
        y = pd.Series([0])
        ids = pd.Series(["c0"])
        dates = pd.Series(pd.to_datetime(["2024-01-01"]))
        with pytest.raises(InsufficientDataError):
            self.splitter.split(X, y, ids, dates)
