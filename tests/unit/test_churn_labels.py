"""Unit tests for the churn label builder."""
from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.churn_labels import (
    LABEL_LEAKAGE_COLUMNS,
    ChurnLabelBuilder,
)
from src.utils.exceptions import TemporalLeakageError


def _make_df(
    customer_ids: list[str],
    last_dates: list[str],
    amounts: list[float] | None = None,
) -> pd.DataFrame:
    if amounts is None:
        amounts = [100.0] * len(customer_ids)
    return pd.DataFrame({
        "customer_id": customer_ids,
        "transaction_date": last_dates,
        "transaction_amount": amounts,
    })


class TestActiveVsChurned:
    def setup_method(self) -> None:
        self.builder = ChurnLabelBuilder()
        # reference_date = "2024-04-01" (passed explicitly)
        self.ref = pd.Timestamp("2024-04-01")

    def test_recent_customer_is_active(self) -> None:
        df = _make_df(["c1"], ["2024-03-25"])   # 7 days before ref
        result = self.builder.build(df, churn_window_days=90, reference_date=self.ref)
        row = result.labels[result.labels["customer_id"] == "c1"].iloc[0]
        assert row["is_churned"] is False or row["is_churned"] == 0

    def test_stale_customer_is_churned(self) -> None:
        df = _make_df(["c1"], ["2023-12-01"])   # 122 days before ref
        result = self.builder.build(df, churn_window_days=90, reference_date=self.ref)
        row = result.labels[result.labels["customer_id"] == "c1"].iloc[0]
        assert row["is_churned"] is True or row["is_churned"] == 1

    def test_exactly_at_boundary_is_active(self) -> None:
        # Exactly churn_window_days ago should NOT be churned (> not >=)
        df = _make_df(["c1"], ["2024-01-01"])   # 91 days before ref
        result = self.builder.build(df, churn_window_days=91, reference_date=self.ref)
        row = result.labels[result.labels["customer_id"] == "c1"].iloc[0]
        assert row["is_churned"] is False or row["is_churned"] == 0

    def test_one_day_over_boundary_is_churned(self) -> None:
        df = _make_df(["c1"], ["2023-12-31"])   # 92 days before ref
        result = self.builder.build(df, churn_window_days=91, reference_date=self.ref)
        row = result.labels[result.labels["customer_id"] == "c1"].iloc[0]
        assert row["is_churned"] is True or row["is_churned"] == 1


class TestChurnRate:
    def setup_method(self) -> None:
        self.builder = ChurnLabelBuilder()

    def test_churn_rate_zero_when_all_active(self) -> None:
        ref = pd.Timestamp("2024-04-01")
        df = _make_df(["c1", "c2", "c3"], ["2024-03-28", "2024-03-29", "2024-03-30"])
        result = self.builder.build(df, churn_window_days=90, reference_date=ref)
        assert result.churn_rate == pytest.approx(0.0)

    def test_churn_rate_one_when_all_churned(self) -> None:
        ref = pd.Timestamp("2024-04-01")
        df = _make_df(["c1", "c2"], ["2023-01-01", "2023-01-01"])
        result = self.builder.build(df, churn_window_days=90, reference_date=ref)
        assert result.churn_rate == pytest.approx(1.0)

    def test_churn_rate_partial(self) -> None:
        ref = pd.Timestamp("2024-04-01")
        df = _make_df(
            ["c1", "c2", "c3", "c4"],
            ["2024-03-28", "2024-03-29", "2023-01-01", "2023-01-01"],
        )
        result = self.builder.build(df, churn_window_days=90, reference_date=ref)
        assert result.churn_rate == pytest.approx(0.5)
        assert result.n_churned == 2
        assert result.n_active == 2

    def test_uses_most_recent_transaction_per_customer(self) -> None:
        # c1 has an old tx and a recent tx — should be active
        ref = pd.Timestamp("2024-04-01")
        df = pd.DataFrame({
            "customer_id": ["c1", "c1", "c2"],
            "transaction_date": ["2023-01-01", "2024-03-28", "2024-03-28"],
            "transaction_amount": [100.0, 100.0, 100.0],
        })
        result = self.builder.build(df, churn_window_days=90, reference_date=ref)
        row = result.labels[result.labels["customer_id"] == "c1"].iloc[0]
        assert row["is_churned"] is False or row["is_churned"] == 0


class TestReferenceDate:
    def setup_method(self) -> None:
        self.builder = ChurnLabelBuilder()

    def test_reference_date_defaults_to_max_transaction_date(self) -> None:
        df = pd.DataFrame({
            "customer_id": ["c1", "c2"],
            "transaction_date": ["2024-01-01", "2024-06-15"],
            "transaction_amount": [100.0, 100.0],
        })
        result = self.builder.build(df, churn_window_days=90)
        assert result.reference_date == pd.Timestamp("2024-06-15")


class TestLeakageGuard:
    def setup_method(self) -> None:
        self.builder = ChurnLabelBuilder()

    def test_assert_no_leakage_raises_for_subscription_status(self) -> None:
        df_features = pd.DataFrame({
            "recency_days": [10],
            "frequency": [3],
            "subscription_status": ["active"],  # leaky column
        })
        with pytest.raises(TemporalLeakageError):
            self.builder.assert_no_leakage(df_features)

    def test_assert_no_leakage_raises_for_any_leakage_column(self) -> None:
        for col in LABEL_LEAKAGE_COLUMNS:
            df = pd.DataFrame({col: ["x"], "frequency": [1]})
            with pytest.raises(TemporalLeakageError):
                self.builder.assert_no_leakage(df)

    def test_assert_no_leakage_passes_for_clean_features(self) -> None:
        df_clean = pd.DataFrame({
            "recency_days": [10],
            "frequency": [3],
            "monetary_total": [500.0],
        })
        # Should not raise
        self.builder.assert_no_leakage(df_clean)

    def test_label_leakage_columns_includes_subscription_status(self) -> None:
        assert "subscription_status" in LABEL_LEAKAGE_COLUMNS
