"""Unit tests for the RFM feature builder."""
from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.rfm import RFMBuilder
from src.utils.exceptions import InsufficientDataError

_N_FILLERS = 47  # brings 3-customer core fixtures up to 50 total


def _make_df() -> pd.DataFrame:
    """50 customers: c1/c2/c3 with real data + 47 single-tx fillers."""
    rows = [
        {"customer_id": "c1", "transaction_date": "2024-01-01", "transaction_amount": 100.0},
        {"customer_id": "c1", "transaction_date": "2024-02-01", "transaction_amount": 150.0},
        {"customer_id": "c2", "transaction_date": "2024-01-15", "transaction_amount": 200.0},
        {"customer_id": "c2", "transaction_date": "2024-03-15", "transaction_amount": 200.0},
        {"customer_id": "c3", "transaction_date": "2024-02-10", "transaction_amount": 300.0},
    ]
    for i in range(_N_FILLERS):
        rows.append({"customer_id": f"filler_{i}",
                     "transaction_date": "2024-01-01",
                     "transaction_amount": 50.0})
    return pd.DataFrame(rows)


def _make_df_with_c1(c1_dates: list[str], c1_amounts: list[float]) -> pd.DataFrame:
    """50+ customer df where c1 has specific transaction history."""
    rows = [{"customer_id": "c1",
             "transaction_date": d,
             "transaction_amount": a}
            for d, a in zip(c1_dates, c1_amounts)]
    for i in range(50):
        rows.append({"customer_id": f"filler_{i}",
                     "transaction_date": "2024-01-01",
                     "transaction_amount": 50.0})
    return pd.DataFrame(rows)


class TestRFMBasics:
    def setup_method(self) -> None:
        self.builder = RFMBuilder()

    def test_returns_one_row_per_customer(self) -> None:
        df = _make_df()
        result = self.builder.build(df)
        assert len(result.features) == df["customer_id"].nunique()

    def test_customer_ids_present(self) -> None:
        df = _make_df()
        result = self.builder.build(df)
        assert set(result.features["customer_id"]) == set(df["customer_id"])

    def test_all_feature_cols_present(self) -> None:
        df = _make_df()
        result = self.builder.build(df)
        for col in RFMBuilder.FEATURE_COLS:
            assert col in result.features.columns, f"Missing column: {col}"

    def test_n_customers_correct(self) -> None:
        df = _make_df()
        result = self.builder.build(df)
        assert result.n_customers == 50


class TestRecency:
    def setup_method(self) -> None:
        self.builder = RFMBuilder()

    def test_recency_is_days_from_reference_to_last_tx(self) -> None:
        df = _make_df_with_c1(
            c1_dates=["2024-01-01", "2024-02-20"],
            c1_amounts=[100.0, 100.0],
        )
        ref = pd.Timestamp("2024-04-01")
        result = self.builder.build(df, reference_date=ref)
        c1 = result.features[result.features["customer_id"] == "c1"].iloc[0]
        expected = (ref - pd.Timestamp("2024-02-20")).days
        assert c1["recency_days"] == expected

    def test_reference_date_defaults_to_max_transaction_date(self) -> None:
        df = _make_df()
        result = self.builder.build(df)
        assert result.reference_date == pd.Timestamp("2024-03-15")

    def test_reference_date_override_respected(self) -> None:
        df = _make_df()
        ref = pd.Timestamp("2025-01-01")
        result = self.builder.build(df, reference_date=ref)
        assert result.reference_date == ref


class TestFrequencyAndMonetary:
    def setup_method(self) -> None:
        self.builder = RFMBuilder()

    def test_frequency_counts_transactions(self) -> None:
        df = _make_df()
        result = self.builder.build(df)
        row = result.features[result.features["customer_id"] == "c1"].iloc[0]
        assert row["frequency"] == 2

    def test_monetary_total_sums_amounts(self) -> None:
        df = _make_df()
        result = self.builder.build(df)
        row = result.features[result.features["customer_id"] == "c1"].iloc[0]
        assert row["monetary_total"] == pytest.approx(250.0)

    def test_aov_equals_total_divided_by_frequency(self) -> None:
        df = _make_df()
        result = self.builder.build(df)
        feat = result.features.copy()
        feat["expected_aov"] = feat["monetary_total"] / feat["frequency"]
        assert (feat["aov"] - feat["expected_aov"]).abs().max() < 1e-6


class TestTenure:
    def setup_method(self) -> None:
        self.builder = RFMBuilder()

    def test_tenure_is_last_minus_first_tx(self) -> None:
        df = _make_df()
        result = self.builder.build(df)
        # c1: 2024-02-01 - 2024-01-01 = 31 days
        row = result.features[result.features["customer_id"] == "c1"].iloc[0]
        assert row["tenure_days"] == 31

    def test_single_tx_customer_has_positive_tenure(self) -> None:
        # tenure_days is clipped to ≥1 to avoid div/0 in tx_per_month
        many_customers = [f"c{i}" for i in range(50)]
        df = pd.DataFrame({
            "customer_id": many_customers,
            "transaction_date": ["2024-01-01"] * 50,
            "transaction_amount": [100.0] * 50,
        })
        result = self.builder.build(df)
        assert (result.features["tenure_days"] >= 1).all()


class TestGapStd:
    def setup_method(self) -> None:
        self.builder = RFMBuilder()

    def test_single_tx_gap_std_is_zero(self) -> None:
        many_customers = [f"c{i}" for i in range(50)]
        df = pd.DataFrame({
            "customer_id": many_customers,
            "transaction_date": ["2024-01-01"] * 50,
            "transaction_amount": [100.0] * 50,
        })
        result = self.builder.build(df)
        assert (result.features["gap_std_days"] == 0.0).all()

    def test_regular_payer_has_low_gap_std(self) -> None:
        cids = [f"c{i}" for i in range(50)]
        rows = []
        for cid in cids:
            for month in range(6):
                rows.append({"customer_id": cid,
                              "transaction_date": f"2024-{month + 1:02d}-01",
                              "transaction_amount": 100.0})
        df = pd.DataFrame(rows)
        result = self.builder.build(df)
        # Calendar gap variation between months is small (28–31 days)
        assert result.features["gap_std_days"].max() < 5.0


class TestInsufficientData:
    def setup_method(self) -> None:
        self.builder = RFMBuilder()

    def test_empty_dataframe_raises(self) -> None:
        df = pd.DataFrame(columns=["customer_id", "transaction_date", "transaction_amount"])
        with pytest.raises(InsufficientDataError):
            self.builder.build(df)

    def test_too_few_customers_raises(self) -> None:
        df = pd.DataFrame({
            "customer_id": ["c1", "c1"],
            "transaction_date": ["2024-01-01", "2024-02-01"],
            "transaction_amount": [100.0, 100.0],
        })
        with pytest.raises(InsufficientDataError):
            self.builder.build(df)
