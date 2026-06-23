"""Unit tests for FeatureMatrixBuilder."""
from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering.churn_labels import LABEL_LEAKAGE_COLUMNS
from src.modeling.features import FeatureMatrix, FeatureMatrixBuilder


def _make_tx_df(n_customers: int = 60, n_months: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n_customers):
        for m in range(n_months):
            rows.append({
                "customer_id": f"c{i}",
                "transaction_date": f"2024-{m + 1:02d}-15",
                "transaction_amount": 100.0 + i,
            })
    return pd.DataFrame(rows)


class TestFeatureMatrixShape:
    def setup_method(self) -> None:
        self.builder = FeatureMatrixBuilder()

    def test_one_row_per_customer(self) -> None:
        df = _make_tx_df(n_customers=60)
        fm = self.builder.build(df)
        assert len(fm.X) == 60

    def test_X_y_customer_ids_aligned(self) -> None:
        df = _make_tx_df(n_customers=60)
        fm = self.builder.build(df)
        assert len(fm.X) == len(fm.y) == len(fm.customer_ids) == len(fm.first_tx_dates)

    def test_feature_names_match_X_columns(self) -> None:
        df = _make_tx_df(n_customers=60)
        fm = self.builder.build(df)
        assert fm.feature_names == list(fm.X.columns)

    def test_y_is_binary(self) -> None:
        df = _make_tx_df(n_customers=60)
        fm = self.builder.build(df)
        assert set(fm.y.unique()).issubset({0, 1})


class TestLeakageProtection:
    def setup_method(self) -> None:
        self.builder = FeatureMatrixBuilder()

    def test_no_leakage_columns_in_X(self) -> None:
        df = _make_tx_df(n_customers=60)
        # Inject a leakage column into the raw data (should be stripped from X)
        df["subscription_status"] = "active"
        fm = self.builder.build(df)
        for col in LABEL_LEAKAGE_COLUMNS:
            assert col not in fm.X.columns, f"Leakage column {col!r} found in X"

    def test_is_churned_not_in_X(self) -> None:
        df = _make_tx_df(n_customers=60)
        fm = self.builder.build(df)
        assert "is_churned" not in fm.X.columns

    def test_X_has_no_nan(self) -> None:
        df = _make_tx_df(n_customers=60)
        fm = self.builder.build(df)
        assert not fm.X.isna().any().any()


class TestRFMAndLabelPassthrough:
    def setup_method(self) -> None:
        self.builder = FeatureMatrixBuilder()

    def test_rfm_features_attached(self) -> None:
        df = _make_tx_df(n_customers=60)
        fm = self.builder.build(df)
        assert "customer_id" in fm.rfm_features.columns
        assert "monetary_total" in fm.rfm_features.columns

    def test_churn_label_df_attached(self) -> None:
        df = _make_tx_df(n_customers=60)
        fm = self.builder.build(df)
        assert "customer_id" in fm.churn_label_df.columns
        assert "is_churned" in fm.churn_label_df.columns

    def test_first_tx_dates_are_timestamps(self) -> None:
        df = _make_tx_df(n_customers=60)
        fm = self.builder.build(df)
        assert pd.api.types.is_datetime64_any_dtype(fm.first_tx_dates)
