"""Unit tests for schema enforcement and column normalisation."""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.schema import SchemaValidator
from src.utils.exceptions import SchemaError


def _df(**kwargs: list) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


class TestColumnNormalisation:
    def setup_method(self) -> None:
        self.validator = SchemaValidator()

    def test_canonical_columns_pass_unchanged(self) -> None:
        df = _df(
            customer_id=["C1"],
            transaction_date=["2024-01-01"],
            transaction_amount=["99.00"],
        )
        result = self.validator.validate(df)
        assert "customer_id" in result.normalised_df.columns

    def test_alias_revenue_mapped_to_transaction_amount(self) -> None:
        df = _df(
            customer_id=["C1"],
            transaction_date=["2024-01-01"],
            revenue=["99.00"],
        )
        result = self.validator.validate(df)
        assert "transaction_amount" in result.normalised_df.columns
        assert "revenue" not in result.normalised_df.columns

    def test_alias_user_id_mapped_to_customer_id(self) -> None:
        df = _df(
            user_id=["C1"],
            transaction_date=["2024-01-01"],
            transaction_amount=["99.00"],
        )
        result = self.validator.validate(df)
        assert "customer_id" in result.normalised_df.columns

    def test_alias_order_date_mapped_to_transaction_date(self) -> None:
        df = _df(
            customer_id=["C1"],
            order_date=["2024-01-01"],
            transaction_amount=["99.00"],
        )
        result = self.validator.validate(df)
        assert "transaction_date" in result.normalised_df.columns

    def test_mapped_columns_reported(self) -> None:
        df = _df(
            customer_id=["C1"],
            transaction_date=["2024-01-01"],
            revenue=["99.00"],
        )
        result = self.validator.validate(df)
        assert "revenue" in result.mapped_columns

    def test_extra_columns_kept_and_reported(self) -> None:
        df = _df(
            customer_id=["C1"],
            transaction_date=["2024-01-01"],
            transaction_amount=["99.00"],
            internal_ref=["XYZ"],
        )
        result = self.validator.validate(df)
        assert "internal_ref" in result.normalised_df.columns
        assert "internal_ref" in result.unrecognised_columns


class TestSchemaValidationErrors:
    def setup_method(self) -> None:
        self.validator = SchemaValidator()

    def test_missing_customer_id_raises(self) -> None:
        df = _df(
            transaction_date=["2024-01-01"],
            transaction_amount=["99.00"],
        )
        with pytest.raises(SchemaError, match="customer_id"):
            self.validator.validate(df)

    def test_missing_transaction_date_raises(self) -> None:
        df = _df(
            customer_id=["C1"],
            transaction_amount=["99.00"],
        )
        with pytest.raises(SchemaError):
            self.validator.validate(df)

    def test_missing_transaction_amount_raises(self) -> None:
        df = _df(
            customer_id=["C1"],
            transaction_date=["2024-01-01"],
        )
        with pytest.raises(SchemaError):
            self.validator.validate(df)

    def test_schema_error_lists_missing_columns(self) -> None:
        df = _df(transaction_amount=["99.00"])
        with pytest.raises(SchemaError) as exc_info:
            self.validator.validate(df)
        assert "customer_id" in str(exc_info.value)


class TestTypeCoercion:
    def setup_method(self) -> None:
        self.validator = SchemaValidator()

    def test_transaction_amount_coerced_to_float(self) -> None:
        df = _df(
            customer_id=["C1"],
            transaction_date=["2024-01-01"],
            transaction_amount=["99.50"],
        )
        result = self.validator.validate(df)
        assert pd.api.types.is_float_dtype(result.normalised_df["transaction_amount"])

    def test_invalid_amount_becomes_nan(self) -> None:
        df = _df(
            customer_id=["C1"],
            transaction_date=["2024-01-01"],
            transaction_amount=["not_a_number"],
        )
        result = self.validator.validate(df)
        assert result.normalised_df["transaction_amount"].isna().all()

    def test_subscription_status_lowercased(self) -> None:
        df = _df(
            customer_id=["C1"],
            transaction_date=["2024-01-01"],
            transaction_amount=["99.00"],
            subscription_status=["ACTIVE"],
        )
        result = self.validator.validate(df)
        assert result.normalised_df["subscription_status"].iloc[0] == "active"
