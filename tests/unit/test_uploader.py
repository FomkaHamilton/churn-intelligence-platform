"""Unit tests for the file upload engine."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from src.ingestion.uploader import UploadEngine
from src.utils.exceptions import (
    InsufficientDataError,
    UnsupportedFileTypeError,
    ValidationError,
)


def _csv_bytes(content: str) -> io.BytesIO:
    return io.BytesIO(content.encode("utf-8"))


VALID_CSV = "customer_id,transaction_date,transaction_amount\nCUST_001,2024-01-15,99.00\nCUST_002,2024-02-01,49.00\n"


class TestUploadEngineValidation:
    def setup_method(self) -> None:
        self.engine = UploadEngine()

    def test_valid_csv_succeeds(self) -> None:
        result = self.engine.process(_csv_bytes(VALID_CSV), "test.csv")
        assert result.row_count == 2
        assert result.column_count == 3

    def test_unsupported_extension_raises(self) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            self.engine.process(_csv_bytes(VALID_CSV), "data.json")

    def test_txt_extension_raises(self) -> None:
        with pytest.raises(UnsupportedFileTypeError):
            self.engine.process(_csv_bytes(VALID_CSV), "data.txt")

    def test_empty_file_raises_insufficient_data(self) -> None:
        with pytest.raises(InsufficientDataError):
            self.engine.process(_csv_bytes("customer_id,transaction_date\n"), "empty.csv")

    def test_file_too_large_raises_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.ingestion.uploader.get_settings",
            lambda: type("S", (), {"max_upload_size_bytes": 10, "max_upload_size_mb": 0})(),
        )
        engine = UploadEngine()
        with pytest.raises(ValidationError, match="limit"):
            engine.process(_csv_bytes(VALID_CSV), "test.csv")

    def test_column_names_normalised_to_lowercase(self) -> None:
        csv = "Customer_ID,Transaction_Date,Transaction_Amount\nC001,2024-01-01,50.00\n"
        result = self.engine.process(_csv_bytes(csv), "test.csv")
        assert "customer_id" in result.dataframe.columns
        assert "transaction_date" in result.dataframe.columns

    def test_column_names_strips_whitespace(self) -> None:
        csv = " customer_id , transaction_date , transaction_amount \nC001,2024-01-01,50.00\n"
        result = self.engine.process(_csv_bytes(csv), "test.csv")
        assert "customer_id" in result.dataframe.columns


class TestCSVInjectionDetection:
    def setup_method(self) -> None:
        self.engine = UploadEngine()

    def test_formula_injection_generates_warning(self) -> None:
        csv = "customer_id,transaction_date,transaction_amount\n=SUM(A1),2024-01-01,99.00\n"
        result = self.engine.process(_csv_bytes(csv), "test.csv")
        assert result.has_warnings
        assert any("formula" in w.lower() for w in result.warnings)

    def test_clean_data_has_no_warnings(self) -> None:
        result = self.engine.process(_csv_bytes(VALID_CSV), "test.csv")
        assert not result.has_warnings

    def test_at_prefix_flagged(self) -> None:
        csv = "customer_id,transaction_date,transaction_amount\n@malicious,2024-01-01,99.00\n"
        result = self.engine.process(_csv_bytes(csv), "test.csv")
        assert result.has_warnings


class TestUploadResultMetadata:
    def setup_method(self) -> None:
        self.engine = UploadEngine()

    def test_result_contains_filename(self) -> None:
        result = self.engine.process(_csv_bytes(VALID_CSV), "my_data.csv")
        assert result.filename == "my_data.csv"

    def test_result_file_size_is_positive(self) -> None:
        result = self.engine.process(_csv_bytes(VALID_CSV), "test.csv")
        assert result.file_size_bytes > 0
