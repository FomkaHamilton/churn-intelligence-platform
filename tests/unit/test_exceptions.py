"""Unit tests for the custom exception hierarchy."""
from __future__ import annotations

import pytest

from src.utils.exceptions import (
    DataQualityError,
    ForecastingError,
    InsufficientDataError,
    InsightGenerationError,
    LLMProviderError,
    ModelTrainingError,
    SIPError,
    SchemaError,
    TemporalLeakageError,
    UnsupportedFileTypeError,
    ValidationError,
)


class TestExceptionHierarchy:
    def test_all_exceptions_inherit_from_sip_error(self) -> None:
        exceptions = [
            ValidationError,
            SchemaError,
            DataQualityError,
            UnsupportedFileTypeError,
            ModelTrainingError,
            TemporalLeakageError,
            ForecastingError,
            InsightGenerationError,
            LLMProviderError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, SIPError)

    def test_schema_error_is_validation_error(self) -> None:
        assert issubclass(SchemaError, ValidationError)

    def test_llm_provider_error_is_insight_error(self) -> None:
        assert issubclass(LLMProviderError, InsightGenerationError)

    def test_insufficient_data_error_message(self) -> None:
        err = InsufficientDataError("churn modeling", required=50, actual=12)
        assert "50" in str(err)
        assert "12" in str(err)
        assert err.operation == "churn modeling"
        assert err.required == 50
        assert err.actual == 12

    def test_sip_error_is_catchable_as_exception(self) -> None:
        with pytest.raises(SIPError):
            raise ValidationError("test error")

    def test_temporal_leakage_is_hard_error(self) -> None:
        with pytest.raises(TemporalLeakageError):
            raise TemporalLeakageError("Future data detected in training set")
