"""
Custom exception hierarchy for the Churn Intelligence Platform.

All platform exceptions inherit from SIPError so callers can catch
platform-specific errors separately from unexpected Python exceptions.
"""
from __future__ import annotations


class SIPError(Exception):
    """Base exception for the Churn Intelligence Platform."""


# ── Data layer ────────────────────────────────────────────────────────────────

class ValidationError(SIPError):
    """Raised when uploaded data fails a validation check."""


class SchemaError(ValidationError):
    """Raised when the uploaded file is missing required columns or has wrong types."""


class DataQualityError(ValidationError):
    """Raised when data quality checks fail at a level that blocks analysis."""


class InsufficientDataError(SIPError):
    """Raised when a dataset is too small for a requested operation."""

    def __init__(self, operation: str, required: int, actual: int) -> None:
        super().__init__(
            f"'{operation}' requires at least {required} records, got {actual}."
        )
        self.operation = operation
        self.required = required
        self.actual = actual


class UnsupportedFileTypeError(ValidationError):
    """Raised when an uploaded file has an unsupported format."""


# ── ML layer ──────────────────────────────────────────────────────────────────

class ModelTrainingError(SIPError):
    """Raised when model training fails."""


class TemporalLeakageError(SIPError):
    """
    Raised when a data split or feature would introduce temporal leakage.
    This is a hard error — leakage silently corrupts model metrics.
    """


# ── Forecasting layer ─────────────────────────────────────────────────────────

class ForecastingError(SIPError):
    """Raised when a forecasting operation fails."""


class InsufficientHistoryError(ForecastingError):
    """Raised when there is not enough time-series history to forecast."""


# ── Insight layer ─────────────────────────────────────────────────────────────

class InsightGenerationError(SIPError):
    """Raised when AI insight generation fails."""


class LLMProviderError(InsightGenerationError):
    """Raised when the configured LLM provider returns an error."""


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigurationError(SIPError):
    """Raised when the application configuration is invalid."""
