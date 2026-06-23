"""
Date format detection and parsing for subscription transaction dates.

The core challenge: MM/DD/YYYY and DD/MM/YYYY are visually identical
when the day value is ≤ 12. This module detects ambiguity and
surfaces it for explicit user confirmation rather than guessing silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.utils.exceptions import ValidationError
from src.utils.log import get_logger
from src.utils.types import DataFrame, Series

logger = get_logger(__name__)

DateFormat = Literal["ISO", "US", "EU", "YMD_SLASH", "UNKNOWN"]

# Ordered list of formats to try. ISO is tried first — it's unambiguous.
_FORMAT_CANDIDATES: list[tuple[str, DateFormat]] = [
    ("%Y-%m-%d", "ISO"),
    ("%Y/%m/%d", "YMD_SLASH"),
    ("%m/%d/%Y", "US"),
    ("%d/%m/%Y", "EU"),
    ("%m-%d-%Y", "US"),
    ("%d-%m-%Y", "EU"),
    ("%d.%m.%Y", "EU"),
    ("%m.%d.%Y", "US"),
]


@dataclass
class DateParseResult:
    """Result of date parsing, including confidence and ambiguity flag."""

    parsed_series: Series
    detected_format: str
    format_label: DateFormat
    is_ambiguous: bool        # True when MM/DD vs DD/MM cannot be determined
    sample_pairs: list[tuple[str, str]]  # [(raw, parsed_str), ...] for UI display
    parse_failure_count: int


class DateFormatDetector:
    """
    Detects, validates, and parses transaction date columns.

    Ambiguity is defined as: all day-component values in the first 200
    non-null rows are ≤ 12, making US and EU format indistinguishable.
    When ambiguous, the caller should present sample_pairs to the user
    and ask for explicit confirmation before proceeding.
    """

    def detect_and_parse(
        self,
        series: Series,
        *,
        force_format: str | None = None,
        sample_size: int = 5,
    ) -> DateParseResult:
        """
        Auto-detect the date format and parse the full series.

        Parameters
        ----------
        series:
            Raw string date column from the uploaded file.
        force_format:
            If provided, skip detection and parse with this strptime format.
            Use this when the user has confirmed the format after seeing samples.
        sample_size:
            Number of (raw, parsed) pairs to return for UI display.

        Raises
        ------
        ValidationError
            If no format can parse more than 50% of non-null values.
        """
        clean = series.dropna().astype(str).str.strip()
        sample_raw = clean.head(200)

        if force_format:
            fmt = force_format
            label: DateFormat = "UNKNOWN"
            is_ambiguous = False
        else:
            fmt, label = self._detect_format(sample_raw)
            is_ambiguous = self._check_ambiguity(sample_raw, label)

        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        failure_count = int(parsed.isna().sum() - series.isna().sum())

        if failure_count > len(series) * 0.5:
            raise ValidationError(
                f"More than 50% of dates could not be parsed with format '{fmt}'. "
                "Use the format override option to specify the correct format."
            )

        if failure_count > 0:
            logger.warning(
                "date_parse_failures",
                count=failure_count,
                format=fmt,
            )

        sample_pairs = self._build_sample_pairs(series, parsed, sample_size)

        logger.info(
            "dates_parsed",
            format=fmt,
            label=label,
            ambiguous=is_ambiguous,
            failures=failure_count,
        )

        return DateParseResult(
            parsed_series=parsed,
            detected_format=fmt,
            format_label=label,
            is_ambiguous=is_ambiguous,
            sample_pairs=sample_pairs,
            parse_failure_count=failure_count,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_format(self, sample: Series) -> tuple[str, DateFormat]:
        """
        Try each candidate format against a sample; return the best match.
        """
        best_fmt = ""
        best_label: DateFormat = "UNKNOWN"
        best_score = 0

        for fmt, label in _FORMAT_CANDIDATES:
            parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
            score = int(parsed.notna().sum())
            if score > best_score:
                best_score = score
                best_fmt = fmt
                best_label = label

        if best_score == 0:
            # Last resort: let pandas infer
            try:
                pd.to_datetime(sample.iloc[0], infer_datetime_format=True)
                return "mixed", "UNKNOWN"
            except Exception:
                raise ValidationError(
                    "Could not detect a date format. "
                    "Ensure transaction_date contains valid dates "
                    "(e.g. 2024-01-15 or 01/15/2024)."
                )

        return best_fmt, best_label

    def _check_ambiguity(self, sample: Series, label: DateFormat) -> bool:
        """
        Flag ambiguity when all day-position values are ≤ 12,
        making US/EU formats indistinguishable.
        """
        if label not in ("US", "EU"):
            return False

        # Extract the first positional component (day in EU, month in US)
        first_components = (
            sample.str.split(r"[/\-.]").str[0].pipe(pd.to_numeric, errors="coerce")
        )
        max_first = first_components.max()

        # If every first component is ≤ 12, format is ambiguous
        return bool(max_first <= 12)

    def _build_sample_pairs(
        self,
        raw: Series,
        parsed: Series,
        n: int,
    ) -> list[tuple[str, str]]:
        """Return n (raw_value, parsed_date_string) pairs for UI confirmation."""
        pairs: list[tuple[str, str]] = []
        for raw_val, parsed_val in zip(raw.dropna().head(n * 2), parsed.dropna().head(n * 2)):
            if len(pairs) >= n:
                break
            pairs.append((str(raw_val), parsed_val.strftime("%B %d, %Y")))
        return pairs


def apply_parsed_dates(df: DataFrame, parsed_series: Series) -> DataFrame:
    """Replace the transaction_date column with the parsed datetime Series."""
    df = df.copy()
    df["transaction_date"] = parsed_series
    return df
