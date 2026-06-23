"""Unit tests for the date format detector."""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.date_parser import DateFormatDetector
from src.utils.exceptions import ValidationError


class TestFormatDetection:
    def setup_method(self) -> None:
        self.detector = DateFormatDetector()

    def test_iso_format_detected(self) -> None:
        series = pd.Series(["2024-01-15", "2024-02-20", "2024-03-01"])
        result = self.detector.detect_and_parse(series)
        assert result.format_label == "ISO"
        assert result.parse_failure_count == 0

    def test_us_format_detected(self) -> None:
        series = pd.Series(["01/15/2024", "02/20/2024", "03/25/2024"])
        result = self.detector.detect_and_parse(series)
        assert result.format_label == "US"

    def test_eu_format_detected(self) -> None:
        series = pd.Series(["15/01/2024", "20/02/2024", "25/03/2024"])
        result = self.detector.detect_and_parse(series)
        assert result.format_label == "EU"

    def test_iso_parses_correctly(self) -> None:
        series = pd.Series(["2024-03-15"])
        result = self.detector.detect_and_parse(series)
        assert result.parsed_series.iloc[0] == pd.Timestamp("2024-03-15")

    def test_null_values_preserved(self) -> None:
        series = pd.Series(["2024-01-15", None, "2024-03-01"])
        result = self.detector.detect_and_parse(series)
        assert pd.isna(result.parsed_series.iloc[1])
        assert result.parse_failure_count == 0


class TestAmbiguityDetection:
    def setup_method(self) -> None:
        self.detector = DateFormatDetector()

    def test_ambiguous_dates_flagged(self) -> None:
        # All days ≤ 12 — could be MM/DD or DD/MM
        series = pd.Series(["01/02/2024", "03/04/2024", "05/06/2024"])
        result = self.detector.detect_and_parse(series)
        assert result.is_ambiguous is True

    def test_unambiguous_eu_dates_not_flagged(self) -> None:
        # Day 15+ means it must be DD/MM
        series = pd.Series(["15/01/2024", "20/02/2024", "25/03/2024"])
        result = self.detector.detect_and_parse(series)
        assert result.is_ambiguous is False

    def test_iso_format_never_ambiguous(self) -> None:
        series = pd.Series(["2024-01-02", "2024-03-04"])
        result = self.detector.detect_and_parse(series)
        assert result.is_ambiguous is False


class TestForceFormat:
    def setup_method(self) -> None:
        self.detector = DateFormatDetector()

    def test_force_format_overrides_detection(self) -> None:
        series = pd.Series(["01/02/2024"])
        result = self.detector.detect_and_parse(series, force_format="%d/%m/%Y")
        assert result.parsed_series.iloc[0] == pd.Timestamp("2024-02-01")

    def test_force_us_format(self) -> None:
        series = pd.Series(["01/02/2024"])
        result = self.detector.detect_and_parse(series, force_format="%m/%d/%Y")
        assert result.parsed_series.iloc[0] == pd.Timestamp("2024-01-02")


class TestSamplePairs:
    def setup_method(self) -> None:
        self.detector = DateFormatDetector()

    def test_sample_pairs_returned(self) -> None:
        series = pd.Series(["2024-01-15", "2024-02-20", "2024-03-01"])
        result = self.detector.detect_and_parse(series, sample_size=3)
        assert len(result.sample_pairs) == 3

    def test_sample_pairs_are_tuples_of_strings(self) -> None:
        series = pd.Series(["2024-01-15"])
        result = self.detector.detect_and_parse(series, sample_size=1)
        raw, parsed = result.sample_pairs[0]
        assert isinstance(raw, str)
        assert isinstance(parsed, str)
        assert "2024" in parsed


class TestParseFailures:
    def setup_method(self) -> None:
        self.detector = DateFormatDetector()

    def test_majority_unparseable_raises(self) -> None:
        series = pd.Series(["not_a_date"] * 10)
        with pytest.raises(ValidationError):
            self.detector.detect_and_parse(series)

    def test_minority_failures_reported_not_raised(self) -> None:
        series = pd.Series(["2024-01-15"] * 9 + ["bad_date"])
        result = self.detector.detect_and_parse(series)
        assert result.parse_failure_count == 1
