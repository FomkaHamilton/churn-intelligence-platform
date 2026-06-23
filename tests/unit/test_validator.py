"""Unit tests for the data quality framework."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.preprocessing.validator import DataQualityChecker, apply_quality_fixes


def _base_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "transaction_date": pd.to_datetime(
                [date(2024, 1, i + 1) for i in range(n)]
            ),
            "transaction_amount": [99.0] * n,
        }
    )


class TestNullChecks:
    def setup_method(self) -> None:
        self.checker = DataQualityChecker()

    def test_null_customer_id_is_error(self) -> None:
        df = _base_df()
        df.loc[0, "customer_id"] = None
        report = self.checker.run_all_checks(df)
        errors = [i for i in report.issues if i.category == "Missing Values" and "customer_id" in i.message]
        assert len(errors) == 1
        assert errors[0].is_blocking

    def test_null_transaction_date_is_error(self) -> None:
        df = _base_df()
        df.loc[0, "transaction_date"] = pd.NaT
        report = self.checker.run_all_checks(df)
        errors = [i for i in report.errors if "transaction_date" in i.message]
        assert len(errors) == 1

    def test_null_amount_is_warning_not_error(self) -> None:
        df = _base_df()
        df.loc[0, "transaction_amount"] = None
        report = self.checker.run_all_checks(df)
        amount_issues = [i for i in report.issues if "transaction_amount" in i.message]
        assert all(i.severity == "warning" for i in amount_issues)

    def test_clean_dataframe_passes(self) -> None:
        report = self.checker.run_all_checks(_base_df())
        assert report.passed
        assert len(report.errors) == 0


class TestNegativeAmounts:
    def setup_method(self) -> None:
        self.checker = DataQualityChecker()

    def test_negative_amount_raises_warning(self) -> None:
        df = _base_df()
        df.loc[0, "transaction_amount"] = -50.0
        report = self.checker.run_all_checks(df)
        neg_issues = [i for i in report.issues if "negative" in i.message.lower()]
        assert len(neg_issues) == 1
        assert neg_issues[0].severity == "warning"

    def test_negative_amount_does_not_block(self) -> None:
        df = _base_df()
        df.loc[0, "transaction_amount"] = -50.0
        report = self.checker.run_all_checks(df)
        assert report.passed  # warnings don't block


class TestFutureDates:
    def setup_method(self) -> None:
        self.checker = DataQualityChecker()

    def test_future_date_is_blocking_error(self) -> None:
        df = _base_df()
        df.loc[0, "transaction_date"] = pd.Timestamp(date.today() + timedelta(days=30))
        report = self.checker.run_all_checks(df)
        future_errors = [i for i in report.errors if "future" in i.message.lower()]
        assert len(future_errors) == 1

    def test_todays_date_passes(self) -> None:
        df = _base_df()
        df.loc[0, "transaction_date"] = pd.Timestamp(date.today())
        report = self.checker.run_all_checks(df)
        future_errors = [i for i in report.errors if "future" in i.message.lower()]
        assert len(future_errors) == 0


class TestDuplicates:
    def setup_method(self) -> None:
        self.checker = DataQualityChecker()

    def test_duplicate_rows_detected(self) -> None:
        df = _base_df(3)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = self.checker.run_all_checks(df)
        dup_issues = [i for i in report.issues if i.category == "Duplicate Records"]
        assert len(dup_issues) == 1
        assert dup_issues[0].affected_count == 1

    def test_no_duplicates_is_clean(self) -> None:
        report = self.checker.run_all_checks(_base_df())
        dup_issues = [i for i in report.issues if i.category == "Duplicate Records"]
        assert len(dup_issues) == 0


class TestQualityReport:
    def setup_method(self) -> None:
        self.checker = DataQualityChecker()

    def test_report_summary_passes_on_clean_data(self) -> None:
        report = self.checker.run_all_checks(_base_df())
        assert "✅" in report.summary

    def test_report_summary_fails_on_errors(self) -> None:
        df = _base_df()
        df["customer_id"] = None
        report = self.checker.run_all_checks(df)
        assert "❌" in report.summary

    def test_total_rows_matches_input(self) -> None:
        df = _base_df(42)
        report = self.checker.run_all_checks(df)
        assert report.total_rows == 42


class TestApplyQualityFixes:
    def test_duplicates_removed_by_fix(self) -> None:
        df = _base_df(3)
        df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = DataQualityChecker().run_all_checks(df_with_dup)
        fixed = apply_quality_fixes(df_with_dup, report)
        assert len(fixed) == 3

    def test_fix_does_not_drop_null_rows(self) -> None:
        df = _base_df()
        df.loc[0, "customer_id"] = None
        report = DataQualityChecker().run_all_checks(df)
        fixed = apply_quality_fixes(df, report)
        assert len(fixed) == len(df)
