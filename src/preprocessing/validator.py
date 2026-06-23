"""
Data quality framework for the Churn Intelligence Platform.

Design principles:
- Never silently drop records — every issue is reported and counted
- Distinguish errors (block analysis) from warnings (proceed with caution)
- Return a structured report the UI can render, not a blob of text
- All checks are idempotent and work on the normalised DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import pandas as pd

from src.config.settings import get_settings, get_yaml_config
from src.utils.log import get_logger
from src.utils.types import DataFrame

logger = get_logger(__name__)

Severity = Literal["error", "warning", "info"]


@dataclass
class DataQualityIssue:
    """A single data quality finding."""

    severity: Severity
    category: str
    message: str
    affected_count: int
    affected_pct: float
    example_values: list[str] = field(default_factory=list)

    @property
    def is_blocking(self) -> bool:
        return self.severity == "error"


@dataclass
class DataQualityReport:
    """
    Aggregated result of all quality checks.

    passed = True means analysis can proceed (no blocking errors).
    Warnings and info items are advisory only.
    """

    total_rows: int
    issues: list[DataQualityIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[DataQualityIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[DataQualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    @property
    def error_count(self) -> int:
        return sum(i.affected_count for i in self.errors)

    @property
    def summary(self) -> str:
        if self.passed:
            return (
                f"✅ {self.total_rows:,} rows passed quality checks"
                + (f" with {len(self.warnings)} warning(s)." if self.warnings else ".")
            )
        return (
            f"❌ {len(self.errors)} blocking issue(s) found. "
            f"Resolve before proceeding."
        )


class DataQualityChecker:
    """
    Runs all data quality checks against a normalised DataFrame.

    Each check returns a DataQualityIssue (or None if clean).
    run_all_checks() aggregates them into a DataQualityReport.
    """

    def __init__(self) -> None:
        cfg = get_yaml_config()
        dq = cfg.get("data_quality", {})
        self._max_future_days: int = dq.get("max_future_date_days", 1)
        self._min_amount: float = dq.get("min_transaction_amount", 0.0)
        self._max_dup_pct: float = dq.get("max_duplicate_pct", 5.0)

    def run_all_checks(self, df: DataFrame) -> DataQualityReport:
        """Run every quality check and return a consolidated report."""
        report = DataQualityReport(total_rows=len(df))

        checks = [
            self._check_null_customer_ids,
            self._check_null_transaction_dates,
            self._check_null_transaction_amounts,
            self._check_negative_amounts,
            self._check_future_dates,
            self._check_duplicate_transactions,
            self._check_zero_amount_rows,
        ]

        for check in checks:
            issue = check(df)
            if issue is not None:
                report.issues.append(issue)

        logger.info(
            "quality_check_complete",
            total_rows=report.total_rows,
            errors=len(report.errors),
            warnings=len(report.warnings),
            passed=report.passed,
        )

        return report

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_null_customer_ids(self, df: DataFrame) -> DataQualityIssue | None:
        if "customer_id" not in df.columns:
            return None
        null_mask = df["customer_id"].isna() | (df["customer_id"].astype(str) == "")
        count = int(null_mask.sum())
        if count == 0:
            return None
        return DataQualityIssue(
            severity="error",
            category="Missing Values",
            message=(
                f"{count:,} row(s) have no customer_id. "
                "These rows cannot be attributed to a customer and must be resolved."
            ),
            affected_count=count,
            affected_pct=round(count / len(df) * 100, 2),
        )

    def _check_null_transaction_dates(self, df: DataFrame) -> DataQualityIssue | None:
        if "transaction_date" not in df.columns:
            return None
        count = int(df["transaction_date"].isna().sum())
        if count == 0:
            return None
        return DataQualityIssue(
            severity="error",
            category="Missing Values",
            message=(
                f"{count:,} row(s) have no transaction_date. "
                "Dates are required for all time-based analysis."
            ),
            affected_count=count,
            affected_pct=round(count / len(df) * 100, 2),
        )

    def _check_null_transaction_amounts(self, df: DataFrame) -> DataQualityIssue | None:
        if "transaction_amount" not in df.columns:
            return None
        count = int(df["transaction_amount"].isna().sum())
        if count == 0:
            return None
        return DataQualityIssue(
            severity="warning",
            category="Missing Values",
            message=(
                f"{count:,} row(s) have no transaction_amount. "
                "Revenue metrics will exclude these rows."
            ),
            affected_count=count,
            affected_pct=round(count / len(df) * 100, 2),
        )

    def _check_negative_amounts(self, df: DataFrame) -> DataQualityIssue | None:
        if "transaction_amount" not in df.columns:
            return None
        mask = df["transaction_amount"] < self._min_amount
        count = int(mask.sum())
        if count == 0:
            return None
        examples = (
            df.loc[mask, "transaction_amount"]
            .head(3)
            .astype(str)
            .tolist()
        )
        return DataQualityIssue(
            severity="warning",
            category="Invalid Values",
            message=(
                f"{count:,} row(s) have a negative transaction_amount. "
                "These may represent refunds. They are flagged but not removed — "
                "verify whether refunds should be excluded from revenue."
            ),
            affected_count=count,
            affected_pct=round(count / len(df) * 100, 2),
            example_values=examples,
        )

    def _check_future_dates(self, df: DataFrame) -> DataQualityIssue | None:
        if "transaction_date" not in df.columns:
            return None
        if not pd.api.types.is_datetime64_any_dtype(df["transaction_date"]):
            return None
        cutoff = pd.Timestamp(date.today() + timedelta(days=self._max_future_days))
        mask = df["transaction_date"] > cutoff
        count = int(mask.sum())
        if count == 0:
            return None
        examples = (
            df.loc[mask, "transaction_date"]
            .dt.strftime("%Y-%m-%d")
            .head(3)
            .tolist()
        )
        return DataQualityIssue(
            severity="error",
            category="Invalid Values",
            message=(
                f"{count:,} row(s) have transaction dates in the future. "
                "Future dates indicate a data issue — likely incorrect date format parsing."
            ),
            affected_count=count,
            affected_pct=round(count / len(df) * 100, 2),
            example_values=examples,
        )

    def _check_duplicate_transactions(self, df: DataFrame) -> DataQualityIssue | None:
        key_cols = [c for c in ("customer_id", "transaction_date", "transaction_amount") if c in df.columns]
        if len(key_cols) < 2:
            return None
        dup_mask = df.duplicated(subset=key_cols, keep="first")
        count = int(dup_mask.sum())
        if count == 0:
            return None
        pct = count / len(df) * 100
        severity: Severity = "error" if pct > self._max_dup_pct else "warning"
        return DataQualityIssue(
            severity=severity,
            category="Duplicate Records",
            message=(
                f"{count:,} duplicate row(s) found ({pct:.1f}% of total). "
                f"Duplicates will be removed before analysis to prevent double-counting."
            ),
            affected_count=count,
            affected_pct=round(pct, 2),
        )

    def _check_zero_amount_rows(self, df: DataFrame) -> DataQualityIssue | None:
        if "transaction_amount" not in df.columns:
            return None
        mask = df["transaction_amount"] == 0.0
        count = int(mask.sum())
        if count == 0:
            return None
        return DataQualityIssue(
            severity="info",
            category="Zero-Value Records",
            message=(
                f"{count:,} row(s) have a transaction_amount of £0.00. "
                "These may represent free trials or $0 renewals. "
                "They are included by default but will not contribute to revenue metrics."
            ),
            affected_count=count,
            affected_pct=round(count / len(df) * 100, 2),
        )


def apply_quality_fixes(df: DataFrame, report: DataQualityReport) -> DataFrame:
    """
    Apply safe, automatic fixes based on quality report findings.

    Only removes true duplicates — never silently drops rows with
    missing values or invalid amounts (those require human review).
    """
    df = df.copy()
    key_cols = [c for c in ("customer_id", "transaction_date", "transaction_amount") if c in df.columns]
    if len(key_cols) >= 2:
        before = len(df)
        df = df.drop_duplicates(subset=key_cols, keep="first")
        removed = before - len(df)
        if removed:
            logger.info("duplicates_removed", count=removed)
    return df
