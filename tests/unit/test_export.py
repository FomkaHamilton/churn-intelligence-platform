"""Unit tests for src/export.py — CSV and PDF export builders."""
from __future__ import annotations

import csv
import io

import pandas as pd

from src.export import (
    _strip_md,
    _to_latin1_safe,
    build_at_risk_csv,
    build_clv_csv,
    build_full_predictions_csv,
    build_insights_pdf,
)
from src.insights.models import InsightReport


def _make_predictions(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "churn_probability": [0.9, 0.5, 0.7, 0.2][:n],
    })


def _make_rfm(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "recency_days": [5, 30, 15, 60][:n],
        "monetary_total": [1000.0, 200.0, 500.0, 50.0][:n],
        "frequency": [10, 3, 6, 1][:n],
        "tenure_days": [365, 90, 180, 30][:n],
    })


def _make_clv(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "expected_clv": [1200.0, 400.0, 800.0][:n],
        "expected_remaining_months": [12.0, 4.0, 8.0][:n],
        "monthly_spend": [100.0, 100.0, 100.0][:n],
    })


def _make_report(*, model_confidence: str | None = "High confidence (AUC 0.85).") -> InsightReport:
    return InsightReport(
        health_summary="Revenue is **stable** with *mild* growth.",
        churn_analysis="- At-risk segment grew by 5%.\n- Top driver: recency.",
        revenue_outlook="Outlook is positive.",
        customer_segments="Segment breakdown is normal.",
        recommendations="- Retain at-risk customers.\n- Reward high-value customers.",
        model_confidence=model_confidence,
    )


# ── _strip_md ─────────────────────────────────────────────────────────────────

class TestStripMd:
    def test_removes_bold_markers(self) -> None:
        assert _strip_md("Hello **world**") == "Hello world"

    def test_removes_italic_markers(self) -> None:
        assert _strip_md("Hello *world*") == "Hello world"

    def test_removes_both(self) -> None:
        assert _strip_md("**bold** and *italic*") == "bold and italic"

    def test_empty_string(self) -> None:
        assert _strip_md("") == ""

    def test_no_markdown(self) -> None:
        text = "plain text with no markers"
        assert _strip_md(text) == text

    def test_bullet_line_untouched(self) -> None:
        assert _strip_md("- bullet item") == "- bullet item"


class TestToLatin1Safe:
    def test_em_dash_replaced(self) -> None:
        assert _to_latin1_safe("5—ten") == "5-ten"

    def test_en_dash_replaced(self) -> None:
        assert _to_latin1_safe("5–10") == "5-10"

    def test_bullet_replaced(self) -> None:
        assert _to_latin1_safe("• item") == "* item"

    def test_plain_text_unchanged(self) -> None:
        text = "Hello world 123"
        assert _to_latin1_safe(text) == text


# ── build_at_risk_csv ─────────────────────────────────────────────────────────

class TestBuildAtRiskCsv:
    def setup_method(self) -> None:
        self.pred = _make_predictions()
        self.rfm = _make_rfm()

    def test_returns_bytes(self) -> None:
        result = build_at_risk_csv(self.pred, self.rfm)
        assert isinstance(result, bytes)

    def test_sorted_by_churn_probability_descending(self) -> None:
        result = build_at_risk_csv(self.pred, self.rfm)
        rows = list(csv.DictReader(io.StringIO(result.decode("utf-8"))))
        probs = [float(r["Churn Probability"]) for r in rows]
        assert probs == sorted(probs, reverse=True)

    def test_expected_columns(self) -> None:
        result = build_at_risk_csv(self.pred, self.rfm)
        reader = csv.DictReader(io.StringIO(result.decode("utf-8")))
        headers = reader.fieldnames or []
        assert "Customer ID" in headers
        assert "Churn Probability" in headers
        assert "Days Since Last Purchase" in headers
        assert "Total Spend" in headers
        assert "Total Purchases" in headers

    def test_probability_rounded_to_4dp(self) -> None:
        pred = pd.DataFrame({
            "customer_id": ["C0"],
            "churn_probability": [0.123456789],
        })
        rfm = _make_rfm(1)
        result = build_at_risk_csv(pred, rfm)
        rows = list(csv.DictReader(io.StringIO(result.decode("utf-8"))))
        assert rows[0]["Churn Probability"] == "0.1235"

    def test_row_count_matches_customers(self) -> None:
        result = build_at_risk_csv(self.pred, self.rfm)
        rows = list(csv.DictReader(io.StringIO(result.decode("utf-8"))))
        assert len(rows) == 4


# ── build_clv_csv ─────────────────────────────────────────────────────────────

class TestBuildClvCsv:
    def setup_method(self) -> None:
        self.clv = _make_clv()

    def test_returns_bytes(self) -> None:
        assert isinstance(build_clv_csv(self.clv), bytes)

    def test_sorted_by_clv_descending(self) -> None:
        result = build_clv_csv(self.clv)
        rows = list(csv.DictReader(io.StringIO(result.decode("utf-8"))))
        clvs = [float(r["Expected CLV"]) for r in rows]
        assert clvs == sorted(clvs, reverse=True)

    def test_expected_columns(self) -> None:
        result = build_clv_csv(self.clv)
        reader = csv.DictReader(io.StringIO(result.decode("utf-8")))
        headers = reader.fieldnames or []
        assert "Customer ID" in headers
        assert "Expected CLV" in headers
        assert "Remaining Months" in headers
        assert "Monthly Spend" in headers


# ── build_full_predictions_csv ────────────────────────────────────────────────

class TestBuildFullPredictionsCsv:
    def setup_method(self) -> None:
        self.pred = _make_predictions()
        self.rfm = _make_rfm()

    def test_returns_bytes(self) -> None:
        assert isinstance(build_full_predictions_csv(self.pred, self.rfm), bytes)

    def test_includes_tenure_days_column(self) -> None:
        result = build_full_predictions_csv(self.pred, self.rfm)
        reader = csv.DictReader(io.StringIO(result.decode("utf-8")))
        assert "Tenure Days" in (reader.fieldnames or [])

    def test_sorted_by_churn_probability_descending(self) -> None:
        result = build_full_predictions_csv(self.pred, self.rfm)
        rows = list(csv.DictReader(io.StringIO(result.decode("utf-8"))))
        probs = [float(r["Churn Probability"]) for r in rows]
        assert probs == sorted(probs, reverse=True)

    def test_probability_rounded_to_4dp(self) -> None:
        pred = pd.DataFrame({
            "customer_id": ["C0"],
            "churn_probability": [0.987654321],
        })
        rfm = _make_rfm(1)
        result = build_full_predictions_csv(pred, rfm)
        rows = list(csv.DictReader(io.StringIO(result.decode("utf-8"))))
        assert rows[0]["Churn Probability"] == "0.9877"


# ── build_insights_pdf ────────────────────────────────────────────────────────

class TestBuildInsightsPdf:
    def test_returns_bytes(self) -> None:
        result = build_insights_pdf(_make_report(), "2026-06-25")
        assert isinstance(result, bytes)

    def test_valid_pdf_magic_bytes(self) -> None:
        result = build_insights_pdf(_make_report(), "2026-06-25")
        assert result[:4] == b"%PDF"

    def test_non_empty(self) -> None:
        result = build_insights_pdf(_make_report(), "2026-06-25")
        assert len(result) > 1000

    def test_without_model_confidence(self) -> None:
        report = _make_report(model_confidence=None)
        result = build_insights_pdf(report, "2026-06-25")
        assert result[:4] == b"%PDF"

    def test_with_model_confidence(self) -> None:
        report = _make_report(model_confidence="AUC 0.87 — high confidence.")
        result = build_insights_pdf(report, "2026-06-25")
        assert result[:4] == b"%PDF"

    def test_bullet_lines_in_body(self) -> None:
        report = InsightReport(
            health_summary="- Item one.\n- Item two.",
            churn_analysis="Normal text here.",
            revenue_outlook="Positive.",
            customer_segments="Mixed.",
            recommendations="- Action A.\n- Action B.",
            model_confidence=None,
        )
        result = build_insights_pdf(report, "2026-06-25")
        assert result[:4] == b"%PDF"

    def test_empty_lines_in_body_no_crash(self) -> None:
        report = InsightReport(
            health_summary="First paragraph.\n\nSecond paragraph.",
            churn_analysis="",
            revenue_outlook="",
            customer_segments="",
            recommendations="",
            model_confidence=None,
        )
        result = build_insights_pdf(report, "2026-06-25")
        assert isinstance(result, bytes)
