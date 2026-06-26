"""Export utilities — CSV and PDF builders for st.download_button()."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.insights.models import InsightReport


def build_at_risk_csv(predictions: pd.DataFrame, rfm_features: pd.DataFrame) -> bytes:
    """All customers sorted by churn probability, with RFM context columns."""
    df = (
        predictions
        .sort_values("churn_probability", ascending=False)
        .merge(
            rfm_features[["customer_id", "recency_days", "monetary_total", "frequency"]],
            on="customer_id",
            how="left",
        )
    )
    df["churn_probability"] = df["churn_probability"].round(4)
    return (
        df.rename(columns={
            "customer_id": "Customer ID",
            "churn_probability": "Churn Probability",
            "recency_days": "Days Since Last Purchase",
            "monetary_total": "Total Spend",
            "frequency": "Total Purchases",
        })
        .to_csv(index=False)
        .encode("utf-8")
    )


def build_clv_csv(clv_per_customer: pd.DataFrame) -> bytes:
    """CLV table sorted by expected value descending."""
    return (
        clv_per_customer[["customer_id", "expected_clv", "expected_remaining_months", "monthly_spend"]]
        .sort_values("expected_clv", ascending=False)
        .rename(columns={
            "customer_id": "Customer ID",
            "expected_clv": "Expected CLV",
            "expected_remaining_months": "Remaining Months",
            "monthly_spend": "Monthly Spend",
        })
        .to_csv(index=False)
        .encode("utf-8")
    )


def build_full_predictions_csv(predictions: pd.DataFrame, rfm_features: pd.DataFrame) -> bytes:
    """All scored customers with extended RFM context."""
    df = (
        predictions
        .sort_values("churn_probability", ascending=False)
        .merge(
            rfm_features[["customer_id", "recency_days", "monetary_total", "frequency", "tenure_days"]],
            on="customer_id",
            how="left",
        )
    )
    df["churn_probability"] = df["churn_probability"].round(4)
    return (
        df.rename(columns={
            "customer_id": "Customer ID",
            "churn_probability": "Churn Probability",
            "recency_days": "Days Since Last Purchase",
            "monetary_total": "Total Spend",
            "frequency": "Total Purchases",
            "tenure_days": "Tenure Days",
        })
        .to_csv(index=False)
        .encode("utf-8")
    )


def _strip_md(text: str) -> str:
    """Remove bold/italic markers; leave bullet structure intact."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text


def build_insights_pdf(report: "InsightReport", generated_at: str) -> bytes:
    """Render an InsightReport as a formatted PDF and return the raw bytes."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title block
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Churn Intelligence Platform", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Executive Briefing", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, f"Generated: {generated_at}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    sections = [
        ("Business Health", report.health_summary),
        ("Churn Analysis", report.churn_analysis),
        ("Revenue Outlook", report.revenue_outlook),
        ("Customer Segments", report.customer_segments),
        ("Recommended Actions", report.recommendations),
    ]
    if report.model_confidence:
        sections.append(("Model Confidence", report.model_confidence))

    for title, body in sections:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(235, 235, 235)
        pdf.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10)
        for line in _strip_md(body).splitlines():
            line = line.strip()
            if not line:
                pdf.ln(3)
            elif line.startswith("- "):
                pdf.set_x(pdf.l_margin + 4)
                pdf.multi_cell(0, 6, f"•  {line[2:]}")
            else:
                pdf.multi_cell(0, 6, line)
        pdf.ln(6)

    return bytes(pdf.output())
