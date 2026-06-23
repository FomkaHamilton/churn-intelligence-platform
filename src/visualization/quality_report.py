"""
Streamlit rendering helpers for data quality reports.

Keeps all st.* calls out of the business logic layer so
the validator remains testable without a running Streamlit instance.
"""
from __future__ import annotations

import streamlit as st

from src.preprocessing.validator import DataQualityReport, Severity

_SEVERITY_CONFIG: dict[Severity, tuple[str, str]] = {
    "error":   ("❌", "error"),
    "warning": ("⚠️", "warning"),
    "info":    ("ℹ️", "info"),
}


def render_quality_report(report: DataQualityReport) -> None:
    """Render a DataQualityReport as Streamlit UI components."""

    if report.passed:
        st.success(report.summary)
    else:
        st.error(report.summary)

    if not report.issues:
        return

    st.markdown("#### Quality check details")

    for issue in report.issues:
        icon, msg_type = _SEVERITY_CONFIG[issue.severity]
        label = f"{icon} **{issue.category}** — {issue.affected_count:,} row(s) affected ({issue.affected_pct:.1f}%)"

        if issue.severity == "error":
            st.error(f"{label}\n\n{issue.message}")
        elif issue.severity == "warning":
            st.warning(f"{label}\n\n{issue.message}")
        else:
            st.info(f"{label}\n\n{issue.message}")

        if issue.example_values:
            st.caption(f"Examples: {', '.join(issue.example_values[:3])}")
