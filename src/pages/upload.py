from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ingestion.uploader import UploadEngine
from src.preprocessing.date_parser import DateFormatDetector, apply_parsed_dates, DateParseResult
from src.preprocessing.schema import SchemaValidator
from src.preprocessing.validator import DataQualityChecker, apply_quality_fixes
from src.utils.exceptions import SIPError
from src.utils.log import get_logger
from src.visualization.quality_report import render_quality_report

_logger = get_logger(__name__)


def _load_file(file: object, filename: str) -> None:
    try:
        engine = UploadEngine()
        upload_result = engine.process(file, filename)  # type: ignore[arg-type]

        if upload_result.has_warnings:
            for w in upload_result.warnings:
                st.warning(w)

        schema_validator = SchemaValidator()
        schema_result = schema_validator.validate(upload_result.dataframe)
        df = schema_result.normalised_df

        if schema_result.mapped_columns:
            mapped_str = ", ".join(f"`{k}` → `{v}`" for k, v in schema_result.mapped_columns.items())
            st.info(f"Column aliases detected and remapped: {mapped_str}")

        detector = DateFormatDetector()
        parse_result = detector.detect_and_parse(df["transaction_date"])
        st.session_state["parse_result"] = parse_result
        st.session_state["pre_date_df"] = df
        st.session_state["filename"] = filename

        if parse_result.is_ambiguous:
            st.session_state["date_format_confirmed"] = False
            st.rerun()
        else:
            _finalise_upload(df, parse_result, filename)

    except SIPError as exc:
        st.error(f"**Upload failed:** {exc}")
    except Exception as exc:
        st.error(f"**Unexpected error:** {exc}")
        _logger.exception("upload_unexpected_error", error=str(exc))


def _show_date_confirmation() -> None:
    parse_result = st.session_state["parse_result"]
    st.warning(
        "⚠️ **Date format ambiguous.** We detected dates that could be MM/DD/YYYY or DD/MM/YYYY. "
        "Review the samples below and confirm the correct interpretation."
    )
    st.markdown("**Sample dates from your file:**")
    col1, col2 = st.columns(2)
    col1.markdown("**Raw value**")
    col2.markdown("**Parsed as**")
    for raw, parsed in parse_result.sample_pairs:
        col1.code(raw)
        col2.markdown(parsed)

    fmt_choice = st.radio(
        "Which interpretation is correct?",
        options=["MM/DD/YYYY (US format)", "DD/MM/YYYY (EU format)"],
    )
    if st.button("✅ Confirm date format", type="primary"):
        force_fmt = "%m/%d/%Y" if "US" in fmt_choice else "%d/%m/%Y"
        detector = DateFormatDetector()
        df = st.session_state["pre_date_df"]
        parse_result = detector.detect_and_parse(df["transaction_date"], force_format=force_fmt)
        st.session_state["date_format_confirmed"] = True
        _finalise_upload(df, parse_result, st.session_state["filename"])


def _finalise_upload(df, parse_result, filename: str) -> None:
    if isinstance(parse_result, DateParseResult):
        df = apply_parsed_dates(df, parse_result.parsed_series)

    checker = DataQualityChecker()
    report = checker.run_all_checks(df)
    clean_df = apply_quality_fixes(df, report)

    st.session_state["raw_df"] = df
    st.session_state["clean_df"] = clean_df
    st.session_state["quality_report"] = report
    st.session_state["filename"] = filename
    st.session_state.pop("model_results", None)
    st.session_state.pop("insights_report", None)
    st.session_state.pop("insights_churn_window", None)

    st.success(f"✅ **{filename}** loaded — {len(clean_df):,} rows ready for analysis.")
    render_quality_report(report)

    _logger.info(
        "upload_finalised",
        filename=filename,
        rows=len(clean_df),
        quality_passed=report.passed,
    )


def render_upload_page(settings) -> None:
    st.title("📤 Upload Subscription Data")
    st.markdown("Upload your subscription transaction history. Accepted formats: CSV, XLSX.")

    sample_path = Path("data/sample/subscriptions_sample.csv")
    if sample_path.exists():
        if st.button("🎲 Load sample dataset instead", type="secondary"):
            with open(sample_path, "rb") as f:
                _load_file(f, "subscriptions_sample.csv")

    st.divider()
    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help=f"Maximum file size: {settings.max_upload_size_mb} MB",
    )

    if uploaded is not None:
        _load_file(uploaded, uploaded.name)

    if st.session_state.get("parse_result") and not st.session_state.get("date_format_confirmed"):
        _show_date_confirmation()
