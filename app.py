"""
Churn Intelligence Platform — Streamlit entry point.

Session state keys used across pages:
  raw_df          : DataFrame straight from upload (pre-quality-fix)
  clean_df        : DataFrame after schema + quality fixes
  quality_report  : DataQualityReport from the last validation run
  filename        : Name of the uploaded file
  churn_window_days : User-selected churn window (default 90)
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config.settings import get_settings
from src.ingestion.uploader import UploadEngine
from src.preprocessing.date_parser import DateFormatDetector, apply_parsed_dates
from src.preprocessing.schema import SchemaValidator
from src.preprocessing.validator import DataQualityChecker, apply_quality_fixes
from src.utils.exceptions import SIPError
from src.utils.log import configure_logging, get_logger
from src.visualization.quality_report import render_quality_report
from src.analytics.cohort import CohortAnalyzer, CohortResult
from src.analytics.kpis import KPICalculator, KPITimeSeries
from src.feature_engineering.churn_labels import ChurnLabelBuilder, ChurnLabelResult
from src.feature_engineering.rfm import RFMBuilder, RFMResult
from src.visualization.analytics import (
    render_churn_trend,
    render_cohort_heatmap,
    render_kpi_strip,
    render_revenue_trend,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_settings = get_settings()
configure_logging(_settings.log_level)
_logger = get_logger(__name__)

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "churn_window_days" not in st.session_state:
    st.session_state["churn_window_days"] = _settings.churn_window_days
if "date_format_confirmed" not in st.session_state:
    st.session_state["date_format_confirmed"] = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Churn Intelligence")
    st.caption(f"v{_settings.app_version}")
    st.divider()

    page = st.radio(
        "page",
        options=[
            "🏠  Overview",
            "📤  Upload Data",
            "🔍  Data Quality",
            "📈  Analytics",
            "🤖  Predictions",
            "🔮  Forecasting",
            "💡  Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("⚙️ Settings"):
        churn_window = st.selectbox(
            "Churn window (days)",
            options=[30, 60, 90, 120],
            index=[30, 60, 90, 120].index(st.session_state["churn_window_days"]),
            help="A customer is considered churned if they have no activity within this window.",
        )
        st.session_state["churn_window_days"] = churn_window

    if st.session_state.get("filename"):
        st.divider()
        st.caption(f"📂 {st.session_state['filename']}")
        rows = len(st.session_state.get("clean_df", pd.DataFrame()))
        st.caption(f"{rows:,} rows loaded")

    st.divider()
    ai_label = "✅ AI insights active" if _settings.has_ai_provider else "⚠️ Template mode"
    st.caption(ai_label)


# ── Cached analytics computations ────────────────────────────────────────────
# st.cache_data hashes the DataFrame by value so these recompute only when the
# underlying data changes (new upload).  Errors are caught in the page layer.

@st.cache_data
def _compute_kpis(df: pd.DataFrame) -> KPITimeSeries:
    return KPICalculator().calculate(df)


@st.cache_data
def _compute_rfm(df: pd.DataFrame) -> RFMResult | None:
    try:
        return RFMBuilder().build(df)
    except Exception:
        return None


@st.cache_data
def _compute_churn_labels(df: pd.DataFrame, churn_window_days: int) -> ChurnLabelResult:
    return ChurnLabelBuilder().build(df, churn_window_days=churn_window_days)


@st.cache_data
def _compute_cohort(df: pd.DataFrame) -> CohortResult | None:
    try:
        return CohortAnalyzer().build(df)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if "🏠" in (page or ""):
    st.title("📊 Churn Intelligence Platform")
    st.markdown("*Transform subscription transaction data into actionable retention intelligence.*")

    if not st.session_state.get("clean_df") is not None and st.session_state.get("filename"):
        st.success(f"✅ Data loaded: **{st.session_state['filename']}** — use the sidebar to navigate.")
    else:
        st.info("👈 Start by uploading your subscription data using **Upload Data** in the sidebar.", icon="📤")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Version", _settings.app_version)
    col2.metric("Build Phase", "3 / 9")
    col3.metric("Churn Window", f"{st.session_state['churn_window_days']}d")
    col4.metric("AI Mode", "Live" if _settings.has_ai_provider else "Template")

    st.divider()
    st.markdown("### Platform capabilities")
    capabilities = {
        "📤 Data Ingestion": ("CSV/XLSX upload, validation, quality profiling", "✅ Phase 2 — Live"),
        "📈 Cohort Analytics": ("Retention matrices, MRR, ARPU, churn rate", "✅ Phase 3 — Live"),
        "🤖 Churn Prediction": ("ML risk scoring with SHAP explainability", "🔄 Phase 4"),
        "💰 CLV Modeling": ("Survival-analysis-based customer lifetime value", "🔄 Phase 4"),
        "🔮 Revenue Forecasting": ("12-month subscriber and revenue forecasts", "🔄 Phase 5"),
        "💡 AI Insights": ("Executive summaries and intervention recommendations", "🔄 Phase 6"),
    }
    for feature, (desc, status) in capabilities.items():
        c1, c2, c3 = st.columns([2, 4, 2])
        c1.markdown(f"**{feature}**")
        c2.markdown(desc)
        c3.markdown(f"`{status}`")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: UPLOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
elif "📤" in (page or ""):
    st.title("📤 Upload Subscription Data")
    st.markdown("Upload your subscription transaction history. Accepted formats: CSV, XLSX.")

    # ── Demo data shortcut ─────────────────────────────────────────────────
    sample_path = Path("data/sample/subscriptions_sample.csv")
    if sample_path.exists():
        if st.button("🎲 Load sample dataset instead", type="secondary"):
            with open(sample_path, "rb") as f:
                _load_file(f, "subscriptions_sample.csv")

    st.divider()
    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help=f"Maximum file size: {_settings.max_upload_size_mb} MB",
    )

    if uploaded is not None:
        _load_file(uploaded, uploaded.name)

    if st.session_state.get("parse_result") and not st.session_state.get("date_format_confirmed"):
        _show_date_confirmation()


def _load_file(file: object, filename: str) -> None:
    """Run the full upload → schema → date-detection pipeline."""
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

        # Parse dates
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
    """Display date format confirmation UI for ambiguous formats."""
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


def _finalise_upload(df: pd.DataFrame, parse_result: object, filename: str) -> None:
    """Apply parsed dates, run quality checks, persist to session state."""
    from src.preprocessing.date_parser import DateParseResult

    if isinstance(parse_result, DateParseResult):
        df = apply_parsed_dates(df, parse_result.parsed_series)

    checker = DataQualityChecker()
    report = checker.run_all_checks(df)
    clean_df = apply_quality_fixes(df, report)

    st.session_state["raw_df"] = df
    st.session_state["clean_df"] = clean_df
    st.session_state["quality_report"] = report
    st.session_state["filename"] = filename

    st.success(f"✅ **{filename}** loaded — {len(clean_df):,} rows ready for analysis.")
    render_quality_report(report)

    _logger.info(
        "upload_finalised",
        filename=filename,
        rows=len(clean_df),
        quality_passed=report.passed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA QUALITY
# ─────────────────────────────────────────────────────────────────────────────
elif "🔍" in (page or ""):
    st.title("🔍 Data Quality Report")

    if not st.session_state.get("quality_report"):
        st.info("Upload data first to see quality results.", icon="📤")
    else:
        report = st.session_state["quality_report"]
        df = st.session_state.get("clean_df", pd.DataFrame())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{report.total_rows:,}")
        col2.metric("After Dedup", f"{len(df):,}")
        col3.metric("Errors", len(report.errors), delta_color="inverse")
        col4.metric("Warnings", len(report.warnings), delta_color="off")

        st.divider()
        render_quality_report(report)

        st.divider()
        st.markdown("### Data preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("### Column summary")
        summary = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.values,
            "Non-null": df.notna().sum().values,
            "Null": df.isna().sum().values,
            "Unique": df.nunique().values,
        })
        st.dataframe(summary, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif "📈" in (page or ""):
    st.title("📈 Analytics")

    df = st.session_state.get("clean_df")
    if df is None or len(df) == 0:
        st.info("Upload data first to see analytics.", icon="📤")
    else:
        churn_window = int(st.session_state["churn_window_days"])

        with st.spinner("Computing KPIs…"):
            kpi_ts = _compute_kpis(df)
        with st.spinner("Building RFM features…"):
            rfm_result = _compute_rfm(df)
        with st.spinner("Labelling churn…"):
            label_result = _compute_churn_labels(df, churn_window)
        with st.spinner("Building cohort matrix…"):
            cohort_result = _compute_cohort(df)

        # ── KPI headline strip ─────────────────────────────────────────────
        render_kpi_strip(kpi_ts.snapshot)

        st.divider()
        # ── Revenue and churn trends ───────────────────────────────────────
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Monthly Revenue")
            render_revenue_trend(kpi_ts)
        with col_right:
            st.markdown("#### Monthly Churn Rate")
            render_churn_trend(kpi_ts)

        st.divider()
        # ── Cohort retention heatmap ───────────────────────────────────────
        st.markdown("#### Cohort Retention Heatmap")
        if cohort_result is not None:
            render_cohort_heatmap(cohort_result)
        else:
            st.warning(
                "Not enough data to build cohort matrix — need at least 10 customers "
                "per cohort month.",
                icon="⚠️",
            )

        st.divider()
        # ── Churn & RFM summary cards ──────────────────────────────────────
        col_churn, col_rfm = st.columns(2)

        with col_churn:
            st.markdown("#### Churn Summary")
            cc1, cc2 = st.columns(2)
            cc1.metric("Churned", f"{label_result.n_churned:,}",
                       delta=f"-{label_result.churn_rate:.1%}", delta_color="inverse")
            cc2.metric("Active", f"{label_result.n_active:,}",
                       delta=f"+{1 - label_result.churn_rate:.1%}", delta_color="normal")
            st.caption(
                f"Window: {churn_window} days · "
                f"Reference date: {label_result.reference_date.date()}"
            )

        with col_rfm:
            st.markdown("#### RFM Summary")
            if rfm_result is not None:
                feat = rfm_result.features
                rc1, rc2 = st.columns(2)
                rc1.metric("Avg Recency", f"{feat['recency_days'].mean():.0f} days")
                rc2.metric("Avg Frequency", f"{feat['frequency'].mean():.1f}×")
                rc1.metric("Avg Total Spend", f"${feat['monetary_total'].mean():,.0f}")
                rc2.metric("Avg AOV", f"${feat['aov'].mean():.2f}")
            else:
                st.info("Not enough customers to compute RFM features.", icon="ℹ️")

# ─────────────────────────────────────────────────────────────────────────────
# PAGES: COMING IN LATER PHASES
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.title(page or "")
    st.info("This section is under active development. Check back soon.", icon="🔄")

_logger.info("page_rendered", page=page)
