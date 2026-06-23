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
from src.visualization.predictions import (
    render_at_risk_table,
    render_model_metrics,
    render_roc_curve,
    render_segment_distribution,
    render_shap_importance,
)
from src.visualization.forecasting import render_forecast_chart, render_forecast_metrics

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


def _train_model_pipeline(df: pd.DataFrame, churn_window_days: int) -> None:
    """Run the full ML pipeline and store results in st.session_state."""
    from src.modeling.churn_model import ChurnModel
    from src.modeling.clv import CLVModel
    from src.modeling.explainability import SHAPExplainer
    from src.modeling.features import FeatureMatrixBuilder
    from src.modeling.segmentation import CustomerSegmenter
    from src.modeling.validation import TimeSeriesChurnSplit

    try:
        with st.spinner("Building feature matrix…"):
            fm = FeatureMatrixBuilder().build(df, churn_window_days=churn_window_days)

        with st.spinner("Temporal train/test split…"):
            split = TimeSeriesChurnSplit().split(
                fm.X, fm.y, fm.customer_ids, fm.first_tx_dates
            )

        with st.spinner("Training LR + RF ensemble…"):
            model = ChurnModel()
            model.train(split.X_train, split.y_train)
            metrics = model.evaluate(split.X_test, split.y_test)

        with st.spinner("Computing SHAP values…"):
            shap_result = SHAPExplainer().explain(model.rf, split.X_test)

        with st.spinner("Scoring all customers…"):
            proba = model.predict_proba(fm.X)
            predictions = pd.DataFrame({
                "customer_id": fm.customer_ids.values,
                "churn_probability": proba,
            })

        with st.spinner("Kaplan-Meier CLV…"):
            clv_result = CLVModel().fit_and_predict(fm.rfm_features, fm.churn_label_df)

        with st.spinner("Segmenting customers…"):
            churn_prob_s = pd.Series(proba, index=fm.customer_ids.values)
            segments = CustomerSegmenter().segment(
                fm.rfm_features, fm.churn_label_df, churn_prob_s
            )

        st.session_state["model_results"] = {
            "model": model,
            "metrics": metrics,
            "shap": shap_result,
            "predictions": predictions,
            "rfm_features": fm.rfm_features,
            "clv": clv_result,
            "segments": segments,
            "split_info": {
                "n_train": len(split.X_train),
                "n_test": len(split.X_test),
                "cutoff_date": split.cutoff_date,
            },
        }
        st.success("✅ Model trained successfully!")

    except Exception as exc:
        st.error(f"**Model training failed:** {exc}")
        _logger.exception("model_training_failed", error=str(exc))


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
    col2.metric("Build Phase", "5 / 9")
    col3.metric("Churn Window", f"{st.session_state['churn_window_days']}d")
    col4.metric("AI Mode", "Live" if _settings.has_ai_provider else "Template")

    st.divider()
    st.markdown("### Platform capabilities")
    capabilities = {
        "📤 Data Ingestion": ("CSV/XLSX upload, validation, quality profiling", "✅ Phase 2 — Live"),
        "📈 Cohort Analytics": ("Retention matrices, MRR, ARPU, churn rate", "✅ Phase 3 — Live"),
        "🤖 Churn Prediction": ("ML risk scoring with SHAP explainability", "✅ Phase 4 — Live"),
        "💰 CLV Modeling": ("Survival-analysis-based customer lifetime value", "✅ Phase 4 — Live"),
        "🔮 Revenue Forecasting": ("12-month subscriber and revenue forecasts", "✅ Phase 5 — Live"),
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
# PAGE: PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
elif "🤖" in (page or ""):
    st.title("🤖 Predictions")
    st.markdown("*ML-powered churn risk scoring with SHAP explainability.*")

    df = st.session_state.get("clean_df")
    if df is None or len(df) == 0:
        st.info("Upload data first to train the churn model.", icon="📤")
    else:
        churn_window = int(st.session_state["churn_window_days"])
        model_results = st.session_state.get("model_results")

        if model_results is None:
            col_info, col_btn = st.columns([3, 1])
            col_info.info(
                "The churn prediction model has not been trained yet. "
                "Click **Train Churn Model** to begin.",
                icon="🤖",
            )
            col_info.markdown(
                "**What trains:**  LR + RF ensemble · Temporal split · SHAP · CLV · Segmentation"
            )
            if col_btn.button("🚀 Train Churn Model", type="primary", use_container_width=True):
                _train_model_pipeline(df, churn_window)
                st.rerun()
        else:
            split_info = model_results["split_info"]
            st.caption(
                f"Trained on {split_info['n_train']:,} customers · "
                f"Test cutoff: {split_info['cutoff_date'].date()} · "
                f"Test set: {split_info['n_test']:,} customers"
            )

            # ── Metrics strip ─────────────────────────────────────────────
            render_model_metrics(model_results["metrics"])

            st.divider()
            # ── ROC curve + SHAP importance ───────────────────────────────
            col_roc, col_shap = st.columns(2)
            with col_roc:
                st.markdown("#### ROC Curve")
                render_roc_curve(model_results["metrics"])
            with col_shap:
                st.markdown("#### Feature Importance (SHAP)")
                render_shap_importance(model_results["shap"])

            st.divider()
            # ── At-risk table ─────────────────────────────────────────────
            st.markdown("#### Top 20 At-Risk Customers")
            render_at_risk_table(
                model_results["predictions"],
                model_results["rfm_features"],
            )

            st.divider()
            # ── Segment distribution + CLV summary ────────────────────────
            col_seg, col_clv = st.columns(2)
            with col_seg:
                st.markdown("#### Segment Distribution")
                render_segment_distribution(model_results["segments"])
            with col_clv:
                st.markdown("#### CLV Summary")
                clv = model_results["clv"]
                rc1, rc2 = st.columns(2)
                rc1.metric("Median Lifetime",
                           f"{clv.median_lifetime_days:.0f} days")
                rc2.metric("Avg Expected CLV",
                           f"${clv.clv_per_customer['expected_clv'].mean():,.0f}")
                rc1.metric("Avg Remaining",
                           f"{clv.clv_per_customer['expected_remaining_months'].mean():.1f} mo")
                rc2.metric("Top CLV Customer",
                           f"${clv.clv_per_customer['expected_clv'].max():,.0f}")

            st.divider()
            if st.button("🔄 Retrain Model", type="secondary"):
                st.session_state.pop("model_results", None)
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FORECASTING
# ─────────────────────────────────────────────────────────────────────────────
elif "🔮" in (page or ""):
    from src.forecasting.pipeline import ForecastingPipeline
    from src.config.settings import get_yaml_config

    st.title("🔮 Forecasting")
    st.markdown("*12-month revenue and subscriber projections with confidence intervals.*")

    df = st.session_state.get("clean_df")
    if df is None or len(df) == 0:
        st.info("Upload data first to run forecasts.", icon="📤")
    else:
        yaml_cfg = get_yaml_config()
        backend = yaml_cfg.get("forecasting", {}).get("backend", "statsmodels")
        horizon = int(yaml_cfg.get("forecasting", {}).get("horizon_months", 12))

        @st.cache_data
        def _run_forecast(df: pd.DataFrame, horizon: int, backend: str):
            return ForecastingPipeline().run(df, horizon_months=horizon, backend=backend)

        try:
            with st.spinner("Running forecast…"):
                bundle = _run_forecast(df, horizon, backend)

            render_forecast_metrics(bundle.revenue, bundle.subscribers)

            st.divider()
            st.markdown("#### Revenue Forecast")
            render_forecast_chart(
                bundle.revenue,
                color="#4F8EF7",
                y_prefix="$",
            )

            st.divider()
            st.markdown("#### Subscriber Forecast")
            render_forecast_chart(
                bundle.subscribers,
                color="#34D399",
                y_suffix=" subs",
            )

            st.divider()
            st.caption(
                f"Backend: **{backend}** · Horizon: **{horizon} months** · "
                "Change backend in `config/settings.yaml` → `forecasting.backend`"
            )

        except Exception as exc:
            st.error(f"**Forecasting failed:** {exc}", icon="⚠️")
            st.info(
                "Forecasting requires at least 6 months of transaction history. "
                "Try the sample dataset (Load sample dataset button on the Upload page).",
                icon="💡",
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGES: COMING IN LATER PHASES
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.title(page or "")
    st.info("This section is under active development. Check back soon.", icon="🔄")

_logger.info("page_rendered", page=page)
