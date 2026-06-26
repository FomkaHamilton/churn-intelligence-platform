"""
Churn Intelligence Platform — Streamlit entry point.

Session state keys used across pages:
  raw_df                : DataFrame straight from upload (pre-quality-fix)
  clean_df              : DataFrame after schema + quality fixes
  quality_report        : DataQualityReport from the last validation run
  filename              : Name of the uploaded file
  churn_window_days     : User-selected churn window (default 90)
  model_results         : Dict of ML artifacts (populated by Predictions page)
  insights_report       : Cached InsightReport (populated by Insights page)
  insights_churn_window : Churn window in effect when insights were last generated
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config.settings import get_settings
from src.utils.log import configure_logging, get_logger
from src.pages.overview import render_overview_page
from src.pages.upload import render_upload_page
from src.pages.quality import render_quality_page
from src.pages.analytics import render_analytics_page
from src.pages.predictions import render_predictions_page
from src.pages.forecasting import render_forecasting_page
from src.pages.insights import render_insights_tab

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
            help=(
                "How many days without a transaction before a customer is considered 'churned'. "
                "This setting affects: churn labels on the Analytics page, the Predictions model (retrain to apply), and the AI Insights report. "
                "It does NOT change revenue trends, cohort retention, or subscriber counts."
            ),
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

    st.divider()
    st.caption("**Pipeline progress**")
    _steps = [
        ("📂 Data loaded", st.session_state.get("clean_df") is not None),
        ("🤖 Model trained", st.session_state.get("model_results") is not None),
        ("💡 Insights ready", st.session_state.get("insights_report") is not None),
    ]
    for _label, _done in _steps:
        st.caption(f"{'✅' if _done else '⬜'} {_label}")

# ── Page routing ──────────────────────────────────────────────────────────────
if "🏠" in (page or ""):
    render_overview_page(_settings)
elif "📤" in (page or ""):
    render_upload_page(_settings)
elif "🔍" in (page or ""):
    render_quality_page()
elif "📈" in (page or ""):
    render_analytics_page()
elif "🤖" in (page or ""):
    render_predictions_page()
elif "🔮" in (page or ""):
    render_forecasting_page()
elif "💡" in (page or ""):
    render_insights_tab(_settings)
else:
    st.title(page or "")
    st.info("This section is under active development. Check back soon.", icon="🔄")

_logger.info("page_rendered", page=page)
