from __future__ import annotations

import pandas as pd
import streamlit as st

from src.visualization.quality_report import render_quality_report


def render_quality_page() -> None:
    st.title("🔍 Data Quality Report")

    if not st.session_state.get("quality_report"):
        st.info("Upload data first to see quality results.", icon="📤")
        return

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
