"""
Plotly/Streamlit rendering helpers for the Analytics page.

All st.* and plotly calls are isolated here so business logic
stays testable without a running Streamlit instance.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analytics.cohort import CohortResult
from src.analytics.kpis import KPISnapshot, KPITimeSeries

# Shared chart layout defaults
_LAYOUT_BASE: dict = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=24, b=0),
    font=dict(size=12),
    hovermode="x unified",
)


def render_kpi_strip(snapshot: KPISnapshot) -> None:
    """Four headline metrics in a single row."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Monthly Revenue",
        f"${snapshot.mrr:,.0f}",
        help="Total subscription payments collected in the most recent complete month. Also called MRR (Monthly Recurring Revenue).",
    )
    col2.metric(
        "Active Subscribers",
        f"{snapshot.active_subscribers:,}",
        help="Unique customers who made at least one payment in the most recent month.",
    )
    col3.metric(
        "Revenue per Customer",
        f"${snapshot.arpu:.2f}",
        help="Monthly revenue divided by the number of active subscribers. Also called ARPU (Average Revenue Per User).",
    )
    col4.metric(
        "Monthly Churn Rate",
        f"{snapshot.monthly_churn_rate:.1f}%",
        delta=f"{snapshot.monthly_churn_rate:.1f}%",
        delta_color="inverse",
        help="Percentage of last month's subscribers who did not make a payment this month. Measures how fast the business is losing customers.",
    )


def render_revenue_trend(kpi_ts: KPITimeSeries) -> None:
    """Area-line chart of monthly revenue."""
    revenue = kpi_ts.monthly_revenue
    x = [str(p) for p in revenue.index]
    y = list(revenue.values)

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color="#4F8EF7", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(79,142,247,0.12)",
            name="Revenue",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        height=260,
        showlegend=False,
        yaxis=dict(tickprefix="$", tickformat=",.0f"),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_churn_trend(kpi_ts: KPITimeSeries) -> None:
    """Area-line chart of monthly churn rate."""
    churn = kpi_ts.monthly_churn_rate
    if len(churn) == 0:
        st.caption("Need ≥ 2 months of data to compute churn trend.")
        return

    x = [str(p) for p in churn.index]
    y = list(churn.values)

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color="#EF4444", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.10)",
            name="Churn %",
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        height=260,
        showlegend=False,
        yaxis=dict(ticksuffix="%"),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cohort_heatmap(cohort_result: CohortResult, max_periods: int = 13) -> None:
    """
    Colour-coded retention heatmap.

    Rows  = cohort months (most recent at top).
    Cols  = period offsets (Month 0 … Month N).
    Colour scale: red → yellow → green (low → high retention).
    """
    retention = cohort_result.retention_matrix.copy()

    # Cap columns to max_periods for readability
    period_cols = [c for c in retention.columns if c.startswith("Month ")][:max_periods]
    retention = retention[period_cols].sort_index(ascending=False)

    z = retention.values.astype(float)
    x = list(retention.columns)
    y = list(retention.index)

    # Cell annotations — blank where there is no data yet (future periods)
    text_vals = [
        [f"{v:.0f}%" if not np.isnan(v) and v > 0 else "" for v in row]
        for row in z
    ]

    z_display = np.where(z == 0, np.nan, z)

    fig = go.Figure(
        go.Heatmap(
            z=z_display,
            x=x,
            y=y,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=[
                [0.0, "#FEE2E2"],
                [0.4, "#FEF3C7"],
                [0.7, "#D1FAE5"],
                [1.0, "#065F46"],
            ],
            zmin=0,
            zmax=100,
            showscale=True,
            colorbar=dict(
                title=dict(text="Retention %", side="right"),
                ticksuffix="%",
                thickness=12,
            ),
            hoverongaps=False,
            hovertemplate="Cohort %{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        height=max(320, len(y) * 32 + 80),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{cohort_result.n_cohorts} cohorts · {len(period_cols)} periods shown · "
        "Green = high retention · Red = low retention"
    )
