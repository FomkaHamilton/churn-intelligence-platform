"""
Plotly/Streamlit rendering for the Forecasting page.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.forecasting.base import ForecastResult

_LAYOUT_BASE: dict = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=28, b=0),
    font=dict(size=12),
    hovermode="x unified",
)


def render_forecast_chart(
    result: ForecastResult,
    *,
    title: str = "",
    color: str = "#4F8EF7",
    y_prefix: str = "",
    y_suffix: str = "",
) -> None:
    """
    Combined historical + forecast area chart with confidence band.

    Historical data is shown as a solid line; the forecast extends it
    with a dashed line and a shaded 80 % prediction interval.
    """
    hist_x = list(result.historical.index)
    hist_y = list(result.historical.values)

    fc_x = list(result.forecast.index)
    fc_y = list(result.forecast.values)
    lo_y = list(result.lower_bound.values)
    hi_y = list(result.upper_bound.values)

    # Join historical and forecast for a continuous visual line
    join_x = [hist_x[-1]] + fc_x
    join_y = [hist_y[-1]] + fc_y
    join_lo = [hist_y[-1]] + lo_y
    join_hi = [hist_y[-1]] + hi_y

    fig = go.Figure()

    # Confidence band (filled region)
    fig.add_trace(go.Scatter(
        x=join_x + join_x[::-1],
        y=join_hi + join_lo[::-1],
        fill="toself",
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.12,)}",
        line=dict(width=0),
        showlegend=True,
        name="80% interval",
        hoverinfo="skip",
    ))

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_x,
        y=hist_y,
        mode="lines",
        line=dict(color=color, width=2),
        name="Historical",
        hovertemplate=f"{y_prefix}%{{y:,.0f}}{y_suffix}<extra>Historical</extra>",
    ))

    # Forecast line (dashed)
    fig.add_trace(go.Scatter(
        x=join_x,
        y=join_y,
        mode="lines",
        line=dict(color=color, width=2, dash="dash"),
        name="Forecast",
        hovertemplate=f"{y_prefix}%{{y:,.0f}}{y_suffix}<extra>Forecast</extra>",
    ))

    # Vertical separator between historical and forecast
    if hist_x:
        fig.add_vline(
            x=hist_x[-1],
            line_dash="dot",
            line_color="#6B7280",
            annotation_text="forecast →",
            annotation_position="top right",
            annotation_font_size=11,
        )

    tick_format = ",.0f"
    fig.update_layout(
        **_LAYOUT_BASE,
        height=320,
        title=dict(text=title, font_size=14) if title else None,
        xaxis=dict(tickangle=-30),
        yaxis=dict(
            tickprefix=y_prefix,
            ticksuffix=y_suffix,
            tickformat=tick_format,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{result.horizon_months}-month forecast · "
        f"Backend: {result.backend} · "
        "Shaded region = 80% prediction interval"
    )


def render_forecast_metrics(revenue: ForecastResult, subscribers: ForecastResult) -> None:
    """Four-column KPI strip summarising the forecast horizon."""
    total_fc_rev = float(revenue.forecast.sum())
    avg_fc_rev = float(revenue.forecast.mean())
    end_subs = float(subscribers.forecast.iloc[-1])
    avg_subs = float(subscribers.forecast.mean())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Forecast {revenue.horizon_months}m Revenue", f"${total_fc_rev:,.0f}")
    col2.metric("Avg Monthly Revenue", f"${avg_fc_rev:,.0f}")
    col3.metric(f"Subscribers (month {subscribers.horizon_months})", f"{end_subs:,.0f}")
    col4.metric("Avg Monthly Subscribers", f"{avg_subs:,.0f}")
