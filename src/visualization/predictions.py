"""
Plotly/Streamlit rendering for the Predictions page.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.modeling.churn_model import ModelMetrics
from src.modeling.explainability import SHAPResult

_SEGMENT_COLORS: dict[str, str] = {
    "New":        "#60A5FA",
    "Healthy":    "#34D399",
    "High Value": "#FBBF24",
    "At Risk":    "#F97316",
    "Churned":    "#EF4444",
}

_LAYOUT_BASE: dict = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=28, b=0),
    font=dict(size=12),
)


def render_model_metrics(metrics: ModelMetrics) -> None:
    """Four-column metric strip."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", f"{metrics.auc:.3f}",
                delta="good" if metrics.auc >= 0.75 else "needs improvement",
                delta_color="normal" if metrics.auc >= 0.75 else "inverse")
    col2.metric("Precision", f"{metrics.precision:.3f}")
    col3.metric("Recall",    f"{metrics.recall:.3f}")
    col4.metric("F1 Score",  f"{metrics.f1:.3f}")
    st.caption(
        f"Trained on {metrics.n_train:,} customers · "
        f"Evaluated on {metrics.n_test:,} held-out customers"
    )


def render_roc_curve(metrics: ModelMetrics) -> None:
    """ROC curve with AUC annotation and random-baseline diagonal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="#9CA3AF", width=1),
        name="Random (AUC = 0.50)",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=metrics.fpr.tolist(),
        y=metrics.tpr.tolist(),
        mode="lines",
        line=dict(color="#4F8EF7", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.10)",
        name=f"Model (AUC = {metrics.auc:.3f})",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=300,
        xaxis=dict(title="False Positive Rate", range=[0, 1]),
        yaxis=dict(title="True Positive Rate", range=[0, 1]),
        legend=dict(x=0.55, y=0.08),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap_importance(shap_result: SHAPResult, top_n: int = 7) -> None:
    """Horizontal bar chart of mean |SHAP| importance (top N features)."""
    df = shap_result.feature_importance.head(top_n).sort_values("mean_abs_shap")
    fig = go.Figure(go.Bar(
        x=df["mean_abs_shap"].tolist(),
        y=df["feature_name"].tolist(),
        orientation="h",
        marker_color="#4F8EF7",
        text=[f"{v:.4f}" for v in df["mean_abs_shap"]],
        textposition="outside",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=max(220, top_n * 38),
        xaxis=dict(title="Mean |SHAP value|"),
        yaxis=dict(tickfont=dict(size=12)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Higher SHAP = stronger influence on churn prediction for this feature.")


def render_at_risk_table(
    predictions: pd.DataFrame,
    rfm_features: pd.DataFrame,
    n: int = 20,
) -> None:
    """
    Top-N customers by churn probability with supporting context.
    predictions: customer_id | churn_probability
    rfm_features: customer_id | monetary_total | recency_days | frequency
    """
    top = (
        predictions
        .sort_values("churn_probability", ascending=False)
        .head(n)
        .merge(
            rfm_features[["customer_id", "monetary_total", "recency_days", "frequency"]],
            on="customer_id",
            how="left",
        )
    )
    top["Churn Risk"] = (top["churn_probability"] * 100).map("{:.1f}%".format)
    top["Total Spend"] = top["monetary_total"].map("${:,.0f}".format)
    top["Days Inactive"] = top["recency_days"].astype(int)
    top["Purchases"] = top["frequency"].astype(int)

    # Warn if customer IDs look like email addresses (potential PII in a shared view)
    sample_ids = top["customer_id"].head(10).astype(str)
    if sample_ids.str.contains("@", na=False).any():
        st.warning(
            "⚠️ **Privacy notice:** Customer IDs appear to contain email addresses. "
            "Consider anonymising your data before sharing this view publicly.",
            icon="🔒",
        )

    st.dataframe(
        top[["customer_id", "Churn Risk", "Total Spend", "Days Inactive", "Purchases"]]
        .rename(columns={"customer_id": "Customer ID"}),
        use_container_width=True,
        hide_index=True,
    )


def render_clv_distribution(clv_per_customer: pd.DataFrame) -> None:
    """Histogram of expected CLV across all customers."""
    values = clv_per_customer["expected_clv"].clip(lower=0)
    fig = go.Figure(go.Histogram(
        x=values,
        nbinsx=30,
        marker_color="#4F8EF7",
        opacity=0.85,
        hovertemplate="CLV ~$%{x:.0f} — %{y} customers<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=260,
        xaxis=dict(title="Expected CLV ($)", tickprefix="$"),
        yaxis=dict(title="Customers"),
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_segment_distribution(segments: pd.Series) -> None:
    """Donut chart showing distribution of customer segments."""
    counts = segments.value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [_SEGMENT_COLORS.get(lbl, "#6B7280") for lbl in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.52,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="percent+label",
        hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
