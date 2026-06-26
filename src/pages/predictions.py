from __future__ import annotations

import pandas as pd
import streamlit as st

from src.export import build_at_risk_csv, build_clv_csv, build_full_predictions_csv
from src.utils.log import get_logger
from src.visualization.predictions import (
    render_at_risk_table,
    render_clv_distribution,
    render_model_metrics,
    render_roc_curve,
    render_segment_distribution,
    render_shap_importance,
)

_logger = get_logger(__name__)


def _train_model_pipeline(df: pd.DataFrame, churn_window_days: int) -> None:
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
            "churn_window_days": churn_window_days,
            "split_info": {
                "n_train": len(split.X_train),
                "n_test": len(split.X_test),
                "cutoff_date": split.cutoff_date,
            },
        }
        st.session_state.pop("insights_report", None)
        st.session_state.pop("insights_churn_window", None)
        st.success("✅ Model trained successfully!")

    except Exception as exc:
        st.error(f"**Model training failed:** {exc}")
        _logger.exception("model_training_failed", error=str(exc))


def render_predictions_page() -> None:
    st.title("🤖 Predictions")
    st.markdown("*ML-powered churn risk scoring with SHAP explainability.*")

    df = st.session_state.get("clean_df")
    if df is None or len(df) == 0:
        st.info("Upload data first to train the churn model.", icon="📤")
        return

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
            "**What the model does:** Scores every customer by churn probability, "
            "identifies the behaviors driving risk, estimates future revenue per customer, "
            "and groups customers into five action-ready segments."
        )
        if col_btn.button("🚀 Train Churn Model", type="primary", use_container_width=True):
            _train_model_pipeline(df, churn_window)
            st.rerun()
        return

    split_info = model_results["split_info"]
    trained_window = model_results.get("churn_window_days", churn_window)
    if trained_window != churn_window:
        st.warning(
            f"⚠️ Model was trained with a **{trained_window}-day** churn window but the "
            f"sidebar is now set to **{churn_window} days**. Click **Retrain Model** to realign.",
            icon="🔄",
        )
    else:
        st.caption(f"📊 Trained with a **{trained_window}-day** churn window — adjust in ⚙️ Settings")
    st.caption(
        f"Trained on {split_info['n_train']:,} customers · "
        f"Test cutoff: {split_info['cutoff_date'].date()} · "
        f"Test set: {split_info['n_test']:,} customers"
    )

    render_model_metrics(model_results["metrics"])

    st.divider()
    col_roc, col_shap = st.columns(2)
    with col_roc:
        st.markdown("#### How Well Does the Model Predict?")
        st.caption("The curve shows how the model trades off catching all churners vs. avoiding false alarms. Higher and further left = better.")
        render_roc_curve(model_results["metrics"])
    with col_shap:
        st.markdown("#### What's Driving Churn Risk?")
        st.caption("Which customer behaviors most strongly predict whether someone will cancel.")
        render_shap_importance(model_results["shap"])
        st.caption(
            "**Note on recency:** Days-since-last-purchase is expected to rank highly here "
            "because churn is defined as inactivity beyond the selected window. "
            "This is by design — recency is the primary behavioural signal for subscription churn."
        )

    st.divider()
    st.markdown("#### Top 20 At-Risk Customers")
    render_at_risk_table(
        model_results["predictions"],
        model_results["rfm_features"],
    )
    st.download_button(
        "⬇️ Download At-Risk Customers (CSV)",
        data=build_at_risk_csv(model_results["predictions"], model_results["rfm_features"]),
        file_name="at_risk_customers.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    col_seg, col_clv = st.columns(2)
    with col_seg:
        st.markdown("#### Segment Distribution")
        render_segment_distribution(model_results["segments"])
    with col_clv:
        st.markdown("#### Customer Lifetime Value")
        st.caption("How much revenue each customer is expected to generate before they leave.")
        clv = model_results["clv"]
        rc1, rc2 = st.columns(2)
        rc1.metric(
            "Median Customer Lifetime",
            f"{clv.median_lifetime_days:.0f} days",
            help="Half of customers stay longer than this, half leave sooner.",
        )
        rc2.metric(
            "Avg Projected Value",
            f"${clv.clv_per_customer['expected_clv'].mean():,.0f}",
            help="Expected total revenue per customer over their remaining lifetime.",
        )
        rc1.metric(
            "Avg Remaining Lifetime",
            f"{clv.clv_per_customer['expected_remaining_months'].mean():.1f} months",
            help="Average number of months the typical active customer is expected to remain subscribed.",
        )
        rc2.metric(
            "Highest-Value Customer",
            f"${clv.clv_per_customer['expected_clv'].max():,.0f}",
            help="The single highest projected lifetime value among all scored customers.",
        )
        st.download_button(
            "⬇️ Download CLV Table (CSV)",
            data=build_clv_csv(clv.clv_per_customer),
            file_name="customer_lifetime_value.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    st.markdown("#### CLV Distribution")
    render_clv_distribution(model_results["clv"].clv_per_customer)

    st.divider()
    col_export, col_retrain = st.columns([3, 1])
    col_export.download_button(
        "⬇️ Download Full Predictions (CSV)",
        data=build_full_predictions_csv(model_results["predictions"], model_results["rfm_features"]),
        file_name="churn_predictions_full.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if col_retrain.button("🔄 Retrain Model", type="secondary", use_container_width=True):
        st.session_state.pop("model_results", None)
        st.rerun()
