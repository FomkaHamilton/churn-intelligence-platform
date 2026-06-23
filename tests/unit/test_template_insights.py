"""Unit tests for the template insight client."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analytics.kpis import KPICalculator
from src.feature_engineering.churn_labels import ChurnLabelBuilder
from src.feature_engineering.rfm import RFMBuilder
from src.insights.factory import get_insight_client
from src.insights.models import InsightData, InsightReport
from src.insights.template_client import TemplateInsightClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_transaction_df(n_customers: int = 60, n_months: int = 12) -> pd.DataFrame:
    """All customers are active — for tests that don't need ML training."""
    rows = []
    for i in range(n_customers):
        for m in range(n_months):
            rows.append({
                "customer_id": f"c{i}",
                "transaction_date": date(2023, 1, 1) + timedelta(days=m * 30 + i % 28),
                "transaction_amount": 50.0 + i * 2,
            })
    return pd.DataFrame(rows)


def _make_churn_df(n: int = 200, n_months: int = 18) -> pd.DataFrame:
    """
    200 customers staggered by 1 day; ~25% churn early (2 transactions only).
    Ensures both train and test sets contain mixed labels for model training.
    """
    rows = []
    for i in range(n):
        start = date(2022, 1, 1) + timedelta(days=i)
        months = 2 if (i % 4 == 0) else n_months
        for m in range(months):
            rows.append({
                "customer_id": f"c{i}",
                "transaction_date": start + timedelta(days=m * 30),
                "transaction_amount": 50.0 + (i % 100),
            })
    return pd.DataFrame(rows)


def _make_minimal_data(n_customers: int = 60, n_months: int = 12) -> InsightData:
    df = _make_transaction_df(n_customers, n_months)
    kpi_ts = KPICalculator().calculate(df)
    rfm = RFMBuilder().build(df)
    labels = ChurnLabelBuilder().build(df)
    return InsightData(
        kpi_snapshot=kpi_ts.snapshot,
        kpi_ts=kpi_ts,
        churn_label_result=labels,
        rfm_result=rfm,
    )


def _make_data_with_model() -> InsightData:
    """InsightData with SHAP, segments, CLV, and forecast populated."""
    from src.modeling.churn_model import ChurnModel
    from src.modeling.clv import CLVModel
    from src.modeling.explainability import SHAPExplainer
    from src.modeling.features import FeatureMatrixBuilder
    from src.modeling.segmentation import CustomerSegmenter
    from src.modeling.validation import TimeSeriesChurnSplit
    from src.forecasting.pipeline import ForecastingPipeline

    df = _make_churn_df()
    kpi_ts = KPICalculator().calculate(df)
    rfm = RFMBuilder().build(df)
    labels = ChurnLabelBuilder().build(df)

    fm = FeatureMatrixBuilder().build(df)
    split = TimeSeriesChurnSplit().split(fm.X, fm.y, fm.customer_ids, fm.first_tx_dates)
    model = ChurnModel()
    model.train(split.X_train, split.y_train)
    metrics = model.evaluate(split.X_test, split.y_test)
    shap_result = SHAPExplainer().explain(model.rf, split.X_test)
    proba = model.predict_proba(fm.X)
    churn_prob_s = pd.Series(proba, index=fm.customer_ids.values)
    segments = CustomerSegmenter().segment(fm.rfm_features, fm.churn_label_df, churn_prob_s)
    clv = CLVModel().fit_and_predict(fm.rfm_features, fm.churn_label_df)
    bundle = ForecastingPipeline().run(df, horizon_months=6)

    return InsightData(
        kpi_snapshot=kpi_ts.snapshot,
        kpi_ts=kpi_ts,
        churn_label_result=labels,
        rfm_result=rfm,
        model_metrics=metrics,
        shap_result=shap_result,
        segments=segments,
        clv_result=clv,
        forecast_bundle=bundle,
    )


# ── Report structure tests ────────────────────────────────────────────────────

class TestReportStructure:
    def setup_method(self) -> None:
        self.client = TemplateInsightClient()

    def test_returns_insight_report(self) -> None:
        data = _make_minimal_data()
        result = self.client.generate(data)
        assert isinstance(result, InsightReport)

    def test_all_required_fields_non_empty(self) -> None:
        data = _make_minimal_data()
        result = self.client.generate(data)
        assert len(result.health_summary) > 0
        assert len(result.churn_analysis) > 0
        assert len(result.revenue_outlook) > 0
        assert len(result.customer_segments) > 0
        assert len(result.recommendations) > 0

    def test_client_type_is_template(self) -> None:
        data = _make_minimal_data()
        result = self.client.generate(data)
        assert result.client_type == "template"

    def test_model_confidence_none_without_model(self) -> None:
        data = _make_minimal_data()
        result = self.client.generate(data)
        assert result.model_confidence is None

    def test_model_confidence_populated_with_model(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        assert result.model_confidence is not None
        assert len(result.model_confidence) > 0

    def test_sections_with_ml_flag(self) -> None:
        data = _make_minimal_data()
        report = self.client.generate(data)
        assert report.sections_with_ml is False

        data_full = _make_data_with_model()
        report_full = self.client.generate(data_full)
        assert report_full.sections_with_ml is True


# ── Health summary tests ──────────────────────────────────────────────────────

class TestHealthSummary:
    def setup_method(self) -> None:
        self.client = TemplateInsightClient()

    def _data_with_churn_rate(self, target_cr: float) -> InsightData:
        """Create data with approximately the target churn rate."""
        from src.analytics.kpis import KPICalculator
        from src.feature_engineering.churn_labels import ChurnLabelBuilder, ChurnLabelResult
        from src.feature_engineering.rfm import RFMBuilder
        import datetime

        n = 100
        rows = []
        for i in range(n):
            rows.append({
                "customer_id": f"c{i}",
                "transaction_date": datetime.date(2023, 1, 15),
                "transaction_amount": 50.0,
            })
        # Force some customers to churn by giving them an old date
        n_churn = max(1, int(n * target_cr))
        churn_rows = []
        for i in range(n_churn):
            churn_rows.append({
                "customer_id": f"ch{i}",
                "transaction_date": datetime.date(2022, 1, 1),
                "transaction_amount": 50.0,
            })
        df = pd.DataFrame(rows + churn_rows)
        kpi_ts = KPICalculator().calculate(df)
        rfm = RFMBuilder().build(df)
        labels = ChurnLabelBuilder().build(df, churn_window_days=90)
        return InsightData(
            kpi_snapshot=kpi_ts.snapshot,
            kpi_ts=kpi_ts,
            churn_label_result=labels,
            rfm_result=rfm,
        )

    def test_low_churn_mentions_strong_retention(self) -> None:
        data = self._data_with_churn_rate(0.02)
        result = self.client.generate(data)
        lower = result.health_summary.lower()
        assert "strong" in lower or "fewer than 1 in" in lower

    def test_high_churn_mentions_urgent_or_elevated(self) -> None:
        data = self._data_with_churn_rate(0.40)
        result = self.client.generate(data)
        lower = result.health_summary.lower()
        assert "high" in lower or "urgent" in lower or "elevated" in lower

    def test_health_summary_contains_mrr(self) -> None:
        data = _make_minimal_data()
        result = self.client.generate(data)
        assert "$" in result.health_summary

    def test_health_summary_contains_active_subscriber_count(self) -> None:
        data = _make_minimal_data()
        result = self.client.generate(data)
        snap = data.kpi_snapshot
        assert str(snap.active_subscribers) in result.health_summary.replace(",", "")


# ── Churn analysis tests ──────────────────────────────────────────────────────

class TestChurnAnalysis:
    def setup_method(self) -> None:
        self.client = TemplateInsightClient()

    def test_without_shap_mentions_transaction_count(self) -> None:
        data = _make_minimal_data()
        result = self.client.generate(data)
        assert "customers" in result.churn_analysis.lower()

    def test_without_shap_mentions_predictions_page(self) -> None:
        data = _make_minimal_data()
        result = self.client.generate(data)
        assert "predictions" in result.churn_analysis.lower()

    def test_with_shap_uses_feature_plain_name(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        plain_names = [
            "time since last purchase",
            "purchase frequency",
            "total lifetime spend",
            "average order value",
            "length of customer relationship",
            "monthly transaction rate",
            "consistency of purchase timing",
        ]
        lower = result.churn_analysis.lower()
        assert any(name in lower for name in plain_names)

    def test_with_shap_no_raw_feature_names(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        # Raw feature names should be translated — recency_days should not appear verbatim
        assert "recency_days" not in result.churn_analysis

    def test_churn_analysis_contains_churn_rate(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        assert "%" in result.churn_analysis


# ── Revenue outlook tests ─────────────────────────────────────────────────────

class TestRevenueOutlook:
    def setup_method(self) -> None:
        self.client = TemplateInsightClient()

    def test_without_forecast_mentions_forecasting_page(self) -> None:
        data = _make_minimal_data()
        result = self.client.generate(data)
        assert "forecasting" in result.revenue_outlook.lower()

    def test_with_forecast_mentions_horizon(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        assert "6-month" in result.revenue_outlook or "month" in result.revenue_outlook.lower()

    def test_with_forecast_contains_dollar_amount(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        assert "$" in result.revenue_outlook

    def test_revenue_outlook_non_empty_both_modes(self) -> None:
        for data in [_make_minimal_data(), _make_data_with_model()]:
            result = self.client.generate(data)
            assert len(result.revenue_outlook) > 20


# ── Customer segments tests ───────────────────────────────────────────────────

class TestCustomerSegments:
    def setup_method(self) -> None:
        self.client = TemplateInsightClient()

    def test_without_segments_mentions_predictions_page(self) -> None:
        data = _make_minimal_data()
        result = self.client.generate(data)
        assert "predictions" in result.customer_segments.lower()

    def test_with_segments_shows_total_count(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        total = len(data.segments)
        assert str(total) in result.customer_segments.replace(",", "")

    def test_with_segments_mentions_at_least_one_segment_name(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        lower = result.customer_segments.lower()
        segments = ["churned", "at risk", "new", "high value", "healthy"]
        assert any(s in lower for s in segments)


# ── Recommendations tests ─────────────────────────────────────────────────────

class TestRecommendations:
    def setup_method(self) -> None:
        self.client = TemplateInsightClient()

    def test_always_produces_three_recommendations(self) -> None:
        for data in [_make_minimal_data(), _make_data_with_model()]:
            result = self.client.generate(data)
            assert result.recommendations.count("**1.**") == 1
            assert result.recommendations.count("**2.**") == 1
            assert result.recommendations.count("**3.**") == 1

    def test_recommendations_non_empty(self) -> None:
        data = _make_minimal_data()
        result = self.client.generate(data)
        assert len(result.recommendations) > 50

    def test_with_clv_mentions_top_tier(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        lower = result.recommendations.lower()
        assert "top" in lower or "high value" in lower or "clv" in lower


# ── Model confidence tests ────────────────────────────────────────────────────

class TestModelConfidence:
    def setup_method(self) -> None:
        self.client = TemplateInsightClient()

    def test_contains_auc_value(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        assert result.model_confidence is not None
        auc = data.model_metrics.auc
        assert f"{auc:.3f}" in result.model_confidence

    def test_contains_train_test_counts(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        assert result.model_confidence is not None
        m = data.model_metrics
        assert str(m.n_train) in result.model_confidence.replace(",", "")

    def test_high_auc_describes_excellent_or_strong(self) -> None:
        data = _make_data_with_model()
        result = self.client.generate(data)
        if result.model_confidence and data.model_metrics.auc >= 0.80:
            lower = result.model_confidence.lower()
            assert "excellent" in lower or "strong" in lower or "good" in lower


# ── Factory tests ─────────────────────────────────────────────────────────────

class TestFactory:
    def test_no_keys_returns_template_client(self) -> None:
        client = get_insight_client(anthropic_api_key=None, openai_api_key=None)
        assert isinstance(client, TemplateInsightClient)

    def test_factory_client_generates_report(self) -> None:
        client = get_insight_client()
        data = _make_minimal_data()
        result = client.generate(data)
        assert isinstance(result, InsightReport)
