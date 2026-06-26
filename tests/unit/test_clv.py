"""Unit tests covering CLV model edge cases and error paths."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.modeling.clv import CLVModel, CLVResult
from src.utils.exceptions import InsufficientDataError


def _make_rfm(n: int, *, tenure_days: int = 180) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "tenure_days": [tenure_days] * n,
        "aov": [50.0] * n,
        "tx_per_month": [1.0] * n,
        "recency_days": [10] * n,
        "monetary_total": [float(tenure_days)] * n,
        "frequency": [6] * n,
        "gap_std_days": [30.0] * n,
    })


def _make_churn(n: int, *, n_churned: int = 3) -> pd.DataFrame:
    is_churned = [True] * n_churned + [False] * (n - n_churned)
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "is_churned": is_churned,
    })


class TestImportGuard:
    def test_raises_import_error_when_lifelines_missing(self) -> None:
        rfm = _make_rfm(15)
        churn = _make_churn(15)
        with patch.dict(sys.modules, {"lifelines": None}), pytest.raises(ImportError, match="lifelines is required"):
            CLVModel().fit_and_predict(rfm, churn)


class TestInsufficientData:
    def test_raises_when_fewer_than_min_customers(self) -> None:
        rfm = _make_rfm(5)
        churn = _make_churn(5)
        with pytest.raises(InsufficientDataError):
            CLVModel().fit_and_predict(rfm, churn)

    def test_raises_at_exactly_min_minus_one(self) -> None:
        n = CLVModel.MIN_CUSTOMERS - 1
        rfm = _make_rfm(n)
        churn = _make_churn(n)
        with pytest.raises(InsufficientDataError):
            CLVModel().fit_and_predict(rfm, churn)

    def test_succeeds_at_exactly_min_customers(self) -> None:
        n = CLVModel.MIN_CUSTOMERS
        rfm = _make_rfm(n)
        churn = _make_churn(n, n_churned=3)
        result = CLVModel().fit_and_predict(rfm, churn)
        assert isinstance(result, CLVResult)


class TestNanMedianProxy:
    def test_proxy_used_when_median_is_nan(self) -> None:
        # Patch KaplanMeierFitter so median_survival_time_ is NaN
        mock_kmf = MagicMock()
        mock_kmf.median_survival_time_ = float("nan")
        mock_sf = MagicMock()
        mock_sf.reset_index.return_value = pd.DataFrame({
            "timeline_days": [0.0, 90.0, 180.0],
            "Customer Lifetime": [1.0, 0.8, 0.6],
        })
        mock_kmf.survival_function_ = mock_sf

        mock_lifelines = MagicMock()
        mock_lifelines.KaplanMeierFitter.return_value = mock_kmf

        rfm = _make_rfm(15, tenure_days=180)
        churn = _make_churn(15, n_churned=3)

        with patch.dict(sys.modules, {"lifelines": mock_lifelines}):
            result = CLVModel().fit_and_predict(rfm, churn)

        # Proxy = mean(tenure_days) * 2 = 180 * 2 = 360
        assert result.median_lifetime_days == pytest.approx(360.0)

    def test_proxy_used_when_median_is_zero(self) -> None:
        mock_kmf = MagicMock()
        mock_kmf.median_survival_time_ = 0.0
        mock_sf = MagicMock()
        mock_sf.reset_index.return_value = pd.DataFrame({
            "timeline_days": [0.0],
            "Customer Lifetime": [1.0],
        })
        mock_kmf.survival_function_ = mock_sf

        mock_lifelines = MagicMock()
        mock_lifelines.KaplanMeierFitter.return_value = mock_kmf

        rfm = _make_rfm(15, tenure_days=90)
        churn = _make_churn(15, n_churned=3)

        with patch.dict(sys.modules, {"lifelines": mock_lifelines}):
            result = CLVModel().fit_and_predict(rfm, churn)

        # Proxy = mean(90) * 2 = 180
        assert result.median_lifetime_days == pytest.approx(180.0)


class TestSurvivalCurveExtraction:
    def test_survival_curve_none_when_reset_index_raises(self) -> None:
        mock_kmf = MagicMock()
        mock_kmf.median_survival_time_ = 365.0
        mock_sf = MagicMock()
        mock_sf.reset_index.side_effect = RuntimeError("extraction failed")
        mock_kmf.survival_function_ = mock_sf

        mock_lifelines = MagicMock()
        mock_lifelines.KaplanMeierFitter.return_value = mock_kmf

        rfm = _make_rfm(15)
        churn = _make_churn(15, n_churned=5)

        with patch.dict(sys.modules, {"lifelines": mock_lifelines}):
            result = CLVModel().fit_and_predict(rfm, churn)

        assert result.population_survival_curve is None

    def test_survival_curve_present_on_success(self) -> None:
        # Real lifelines — survival curve should be populated
        rfm = _make_rfm(15)
        churn = _make_churn(15, n_churned=5)
        result = CLVModel().fit_and_predict(rfm, churn)
        assert result.population_survival_curve is not None
        assert "timeline_days" in result.population_survival_curve.columns


class TestClvOutput:
    def test_output_has_expected_columns(self) -> None:
        rfm = _make_rfm(15)
        churn = _make_churn(15, n_churned=5)
        result = CLVModel().fit_and_predict(rfm, churn)
        cols = result.clv_per_customer.columns.tolist()
        assert "customer_id" in cols
        assert "expected_clv" in cols
        assert "expected_remaining_months" in cols
        assert "monthly_spend" in cols

    def test_active_customers_have_nonnegative_clv(self) -> None:
        rfm = _make_rfm(15)
        churn = _make_churn(15, n_churned=5)
        result = CLVModel().fit_and_predict(rfm, churn)
        assert (result.clv_per_customer["expected_clv"] >= 0).all()
