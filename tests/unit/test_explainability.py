"""Unit tests covering SHAPExplainer edge cases and error paths."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.modeling.explainability import SHAPExplainer


def _make_features(n: int = 20, n_features: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )


def _make_trained_rf(x: pd.DataFrame) -> RandomForestClassifier:
    y = np.array([i % 2 for i in range(len(x))])
    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(x, y)
    return rf


class TestImportGuard:
    def test_raises_import_error_when_shap_missing(self) -> None:
        x = _make_features()
        rf = _make_trained_rf(x)
        with patch.dict(sys.modules, {"shap": None}), pytest.raises(ImportError, match="shap is required"):
            SHAPExplainer().explain(rf, x)


class TestShapOutputShapes:
    def setup_method(self) -> None:
        self.x = _make_features(n=20, n_features=4)
        self.rf = _make_trained_rf(self.x)

    def _mock_shap_module(self, shap_values_return: object) -> MagicMock:
        mock_shap = MagicMock()
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = shap_values_return
        mock_shap.TreeExplainer.return_value = mock_explainer
        return mock_shap

    def test_list_format_uses_class1(self) -> None:
        # Older SHAP: returns list [class0_arr, class1_arr]
        n, f = len(self.x), len(self.x.columns)
        class0 = np.zeros((n, f))
        class1 = np.ones((n, f)) * 0.5
        mock_shap = self._mock_shap_module([class0, class1])

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = SHAPExplainer().explain(self.rf, self.x)

        assert result.shap_values.shape == (n, f)
        assert np.allclose(result.shap_values, 0.5)

    def test_3d_ndarray_format_uses_class1(self) -> None:
        # Newer SHAP: returns ndarray (n_samples, n_features, n_classes)
        n, f = len(self.x), len(self.x.columns)
        raw = np.zeros((n, f, 2))
        raw[:, :, 1] = 0.7
        mock_shap = self._mock_shap_module(raw)

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = SHAPExplainer().explain(self.rf, self.x)

        assert result.shap_values.shape == (n, f)
        assert np.allclose(result.shap_values, 0.7)

    def test_2d_ndarray_format_uses_as_is(self) -> None:
        # Single-output SHAP: returns ndarray (n_samples, n_features)
        n, f = len(self.x), len(self.x.columns)
        raw = np.full((n, f), 0.3)
        mock_shap = self._mock_shap_module(raw)

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = SHAPExplainer().explain(self.rf, self.x)

        assert result.shap_values.shape == (n, f)
        assert np.allclose(result.shap_values, 0.3)


class TestSHAPResult:
    def test_feature_importance_has_expected_columns(self) -> None:
        x = _make_features(n=30, n_features=3)
        rf = _make_trained_rf(x)
        result = SHAPExplainer().explain(rf, x)
        cols = result.feature_importance.columns.tolist()
        assert "feature_name" in cols
        assert "mean_abs_shap" in cols
        assert "rank" in cols

    def test_feature_importance_sorted_descending(self) -> None:
        x = _make_features(n=30, n_features=4)
        rf = _make_trained_rf(x)
        result = SHAPExplainer().explain(rf, x)
        values = result.feature_importance["mean_abs_shap"].tolist()
        assert values == sorted(values, reverse=True)

    def test_rank_starts_at_one(self) -> None:
        x = _make_features(n=30, n_features=3)
        rf = _make_trained_rf(x)
        result = SHAPExplainer().explain(rf, x)
        assert result.feature_importance["rank"].iloc[0] == 1

    def test_sampling_applied_when_over_limit(self) -> None:
        x = _make_features(n=50, n_features=3)
        rf = _make_trained_rf(x)
        result = SHAPExplainer().explain(rf, x, max_samples=20)
        # shap_values rows should be capped at max_samples
        assert result.shap_values.shape[0] <= 20

    def test_nan_in_features_filled_before_shap(self) -> None:
        x = _make_features(n=30, n_features=3)
        x.iloc[0, 0] = float("nan")
        rf = _make_trained_rf(x.fillna(0.0))
        result = SHAPExplainer().explain(rf, x)
        assert result.feature_importance is not None
