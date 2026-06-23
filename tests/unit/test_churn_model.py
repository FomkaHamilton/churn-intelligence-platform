"""Unit tests for the ChurnModel ensemble."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.churn_model import ChurnModel, ModelMetrics
from src.utils.exceptions import ModelTrainingError


def _make_training_data(n_each: int = 60):
    """60 active + 60 churned customers with separable feature values."""
    n = n_each * 2
    # Active customers: low recency, high frequency
    # Churned customers: high recency, low frequency
    recency = [10] * n_each + [200] * n_each
    frequency = [12] * n_each + [1] * n_each
    monetary = [500.0] * n_each + [50.0] * n_each
    aov = [50.0] * n_each + [50.0] * n_each
    tenure = [365] * n_each + [90] * n_each
    tx_pm = [1.0] * n_each + [0.1] * n_each
    gap_std = [5.0] * n_each + [30.0] * n_each

    X = pd.DataFrame({
        "recency_days": recency,
        "frequency": frequency,
        "monetary_total": monetary,
        "aov": aov,
        "tenure_days": tenure,
        "tx_per_month": tx_pm,
        "gap_std_days": gap_std,
    })
    y = pd.Series([0] * n_each + [1] * n_each)
    return X, y


class TestTraining:
    def setup_method(self) -> None:
        self.model = ChurnModel(rf_n_estimators=10, random_state=42)

    def test_train_completes_without_error(self) -> None:
        X, y = _make_training_data()
        self.model.train(X, y)
        assert self.model.is_trained

    def test_single_class_raises_model_training_error(self) -> None:
        X, _ = _make_training_data()
        y_single = pd.Series([0] * len(X))
        with pytest.raises(ModelTrainingError):
            self.model.train(X, y_single)


class TestPrediction:
    def setup_method(self) -> None:
        self.model = ChurnModel(rf_n_estimators=10, random_state=42)
        X, y = _make_training_data()
        self.model.train(X, y)
        self.X, self.y = X, y

    def test_predict_proba_in_zero_one_range(self) -> None:
        proba = self.model.predict_proba(self.X)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_proba_length_matches_input(self) -> None:
        proba = self.model.predict_proba(self.X)
        assert len(proba) == len(self.X)

    def test_predict_before_training_raises(self) -> None:
        untrained = ChurnModel()
        X, _ = _make_training_data()
        with pytest.raises(ModelTrainingError):
            untrained.predict_proba(X)

    def test_higher_recency_gets_higher_churn_probability(self) -> None:
        # Active customer
        active = pd.DataFrame({
            "recency_days": [5], "frequency": [12], "monetary_total": [500.0],
            "aov": [50.0], "tenure_days": [365], "tx_per_month": [1.0], "gap_std_days": [5.0]
        })
        # Churned customer
        churned = pd.DataFrame({
            "recency_days": [250], "frequency": [1], "monetary_total": [50.0],
            "aov": [50.0], "tenure_days": [90], "tx_per_month": [0.1], "gap_std_days": [30.0]
        })
        assert self.model.predict_proba(churned)[0] > self.model.predict_proba(active)[0]


class TestEvaluation:
    def setup_method(self) -> None:
        self.model = ChurnModel(rf_n_estimators=10, random_state=42)
        X, y = _make_training_data()
        self.model.train(X, y)
        self.metrics = self.model.evaluate(X, y)

    def test_auc_above_random(self) -> None:
        assert self.metrics.auc > 0.50

    def test_auc_on_separable_data_is_high(self) -> None:
        # With clearly separated classes the AUC should be near 1.0
        assert self.metrics.auc > 0.90

    def test_metrics_fields_present(self) -> None:
        assert isinstance(self.metrics, ModelMetrics)
        assert 0.0 <= self.metrics.precision <= 1.0
        assert 0.0 <= self.metrics.recall <= 1.0
        assert 0.0 <= self.metrics.f1 <= 1.0

    def test_roc_curve_starts_at_origin(self) -> None:
        assert self.metrics.fpr[0] == pytest.approx(0.0)
        assert self.metrics.tpr[0] == pytest.approx(0.0)

    def test_confusion_matrix_shape(self) -> None:
        assert self.metrics.confusion_matrix.shape == (2, 2)

    def test_n_test_recorded(self) -> None:
        X, y = _make_training_data()
        assert self.metrics.n_test == len(y)

    def test_rf_property_accessible(self) -> None:
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(self.model.rf, RandomForestClassifier)
