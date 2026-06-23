"""
Churn prediction model.

Ensemble of Logistic Regression + Random Forest.
Both use class_weight='balanced' to handle the typical 10–25 % churn rate
without requiring SMOTE or manual resampling.

The final churn probability is the average of both model outputs so we
get the calibration of LR and the non-linearity of RF in one score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from src.utils.exceptions import ModelTrainingError
from src.utils.types import DataFrame


@dataclass
class ModelMetrics:
    auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    fpr: np.ndarray    # for ROC curve
    tpr: np.ndarray    # for ROC curve
    n_train: int
    n_test: int


class ChurnModel:
    """LR + RF ensemble churn classifier."""

    def __init__(
        self,
        rf_n_estimators: int = 200,
        rf_max_depth: int | None = 10,
        lr_max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self._random_state = random_state
        self._scaler = StandardScaler()
        self._lr: LogisticRegression | None = None
        self._rf: RandomForestClassifier | None = None
        self._rf_n_estimators = rf_n_estimators
        self._rf_max_depth = rf_max_depth
        self._lr_max_iter = lr_max_iter
        self.is_trained: bool = False
        self.n_train: int = 0

    def train(self, X: DataFrame, y: pd.Series) -> None:
        if len(y.unique()) < 2:
            raise ModelTrainingError(
                "Training data has only one class. "
                "Need both churned and active customers."
            )

        X_arr = X.fillna(0.0).values.astype(float)
        X_scaled = self._scaler.fit_transform(X_arr)
        y_arr = y.values

        self._lr = LogisticRegression(
            class_weight="balanced",
            max_iter=self._lr_max_iter,
            random_state=self._random_state,
        )
        self._rf = RandomForestClassifier(
            n_estimators=self._rf_n_estimators,
            max_depth=self._rf_max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self._random_state,
        )

        self._lr.fit(X_scaled, y_arr)
        self._rf.fit(X_arr, y_arr)
        self.is_trained = True
        self.n_train = len(y)

    def predict_proba(self, X: DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ModelTrainingError("Model must be trained before predicting.")

        X_arr = X.fillna(0.0).values.astype(float)
        X_scaled = self._scaler.transform(X_arr)

        lr_p = self._lr.predict_proba(X_scaled)[:, 1]    # type: ignore[union-attr]
        rf_p = self._rf.predict_proba(X_arr)[:, 1]       # type: ignore[union-attr]
        return (lr_p + rf_p) / 2.0

    def evaluate(self, X: DataFrame, y: pd.Series) -> ModelMetrics:
        proba = self.predict_proba(X)
        y_pred = (proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y.values, proba)

        return ModelMetrics(
            auc=float(roc_auc_score(y.values, proba)),
            precision=float(precision_score(y.values, y_pred, zero_division=0)),
            recall=float(recall_score(y.values, y_pred, zero_division=0)),
            f1=float(f1_score(y.values, y_pred, zero_division=0)),
            confusion_matrix=confusion_matrix(y.values, y_pred),
            fpr=fpr,
            tpr=tpr,
            n_train=self.n_train,
            n_test=len(y),
        )

    @property
    def rf(self) -> RandomForestClassifier:
        if self._rf is None:
            raise ModelTrainingError("Model not trained.")
        return self._rf
