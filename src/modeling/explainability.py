"""
SHAP-based model explainability.

Uses TreeExplainer with the Random Forest component of the ChurnModel
ensemble.  Returns per-feature mean |SHAP| values (global importance)
and the raw SHAP value matrix for downstream plotting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils.types import DataFrame


@dataclass
class SHAPResult:
    feature_importance: DataFrame  # feature_name | mean_abs_shap | rank
    shap_values: np.ndarray        # shape (n_samples, n_features), class-1 values


class SHAPExplainer:
    """Compute SHAP values for a trained Random Forest."""

    def explain(
        self,
        model: RandomForestClassifier,
        X: DataFrame,
        *,
        max_samples: int = 1000,
    ) -> SHAPResult:
        """
        Args:
            model: Fitted RandomForestClassifier.
            X: Feature matrix to explain (ideally the test set).
            max_samples: Cap on rows to explain (SHAP is O(n × trees)).
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required. Run: pip install shap") from exc

        X_arr = X.fillna(0.0)
        if len(X_arr) > max_samples:
            X_arr = X_arr.sample(max_samples, random_state=42)

        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_arr.values)

        # Binary classification may return a list [class0_shap, class1_shap]
        sv = raw[1] if isinstance(raw, list) else raw

        mean_abs = np.abs(sv).mean(axis=0)
        importance = (
            pd.DataFrame({
                "feature_name": X.columns.tolist(),
                "mean_abs_shap": mean_abs,
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        importance["rank"] = range(1, len(importance) + 1)

        return SHAPResult(feature_importance=importance, shap_values=sv)
