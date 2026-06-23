"""
TimeSeriesChurnSplit — temporally safe train/test split for churn models.

Splits the customer population by first transaction date.
Customers who first appeared on or before the 80th-percentile cutoff date
go to the training set; those who joined later go to the test set.

This guarantees the model is evaluated on customers it has never seen,
preserving real-world temporal ordering.  An explicit leakage assertion
verifies the invariant after splitting.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.exceptions import InsufficientDataError, TemporalLeakageError
from src.utils.types import DataFrame


@dataclass
class TemporalSplitResult:
    X_train: DataFrame
    X_test: DataFrame
    y_train: pd.Series
    y_test: pd.Series
    customer_ids_train: pd.Series
    customer_ids_test: pd.Series
    cutoff_date: pd.Timestamp


class TimeSeriesChurnSplit:
    """
    Train/test split that respects the temporal order of customer acquisition.

    The split point is the train_ratio quantile of first-transaction dates.
    After splitting, a leakage assertion confirms every test customer joined
    after the cutoff — they were invisible during training.
    """

    def split(
        self,
        X: DataFrame,
        y: pd.Series,
        customer_ids: pd.Series,
        first_tx_dates: pd.Series,
        *,
        train_ratio: float = 0.80,
    ) -> TemporalSplitResult:
        """
        Args:
            X: Feature matrix aligned with the other series.
            y: Binary churn labels.
            customer_ids: Customer identifiers.
            first_tx_dates: Each customer's earliest transaction date.
            train_ratio: Fraction for the training set.
        """
        n = len(X)
        if n < 2:
            raise InsufficientDataError("temporal split", 2, n)

        cutoff_date = pd.Timestamp(first_tx_dates.quantile(train_ratio))
        train_mask = first_tx_dates <= cutoff_date
        test_mask = ~train_mask

        # Edge case: all customers share the same date → put the latest n×(1-ratio) in test
        if test_mask.sum() == 0:
            n_test = max(1, round(n * (1 - train_ratio)))
            test_indices = first_tx_dates.nlargest(n_test).index
            train_mask.loc[test_indices] = False
            test_mask = ~train_mask
            cutoff_date = first_tx_dates[train_mask].max()

        result = TemporalSplitResult(
            X_train=X[train_mask.values].reset_index(drop=True),
            X_test=X[test_mask.values].reset_index(drop=True),
            y_train=y[train_mask.values].reset_index(drop=True),
            y_test=y[test_mask.values].reset_index(drop=True),
            customer_ids_train=customer_ids[train_mask.values].reset_index(drop=True),
            customer_ids_test=customer_ids[test_mask.values].reset_index(drop=True),
            cutoff_date=cutoff_date,
        )

        # Invariant: no test customer joined before the cutoff
        test_dates = first_tx_dates[test_mask.values].reset_index(drop=True)
        leaking = test_dates[test_dates <= cutoff_date]
        if not leaking.empty:
            raise TemporalLeakageError(
                f"{len(leaking)} test customers have first_tx_date ≤ training cutoff "
                f"({cutoff_date.date()}). This indicates a split logic bug."
            )

        return result
