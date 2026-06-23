"""Unit tests for CustomerSegmenter."""
from __future__ import annotations

import pandas as pd
import pytest

from src.config.constants import ALL_SEGMENTS, SEGMENT_AT_RISK, SEGMENT_CHURNED, SEGMENT_HIGH_VALUE, SEGMENT_NEW
from src.modeling.segmentation import CustomerSegmenter


def _make_rfm(n: int = 50, base_monetary: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(n)],
        "recency_days": [10 + i * 2 for i in range(n)],
        "frequency": [5] * n,
        "monetary_total": [base_monetary * (i + 1) for i in range(n)],
        "aov": [base_monetary] * n,
        "tenure_days": [365] * n,
        "tx_per_month": [1.0] * n,
        "gap_std_days": [5.0] * n,
    })


def _make_labels(customer_ids, is_churned_flags) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": customer_ids,
        "is_churned": is_churned_flags,
        "last_tx_date": pd.Timestamp("2024-01-01"),
        "days_since_last_tx": [10] * len(customer_ids),
    })


class TestSegmentCoverage:
    def setup_method(self) -> None:
        self.segmenter = CustomerSegmenter()

    def test_every_customer_gets_a_segment(self) -> None:
        rfm = _make_rfm(50)
        labels = _make_labels(rfm["customer_id"], [False] * 50)
        segments = self.segmenter.segment(rfm, labels)
        assert len(segments) == 50
        assert segments.notna().all()

    def test_all_segment_labels_are_valid(self) -> None:
        rfm = _make_rfm(50)
        labels = _make_labels(rfm["customer_id"], [False] * 50)
        segments = self.segmenter.segment(rfm, labels)
        for s in segments:
            assert s in ALL_SEGMENTS

    def test_index_is_customer_id(self) -> None:
        rfm = _make_rfm(50)
        labels = _make_labels(rfm["customer_id"], [False] * 50)
        segments = self.segmenter.segment(rfm, labels)
        assert set(segments.index) == set(rfm["customer_id"])


class TestChurnedSegment:
    def setup_method(self) -> None:
        self.segmenter = CustomerSegmenter()

    def test_churned_customers_get_churned_segment(self) -> None:
        rfm = _make_rfm(10)
        is_churned = [True, False, True, False, False, False, False, False, False, False]
        labels = _make_labels(rfm["customer_id"], is_churned)
        segments = self.segmenter.segment(rfm, labels)
        assert segments["c0"] == SEGMENT_CHURNED
        assert segments["c2"] == SEGMENT_CHURNED

    def test_active_customers_not_churned(self) -> None:
        rfm = _make_rfm(10)
        labels = _make_labels(rfm["customer_id"], [False] * 10)
        segments = self.segmenter.segment(rfm, labels)
        assert SEGMENT_CHURNED not in segments.values


class TestNewSegment:
    def setup_method(self) -> None:
        self.segmenter = CustomerSegmenter()

    def test_short_tenure_customer_gets_new_segment(self) -> None:
        rfm = _make_rfm(50)
        rfm.loc[0, "tenure_days"] = 10  # very new customer
        rfm.loc[0, "recency_days"] = 5  # low recency (active)
        labels = _make_labels(rfm["customer_id"], [False] * 50)
        segments = self.segmenter.segment(rfm, labels, new_threshold_days=90)
        assert segments["c0"] == SEGMENT_NEW


class TestAtRiskSegment:
    def setup_method(self) -> None:
        self.segmenter = CustomerSegmenter()

    def test_high_churn_prob_gets_at_risk(self) -> None:
        rfm = _make_rfm(10)
        rfm["tenure_days"] = 365  # not new
        labels = _make_labels(rfm["customer_id"], [False] * 10)
        # Provide high churn probabilities for first customer
        churn_probs = pd.Series(
            [0.95] + [0.1] * 9,
            index=rfm["customer_id"],
        )
        segments = self.segmenter.segment(rfm, labels, churn_probs, at_risk_threshold=0.60)
        assert segments["c0"] == SEGMENT_AT_RISK


class TestHighValueSegment:
    def setup_method(self) -> None:
        self.segmenter = CustomerSegmenter()

    def test_top_spender_gets_high_value(self) -> None:
        rfm = _make_rfm(50)
        rfm["tenure_days"] = 365  # not new
        labels = _make_labels(rfm["customer_id"], [False] * 50)
        # Low churn probs so at-risk threshold not met
        churn_probs = pd.Series([0.1] * 50, index=rfm["customer_id"])
        segments = self.segmenter.segment(
            rfm, labels, churn_probs,
            at_risk_threshold=0.60,
            high_value_percentile=0.75,
        )
        # Top 25% spenders should be HIGH VALUE
        top_spender = rfm.sort_values("monetary_total", ascending=False).iloc[0]["customer_id"]
        assert segments[top_spender] == SEGMENT_HIGH_VALUE
