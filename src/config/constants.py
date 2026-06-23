"""
Shared constants used across the platform.
Values here are fixed by design; tuneable values belong in config/settings.yaml.
"""
from __future__ import annotations

# ── Required input schema ─────────────────────────────────────────────────────
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "customer_id",
        "transaction_date",
        "transaction_amount",
    }
)

OPTIONAL_COLUMNS: frozenset[str] = frozenset(
    {
        "subscription_id",
        "subscription_status",
        "country",
        "product",
    }
)

# ── Customer segments ─────────────────────────────────────────────────────────
SEGMENT_NEW = "New"
SEGMENT_HEALTHY = "Healthy"
SEGMENT_AT_RISK = "At Risk"
SEGMENT_HIGH_VALUE = "High Value"
SEGMENT_CHURNED = "Churned"

ALL_SEGMENTS: list[str] = [
    SEGMENT_NEW,
    SEGMENT_HEALTHY,
    SEGMENT_AT_RISK,
    SEGMENT_HIGH_VALUE,
    SEGMENT_CHURNED,
]

# ── Churn window options exposed in the UI ────────────────────────────────────
CHURN_WINDOW_OPTIONS: list[int] = [30, 60, 90, 120]

# ── Upload constraints ────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".xlsx", ".xls"})
ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
)

# ── Minimum viable dataset sizes ──────────────────────────────────────────────
MIN_CUSTOMERS_FOR_MODELING = 50
MIN_CUSTOMERS_FOR_COHORT = 10
MIN_MONTHS_FOR_FORECASTING = 6

# ── Column name normalisation map ─────────────────────────────────────────────
# Maps common alternative column names to the canonical schema.
COLUMN_ALIASES: dict[str, str] = {
    "cust_id": "customer_id",
    "client_id": "customer_id",
    "user_id": "customer_id",
    "order_date": "transaction_date",
    "purchase_date": "transaction_date",
    "date": "transaction_date",
    "amount": "transaction_amount",
    "revenue": "transaction_amount",
    "price": "transaction_amount",
    "value": "transaction_amount",
}
