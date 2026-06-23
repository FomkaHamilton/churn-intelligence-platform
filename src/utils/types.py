"""
Shared type aliases used across the platform.

Centralising these avoids repeating verbose type annotations and makes
future type changes (e.g. switching DataFrame library) a one-line edit.
"""
from __future__ import annotations

from typing import Any, TypeAlias

import pandas as pd

# ── DataFrame aliases ─────────────────────────────────────────────────────────
DataFrame: TypeAlias = pd.DataFrame
Series: TypeAlias = pd.Series  # type: ignore[type-arg]

# ── Domain ID types ───────────────────────────────────────────────────────────
CustomerID: TypeAlias = str
SubscriptionID: TypeAlias = str

# ── Common return types ───────────────────────────────────────────────────────
JsonDict: TypeAlias = dict[str, Any]
MetricsDict: TypeAlias = dict[str, float]
