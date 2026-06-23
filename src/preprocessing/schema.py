"""
Schema enforcement for uploaded subscription data.

Responsibilities:
- Map common column name variants to the canonical schema
- Validate that required columns are present
- Coerce columns to appropriate dtypes without dropping rows
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config.constants import COLUMN_ALIASES, OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from src.utils.exceptions import SchemaError
from src.utils.log import get_logger
from src.utils.types import DataFrame

logger = get_logger(__name__)


@dataclass
class SchemaValidationResult:
    """Outcome of schema validation."""

    normalised_df: DataFrame
    mapped_columns: dict[str, str]   # alias -> canonical name
    unrecognised_columns: list[str]  # extra columns that were kept as-is
    missing_optional: list[str]      # optional columns absent from this file

    @property
    def has_all_optional(self) -> bool:
        return len(self.missing_optional) == 0


class SchemaValidator:
    """
    Normalises uploaded DataFrames to the canonical schema.

    Column matching is case-insensitive and whitespace-tolerant.
    Unrecognised columns are kept (extra context is fine) but flagged.
    """

    def validate(self, df: DataFrame) -> SchemaValidationResult:
        """
        Normalise columns and validate required fields are present.

        Raises
        ------
        SchemaError
            If one or more required columns are absent after alias resolution.
        """
        df = df.copy()

        # Step 1 — apply alias map (columns already lowercased by uploader)
        mapped: dict[str, str] = {}
        rename_map: dict[str, str] = {}

        for col in df.columns:
            canonical = COLUMN_ALIASES.get(col)
            if canonical and canonical not in df.columns:
                rename_map[col] = canonical
                mapped[col] = canonical

        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info("columns_remapped", mapping=rename_map)

        # Step 2 — check required columns
        present = set(df.columns)
        missing_required = REQUIRED_COLUMNS - present
        if missing_required:
            raise SchemaError(
                f"Required column(s) missing: {sorted(missing_required)}. "
                f"Found: {sorted(present)}. "
                f"Common aliases are accepted automatically — "
                f"see docs/SCHEMA.md for the full list."
            )

        # Step 3 — identify optional and unrecognised columns
        known = REQUIRED_COLUMNS | OPTIONAL_COLUMNS
        missing_optional = sorted(OPTIONAL_COLUMNS - present)
        unrecognised = sorted(present - known)

        if unrecognised:
            logger.info("extra_columns_found", columns=unrecognised)

        # Step 4 — coerce types
        df = self._coerce_types(df)

        return SchemaValidationResult(
            normalised_df=df,
            mapped_columns=mapped,
            unrecognised_columns=unrecognised,
            missing_optional=missing_optional,
        )

    # ── Type coercion ─────────────────────────────────────────────────────────

    def _coerce_types(self, df: DataFrame) -> DataFrame:
        """
        Coerce columns to correct types in-place (on a copy).
        Coercion failures leave the value as NaN — caught by DataQualityChecker.
        """
        df = df.copy()

        # transaction_amount → float
        if "transaction_amount" in df.columns:
            df["transaction_amount"] = pd.to_numeric(
                df["transaction_amount"], errors="coerce"
            )

        # customer_id → string (strip whitespace)
        if "customer_id" in df.columns:
            df["customer_id"] = (
                df["customer_id"].astype(str).str.strip().replace("nan", pd.NA)
            )

        # subscription_id → string
        if "subscription_id" in df.columns:
            df["subscription_id"] = (
                df["subscription_id"].astype(str).str.strip().replace("nan", pd.NA)
            )

        # subscription_status → lowercase string
        if "subscription_status" in df.columns:
            df["subscription_status"] = (
                df["subscription_status"].astype(str).str.strip().str.lower()
            )

        # country, product → stripped strings
        for col in ("country", "product"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df
