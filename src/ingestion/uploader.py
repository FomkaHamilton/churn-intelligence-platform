"""
File upload engine for the Churn Intelligence Platform.

Responsibilities:
- Validate file type and size before touching content
- Read CSV and XLSX into a DataFrame
- Detect and warn about CSV formula injection attempts
- Never silently drop data — surface all issues to the caller
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.config.constants import ALLOWED_EXTENSIONS
from src.config.settings import get_settings
from src.utils.exceptions import (
    InsufficientDataError,
    UnsupportedFileTypeError,
    ValidationError,
)
from src.utils.log import get_logger
from src.utils.types import DataFrame

logger = get_logger(__name__)

# Characters that trigger formula execution in spreadsheet software.
# We warn about these rather than silently stripping them.
_INJECTION_PREFIXES = ("=", "+", "-", "@", "|", "\t", "\r")

_MIN_ROWS = 2  # header + at least one data row


@dataclass
class UploadResult:
    """Result of a file upload operation."""

    dataframe: DataFrame
    filename: str
    row_count: int
    column_count: int
    file_size_bytes: int
    warnings: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class UploadEngine:
    """
    Validates and reads subscription data files.

    All validation failures raise typed exceptions so the UI layer
    can present actionable messages rather than raw stack traces.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def process(self, file: BinaryIO, filename: str) -> UploadResult:
        """
        Full upload pipeline: validate → read → inspect.

        Parameters
        ----------
        file:
            File-like object (e.g. from st.file_uploader or open()).
        filename:
            Original filename including extension.

        Returns
        -------
        UploadResult with the parsed DataFrame and any warnings.

        Raises
        ------
        UnsupportedFileTypeError
            If the file extension is not CSV or XLSX.
        ValidationError
            If the file exceeds the size limit.
        InsufficientDataError
            If the file contains fewer than 2 rows (header only).
        """
        content = file.read()
        size_bytes = len(content)

        logger.info(
            "upload_received",
            filename=filename,
            size_bytes=size_bytes,
        )

        self._validate_extension(filename)
        self._validate_size(size_bytes, filename)

        df, warnings = self._read_file(content, filename)

        if len(df) < _MIN_ROWS - 1:
            raise InsufficientDataError(
                operation="upload",
                required=1,
                actual=len(df),
            )

        injection_warnings = self._check_injection(df)
        warnings.extend(injection_warnings)

        result = UploadResult(
            dataframe=df,
            filename=filename,
            row_count=len(df),
            column_count=len(df.columns),
            file_size_bytes=size_bytes,
            warnings=warnings,
        )

        logger.info(
            "upload_complete",
            filename=filename,
            rows=result.row_count,
            columns=result.column_count,
            warnings=len(warnings),
        )

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate_extension(self, filename: str) -> None:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"'{ext}' is not supported. Upload a CSV or XLSX file."
            )

    def _validate_size(self, size_bytes: int, filename: str) -> None:
        limit = self._settings.max_upload_size_bytes
        if size_bytes > limit:
            mb = size_bytes / (1024 * 1024)
            limit_mb = self._settings.max_upload_size_mb
            raise ValidationError(
                f"'{filename}' is {mb:.1f} MB. The limit is {limit_mb} MB."
            )

    def _read_file(
        self, content: bytes, filename: str
    ) -> tuple[DataFrame, list[str]]:
        ext = Path(filename).suffix.lower()
        warnings: list[str] = []

        try:
            if ext == ".csv":
                df = self._read_csv(content)
            else:
                df = self._read_xlsx(content, filename)
        except Exception as exc:
            raise ValidationError(
                f"Could not read '{filename}': {exc}. "
                "Ensure the file is not password-protected or corrupted."
            ) from exc

        # Normalise column names: strip whitespace, lowercase
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        return df, warnings

    def _read_csv(self, content: bytes) -> DataFrame:
        # Try UTF-8 first; fall back to latin-1 for legacy exports
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(
                    io.BytesIO(content),
                    encoding=encoding,
                    dtype=str,          # read everything as str; types enforced later
                    keep_default_na=False,
                    na_values=["", "NULL", "null", "None", "NA", "N/A", "n/a"],
                )
            except UnicodeDecodeError:
                continue
        raise ValidationError("Could not decode the CSV file. Try saving it as UTF-8.")

    def _read_xlsx(self, content: bytes, filename: str) -> DataFrame:
        return pd.read_excel(
            io.BytesIO(content),
            sheet_name=0,
            dtype=str,
            keep_default_na=False,
            na_values=["", "NULL", "null", "None", "NA", "N/A", "n/a"],
        )

    def _check_injection(self, df: DataFrame) -> list[str]:
        """
        Detect cells that start with formula-injection characters.
        Returns warning messages — never modifies the DataFrame.
        """
        warnings: list[str] = []

        for col in df.select_dtypes(include="object").columns:
            mask = df[col].dropna().astype(str).str.startswith(_INJECTION_PREFIXES)
            count = int(mask.sum())
            if count:
                warnings.append(
                    f"Column '{col}' contains {count} cell(s) starting with "
                    f"formula characters (=, +, -, @). These will be treated as "
                    f"plain text, but verify the data is correct."
                )

        return warnings
