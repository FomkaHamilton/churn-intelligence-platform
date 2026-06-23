"""
Application settings loaded from config/settings.yaml and environment variables.
Environment variables take precedence over YAML values.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"


def _load_yaml_defaults() -> dict[str, Any]:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "Churn Intelligence Platform"
    app_version: str = "0.1.0"
    max_upload_size_mb: int = Field(default=50, ge=1, le=500)
    log_level: str = Field(default="INFO")

    # ── Churn ─────────────────────────────────────────────────────────────────
    churn_window_days: int = Field(default=90, ge=1, le=365)

    # ── AI providers (optional — platform works without these) ────────────────
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(default="sqlite:///data/sessions/sip.db")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @property
    def has_ai_provider(self) -> bool:
        return bool(self.anthropic_api_key or self.openai_api_key)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Call get_settings.cache_clear() in tests."""
    return Settings()


def get_yaml_config() -> dict[str, Any]:
    """Return the raw YAML config dict for values not modelled in Settings."""
    return _load_yaml_defaults()
