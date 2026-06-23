"""Unit tests for application settings and configuration."""
from __future__ import annotations

import pytest

from src.config.settings import Settings, get_settings


class TestSettings:
    def test_default_churn_window(self) -> None:
        s = Settings()
        assert s.churn_window_days == 90

    def test_churn_window_bounds(self) -> None:
        with pytest.raises(Exception):
            Settings(churn_window_days=0)
        with pytest.raises(Exception):
            Settings(churn_window_days=366)

    def test_log_level_normalised_to_uppercase(self) -> None:
        s = Settings(log_level="debug")
        assert s.log_level == "DEBUG"

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(Exception):
            Settings(log_level="VERBOSE")

    def test_no_api_keys_by_default(self) -> None:
        s = Settings()
        assert s.anthropic_api_key is None
        assert s.openai_api_key is None
        assert s.has_ai_provider is False

    def test_has_ai_provider_with_anthropic_key(self) -> None:
        s = Settings(anthropic_api_key="sk-ant-test")
        assert s.has_ai_provider is True

    def test_max_upload_size_bytes(self) -> None:
        s = Settings(max_upload_size_mb=10)
        assert s.max_upload_size_bytes == 10 * 1024 * 1024

    def test_get_settings_returns_cached_instance(self) -> None:
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        get_settings.cache_clear()
