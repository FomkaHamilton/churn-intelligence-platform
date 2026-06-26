"""Unit tests for src/utils/log.py — structured logging configuration."""
from __future__ import annotations

from src.utils.log import configure_logging, get_logger


class TestConfigureLogging:
    def test_configure_info_does_not_raise(self) -> None:
        configure_logging("INFO")

    def test_configure_debug_does_not_raise(self) -> None:
        configure_logging("DEBUG")

    def test_configure_warning_does_not_raise(self) -> None:
        configure_logging("WARNING")

    def test_configure_default_level_does_not_raise(self) -> None:
        configure_logging()

    def test_idempotent_multiple_calls(self) -> None:
        configure_logging("INFO")
        configure_logging("DEBUG")
        configure_logging("INFO")


class TestGetLogger:
    def setup_method(self) -> None:
        configure_logging("INFO")

    def test_returns_bound_logger(self) -> None:
        logger = get_logger("test.module")
        assert logger is not None

    def test_logger_info_does_not_raise(self) -> None:
        logger = get_logger("test.info")
        logger.info("test_event", key="value", count=1)

    def test_logger_warning_does_not_raise(self) -> None:
        logger = get_logger("test.warning")
        logger.warning("test_warning", code=42)

    def test_logger_debug_does_not_raise(self) -> None:
        configure_logging("DEBUG")
        logger = get_logger("test.debug")
        logger.debug("test_debug", detail="some value")

    def test_different_names_return_different_loggers(self) -> None:
        a = get_logger("module.a")
        b = get_logger("module.b")
        assert a is not b
