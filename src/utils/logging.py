"""Structured logging helpers for graph execution."""

from __future__ import annotations

import json
import logging
from typing import Any

LOGGER_NAME = "amiio"
_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger once."""
    global _CONFIGURED
    logger = logging.getLogger(LOGGER_NAME)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(level)
        _CONFIGURED = True
    return logger


def get_logger() -> logging.Logger:
    """Return configured project logger."""
    return configure_logging()


def _redact_value(value: Any) -> Any:
    """Redact sensitive-like values recursively."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, nested in value.items():
            lowered = key.lower()
            if any(
                marker in lowered
                for marker in ("api_key", "token", "secret", "password")
            ):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = _redact_value(nested)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith("sk-") or "api_key" in lowered or "token" in lowered:
            return "***REDACTED***"
    return value


def log_event(event: str, **fields: Any) -> None:
    """Log a structured event with redacted sensitive fields."""
    logger = get_logger()
    payload = {"event": event, **_redact_value(fields)}
    logger.info(json.dumps(payload, default=str, ensure_ascii=True))
