"""Guard checks and lightweight routing heuristics."""

from __future__ import annotations

import re
from typing import Any

from config.constants import (
    ADVERSARIAL_MARKERS,
    INTENT_ADVERSARIAL,
    INTENT_DATASET_KNOWLEDGE,
    INTENT_GIBBERISH,
    MSG_CANNOT_PROCEED,
    MSG_GIBBERISH,
)


def split_questions(user_query: str) -> list[str]:
    """Split a user message into question-like segments."""
    parts = [part.strip() for part in re.split(r"[?]+", user_query) if part.strip()]
    return parts


def detect_multiple_questions(user_query: str) -> bool:
    """Detect if user input contains multiple questions in one message."""
    if len(split_questions(user_query)) > 1:
        return True
    lowered = user_query.lower()
    return bool(
        re.search(
            r"\b(and|also|then)\s+(what|which|who|how|when|where|why|is|are|can|could|would|should|do|does)\b",
            lowered,
        )
    )


def detect_adversarial(text: str) -> bool:
    """Detect obvious adversarial/prompt-injection requests."""
    lowered = text.lower()
    return any(marker in lowered for marker in ADVERSARIAL_MARKERS)


def detect_gibberish(text: str) -> bool:
    """Detect unparseable or gibberish-like input."""
    stripped = text.strip()
    if not stripped:
        return True

    alpha_tokens = re.findall(r"[A-Za-z]+", stripped)
    has_digits = bool(re.search(r"\d", stripped))
    if not alpha_tokens and not has_digits:
        return True

    # Treat heavy-symbol, near-nonword inputs as gibberish; allow short normal queries through.
    non_space_chars = re.sub(r"\s+", "", stripped)
    if non_space_chars:
        wordlike_chars = re.findall(r"[A-Za-z0-9]", non_space_chars)
        ratio = len(wordlike_chars) / len(non_space_chars)
        if ratio < 0.30 and len(alpha_tokens) <= 1:
            return True

    return False


def route_query(text: str) -> dict[str, Any]:
    """Run guard checks and return routing decision payload."""
    if detect_adversarial(text):
        return {
            "intent": INTENT_ADVERSARIAL,
            "action": "fallback",
            "fallback_message": MSG_CANNOT_PROCEED,
            "reason": "adversarial content",
        }
    if detect_gibberish(text):
        return {
            "intent": INTENT_GIBBERISH,
            "action": "fallback",
            "fallback_message": MSG_GIBBERISH,
            "reason": "unparseable",
        }
    return {
        "intent": INTENT_DATASET_KNOWLEDGE,
        "action": "continue",
        "fallback_message": "",
        "reason": "non-exit query; defer to LLM intent classifier",
    }
