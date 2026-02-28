"""Response formatting helpers and fallback mapping."""

from __future__ import annotations
from typing import Any
import json
import re
from config.month_labels import MONTH_LABELS
from config.settings import settings
from config.constants import (
    MSG_CANNOT_PROCEED,
    MSG_GIBBERISH,
    MSG_NOT_PRESENT,
    MSG_OUT_OF_SCOPE,
)
from config.prompts import build_answer_prompt
from src.data.profiler import build_minimal_prompt_profile_json, get_startup_profile
from src.services.llm_client import OpenAILLMClient


def _format_month_tokens(text: str) -> str:
    """Convert month tokens like 2025-M01 into human-readable month names."""

    def _replace_year_month(match: re.Match[str]) -> str:
        year = match.group(1)
        month_token = match.group(2)
        month_name = MONTH_LABELS.get(month_token)
        if not month_name:
            return match.group(0)
        return f"{month_name} {year}"

    formatted = re.sub(r"\b(\d{4})-(M\d{2})\b", _replace_year_month, text)

    def _replace_month_only(match: re.Match[str]) -> str:
        month_token = match.group(1)
        return MONTH_LABELS.get(month_token, month_token)

    return re.sub(r"\b(M\d{2})\b", _replace_month_only, formatted)


def fallback_for_error_type(error_type: str) -> str:
    """Map error category to canonical fallback string."""
    mapping = {
        "not_present": MSG_NOT_PRESENT,
        "out_of_scope": MSG_OUT_OF_SCOPE,
        "adversarial": MSG_CANNOT_PROCEED,
        "gibberish": MSG_GIBBERISH,
    }
    return _format_month_tokens(mapping.get(error_type, MSG_NOT_PRESENT))


def answer_from_result_with_llm(
    *,
    user_query: str,
    result_payload: dict[str, Any] | list[dict[str, Any]],
    profile: dict[str, Any] | None = None,
    conversation_messages: list[dict[str, Any]] | None = None,
    client: OpenAILLMClient | None = None,
) -> str:
    """Generate final user-facing answer from extracted result JSON."""
    llm = client or OpenAILLMClient()
    profile_data = profile or get_startup_profile()
    profile_json = build_minimal_prompt_profile_json(profile_data)
    payload_json = json.dumps(result_payload, ensure_ascii=True)
    if payload_json in ("{}", "[]"):
        return _format_month_tokens(MSG_NOT_PRESENT)
    user_prompt = json.dumps(
        {"user_query": user_query, "result_json": result_payload},
        ensure_ascii=True,
    )
    return _format_month_tokens(
        llm.chat_text(
            system_prompt=build_answer_prompt(profile_json),
            user_prompt=user_prompt,
            conversation_messages=conversation_messages,
            max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS_ANSWER,
        ).strip()
    )


def answer_from_profile_with_llm(
    *,
    user_query: str,
    profile: dict[str, Any] | None = None,
    conversation_messages: list[dict[str, Any]] | None = None,
    client: OpenAILLMClient | None = None,
) -> str:
    """Generate final answer from profile context only for metadata/methodology questions."""
    llm = client or OpenAILLMClient()
    profile_data = profile or get_startup_profile()
    profile_json = build_minimal_prompt_profile_json(profile_data)
    user_prompt = json.dumps(
        {"user_query": user_query, "result_json": {}},
        ensure_ascii=True,
    )
    return _format_month_tokens(
        llm.chat_text(
            system_prompt=build_answer_prompt(profile_json),
            user_prompt=user_prompt,
            conversation_messages=conversation_messages,
            max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS_ANSWER,
        ).strip()
    )
