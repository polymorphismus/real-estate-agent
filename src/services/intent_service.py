"""Intent service helpers for router output handling."""

from __future__ import annotations

from typing import Any

from config.constants import (
    INTENT_AMBIGUOUS,
    INTENT_DATASET_KNOWLEDGE,
)
from config.settings import settings
from config.prompts import build_intent_extractor_prompt
from src.data.profiler import build_minimal_prompt_profile_json
from src.contracts.models import (
    ALLOWED_INTENTS,
    IntentDecision,
    IntentExtractionSchema,
)
from src.services.llm_client import OpenAILLMClient


def normalize_router_output(payload: dict[str, Any]) -> IntentDecision:
    """Normalize and validate router output JSON."""
    intent = str(payload.get("intent", INTENT_DATASET_KNOWLEDGE))
    if intent not in ALLOWED_INTENTS:
        intent = INTENT_AMBIGUOUS

    action = str(payload.get("action", "clarify"))
    if action not in {"continue", "fallback", "clarify"}:
        action = "clarify"

    return IntentDecision(
        intent=intent,
        action=action,
        fallback_message=str(payload.get("fallback_message", "")),
        reason=str(payload.get("reason", "")),
        clarification_prompt=str(payload.get("clarification_prompt", "")),
    )


def classify_intent_and_extract_with_llm(
    user_query: str,
    profile: dict[str, Any],
    conversation_messages: list[dict[str, Any]] | None = None,
    client: OpenAILLMClient | None = None,
) -> tuple[IntentDecision, dict[str, Any]]:
    """Classify intent and extract entities in a single structured LLM call."""
    llm = client or OpenAILLMClient()
    profile_json = build_minimal_prompt_profile_json(profile)
    parsed = llm.parse_structured(
        system_prompt=build_intent_extractor_prompt(profile_json),
        user_prompt=user_query,
        response_format=IntentExtractionSchema,
        conversation_messages=conversation_messages,
        max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS_EXTRACTOR,
    )
    payload = parsed.model_dump()
    decision = normalize_router_output(payload)
    entities = payload.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}
    return decision, entities
