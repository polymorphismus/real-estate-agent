"""OpenAI client wrapper used by service-layer agents."""

from __future__ import annotations

import time
from typing import Any
from typing import TypeVar

from config.settings import settings
from src.contracts.policies import LLM_COMPATIBILITY_MARKERS
from src.utils.logging import log_event


class LLMClientError(RuntimeError):
    """Raised when LLM client initialization or invocation fails."""


class OpenAILLMClient:
    """Minimal OpenAI chat wrapper with lazy dependency import."""

    def __init__(self) -> None:
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Initialize and cache OpenAI client lazily."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            raise LLMClientError(
                "openai package is required for runtime LLM calls."
            ) from exc

        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        conversation_messages: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """Send a chat request and return message text.

        Args:
            system_prompt: Instruction prompt for assistant role.
            user_prompt: Query/input payload.
            temperature: Optional override for model temperature.
            max_output_tokens: Optional override for max completion tokens.

        Returns:
            Model response content as text.
        """
        client = self._get_client()
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if conversation_messages:
            for message in conversation_messages:
                role = str(message.get("role", "")).strip()
                content = str(message.get("content", "")).strip()
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            temperature=(
                temperature if temperature is not None else settings.OPENAI_TEMPERATURE
            ),
            max_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else settings.OPENAI_MAX_OUTPUT_TOKENS
            ),
            messages=messages,
            timeout=settings.OPENAI_TIMEOUT_SEC,
        )

        content = response.choices[0].message.content if response.choices else ""
        return content or ""

    TModel = TypeVar("TModel")

    def parse_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: type[TModel],
        conversation_messages: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> TModel:
        """Request structured output parsed into the provided response_format type."""
        client = self._get_client()
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if conversation_messages:
            for message in conversation_messages:
                role = str(message.get("role", "")).strip()
                content = str(message.get("content", "")).strip()
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})

        # Preferred path: Responses API with parsing.
        primary_error: Exception | None = None
        try:
            t0 = time.perf_counter()
            response = client.responses.parse(
                model=settings.OPENAI_MODEL,
                input=messages,
                response_format=response_format,
                temperature=(
                    temperature
                    if temperature is not None
                    else settings.OPENAI_TEMPERATURE
                ),
                max_output_tokens=(
                    max_output_tokens
                    if max_output_tokens is not None
                    else settings.OPENAI_MAX_OUTPUT_TOKENS
                ),
                timeout=settings.OPENAI_TIMEOUT_SEC,
            )
            duration_ms = int((time.perf_counter() - t0) * 1000)
            parsed = getattr(response, "output_parsed", None)
            if parsed is not None:
                log_event(
                    "llm_parse_structured_attempt",
                    path="responses.parse_response_format",
                    success=True,
                    duration_ms=duration_ms,
                    response_format=str(
                        getattr(response_format, "__name__", response_format)
                    ),
                )
                return parsed
            log_event(
                "llm_parse_structured_attempt",
                path="responses.parse_response_format",
                success=False,
                duration_ms=duration_ms,
                reason="output_parsed_empty",
                response_format=str(
                    getattr(response_format, "__name__", response_format)
                ),
            )
        except TypeError:
            # Some SDK versions use text_format instead of response_format.
            t0 = time.perf_counter()
            response = client.responses.parse(
                model=settings.OPENAI_MODEL,
                input=messages,
                text_format=response_format,
                temperature=(
                    temperature
                    if temperature is not None
                    else settings.OPENAI_TEMPERATURE
                ),
                max_output_tokens=(
                    max_output_tokens
                    if max_output_tokens is not None
                    else settings.OPENAI_MAX_OUTPUT_TOKENS
                ),
                timeout=settings.OPENAI_TIMEOUT_SEC,
            )
            duration_ms = int((time.perf_counter() - t0) * 1000)
            parsed = getattr(response, "output_parsed", None)
            if parsed is not None:
                log_event(
                    "llm_parse_structured_attempt",
                    path="responses.parse_text_format",
                    success=True,
                    duration_ms=duration_ms,
                    response_format=str(
                        getattr(response_format, "__name__", response_format)
                    ),
                )
                return parsed
            log_event(
                "llm_parse_structured_attempt",
                path="responses.parse_text_format",
                success=False,
                duration_ms=duration_ms,
                reason="output_parsed_empty",
                response_format=str(
                    getattr(response_format, "__name__", response_format)
                ),
            )
        except Exception as exc:
            # Only fallback for likely SDK capability/signature issues.
            primary_error = exc
            message = str(exc).lower()
            if not any(marker in message for marker in LLM_COMPATIBILITY_MARKERS):
                raise LLMClientError("Structured parse request failed.") from exc

        # Fallback path: beta chat completions parse.
        try:
            t0 = time.perf_counter()
            response = client.beta.chat.completions.parse(
                model=settings.OPENAI_MODEL,
                temperature=(
                    temperature
                    if temperature is not None
                    else settings.OPENAI_TEMPERATURE
                ),
                max_tokens=(
                    max_output_tokens
                    if max_output_tokens is not None
                    else settings.OPENAI_MAX_OUTPUT_TOKENS
                ),
                messages=messages,
                response_format=response_format,
                timeout=settings.OPENAI_TIMEOUT_SEC,
            )
            duration_ms = int((time.perf_counter() - t0) * 1000)
            parsed = response.choices[0].message.parsed if response.choices else None
            if parsed is not None:
                log_event(
                    "llm_parse_structured_attempt",
                    path="beta.chat.completions.parse",
                    success=True,
                    duration_ms=duration_ms,
                    response_format=str(
                        getattr(response_format, "__name__", response_format)
                    ),
                )
                return parsed
            log_event(
                "llm_parse_structured_attempt",
                path="beta.chat.completions.parse",
                success=False,
                duration_ms=duration_ms,
                reason="parsed_empty",
                response_format=str(
                    getattr(response_format, "__name__", response_format)
                ),
            )
        except Exception as exc:
            if primary_error is not None:
                raise LLMClientError(
                    "Structured parse failed in primary and fallback paths."
                ) from primary_error
            raise LLMClientError("Structured parse request failed.") from exc

        raise LLMClientError("Structured parse returned no parsed output.")
