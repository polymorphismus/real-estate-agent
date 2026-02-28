"""LLM code generation service for dataset querying."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from config.settings import settings
from config.prompts import build_codegen_prompt
from src.data.profiler import build_minimal_prompt_profile_json
from src.contracts.models import CodegenPlanSchema
from src.contracts.policies import (
    FORBIDDEN_CODE_PATTERNS,
    build_exec_locals,
    build_safe_exec_globals,
)
from src.services.llm_client import OpenAILLMClient


def generate_query_code_with_llm(
    user_query: str,
    extracted_entities: dict[str, Any],
    profile: dict[str, Any],
    conversation_messages: list[dict[str, Any]] | None = None,
    client: OpenAILLMClient | None = None,
) -> dict[str, Any]:
    """Generate Python query code using LLM with strict schema validation."""
    llm = client or OpenAILLMClient()
    profile_json = build_minimal_prompt_profile_json(profile)
    user_payload = json.dumps(
        {
            "user_query": user_query,
            "request_target": extracted_entities.get("request_target", []),
            "ranking": extracted_entities.get(
                "ranking", {"mode": "none", "top_k": None}
            ),
            "time_scope": extracted_entities.get("time_scope", {"mode": "none"}),
            "extracted_entities": extracted_entities,
        },
        ensure_ascii=True,
    )
    parsed = llm.parse_structured(
        system_prompt=build_codegen_prompt(profile_json),
        user_prompt=user_payload,
        response_format=CodegenPlanSchema,
        conversation_messages=conversation_messages,
        max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS_CODEGEN,
    )
    return parsed.model_dump()


def execute_generated_python_code(
    dataframe: pd.DataFrame, python_code: str
) -> dict[str, Any]:
    """Execute generated pandas code in restricted namespace.

    This step is for information gathering only and expects `result_df`.
    """
    if not python_code.strip():
        return {"result_df": None, "result_payload": None}

    for pattern in FORBIDDEN_CODE_PATTERNS:
        if re.search(pattern, python_code):
            raise ValueError("Generated code contains forbidden operations.")

    local_vars = build_exec_locals(dataframe)
    safe_globals = build_safe_exec_globals()
    exec(python_code, safe_globals, local_vars)
    result_df = local_vars.get("result_df")
    filtered_df = local_vars.get("filtered_df")
    if result_df is not None and not isinstance(result_df, pd.DataFrame):
        raise ValueError("Generated code must set result_df as a pandas DataFrame.")
    filtered_row_count: int | None = None
    if isinstance(filtered_df, pd.DataFrame):
        filtered_row_count = int(len(filtered_df))
    return {
        "result_df": result_df,
        "filtered_row_count": filtered_row_count,
        "result_payload": local_vars.get("result_payload"),
    }
