"""LangGraph state schema and helper constructors."""

from __future__ import annotations

from typing import Any
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field


class GraphStateDict(TypedDict, total=False):
    """Mapping-based state shape for LangGraph runtime."""

    user_query: str
    intent: str | None
    entities: dict[str, Any]
    entities_preextracted: bool
    task_type: str | None
    python_code: str | None
    data_profile: dict[str, Any] | None
    retrieved_rows: list[dict[str, Any]]
    computed_result: dict[str, Any] | None
    messages: list[dict[str, Any]]
    needs_clarification: bool
    clarification_question: str | None
    error_type: str | None
    final_answer: str | None
    routing_action: str


class GraphState(BaseModel):
    """Strict shared state for graph node execution."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    user_query: str = Field(description="Raw user input for the current turn.")
    intent: str | None = Field(
        default=None, description="Detected top-level intent label for routing."
    )
    entities: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "LLM-extracted, column-aligned entities used to ground code generation "
            "and drive clarification checks (not used as deterministic query filters)."
        ),
    )
    entities_preextracted: bool = Field(
        default=False,
        description="Whether entities were already extracted in guard stage via combined intent+extract call.",
    )
    task_type: str | None = Field(
        default=None, description="LLM-generated task label for current query."
    )
    python_code: str | None = Field(
        default=None,
        description="Generated pandas code for information gathering; must assign result_df.",
    )
    data_profile: dict[str, Any] | None = Field(
        default=None, description="Optional startup metadata profile passed to nodes."
    )
    retrieved_rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Rows produced by executed generated code (result_df serialized to records).",
    )
    computed_result: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Bounded JSON payload derived from result_df and passed to the final answer LLM."
        ),
    )
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation messages for the current session.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether additional user input is required to proceed.",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Clarification prompt shown when query is incomplete/ambiguous.",
    )
    error_type: str | None = Field(
        default=None,
        description="Error or fallback category set by guard/executor nodes.",
    )
    final_answer: str | None = Field(
        default=None, description="Final user-facing response for the turn."
    )


def build_initial_state(
    *,
    user_query: str,
    messages: list[dict[str, Any]] | None = None,
) -> GraphState:
    """Create a default graph state for a new user query."""
    return GraphState(
        user_query=user_query,
        messages=messages or [{"role": "user", "content": user_query}],
    )


def to_state_dict(state: GraphState) -> dict[str, Any]:
    """Convert model to plain dict for graph runtimes expecting mappings."""
    return state.model_dump()


def build_initial_state_dict(
    *,
    user_query: str,
    messages: list[dict[str, Any]] | None = None,
) -> GraphStateDict:
    """Create initial mapping-based state for graph invocation."""
    model_state = build_initial_state(user_query=user_query, messages=messages)
    return to_state_dict(model_state)
