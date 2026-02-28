"""Shared schema/data contracts for LLM stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from config.constants import INTENT_DESCRIPTIONS, INTENT_LITERALS

ALLOWED_INTENTS = set(INTENT_LITERALS)


class IntentDecision(BaseModel):
    """Normalized intent decision produced by router stage."""

    intent: str = Field(description="Validated routing intent.")
    action: Literal["continue", "fallback", "clarify"] = Field(
        description="Next graph action after routing."
    )
    fallback_message: str = Field(
        default="",
        description="Fallback message when action is fallback.",
    )
    reason: str = Field(
        default="",
        description="Short explanation of the routing decision.",
    )
    clarification_prompt: str = Field(
        default="",
        description="Clarification message when action is clarify.",
    )


class TimeScopeSchema(BaseModel):
    """Extracted time scope for dataset filtering."""

    mode: str = Field(
        default="none",
        description="Time mode: none|exact|range|relative.",
    )
    month: str | None = Field(
        default=None,
        description="Exact month token in YYYY-MNN format when mode=exact and month is provided.",
    )
    quarter: str | None = Field(
        default=None,
        description="Exact quarter token in YYYY-QN format when mode=exact and quarter is provided.",
    )
    year: str | None = Field(
        default=None,
        description="Exact year token in YYYY format when mode=exact and year is provided.",
    )
    column: str | None = Field(
        default=None,
        description="Column used for range filtering when mode=range (month|quarter|year).",
    )
    start: str | None = Field(
        default=None,
        description="Inclusive range start token when mode=range.",
    )
    end: str | None = Field(
        default=None,
        description="Inclusive range end token when mode=range.",
    )
    relative_period: str | None = Field(
        default=None,
        description=(
            "Relative time label when mode=relative. "
            "Examples: current_year, last_year, next_year, "
            "current_quarter, last_quarter, next_quarter, "
            "current_month, last_month, next_month."
        ),
    )


class RankingSchema(BaseModel):
    """Ranking intent for highest/lowest/top-k style requests."""

    mode: Literal["none", "highest", "lowest"] = Field(
        default="none",
        description="Ranking mode: none|highest|lowest.",
    )
    top_k: int | None = Field(
        default=None,
        description="Requested rank cutoff (e.g., 1 for top result, 5 for top-5).",
    )


class ExtractedEntitiesSchema(BaseModel):
    """Column-aligned extraction output used to guide query code generation."""

    entity_name: list[str] = Field(
        default_factory=list,
        description="Extracted entity_name values from query text.",
    )
    property_name: list[str] = Field(
        default_factory=list,
        description="Extracted property_name values (e.g., Building 160).",
    )
    tenant_name: list[str] = Field(
        default_factory=list,
        description="Extracted tenant_name values from query text.",
    )
    ledger_type: list[str] = Field(
        default_factory=list,
        description="Extracted ledger_type values (e.g., revenue, expenses).",
    )
    ledger_group: list[str] = Field(
        default_factory=list,
        description="Extracted ledger_group values.",
    )
    ledger_category: list[str] = Field(
        default_factory=list,
        description="Extracted ledger_category values.",
    )
    ledger_code: list[str] = Field(
        default_factory=list,
        description="Extracted ledger_code values represented as strings.",
    )
    ledger_description: list[str] = Field(
        default_factory=list,
        description="Extracted ledger_description keywords/values.",
    )
    ledger_raw_mentions: list[str] = Field(
        default_factory=list,
        description=(
            "Raw ledger-like literals from user query when column is uncertain "
            "(for example snake_case/code-like tokens)."
        ),
    )
    request_target: list[str] = Field(
        default_factory=list,
        description="Requested answer target fields (e.g., property_name, tenant_name).",
    )
    requested_metric: str = Field(
        default="",
        description=(
            "Canonical requested metric name (for example: pnl, net_pnl, revenue_total, "
            "expenses_total, count, sum_profit, cap_rate, unknown)."
        ),
    )
    ranking: RankingSchema = Field(
        default_factory=RankingSchema,
        description="Ranking requirement extracted from query wording.",
    )
    time_scope: TimeScopeSchema = Field(
        default_factory=TimeScopeSchema,
        description="Extracted time scope; defaults to whole period with mode=none.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether additional user clarification is required before query generation.",
    )
    clarification_prompt: str = Field(
        default="",
        description="Clarification question when needs_clarification=true.",
    )


class CodegenPlanSchema(BaseModel):
    """Schema for generated Python query plan."""

    task_type: str = Field(description="Short label describing generated query type.")
    python_code: str = Field(default="")
    needs_clarification: bool = False
    clarification_prompt: str = ""


class IntentExtractionSchema(BaseModel):
    """Combined guard output: routing intent plus extracted entities."""

    intent: str = Field(
        description=(
            "Top-level intent selected for routing. "
            + "; ".join(f"{k}={v}" for k, v in INTENT_DESCRIPTIONS.items())
        )
    )
    action: Literal["continue", "fallback", "clarify"] = Field(
        description="Next action for graph routing."
    )
    fallback_message: str = Field(
        default="",
        description="Fallback message when action is fallback, otherwise empty.",
    )
    clarification_prompt: str = Field(
        default="",
        description="Clarification question when action is clarify, otherwise empty.",
    )
    reason: str = Field(description="Short explanation of classification decision.")
    entities: ExtractedEntitiesSchema = Field(
        default_factory=ExtractedEntitiesSchema,
        description="Column-aligned extracted entities for code generation.",
    )

    @field_validator("intent")
    @classmethod
    def validate_combined_intent(cls, value: str) -> str:
        """Ensure combined intent is one of configured literals."""
        if value not in ALLOWED_INTENTS:
            raise ValueError(f"Invalid intent: {value}")
        return value
