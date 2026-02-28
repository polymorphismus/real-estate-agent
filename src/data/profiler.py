"""Startup metadata profiler for cortex parquet."""

from __future__ import annotations

from functools import lru_cache
import json
from typing import Any

import pandas as pd

from config.constants import DATASET_PATH
from config.metric_registry import SUPPORTED_METRICS
from src.data.repository import get_dataframe
from src.data.constants import PROFILE_VALUE_COLUMNS


def _build_dataset_guide(_dataframe: pd.DataFrame) -> dict[str, Any]:
    """Build compact semantic guide for routing/extraction/codegen prompts."""
    return {
        "column_definitions": {
            "entity_name": "Company/entity managing the assets. The only present is PropCo.",
            "property_name": "Property identifier (e.g., Building 180).",
            "tenant_name": "Tenant identifier where available; may be null.",
            "ledger_type": "High-level financial type, typically revenue or expenses.",
            "ledger_group": "Ledger grouping under a type (e.g., general_expenses, rental_income).",
            "ledger_category": "Detailed financial category under ledger_group.",
            "ledger_code": "Numeric code for accounting line item; if a number like 4xxx/8xxx is mentioned, map here.",
            "ledger_description": "Human-readable description of ledger line item.",
            "month": "Month period in YYYY-MMM format (e.g., 2025-M01).",
            "quarter": "Quarter period in YYYY-QN format (e.g., 2025-Q1).",
            "year": "Year period (e.g., 2025).",
            "profit": "Signed financial value. Positive=Revenue, Negative=Loss.",
        },
        "query_hints": [
            "If query includes compare/comparison, likely comparison task across property_name.",
            "If query includes P&L/profit/loss/revenue/expenses, likely pnl task using profit column.",
            "If query includes a 4-digit accounting number, map it to ledger_code.",
            "If query includes YYYY-MNN, filter month exactly.",
            "If query includes YYYY-QN, filter quarter exactly.",
            "If query includes YYYY only, filter year exactly.",
            "If no timeframe is provided, do not apply a time filter.",
        ],
    }


def _unique_non_null_values(dataframe: pd.DataFrame, column: str) -> list[Any]:
    """Return sorted unique non-null values for a column."""
    values = [value for value in dataframe[column].dropna().unique().tolist()]
    return sorted(values)


def _build_time_ranges(unique_values: dict[str, list[Any]]) -> dict[str, Any]:
    """Build explicit min/max time ranges from profiled unique values."""

    def _min_max(column: str) -> tuple[Any | None, Any | None]:
        values = unique_values.get(column, [])
        if not isinstance(values, list) or not values:
            return None, None
        return values[0], values[-1]

    min_month, max_month = _min_max("month")
    min_quarter, max_quarter = _min_max("quarter")
    min_year, max_year = _min_max("year")
    return {
        "min_month": min_month,
        "max_month": max_month,
        "min_quarter": min_quarter,
        "max_quarter": max_quarter,
        "min_year": min_year,
        "max_year": max_year,
    }


def build_data_profile(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Build compact startup profile from dataframe."""
    unique_values: dict[str, list[Any]] = {}
    for column in PROFILE_VALUE_COLUMNS:
        unique_values[column] = _unique_non_null_values(dataframe, column)
    null_counts = {
        column: int(dataframe[column].isna().sum()) for column in dataframe.columns
    }
    time_ranges = _build_time_ranges(unique_values)

    return {
        "columns": dataframe.columns.tolist(),
        "unique_values": unique_values,
        "null_counts": null_counts,
        **time_ranges,
        "supported_metrics": SUPPORTED_METRICS,
        "dataset_guide": _build_dataset_guide(dataframe),
    }


@lru_cache(maxsize=4)
def get_startup_profile(dataset_path: str = DATASET_PATH) -> dict[str, Any]:
    """Load and cache profile at startup scope for a dataset path."""
    dataframe = get_dataframe(copy=False, dataset_path=dataset_path)
    return build_data_profile(dataframe)


def build_minimal_prompt_profile_json(profile: dict[str, Any] | None) -> str:
    """Build compact JSON profile for prompt context to reduce token usage."""
    profile_data = profile or {}
    columns = profile_data.get("columns", [])
    minimal_profile = {
        "columns": columns if isinstance(columns, list) else [],
        "dataset_guide": profile_data.get("dataset_guide", {}),
        "time_columns": ["month", "quarter", "year"],
        "metric_column": "profit",
        "supported_metrics": profile_data.get("supported_metrics", {}),
        "pnl_definition": (
            "P&L uses ledger_type buckets: revenue_total = sum(profit) for ledger_type='revenue', "
            "expenses_total = sum(profit) for ledger_type='expenses' (typically negative), "
            "and net_pnl = revenue_total + expenses_total"
        ),
    }
    return json.dumps(minimal_profile, ensure_ascii=True)
