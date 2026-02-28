"""Helper functions for graph node state normalization, entity resolution, and time handling."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from config.constants import MSG_NOT_PRESENT
from config.matching_rules import MISSING_CHECK_COLUMNS
from config.month_labels import MONTH_LABELS
from src.data.profiler import get_startup_profile
from src.services.llm_client import OpenAILLMClient


def _ensure_state(state: dict[str, Any]) -> dict[str, Any]:
    """Ensure mutable dict with required keys exists."""
    state.setdefault("messages", [])
    state.setdefault("entities", {})
    if not isinstance(state.get("llm_client"), OpenAILLMClient):
        state["llm_client"] = OpenAILLMClient()
    if not isinstance(state.get("data_profile"), dict):
        state["data_profile"] = get_startup_profile()
    state.setdefault("needs_clarification", False)
    state.setdefault("clarification_question", None)
    state.setdefault("entities_preextracted", False)
    return state


def _normalize_text(value: str) -> str:
    """Normalize free text for tolerant string matching."""
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _resolve_requested_values(
    requested_values: list[str],
    allowed_raw: list[Any],
) -> tuple[list[str], list[str]]:
    """Resolve extracted requested values to canonical dataset values using exact/partial matching."""
    allowed_strings = [str(value) for value in allowed_raw if value is not None]
    if not allowed_strings:
        return requested_values, []

    normalized_to_originals: dict[str, list[str]] = {}
    for value in allowed_strings:
        norm = _normalize_text(value)
        if norm:
            normalized_to_originals.setdefault(norm, []).append(value)

    def _find_best_matches(raw_value: str) -> list[str]:
        norm_value = _normalize_text(raw_value)
        if not norm_value:
            return []
        if norm_value in normalized_to_originals:
            return normalized_to_originals[norm_value]

        substring_matches: list[str] = []
        for allowed_norm, originals in normalized_to_originals.items():
            if norm_value in allowed_norm or allowed_norm in norm_value:
                substring_matches.extend(originals)

        if substring_matches:
            return sorted(set(substring_matches), key=lambda item: len(item))
        return []

    resolved: list[str] = []
    unresolved: list[str] = []

    for value in requested_values:
        raw_value = str(value).strip()
        if not raw_value:
            continue
        matches = _find_best_matches(raw_value)
        if not matches:
            unresolved.append(raw_value)
            continue
        resolved.append(matches[0])

    if len(requested_values) >= 2:
        joined = " ".join(
            str(value).strip() for value in requested_values if str(value).strip()
        )
        if joined:
            joined_matches = _find_best_matches(joined)
            if joined_matches:
                canonical = joined_matches[0]
                if canonical not in resolved:
                    resolved = [
                        item
                        for item in resolved
                        if _normalize_text(item) != _normalize_text(canonical)
                    ]
                    resolved.append(canonical)
                unresolved = [
                    item
                    for item in unresolved
                    if _normalize_text(item) not in _normalize_text(canonical)
                ]

    deduped_resolved: list[str] = []
    seen = set()
    for item in resolved:
        key = _normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            deduped_resolved.append(item)

    return deduped_resolved, unresolved


def _missing_requested_values(
    entities: dict[str, Any],
    data_profile: dict[str, Any],
) -> dict[str, list[str]]:
    """Return requested values that do not exist in dataset unique values."""
    ledger_columns = (
        "ledger_type",
        "ledger_group",
        "ledger_category",
        "ledger_code",
        "ledger_description",
    )

    def _is_explicit_identifier(value: str) -> bool:
        normalized = _normalize_text(value)
        if not normalized:
            return False
        if any(char.isdigit() for char in normalized):
            return True
        tokens = normalized.split()
        if len(tokens) == 1:
            return False
        if all(len(token) <= 2 for token in tokens):
            return False
        return len(tokens) >= 2

    def _try_cross_column_ledger_rescue(
        *,
        source_column: str,
        raw_value: str,
        unique_values: dict[str, Any],
    ) -> tuple[str, str] | None:
        """Rescue unresolved ledger value by matching strongly in sibling ledger columns."""
        if source_column not in ledger_columns:
            return None
        normalized_raw = _normalize_text(raw_value)
        if not normalized_raw:
            return None

        candidates: list[tuple[str, str]] = []
        for column in ledger_columns:
            allowed_raw = unique_values.get(column, [])
            if not isinstance(allowed_raw, list):
                continue
            for allowed_value in allowed_raw:
                if allowed_value is None:
                    continue
                allowed_str = str(allowed_value)
                if _normalize_text(allowed_str) == normalized_raw:
                    candidates.append((column, allowed_str))

        deduped: list[tuple[str, str]] = []
        seen = set()
        for column, value in candidates:
            key = (column, _normalize_text(value))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((column, value))

        if len(deduped) != 1:
            return None
        return deduped[0]

    unique_values = (
        data_profile.get("unique_values", {}) if isinstance(data_profile, dict) else {}
    )
    missing: dict[str, list[str]] = {}
    for column in MISSING_CHECK_COLUMNS:
        requested = entities.get(column, [])
        if not isinstance(requested, list) or not requested:
            continue
        allowed_raw = unique_values.get(column, [])
        resolved_values, absent = _resolve_requested_values(
            [str(value) for value in requested],
            allowed_raw if isinstance(allowed_raw, list) else [],
        )
        entities[column] = resolved_values
        if absent:
            rescued_absent: list[str] = []
            for raw_value in absent:
                rescue = _try_cross_column_ledger_rescue(
                    source_column=column,
                    raw_value=raw_value,
                    unique_values=unique_values,
                )
                if not rescue:
                    rescued_absent.append(raw_value)
                    continue

                destination_column, canonical_value = rescue
                destination_values = entities.get(destination_column, [])
                if not isinstance(destination_values, list):
                    destination_values = []
                if not any(
                    _normalize_text(existing) == _normalize_text(canonical_value)
                    for existing in destination_values
                ):
                    destination_values.append(canonical_value)
                entities[destination_column] = destination_values

            absent = rescued_absent

        if not absent:
            continue

        explicit_absent = [item for item in absent if _is_explicit_identifier(item)]
        if explicit_absent:
            missing[column] = explicit_absent
    return missing


def _resolve_ledger_raw_mentions(
    entities: dict[str, Any],
    data_profile: dict[str, Any],
) -> dict[str, list[str]]:
    """Resolve raw ledger mentions into canonical ledger_* columns when uniquely matched."""
    raw_mentions = entities.get("ledger_raw_mentions", [])
    if not isinstance(raw_mentions, list) or not raw_mentions:
        return {}

    unique_values = (
        data_profile.get("unique_values", {}) if isinstance(data_profile, dict) else {}
    )
    ledger_columns = (
        "ledger_type",
        "ledger_group",
        "ledger_category",
        "ledger_code",
        "ledger_description",
    )

    unresolved: list[str] = []
    for raw in raw_mentions:
        raw_value = str(raw).strip()
        if not raw_value:
            continue
        raw_norm = _normalize_text(raw_value)
        if not raw_norm:
            continue

        exact_candidates: list[tuple[str, str]] = []
        substring_candidates: list[tuple[str, str]] = []
        for column in ledger_columns:
            allowed_raw = unique_values.get(column, [])
            if not isinstance(allowed_raw, list):
                continue
            for allowed in allowed_raw:
                if allowed is None:
                    continue
                canonical = str(allowed)
                canonical_norm = _normalize_text(canonical)
                if not canonical_norm:
                    continue
                if canonical_norm == raw_norm:
                    exact_candidates.append((column, canonical))
                elif raw_norm in canonical_norm:
                    substring_candidates.append((column, canonical))

        def _dedupe(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
            deduped: list[tuple[str, str]] = []
            seen = set()
            for col, val in candidates:
                key = (col, _normalize_text(val))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append((col, val))
            return deduped

        exact_candidates = _dedupe(exact_candidates)
        substring_candidates = _dedupe(substring_candidates)

        chosen: tuple[str, str] | None = None
        if len(exact_candidates) == 1:
            chosen = exact_candidates[0]
        elif len(exact_candidates) > 1:
            unresolved.append(raw_value)
            continue
        elif len(substring_candidates) == 1:
            chosen = substring_candidates[0]
        else:
            unresolved.append(raw_value)
            continue

        destination_column, canonical_value = chosen
        destination_values = entities.get(destination_column, [])
        if not isinstance(destination_values, list):
            destination_values = []
        if not any(
            _normalize_text(existing) == _normalize_text(canonical_value)
            for existing in destination_values
        ):
            destination_values.append(canonical_value)
        entities[destination_column] = destination_values

    entities["ledger_raw_mentions"] = unresolved
    if unresolved:
        return {"ledger_raw_mentions": unresolved}
    return {}


def _resolve_relative_time_scope(entities: dict[str, Any]) -> None:
    """Normalize and resolve structured time_scope into a single canonical form."""
    time_scope = entities.get("time_scope")
    if not isinstance(time_scope, dict):
        time_scope = {
            "mode": "none",
            "month": None,
            "quarter": None,
            "year": None,
            "column": None,
            "start": None,
            "end": None,
            "relative_period": None,
        }
        entities["time_scope"] = time_scope

    relative_period = str(time_scope.get("relative_period") or "").strip().lower()
    if relative_period:
        now = datetime.now()
        if relative_period == "current_year":
            time_scope.update(
                {
                    "mode": "exact",
                    "year": f"{now.year}",
                    "quarter": None,
                    "month": None,
                    "column": None,
                    "start": None,
                    "end": None,
                    "relative_period": None,
                }
            )
        elif relative_period == "last_year":
            time_scope.update(
                {
                    "mode": "exact",
                    "year": f"{now.year - 1}",
                    "quarter": None,
                    "month": None,
                    "column": None,
                    "start": None,
                    "end": None,
                    "relative_period": None,
                }
            )
        elif relative_period == "next_year":
            time_scope.update(
                {
                    "mode": "exact",
                    "year": f"{now.year + 1}",
                    "quarter": None,
                    "month": None,
                    "column": None,
                    "start": None,
                    "end": None,
                    "relative_period": None,
                }
            )
        elif relative_period in ("current_quarter", "last_quarter", "next_quarter"):
            quarter_index = ((now.month - 1) // 3) + 1
            year = now.year
            if relative_period == "last_quarter":
                quarter_index -= 1
                if quarter_index == 0:
                    quarter_index = 4
                    year -= 1
            elif relative_period == "next_quarter":
                quarter_index += 1
                if quarter_index == 5:
                    quarter_index = 1
                    year += 1
            time_scope.update(
                {
                    "mode": "exact",
                    "quarter": f"{year}-Q{quarter_index}",
                    "year": None,
                    "month": None,
                    "column": None,
                    "start": None,
                    "end": None,
                    "relative_period": None,
                }
            )
        elif relative_period in ("current_month", "last_month", "next_month"):
            year = now.year
            month = now.month
            if relative_period == "last_month":
                month -= 1
                if month == 0:
                    month = 12
                    year -= 1
            elif relative_period == "next_month":
                month += 1
                if month == 13:
                    month = 1
                    year += 1
            time_scope.update(
                {
                    "mode": "exact",
                    "month": f"{year}-M{month:02d}",
                    "year": None,
                    "quarter": None,
                    "column": None,
                    "start": None,
                    "end": None,
                    "relative_period": None,
                }
            )

    mode = str(time_scope.get("mode", "none"))
    month = str(time_scope.get("month") or "").strip()
    quarter = str(time_scope.get("quarter") or "").strip()
    year = str(time_scope.get("year") or "").strip()

    if month or quarter or year:
        time_scope["mode"] = "exact"
        time_scope["relative_period"] = None
        if month:
            time_scope["month"] = month
            time_scope["quarter"] = None
            time_scope["year"] = None
        elif quarter:
            time_scope["month"] = None
            time_scope["quarter"] = quarter
            time_scope["year"] = None
        else:
            time_scope["month"] = None
            time_scope["quarter"] = None
            time_scope["year"] = year
    elif mode == "exact":
        time_scope.update(
            {
                "mode": "none",
                "relative_period": None,
            }
        )

    if str(time_scope.get("mode", "none")) == "exact":
        entities["needs_clarification"] = False
        entities["clarification_prompt"] = ""


def _format_time_scope_request(time_scope: dict[str, Any]) -> str | None:
    """Return a human-readable requested time period from extracted time_scope."""
    if not isinstance(time_scope, dict):
        return None
    if str(time_scope.get("mode", "none")) != "exact":
        return None

    month = str(time_scope.get("month") or "").strip()
    quarter = str(time_scope.get("quarter") or "").strip()
    year = str(time_scope.get("year") or "").strip()

    if month:
        return f"month {month}"
    if quarter:
        return f"quarter {quarter}"
    if year:
        return f"year {year}"
    return None


def _format_available_time_range(data_profile: dict[str, Any]) -> str | None:
    """Return a human-readable available dataset time range."""

    def _format_month_token(value: str) -> str:
        match = re.fullmatch(r"(\d{4})-(M\d{2})", value)
        if not match:
            return value
        year, month_token = match.groups()
        month_name = MONTH_LABELS.get(month_token)
        if not month_name:
            return value
        return f"{month_name} {year}"

    if not isinstance(data_profile, dict):
        return None

    min_month = str(data_profile.get("min_month") or "").strip()
    max_month = str(data_profile.get("max_month") or "").strip()
    if min_month and max_month:
        return (
            f"from {_format_month_token(min_month)} to {_format_month_token(max_month)}"
        )

    min_quarter = str(data_profile.get("min_quarter") or "").strip()
    max_quarter = str(data_profile.get("max_quarter") or "").strip()
    if min_quarter and max_quarter:
        return f"from {min_quarter} to {max_quarter}"

    min_year = str(data_profile.get("min_year") or "").strip()
    max_year = str(data_profile.get("max_year") or "").strip()
    if min_year and max_year:
        return f"from {min_year} to {max_year}"

    return None


def _time_range_not_present_answer(
    entities: dict[str, Any],
    data_profile: dict[str, Any],
) -> str | None:
    """Return a specific out-of-range time message when available."""
    requested = _format_time_scope_request(entities.get("time_scope", {}))
    available = _format_available_time_range(data_profile)
    if not requested or not available:
        return None
    return f"You are asking for information in {requested}, but the information I have is {available}."


def _is_supported_metric_request(
    entities: dict[str, Any],
    data_profile: dict[str, Any],
) -> bool:
    """Validate extracted requested_metric against profile-supported metrics."""
    requested_metric = str(entities.get("requested_metric", "") or "").strip().lower()
    if not requested_metric or requested_metric == "unknown":
        return True

    supported_metrics = (
        data_profile.get("supported_metrics", {})
        if isinstance(data_profile, dict)
        else {}
    )
    if not isinstance(supported_metrics, dict):
        return True
    metric_spec = supported_metrics.get(requested_metric)
    if not isinstance(metric_spec, dict):
        return False

    required = metric_spec.get("required_columns", [])
    columns = data_profile.get("columns", []) if isinstance(data_profile, dict) else []
    if not isinstance(required, list) or not isinstance(columns, list):
        return True
    column_set = {str(column) for column in columns}
    return all(str(column) in column_set for column in required)


def _definitions_intent_is_eligible(entities: dict[str, Any]) -> bool:
    """Return True only for pure profile-only explanatory questions."""
    concrete_value_columns = (
        "entity_name",
        "property_name",
        "tenant_name",
        "ledger_type",
        "ledger_group",
        "ledger_category",
        "ledger_code",
        "ledger_description",
        "ledger_raw_mentions",
    )
    for column in concrete_value_columns:
        value = entities.get(column, [])
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return False

    request_target = entities.get("request_target", [])
    if isinstance(request_target, list) and any(
        str(item).strip() for item in request_target
    ):
        return False

    requested_metric = str(entities.get("requested_metric", "") or "").strip().lower()
    if requested_metric and requested_metric != "unknown":
        return False

    ranking = entities.get("ranking", {})
    if isinstance(ranking, dict):
        if str(ranking.get("mode", "none")) != "none":
            return False
        if ranking.get("top_k") is not None:
            return False

    time_scope = entities.get("time_scope", {})
    if isinstance(time_scope, dict) and str(time_scope.get("mode", "none")) != "none":
        return False

    if bool(entities.get("needs_clarification")):
        return False

    return True
