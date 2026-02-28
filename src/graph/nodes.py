"""Graph nodes for LLM-driven extraction, code generation, execution, and response."""

from __future__ import annotations

import time
from typing import Any

from config.constants import (
    INTENT_DATASET_KNOWLEDGE,
    INTENT_DEFINITIONS,
    MSG_MULTIPLE_QUESTION,
    MSG_NOT_PRESENT,
)
from src.data.repository import get_dataframe
from src.graph.guards import (
    detect_multiple_questions,
    route_query,
)
from src.graph.resolvers import (
    _definitions_intent_is_eligible,
    _ensure_state,
    _is_supported_metric_request,
    _missing_requested_values,
    _resolve_ledger_raw_mentions,
    _resolve_relative_time_scope,
    _time_range_not_present_answer,
)
from src.services.codegen_service import (
    execute_generated_python_code,
    generate_query_code_with_llm,
)
from src.services.intent_service import (
    classify_intent_and_extract_with_llm,
)
from src.services.response_service import (
    answer_from_profile_with_llm,
    answer_from_result_with_llm,
    fallback_for_error_type,
)
from src.utils.logging import log_event


def guard_router_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Run guard checks and combined intent+entity extraction routing."""
    state = _ensure_state(state)
    user_query = str(state.get("user_query", "")).strip()
    if user_query:
        messages = state.setdefault("messages", [])
        if (
            not messages
            or messages[-1].get("role") != "user"
            or str(messages[-1].get("content", "")).strip() != user_query
        ):
            messages.append({"role": "user", "content": user_query})
    log_event("query_received", user_query=user_query)

    if detect_multiple_questions(user_query):
        state["final_answer"] = MSG_MULTIPLE_QUESTION
        state["error_type"] = "not_present"
        state["routing_action"] = "finalize"
        log_event("multiple_questions_blocked", user_query=user_query)
        return state

    decision = route_query(user_query)
    if decision.get("action") == "continue":
        try:
            t0 = time.perf_counter()
            llm_decision, llm_entities = classify_intent_and_extract_with_llm(
                user_query,
                profile=state.get("data_profile", {}),
                conversation_messages=state.get("messages"),
                client=state.get("llm_client"),
            )
            duration_ms = int((time.perf_counter() - t0) * 1000)
            log_event(
                "intent_stage_timing", duration_ms=duration_ms, source="llm_combined"
            )
            decision = {
                "intent": llm_decision.intent,
                "action": llm_decision.action,
                "fallback_message": llm_decision.fallback_message,
                "clarification_prompt": llm_decision.clarification_prompt,
                "reason": llm_decision.reason,
            }
            if (
                decision["action"] == "continue"
                and isinstance(llm_entities, dict)
                and llm_entities
            ):
                if decision[
                    "intent"
                ] == INTENT_DEFINITIONS and not _definitions_intent_is_eligible(
                    llm_entities
                ):
                    decision["intent"] = INTENT_DATASET_KNOWLEDGE
                    log_event(
                        "definitions_downgraded_to_dataset_knowledge", status="ok"
                    )
                state["entities"] = llm_entities
                state["entities_preextracted"] = True
                log_event("guard_preextracted_entities", status="ok")
            elif (
                decision["action"] == "clarify"
                and decision["intent"] == INTENT_DATASET_KNOWLEDGE
                and isinstance(llm_entities, dict)
                and llm_entities
            ):
                # Reorder checks: validate explicit entities first (not_present)
                # before asking clarification (e.g., metric clarification).
                explicit_columns = (
                    "property_name",
                    "tenant_name",
                    "entity_name",
                    "ledger_code",
                )
                has_explicit_entity = any(
                    isinstance(llm_entities.get(column), list)
                    and len(llm_entities.get(column, [])) > 0
                    for column in explicit_columns
                )
                if has_explicit_entity:
                    state["entities"] = llm_entities
                    state["entities_preextracted"] = True
                    decision["action"] = "continue"
                    log_event(
                        "guard_clarify_deferred_for_entity_validation", status="ok"
                    )
            log_event(
                "intent_llm_override",
                intent=decision["intent"],
                action=decision["action"],
            )
        except Exception as exc:
            log_event("intent_extractor_llm_failed", error=str(exc))
            decision = {
                "intent": "ambiguous",
                "action": "clarify",
                "fallback_message": "",
                "clarification_prompt": "Please rephrase your request with the target and time scope.",
                "reason": "combined intent+extraction failed",
            }
    log_event(
        "intent_detected",
        intent=decision.get("intent"),
        action=decision.get("action"),
        reason=decision.get("reason"),
    )
    state["intent"] = decision["intent"]
    state["routing_action"] = decision["action"]

    if decision["action"] == "fallback":
        error_type_map = {
            "adversarial": "adversarial",
            "gibberish": "gibberish",
            "general_knowledge": "out_of_scope",
        }
        state["error_type"] = error_type_map.get(decision["intent"], "not_present")
        state["final_answer"] = decision["fallback_message"]
        log_event(
            "routing_fallback",
            intent=decision.get("intent"),
            error_type=state.get("error_type"),
            fallback_message=state.get("final_answer"),
        )
    elif decision["action"] == "clarify":
        state["needs_clarification"] = True
        state["clarification_question"] = (
            str(decision.get("clarification_prompt", "")).strip()
            or "Please clarify your question."
        )
        log_event(
            "routing_clarification",
            intent=decision.get("intent"),
            clarification_question=state.get("clarification_question"),
            reason=decision.get("reason"),
        )
    return state


def entity_extractor_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Consume preextracted entities from guard stage and run validation."""
    state = _ensure_state(state)
    user_query = str(state.get("user_query", ""))
    data_profile = state.get("data_profile", {})
    entities: dict[str, Any] = {}
    if state.get("entities_preextracted") and isinstance(state.get("entities"), dict):
        entities = state["entities"]
        state["entities_preextracted"] = False
        log_event("extractor_used_preextracted", status="ok")
    else:
        state["needs_clarification"] = True
        state["clarification_question"] = (
            "Please rephrase your request with the target and time scope."
        )
        log_event("extractor_missing_preextracted", status="clarify")
        return state
    state["entities"] = entities
    _resolve_relative_time_scope(entities)
    log_event("entities_extracted", entities=entities)

    unresolved_raw_mentions = _resolve_ledger_raw_mentions(entities, data_profile)
    if unresolved_raw_mentions:
        log_event("ledger_raw_mentions_unresolved", values=unresolved_raw_mentions)

    missing_values = _missing_requested_values(entities, data_profile)
    if unresolved_raw_mentions:
        missing_values.update(unresolved_raw_mentions)
    if missing_values:
        state["error_type"] = "not_present"
        state["final_answer"] = MSG_NOT_PRESENT
        log_event("entities_not_present", missing_values=missing_values)
        return state

    if not _is_supported_metric_request(entities, data_profile):
        state["error_type"] = "not_present"
        state["final_answer"] = MSG_NOT_PRESENT
        log_event(
            "unsupported_metric_not_present",
            requested_metric=entities.get("requested_metric", ""),
        )
        return state

    if entities.get("needs_clarification"):
        state["needs_clarification"] = True
        state["clarification_question"] = entities.get(
            "clarification_prompt", "Please clarify the missing details."
        )
    if state.get("needs_clarification"):
        log_event(
            "clarification_required", question=state.get("clarification_question")
        )

    return state


def query_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Generate executable query code contract from extracted entities."""
    state = _ensure_state(state)
    if state.get("final_answer"):
        return state
    if state.get("needs_clarification"):
        return state

    entities = state.get("entities", {})
    data_profile = state.get("data_profile", {})

    if str(state.get("intent", "")) == INTENT_DEFINITIONS:
        try:
            state["final_answer"] = answer_from_profile_with_llm(
                user_query=str(state.get("user_query", "")),
                profile=data_profile,
                conversation_messages=state.get("messages"),
                client=state.get("llm_client"),
            )
            log_event("definitions_answer_llm_used", status="ok")
        except Exception as exc:
            log_event("definitions_answer_llm_failed", error=str(exc))
            state["needs_clarification"] = True
            state["clarification_question"] = (
                "Please clarify what information you want me to extract."
            )
        return state

    try:
        generated = generate_query_code_with_llm(
            str(state.get("user_query", "")),
            entities,
            profile=data_profile,
            conversation_messages=state.get("messages"),
            client=state.get("llm_client"),
        )
        if generated.get("needs_clarification"):
            state["needs_clarification"] = True
            state["clarification_question"] = generated.get(
                "clarification_prompt", "Please clarify the query."
            )
            return state

        python_code = str(generated.get("python_code", "") or "").strip()
        if not python_code:
            state["needs_clarification"] = True
            state["clarification_question"] = (
                "Please clarify what information you want me to extract."
            )
            return state

        task_type = str(generated.get("task_type", "asset_details"))
        state["task_type"] = task_type
        state["python_code"] = python_code
        log_event("codegen_llm_used", task_type=task_type, python_code=python_code)
    except Exception as exc:
        log_event("codegen_llm_failed", error=str(exc))
        state["needs_clarification"] = True
        state["clarification_question"] = (
            "Please clarify what information you want me to extract."
        )
        return state

    log_event(
        "query_agent_output",
        task_type=state.get("task_type"),
        python_code_present=bool(str(state.get("python_code", "")).strip()),
        python_code=str(state.get("python_code", "") or ""),
    )
    return state


def executor_response_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Execute generated pandas code, serialize result_df, and produce final answer."""
    state = _ensure_state(state)
    if state.get("needs_clarification"):
        return state
    if state.get("final_answer"):
        return state

    python_code = str(state.get("python_code", "") or "").strip()
    if not python_code:
        state["error_type"] = "not_present"
        state["final_answer"] = fallback_for_error_type("not_present")
        return state

    task_type = state.get("task_type")
    data_profile = state.get("data_profile", {})

    try:
        full_df = get_dataframe(copy=True)
        execution = execute_generated_python_code(full_df, python_code)
        filtered_row_count = execution.get("filtered_row_count")
        if filtered_row_count == 0:
            state["error_type"] = "not_present"
            state["final_answer"] = (
                _time_range_not_present_answer(
                    state.get("entities", {}),
                    data_profile,
                )
                or MSG_NOT_PRESENT
            )
            log_event("code_execution_empty_filtered_df", task_type=task_type)
            return state
        result_df = execution.get("result_df")
        if result_df is None:
            state["error_type"] = "not_present"
            state["final_answer"] = MSG_NOT_PRESENT
            return state

        query_df = result_df
        state["retrieved_rows"] = query_df.to_dict(orient="records")
        log_event(
            "code_execution_result_df",
            entities=state.get("entities", {}),
            row_count=len(query_df),
            columns=list(query_df.columns),
            task_type=task_type,
        )

        records = query_df.to_dict(orient="records")
        bounded_records = records[:500]
        state["computed_result"] = {
            "rows": bounded_records,
            "total_rows": len(records),
            "truncated": len(records) > len(bounded_records),
            "task_type": task_type,
        }

        try:
            state["final_answer"] = answer_from_result_with_llm(
                user_query=str(state.get("user_query", "")),
                result_payload=state["computed_result"],
                profile=state.get("data_profile", {}),
                conversation_messages=state.get("messages"),
                client=state.get("llm_client"),
            )
            log_event("answer_llm_used", status="ok")
        except Exception as exc:
            log_event("answer_llm_failed", error=str(exc))
            state["error_type"] = "not_present"
            state["final_answer"] = MSG_NOT_PRESENT
    except Exception as exc:
        log_event("code_execution_failed", error=str(exc))
        state["error_type"] = "not_present"
        state["final_answer"] = MSG_NOT_PRESENT
    return state


def clarification_node(state: dict[str, Any]) -> dict[str, Any]:
    """Render clarification prompt as final answer when needed."""
    state = _ensure_state(state)
    if state.get("needs_clarification") and state.get("clarification_question"):
        state["final_answer"] = str(state["clarification_question"])
    return state


def finalize_node(state: dict[str, Any]) -> dict[str, Any]:
    """Finalize response and append assistant message to conversation."""
    state = _ensure_state(state)

    if not state.get("final_answer") and state.get("error_type"):
        state["final_answer"] = fallback_for_error_type(str(state["error_type"]))

    if state.get("final_answer"):
        state["messages"].append(
            {"role": "assistant", "content": str(state.get("final_answer", ""))}
        )

    log_event(
        "final_response",
        outcome_category=state.get("error_type") or "factual",
        final_answer=state.get("final_answer"),
    )
    return state
