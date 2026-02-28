"""LangGraph flow wiring for the real-estate agent."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    clarification_node,
    entity_extractor_agent,
    executor_response_agent,
    finalize_node,
    guard_router_agent,
    query_agent,
)
from src.graph.states import GraphStateDict

GuardRoute = Literal["extract", "clarify", "finalize"]
ExtractorRoute = Literal["query", "clarify", "finalize"]
QueryRoute = Literal["executor", "clarify", "finalize"]
SharedRoute = Literal["continue", "clarify", "finalize"]


def _shared_route_decision(state: GraphStateDict) -> SharedRoute:
    """Shared routing decision: finalize > clarify > continue."""
    if state.get("final_answer"):
        return "finalize"
    if state.get("needs_clarification"):
        return "clarify"
    return "continue"


def _route_after_guard(state: GraphStateDict) -> GuardRoute:
    """Route after guard using shared decision helper."""
    decision = _shared_route_decision(state)
    if decision == "continue":
        return "extract"
    return decision


def _route_after_extractor(state: GraphStateDict) -> ExtractorRoute:
    """Route after extractor using shared decision helper."""
    decision = _shared_route_decision(state)
    if decision == "continue":
        return "query"
    return decision


def _route_after_query_agent(state: GraphStateDict) -> QueryRoute:
    """Route after query agent using shared decision helper."""
    decision = _shared_route_decision(state)
    if decision == "continue":
        return "executor"
    return decision


def build_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(GraphStateDict)

    graph.add_node("guard_router_agent", guard_router_agent)
    graph.add_node("entity_extractor_agent", entity_extractor_agent)
    graph.add_node("query_agent", query_agent)
    graph.add_node("executor_response_agent", executor_response_agent)
    graph.add_node("clarification_node", clarification_node)
    graph.add_node("finalize_node", finalize_node)

    graph.add_edge(START, "guard_router_agent")
    graph.add_conditional_edges(
        "guard_router_agent",
        _route_after_guard,
        {
            "extract": "entity_extractor_agent",
            "clarify": "clarification_node",
            "finalize": "finalize_node",
        },
    )
    graph.add_conditional_edges(
        "entity_extractor_agent",
        _route_after_extractor,
        {
            "query": "query_agent",
            "clarify": "clarification_node",
            "finalize": "finalize_node",
        },
    )
    graph.add_conditional_edges(
        "query_agent",
        _route_after_query_agent,
        {
            "executor": "executor_response_agent",
            "clarify": "clarification_node",
            "finalize": "finalize_node",
        },
    )
    graph.add_edge("executor_response_agent", "finalize_node")
    graph.add_edge("clarification_node", "finalize_node")
    graph.add_edge("finalize_node", END)

    return graph.compile()
