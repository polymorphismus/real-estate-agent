"""Minimal Streamlit chat UI for the LangGraph agent."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError
from pathlib import Path
import sys
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.profiler import get_startup_profile
from src.graph.flow import build_graph
from src.graph.states import build_initial_state_dict
from src.services.llm_client import OpenAILLMClient

WAIT_MESSAGES = ("thinking...", "getting your data...", "evaluating...")
WAIT_INTERVAL_SEC = 2.0


def _init_session() -> None:
    """Initialize long-lived app/session objects once."""
    if "graph_app" not in st.session_state:
        st.session_state.graph_app = build_graph()
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = OpenAILLMClient()
    if "data_profile" not in st.session_state:
        st.session_state.data_profile = get_startup_profile()
    if "graph_state" not in st.session_state:
        st.session_state.graph_state = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def _prepare_state_for_query(query: str) -> dict[str, Any]:
    """Build or refresh graph state for the next user query."""
    if st.session_state.graph_state is None:
        state = build_initial_state_dict(user_query=query)
    else:
        state = dict(st.session_state.graph_state)
        state["user_query"] = query
        state["intent"] = None
        state["entities"] = {}
        state["entities_preextracted"] = False
        state["task_type"] = None
        state["python_code"] = None
        state["retrieved_rows"] = []
        state["computed_result"] = None
        state["needs_clarification"] = False
        state["clarification_question"] = None
        state["error_type"] = None
        state["final_answer"] = None
        state["routing_action"] = ""

    state["llm_client"] = st.session_state.llm_client
    state["data_profile"] = st.session_state.data_profile
    return state


def _invoke_with_wait_status(state: dict[str, Any], placeholder: Any) -> dict[str, Any]:
    """Invoke graph and rotate waiting status text every 2.5 seconds."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(st.session_state.graph_app.invoke, state)
        index = 1
        while True:
            try:
                result = future.result(timeout=WAIT_INTERVAL_SEC)
                placeholder.empty()
                return result
            except TimeoutError:
                placeholder.info(WAIT_MESSAGES[index % len(WAIT_MESSAGES)])
                index += 1


def main() -> None:
    """Render chat UI and run LangGraph flow."""
    st.set_page_config(page_title="Real Estate Asset Manager Agent", layout="centered")
    _init_session()

    st.title("Real Estate Asset Manager Agent")
    st.caption("Ask questions about your real-estate assets")

    for item in st.session_state.chat_messages:
        role = item.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(str(item.get("content", "")))

    prompt = st.chat_input("Ask me about your real-estate assets...")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.info(WAIT_MESSAGES[0])
        state = _prepare_state_for_query(prompt)
        try:
            result = _invoke_with_wait_status(state, response_placeholder)
            answer = str(result.get("final_answer", "") or "")
        except Exception:
            answer = "The requested information is not present in the dataset"
            result = state
            result["final_answer"] = answer

        response_placeholder.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.session_state.graph_state = result


if __name__ == "__main__":
    main()
