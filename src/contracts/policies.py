"""Shared execution and compatibility policies."""

from __future__ import annotations

from typing import Any

import pandas as pd

LLM_COMPATIBILITY_MARKERS: tuple[str, ...] = (
    "responses",
    "parse",
    "response_format",
    "text_format",
    "unexpected keyword argument",
    "no attribute",
    "not implemented",
    "unsupported",
)


FORBIDDEN_CODE_PATTERNS: tuple[str, ...] = (
    r"__import__",
    r"\bimport\b",
    r"\beval\(",
    r"\bexec\(",
    r"\bcompile\(",
    r"\bexecfile\(",
    r"\bopen\(",
    r"\bos\.",
    r"\bpathlib\.",
    r"\bshutil\.",
    r"\bglob\.",
    r"\btempfile\.",
    r"\bfnmatch\.",
    r"\bfileinput\.",
    r"\bsubprocess\.",
    r"\bos\.system\(",
    r"\bos\.popen\(",
    r"\bos\.spawn",
    r"\bos\.exec",
    r"\bpexpect\.",
    r"\bpty\.",
    r"\bcommands\.",
    r"\bsocket\.",
    r"\burllib\.",
    r"\burllib2\.",
    r"\brequests\.",
    r"\bhttplib\.",
    r"\bhttp\.client",
    r"\bftplib\.",
    r"\bsmtplib\.",
    r"\bimaplib\.",
    r"\bparamiko\.",
    r"\btwisted\.",
    r"\baiohttp\.",
    r"\bhttpx\.",
    r"\bpickle\.",
    r"\bcPickle\.",
    r"\bmarshal\.",
    r"\bshelve\.",
    r"\byaml\.load\(",
    r"\bjsonpickle\.",
    r"\bgetattr\(",
    r"\bsetattr\(",
    r"\bdelattr\(",
    r"\bhasattr\(",
    r"\bvars\(",
    r"\bdir\(",
    r"\bglobals\(",
    r"\blocals\(",
    r"\b__builtins__",
    r"\b__globals__",
    r"\b__locals__",
    r"\b__code__",
    r"\b__class__",
    r"\b__bases__",
    r"\b__subclasses__\(",
    r"\b__mro__",
    r"\bast\.",
    r"\bdis\.",
    r"\binspect\.",
    r"\btypes\.",
    r"\bctypes\.",
    r"\bcffi\.",
    r"\bmultiprocessing\.",
    r"\bthreading\.",
    r"\bconcurrent\.",
    r"\basyncio\.",
    r"\bsignal\.",
    r"\bos\.environ",
    r"\bos\.getenv\(",
    r"\bdotenv\.",
    r"\bconfigparser\.",
    r"\b\.execute\s*\(\s*['\"]?\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)",
    r"\bDROP\b",
    r"\bTRUNCATE\b",
    r"\bALTER\b",
    r"\bshlex\.",
    r"__class__.*__init__.*__globals__",
    r"\bbreakpoint\(",
    r"\binput\(",
    r"\bmemoryview\(",
    r"\b__debug__",
)


def build_exec_locals(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Build local exec namespace for generated code."""
    return {
        "dataframe": dataframe.copy(),
        "filtered_df": None,
        "result_df": None,
        "result_payload": None,
    }


def build_safe_exec_globals() -> dict[str, Any]:
    """Build restricted globals namespace for generated code."""
    return {
        "pd": pd,
        "__builtins__": {
            "float": float,
            "int": int,
        },
    }
