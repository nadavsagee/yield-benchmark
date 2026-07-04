"""Dispatch tool calls to deterministic agent tools (with error guardrails)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from agent.tools.chain_correlate import chain_correlate
from agent.tools.commonality import commonality
from agent.tools.excursion_confirm import excursion_confirm
from agent.tools.inline_trace import inline_trace
from agent.tools.spatial_signature import spatial_signature
from agent.tools.wat_profile import wat_profile


def _json_safe(obj: Any) -> Any:
    """Ensure tool output is JSON-serializable and compact."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return str(obj)


def dispatch(
    name: str,
    args: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """
    Run a named tool. Unknown tools, bad args, or runtime errors return {error: ...}.
    """
    try:
        if name == "spatial_signature":
            lot = args["lot"]
            population = args.get("population", "speed")
            result = spatial_signature(lot, tables, population=population)
        elif name == "wat_profile":
            result = wat_profile(args["lot"], args["param"], tables)
        elif name == "commonality":
            lot = args["lot"]
            population = args.get("population", "yield")
            result = commonality(lot, tables, population=population)
        elif name == "chain_correlate":
            result = chain_correlate(args["lot"], tables)
        elif name == "excursion_confirm":
            lot = args["lot"]
            param = args["param"]
            population = args.get("population", "yield")
            result = excursion_confirm(lot, param, tables, population=population)
        elif name == "inline_trace":
            result = inline_trace(args["lot"], tables)
        elif name == "submit_findings":
            return {"error": "submit_findings is handled by the agent loop, not dispatch"}
        else:
            return {"error": f"unknown tool: {name}"}
        return _json_safe(result)
    except TypeError as exc:
        return {"error": f"bad args: {exc}"}
    except KeyError as exc:
        return {"error": f"bad args: missing {exc}"}
    except Exception as exc:
        return {"error": str(exc)}


def tool_result_content(result: dict[str, Any]) -> str:
    """Compact JSON string for Anthropic tool_result blocks."""
    return json.dumps(result, separators=(",", ":"))
