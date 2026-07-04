"""Anthropic tool schemas for v0_baseline agent."""

from __future__ import annotations

from schema import ANOMALY_TYPES, INLINE_STEPS, SOFT_BINS, WAT_PARAMS

_ANOMALY_LIST = ", ".join(ANOMALY_TYPES)
_WAT_LIST = ", ".join(WAT_PARAMS)
_STEPS_LIST = ", ".join(INLINE_STEPS)
_BINS_LIST = ", ".join(SOFT_BINS)


def _inline_trace_schema() -> dict:
    return {
        "name": "inline_trace",
        "description": (
            "Inline metrology level and multi-lot trend for one lot across all inline "
            "steps and metrics (overlay_nm, cd_nm, film_A, defect_density). Returns the "
            "strongest step/metric with level_sigma, trend_slope, sustained drift, and "
            "out_of_control. Use to detect slow sustained inline drift before WAT/Sort move."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lot": {"type": "string", "description": "Lot id, e.g. lot_038"},
            },
            "required": ["lot"],
        },
    }


def tool_schemas(*, version: str = "v0") -> list[dict]:
    """Return tool definitions for the Anthropic messages API."""
    pop_desc = f"Sort view: 'yield', 'fail', or soft_bin ({_BINS_LIST})."
    schemas = [
        {
            "name": "spatial_signature",
            "description": (
                "Spatial fail pattern on sort data for one lot vs the population. "
                "Use to detect edge_ring, center, or gradient signatures."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "lot": {"type": "string", "description": "Lot id, e.g. lot_012"},
                    "population": {
                        "type": "string",
                        "description": f"{pop_desc} Use 'fail' for overall spatial fails.",
                    },
                },
                "required": ["lot"],
            },
        },
        {
            "name": "wat_profile",
            "description": (
                "WAT parametric profile for one lot vs population. "
                f"Params: {_WAT_LIST}. Returns sigma_shift, radial edge gradient, bimodal flag."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "lot": {"type": "string"},
                    "param": {"type": "string", "enum": WAT_PARAMS},
                },
                "required": ["lot", "param"],
            },
        },
        {
            "name": "commonality",
            "description": (
                "Chamber/tool commonality within a lot at gate_etch. "
                "Finds which chamber groups wafers with distinct yield or fail rate."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "lot": {"type": "string"},
                    "population": {
                        "type": "string",
                        "description": pop_desc,
                    },
                },
                "required": ["lot"],
            },
        },
        {
            "name": "chain_correlate",
            "description": (
                "Test inline defect -> WAT param -> sort speed chain for one lot. "
                f"Inline steps: {_STEPS_LIST}. Returns chain_intact, links, break_at."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "lot": {"type": "string"},
                },
                "required": ["lot"],
            },
        },
        {
            "name": "excursion_confirm",
            "description": (
                "SPC-style confirmation: is a lot out of control on a WAT param or sort metric? "
                "Use before concluding an excursion is real (not noise)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "lot": {"type": "string"},
                    "param": {
                        "type": "string",
                        "description": f"WAT param ({_WAT_LIST}) or sort metric (yield, fail, soft_bin).",
                    },
                    "population": {
                        "type": "string",
                        "description": "Sort population when param is a WAT param (default yield).",
                    },
                },
                "required": ["lot", "param"],
            },
        },
    ]
    if version == "v1":
        schemas.append(_inline_trace_schema())
    schemas.append(
        {
            "name": "submit_findings",
            "description": (
                "Submit final investigation findings. Call only after confirming whether a "
                f"real excursion exists. type must be one of: {_ANOMALY_LIST}."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "detected": {
                        "type": "boolean",
                        "description": "True if a real process excursion was confirmed.",
                    },
                    "type": {
                        "type": "string",
                        "enum": ANOMALY_TYPES,
                        "description": "Anomaly type; use 'clean' when no excursion.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Chamber/tool (e.g. CH-A) when chamber_specific; omit or null if unknown.",
                    },
                    "origin_step": {
                        "type": "string",
                        "description": f"Inline origin step ({_STEPS_LIST}) when supported; omit or null.",
                    },
                    "affected_param": {
                        "type": "string",
                        "enum": WAT_PARAMS,
                        "description": "Primary WAT param when supported by evidence; omit or null.",
                    },
                    "cause": {
                        "type": "string",
                        "description": "Short physical explanation of root cause or why clean.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "low"],
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief investigation narrative (not scored).",
                    },
                },
                "required": [
                    "detected",
                    "type",
                    "cause",
                    "confidence",
                    "reasoning",
                ],
            },
        }
    )
    return schemas
