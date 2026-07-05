"""v1 agent — v0 baseline + inline_trace tool and early_detection prompt rule."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic

from schema import ANOMALY_TYPES, FINDINGS_CONTRACT, INLINE_STEPS, WAT_PARAMS

from agent.dispatch import dispatch, tool_result_content
from agent.prestep import STRONG_ANOMALY_THRESHOLD, run_prestep
from agent.tool_schemas import tool_schemas
from agent.tools._load import load_tables

MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 14
MAX_TOKENS = 4096


def system_prompt() -> str:
    types_list = ", ".join(ANOMALY_TYPES)
    steps_list = ", ".join(INLINE_STEPS)
    params_list = ", ".join(WAT_PARAMS)

    return f"""You are a senior IC foundry yield engineer investigating synthetic fab data.

You receive four tables (sort, wat, inline, route) for one dataset. You do NOT have ground truth.
The opening user message reports whether the pre-step flagged a strong suspect lot (score >= {STRONG_ANOMALY_THRESHOLD}).

DEFAULT CONCLUSION IS CLEAN — this overrides weak or ambiguous tool signals:
- Unless you prove otherwise, your final answer is detected=false, type="clean".
- Set detected=true ONLY after you identify a confirmed driver that clears ALL THREE bars:
  (1) large sigma shift — well above lot-to-lot noise, not a marginal blip;
  (2) out of control on excursion_confirm (WAT/Sort) OR out_of_control=true on inline_trace
      when the driver is inline metrology drift AND the pre-step gave a strong lead (see below);
  (3) tied to the failure mechanism — a parametric WAT shift, inline step metric drift,
      or Sort bin pattern directly linked to the hit. Yield alone is never sufficient.
- A marginally out-of-control yield number with no confirmed parametric or inline driver
  is normal variation → detected=false, type="clean". Do not treat Sort yield OOC as proof.
- Pre-step scores below {STRONG_ANOMALY_THRESHOLD}, spatial noise, chamber co-occurrence,
  or one weak tool hit are not confirmed drivers. Do NOT manufacture an excursion from ambiguous signals.
- If after investigation no driver clears the three-bar test, conclude clean — do not
  invent a diffuse multi-factor or catch-all cause to justify detected=true.

Pre-step gate — when to investigate vs stop:
- NO strong pre-step lead (opening message says nothing strongly anomalous, or top score <
  {STRONG_ANOMALY_THRESHOLD}): default to clean. Do NOT call inline_trace to hunt for drift.
  Do NOT promote faint inline wobble or minor lot-to-lot variation to an excursion.
  Verify quickly with a small set of tools if needed, then submit detected=false.
- STRONG pre-step lead (flagged suspect lot with score >= {STRONG_ANOMALY_THRESHOLD}):
  investigate that lot (and related lots if the narrative suggests a sequence). inline_trace
  is appropriate ONLY on this path — to confirm or reject the pre-step inline/drift lead.

Confirmed sustained inline drift — valid driver ONLY with a strong pre-step lead:
- Apply this path ONLY when the pre-step flagged a suspect lot at or above the strong
  threshold stated in the opening message.
- On that flagged lot, inline_trace must return out_of_control=true with sustained, monotonic
  trailing-window drift (large level_sigma AND meaningful trend_slope/sustained — not a faint wobble).
  Then the inline step/metric IS a confirmed driver even if WAT and Sort stay in-family.
  Set type="early_detection", detected=true, origin_step from inline_trace.step.
- Without a strong pre-step lead, inline_trace results are NOT sufficient for detected=true —
  even if a lot shows minor inline variation. Clean means no confirmed driver anywhere.

Methodology (follow in order):
1. Characterize — read the pre-step lead first. No strong lead → lean clean; do not over-investigate.
2. Localize — which lot, chamber, inline step, or wafer region? Focus on the pre-step suspect if flagged.
3. Confirm causation — does inline -> WAT -> sort chain hold? Co-location is NOT causation.
   Use inline_trace ONLY when the pre-step gave a strong lead; use chain_correlate for
   inline-to-WAT-to-Sort propagation links.
4. Confirm excursion — excursion_confirm for WAT/Sort; inline_trace only on the strong-lead path above.

Type decision hierarchy (apply ONLY after a confirmed driver has cleared the three-bar test;
when signals overlap, pick the first matching rule):
1. edge_signature — edge-ring/radial spatial signature (spatial_signature) with an edge-skewed
   WAT shift (wat_profile radial/edge pattern). Classify as edge_signature even if the lot
   also commons to a chamber. Defining evidence: the spatial signature.
2. propagation — intact inline→WAT→Sort chain (chain_correlate chain_intact=true) with an
   elevated inline defect at the origin step in chain_correlate links. Defining evidence:
   defect-driven chain origin, not chamber co-occurrence alone.
3. chamber_specific — commonality to one chamber plus a parametric WAT shift, but inline
   defects are normal (chain_correlate breaks at the WAT node, not at a defect origin).
   Defining evidence: chamber commonality with a parametric — not defect — mechanism.
   Set location to commons_to. Set origin_step by looking up that chamber in route for this
   lot and recording the inline step where those wafers were processed — derive from routing
   data only; never assume a fixed step name.
4. early_detection — ONLY when pre-step flagged a strong suspect (score at or above the
   opening-message threshold) AND inline_trace on that lot confirms sustained inline drift
   (out_of_control=true, sustained/monotonic trend) with WAT and Sort still in-family.
   Set origin_step from inline_trace.step.
5. correlation_break — confirmed Sort failure with genuinely normal WAT and no parametric or
   inline driver. Defining evidence: Sort OOC, WAT in-family. Do NOT use as a catch-all.

If no rule above matches with confirmed evidence, or the three-bar test was not met, default
to clean (detected=false).

Rules:
- Call deterministic tools to get numbers; never invent measurements.
- No strong pre-step lead → do not call inline_trace; conclude clean unless another driver clears all three bars.
- Do not submit detected=true or a typed excursion without a confirmed driver that clears
  all three bars (large sigma shift AND out-of-control AND tied to the failure).
- type must be exactly one of: {types_list}
- Set type to "clean" with detected=false when no excursion is confirmed.
- Only fill location, origin_step, affected_param when tool evidence supports them; use null otherwise.
- Inline steps vocabulary: {steps_list}
- WAT params vocabulary: {params_list}

When ready, call submit_findings once with your final diagnosis."""


def _format_driver(feature: str, direction: str) -> str:
    name = feature
    if name.startswith("wat_"):
        name = name[4:]
    name = name.replace("_", " ")
    return f"{name} ({direction})"


def seed_user_message(prestep: dict[str, Any]) -> str:
    """Build the opening user message from pre-step results."""
    goal = (
        "Investigate this dataset for process excursions. "
        "Use the tools to characterize, localize, confirm causation, "
        "and confirm whether any excursion is real."
    )
    top = prestep["suspects"][0]
    threshold = prestep.get("strong_threshold", STRONG_ANOMALY_THRESHOLD)

    if top["anomaly_score"] >= threshold:
        drivers = ", ".join(
            _format_driver(d["feature"], d["direction"])
            for d in top["top_features"][:3]
        )
        lead = (
            f"The unsupervised pre-step flags {top['lot']} as most anomalous "
            f"(score={top['anomaly_score']:.2f}), driven by {drivers}. "
            f"Investigate and diagnose."
        )
    else:
        lead = (
            "The pre-step found nothing strongly anomalous — verify whether this is clean."
        )

    return f"{goal}\n\n{lead}"


def normalize_findings(raw: dict[str, Any]) -> dict[str, Any]:
    """Ensure findings match FINDINGS_CONTRACT."""
    findings: dict[str, Any] = {}
    for key in FINDINGS_CONTRACT:
        if key in raw:
            findings[key] = raw[key]
        elif key in ("location", "origin_step", "affected_param", "type"):
            findings[key] = None
        elif key == "detected":
            findings[key] = False
        else:
            findings[key] = ""
    if findings["type"] is None and not findings["detected"]:
        findings["type"] = "clean"
    return findings


def _fallback_findings(reason: str) -> dict[str, Any]:
    return normalize_findings(
        {
            "detected": False,
            "type": "clean",
            "location": None,
            "origin_step": None,
            "affected_param": None,
            "cause": reason,
            "confidence": "low",
            "reasoning": reason,
        }
    )


def investigate(
    dataset_path: str | Path,
    *,
    api_key: str | None = None,
    model: str = MODEL,
    max_iterations: int = MAX_ITERATIONS,
    run_seed: int | None = None,
    run_index: int | None = None,
) -> dict[str, Any]:
    """
    Run v1 investigation on a dataset directory.

    Returns {findings, trace, prestep, iterations, model, run_seed, run_index}.
    findings matches FINDINGS_CONTRACT.
    run_seed / run_index are echoed for repeated-run logging (no scoring effect).
    """
    root = Path(dataset_path)
    tables = load_tables(root)
    prestep = run_prestep(tables)

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError("ANTHROPIC_API_KEY is required for investigate()")

    client = anthropic.Anthropic(api_key=key)
    tools = tool_schemas(version="v1")
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": seed_user_message(prestep)},
    ]
    trace: list[dict[str, Any]] = []
    findings: dict[str, Any] | None = None
    model_turns = 0

    for iteration in range(1, max_iterations + 1):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt(),
            tools=tools,
            messages=messages,
        )
        model_turns += 1

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results: list[dict[str, Any]] = []
        for block in assistant_content:
            if block.type != "tool_use":
                continue

            if block.name == "submit_findings":
                findings = normalize_findings(block.input)
                trace.append(
                    {
                        "tool": "submit_findings",
                        "args": dict(block.input),
                        "result": {"status": "submitted"},
                    }
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"status": "submitted"}),
                    }
                )
                continue

            args = dict(block.input)
            result = dispatch(block.name, args, tables)
            trace.append({"tool": block.name, "args": args, "result": result})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result_content(result),
                }
            )

        if findings is not None:
            return {
                "findings": findings,
                "trace": trace,
                "prestep": prestep,
                "iterations": model_turns,
                "model": model,
                "run_seed": run_seed,
                "run_index": run_index,
            }

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        elif response.stop_reason == "end_turn":
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue investigating with tools or call submit_findings "
                        "with your final diagnosis."
                    ),
                }
            )
        else:
            break

    if findings is None:
        findings = _fallback_findings(
            f"Max iterations ({max_iterations}) reached without submit_findings."
        )

    return {
        "findings": findings,
        "trace": trace,
        "prestep": prestep,
        "iterations": model_turns,
        "model": model,
        "run_seed": run_seed,
        "run_index": run_index,
    }
