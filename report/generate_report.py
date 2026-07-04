#!/usr/bin/env python3
"""
Generate a self-contained HTML RCA report for foundry yield excursions.

Takes agent findings (FINDINGS_CONTRACT) plus the four dataset tables and writes
one standalone .html file with embedded matplotlib charts (base64 PNG).

Usage:
    python report/generate_report.py \\
        --dataset datasets/_verify/verify_early_detection \\
        --investigation report/examples/investigation_early_detection.json \\
        --output report/examples/early_detection_rca.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.prestep import STRONG_ANOMALY_THRESHOLD, run_prestep
from agent.tools._baseline import (
    COMMONALITY_STEP,
    DEFAULT_SIGMA_THRESHOLD,
    robust_control_limits,
    robust_shift,
)
from agent.tools._inline import ordered_lots
from agent.tools.chain_correlate import chain_correlate
from agent.tools.commonality import commonality
from agent.tools.excursion_confirm import excursion_confirm
from agent.tools._load import load_tables
from agent.tools.inline_trace import inline_trace
from agent.tools.spatial_signature import spatial_signature
from agent.tools.wat_profile import wat_profile
from schema import FINDINGS_CONTRACT, INLINE_METRICS, INLINE_STEPS, SOFT_BINS, WAT_PARAMS

# Chart selection by excursion type (non-spatial types skip wafer map, etc.)
_CHART_PLAN: dict[str, dict[str, bool]] = {
    "edge_signature": {
        "wafer_map": True,
        "spc_wat": True,
        "spc_inline": False,
        "commonality": False,
        "bin_breakdown": True,
    },
    "chamber_specific": {
        "wafer_map": False,
        "spc_wat": True,
        "spc_inline": False,
        "commonality": True,
        "bin_breakdown": True,
    },
    "propagation": {
        "wafer_map": True,
        "spc_wat": True,
        "spc_inline": True,
        "commonality": False,
        "bin_breakdown": True,
    },
    "early_detection": {
        "wafer_map": False,
        "spc_wat": False,
        "spc_inline": True,
        "commonality": False,
        "bin_breakdown": False,
    },
    "mean_shift": {
        "wafer_map": False,
        "spc_wat": True,
        "spc_inline": False,
        "commonality": False,
        "bin_breakdown": True,
    },
    "correlation_break": {
        "wafer_map": False,
        "spc_wat": False,
        "spc_inline": False,
        "commonality": True,
        "bin_breakdown": True,
    },
    "confounding": {
        "wafer_map": False,
        "spc_wat": True,
        "spc_inline": False,
        "commonality": False,
        "bin_breakdown": True,
    },
    "clean": {
        "wafer_map": False,
        "spc_wat": False,
        "spc_inline": False,
        "commonality": False,
        "bin_breakdown": False,
    },
}

# Negative-evidence charts (ruled-out hypotheses / monitoring gaps) by excursion type.
_NEGATIVE_CHART_PLAN: dict[str, dict[str, bool]] = {
    "early_detection": {
        "spc_yield": True,
        "wafer_map": True,
        "wat_sigma_panel": True,
        "chamber_yield": True,
    },
}

_ACTIONS: dict[str, str] = {
    "edge_signature": (
        "Review litho/etch radial uniformity and edge-exclusion controls on the affected tool set. "
        "Hold affected lots pending edge-ring disposition review and re-metrology at wafer edge."
    ),
    "chamber_specific": (
        "Inspect and PM the identified chamber at gate_etch; quarantine wafers processed on the "
        "culprit chamber pending parametric review. Release only after chamber matching baseline."
    ),
    "propagation": (
        "Stop-line the origin inline step; inspect defect source and downstream WAT correlation. "
        "Hold affected lots until inline defect count and WAT parametrics return to control."
    ),
    "early_detection": (
        "Inspect and recalibrate the inline tool at the drifting step (overlay/CD track). "
        "Hold the drifting lot sequence pending inline disposition — WAT/Sort may still pass spec."
    ),
    "mean_shift": (
        "Hold affected lots; run WAT re-screen and parametric disposition against control limits. "
        "Identify upstream process offset feeding the shifted WAT parameter."
    ),
    "correlation_break": (
        "Escalate to test engineering and sort/test hardware commonality; WAT is in-family so "
        "focus on probe/card/site or between-layer defect not captured in WAT."
    ),
    "confounding": (
        "Disposition on the causal parameter only; do not over-react to the correlated confounder. "
        "Hold lots and verify rc vs rs ownership with process engineering."
    ),
    "clean": (
        "No fab hold required. Continue routine SPC monitoring; document investigation as no excursion."
    ),
}

_CAUSAL_CHAINS: dict[str, str] = {
    "edge_signature": (
        "Spatial edge-ring signature at Sort → elevated edge WAT parametric → localized yield loss"
    ),
    "chamber_specific": (
        "Chamber-specific process offset at gate_etch → WAT parametric shift on affected wafers → yield split"
    ),
    "propagation": (
        "Inline defect excursion at origin step → WAT parametric shift → Sort speed/bin failure"
    ),
    "early_detection": (
        "Sustained inline metrology drift (multi-lot) → WAT still in-family → Sort still in-family (early signal)"
    ),
    "mean_shift": (
        "Lot-wide WAT parametric shift → crosses SPC/spec → downstream yield risk"
    ),
    "correlation_break": (
        "WAT in-family → unexplained Sort failure (test/layer correlation break)"
    ),
    "confounding": (
        "Causal WAT param shift + correlated confounder moving together → Sort fails on causal param"
    ),
    "clean": "No excursion — all signals within expected lot-to-lot variation.",
}


def _esc(text: Any) -> str:
    if text is None:
        return "—"
    return html.escape(str(text))


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _lot_label(lot: str) -> str:
    return lot.replace("lot_", "L")


def _severity(max_sigma: float) -> str:
    a = abs(max_sigma)
    if a >= 10:
        return "Critical"
    if a >= 5:
        return "High"
    if a >= DEFAULT_SIGMA_THRESHOLD:
        return "Moderate"
    if a > 0:
        return "Low"
    return "None"


def _normalize_findings(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in FINDINGS_CONTRACT:
        if key in raw:
            out[key] = raw[key]
        elif key in ("location", "origin_step", "affected_param", "type"):
            out[key] = None
        elif key == "detected":
            out[key] = False
        else:
            out[key] = ""
    if out.get("type") is None and not out.get("detected"):
        out["type"] = "clean"
    return out


def _resolve_lots(
    findings: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    *,
    affected_lots: list[str] | None,
    primary_lot: str | None,
    prestep: dict[str, Any] | None = None,
) -> tuple[str | None, list[str]]:
    if not findings.get("detected"):
        return None, []

    lots = ordered_lots(tables["sort"])
    if affected_lots:
        valid = [lot for lot in affected_lots if lot in lots]
        if valid:
            primary = primary_lot if primary_lot in valid else valid[-1]
            return primary, valid

    pre = prestep if prestep is not None else run_prestep(tables)
    top = pre["suspects"][0]
    if float(top["anomaly_score"]) < STRONG_ANOMALY_THRESHOLD:
        return None, []

    primary = primary_lot if primary_lot in lots else top["lot"]
    if findings.get("type") == "early_detection":
        trace = inline_trace(primary, tables)
        step, metric = trace["step"], trace["metric"]
        inline_df = tables["inline"]
        idx = lots.index(primary)
        start = max(0, idx - 3)
        window = lots[start : idx + 1]
        seq = [
            lot
            for lot in window
            if abs(
                robust_shift(
                    float(
                        inline_df[(inline_df["lot"] == lot) & (inline_df["step"] == step)][
                            metric
                        ].mean()
                    ),
                    inline_df[inline_df["step"] == step].groupby("lot")[metric].mean(),
                    lot,
                )
            )
            >= 2.0
            or lot == primary
        ]
        return primary, seq or [primary]

    return primary, [primary]


def _chart_plan(excursion_type: str | None) -> dict[str, bool]:
    key = excursion_type or "clean"
    return _CHART_PLAN.get(key, _CHART_PLAN["clean"])


def _negative_chart_plan(excursion_type: str | None) -> dict[str, bool]:
    key = excursion_type or "clean"
    return _NEGATIVE_CHART_PLAN.get(key, {})


def _plot_wafer_map(
    sort_df: pd.DataFrame,
    lot: str,
    *,
    title: str,
) -> str:
    lot_df = sort_df[sort_df["lot"] == lot]
    fails = lot_df[~lot_df["pass"]]
    passes = lot_df[lot_df["pass"]]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    if len(passes):
        ax.scatter(
            passes["die_x"],
            passes["die_y"],
            c="#2ecc71",
            s=8,
            alpha=0.35,
            linewidths=0,
            label="Pass",
        )
    if len(fails):
        ax.scatter(
            fails["die_x"],
            fails["die_y"],
            c="#e74c3c",
            s=14,
            alpha=0.85,
            linewidths=0,
            label="Fail",
        )
    ax.set_aspect("equal")
    ax.set_xlabel("Die X")
    ax.set_ylabel("Die Y")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    return _fig_to_b64(fig)


def _plot_spc_series(
    lots: list[str],
    values: pd.Series,
    affected: set[str],
    *,
    ylabel: str,
    title: str,
    highlight_lot: str | None = None,
) -> str:
    xs = np.arange(len(lots))
    ys = [float(values.get(lot, np.nan)) for lot in lots]
    ref_lot = highlight_lot or (next(iter(affected)) if affected else lots[-1])
    if ref_lot not in lots:
        ref_lot = lots[-1]
    ucl, lcl, center, _ = robust_control_limits(values, ref_lot)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(xs, ys, color="#34495e", linewidth=1.5, marker="o", markersize=4, label="Lot mean")
    ax.axhline(center, color="#7f8c8d", linestyle="--", linewidth=1, label="Center")
    ax.axhline(ucl, color="#e74c3c", linestyle=":", linewidth=1.2, label="UCL (3σ)")
    ax.axhline(lcl, color="#e74c3c", linestyle=":", linewidth=1.2, label="LCL (3σ)")

    for i, lot in enumerate(lots):
        if lot in affected:
            ax.scatter([i], [ys[i]], c="#e74c3c", s=55, zorder=5)

    step = max(1, len(lots) // 10)
    tick_idx = list(range(0, len(lots), step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([_lot_label(lots[i]) for i in tick_idx], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    return _fig_to_b64(fig)


def _plot_wat_sigma_bars(lot: str, tables: dict[str, pd.DataFrame]) -> str:
    sigmas = [float(wat_profile(lot, param, tables)["sigma_shift"]) for param in WAT_PARAMS]
    colors = ["#3498db" if abs(s) < DEFAULT_SIGMA_THRESHOLD else "#e74c3c" for s in sigmas]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(WAT_PARAMS, sigmas, color=colors, edgecolor="#2c3e50", linewidth=0.6)
    ax.axhline(DEFAULT_SIGMA_THRESHOLD, color="#e74c3c", linestyle=":", linewidth=1.2, label="+3σ")
    ax.axhline(-DEFAULT_SIGMA_THRESHOLD, color="#e74c3c", linestyle=":", linewidth=1.2, label="−3σ")
    ax.axhline(0, color="#7f8c8d", linestyle="--", linewidth=1, label="Population center")
    ax.set_ylabel("σ vs leave-one-out population")
    ax.set_xlabel("WAT parameter")
    ax.set_title(f"WAT parametric shifts — {lot}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return _fig_to_b64(fig)


def _plot_commonality_bars(
    route_df: pd.DataFrame,
    sort_df: pd.DataFrame,
    lot: str,
    *,
    highlight_chamber: str | None,
    neutral: bool = False,
) -> str:
    lot_route = route_df[(route_df["lot"] == lot) & (route_df["step"] == COMMONALITY_STEP)]
    lot_sort = sort_df[sort_df["lot"] == lot]
    chamber_yields: dict[str, float] = {}
    for chamber, group in lot_route.groupby("chamber"):
        wafers = group["wafer"].unique()
        vals = lot_sort[lot_sort["wafer"].isin(wafers)]
        if len(vals):
            chamber_yields[str(chamber)] = float(vals["pass"].mean())

    chambers = sorted(chamber_yields)
    yields = [chamber_yields[c] for c in chambers]
    if neutral:
        colors = ["#3498db"] * len(chambers)
    else:
        colors = ["#e74c3c" if c == highlight_chamber else "#3498db" for c in chambers]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(chambers, yields, color=colors, edgecolor="#2c3e50", linewidth=0.6)
    ax.set_ylabel("Sort yield")
    ax.set_xlabel(f"Chamber @ {COMMONALITY_STEP}")
    title_suffix = " (all chambers level)" if neutral else ""
    ax.set_title(f"Yield by chamber — {lot}{title_suffix}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    return _fig_to_b64(fig)


def _build_evidence(
    findings: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    primary_lot: str | None,
    affected_lots: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not primary_lot:
        rows.append(
            {
                "metric": "Investigation outcome",
                "value": "No excursion detected",
                "baseline": "—",
                "sigma": "—",
            }
        )
        return rows

    sort_df = tables["sort"]
    wat_df = tables["wat"]
    inline_df = tables["inline"]

    lot_yield = float(sort_df[sort_df["lot"] == primary_lot]["pass"].mean())
    pop_yield = sort_df.groupby("lot")["pass"].mean()
    yield_sigma = robust_shift(lot_yield, pop_yield, primary_lot)
    rows.append(
        {
            "metric": f"Sort yield ({primary_lot})",
            "value": f"{lot_yield:.1%}",
            "baseline": f"{pop_yield.drop(index=primary_lot, errors='ignore').median():.1%}",
            "sigma": f"{yield_sigma:+.2f}",
        }
    )

    param = findings.get("affected_param")
    if param in WAT_PARAMS:
        lot_means = wat_df.groupby("lot")[param].mean()
        val = float(lot_means[primary_lot])
        sigma = robust_shift(val, lot_means, primary_lot)
        rows.append(
            {
                "metric": f"WAT {param}",
                "value": f"{val:.3f}",
                "baseline": f"{lot_means.drop(index=primary_lot, errors='ignore').median():.3f}",
                "sigma": f"{sigma:+.2f}",
            }
        )

    origin = findings.get("origin_step")
    if findings.get("type") == "early_detection" or origin:
        trace = inline_trace(primary_lot, tables)
        step = origin or trace["step"]
        metric = trace["metric"]
        step_series = inline_df[inline_df["step"] == step].groupby("lot")[metric].mean()
        val = float(step_series.get(primary_lot, 0))
        sigma = robust_shift(val, step_series, primary_lot)
        rows.append(
            {
                "metric": f"Inline {step} {metric}",
                "value": f"{val:.3f}",
                "baseline": f"{step_series.drop(index=primary_lot, errors='ignore').median():.3f}",
                "sigma": f"{sigma:+.2f}",
            }
        )
        rows.append(
            {
                "metric": "Inline trend (slope / sustained)",
                "value": f"{trace['trend_slope']:.4f} / {trace['sustained']:.3f}",
                "baseline": "trailing 4-lot window",
                "sigma": f"{trace['level_sigma']:+.2f} level",
            }
        )

    if findings.get("location"):
        comm = commonality(primary_lot, tables, population="yield")
        rows.append(
            {
                "metric": f"Chamber commonality @ {COMMONALITY_STEP}",
                "value": f"{comm['commons_to']} (gap {comm['yield_gap']:.3f})",
                "baseline": findings["location"],
                "sigma": f"{comm['strength']:+.2f}",
            }
        )

    if len(affected_lots) > 1:
        rows.append(
            {
                "metric": "Affected lot sequence",
                "value": ", ".join(affected_lots),
                "baseline": "—",
                "sigma": "—",
            }
        )

    lot_bins = sort_df[sort_df["lot"] == primary_lot]["soft_bin"].value_counts(normalize=True)
    for bin_name in ("speed", "gross", "leakage"):
        rate = float(lot_bins.get(bin_name, 0.0))
        if rate > 0.001:
            rows.append(
                {
                    "metric": f"Sort bin {bin_name}",
                    "value": f"{rate:.1%}",
                    "baseline": "lot fraction",
                    "sigma": "—",
                }
            )

    return rows


def _max_sigma(evidence: list[dict[str, str]]) -> float:
    best = 0.0
    for row in evidence:
        m = re.search(r"([+-]?\d+\.?\d*)", row.get("sigma", ""))
        if m:
            best = max(best, abs(float(m.group(1))))
    return best


def _build_signals_scanned(
    tables: dict[str, pd.DataFrame],
    prestep: dict[str, Any],
    primary_lot: str | None,
) -> dict[str, Any]:
    n_lots = int(prestep["n_lots"])
    n_features = int(prestep["feature_count"])
    prestep_cells = n_lots * n_features
    flagged_lots = [
        row["lot"]
        for row in prestep["suspects"]
        if float(row["anomaly_score"]) >= STRONG_ANOMALY_THRESHOLD
    ]
    n_wat = len(WAT_PARAMS)
    n_inline = len(INLINE_METRICS) * len(INLINE_STEPS)
    n_sort = len(SOFT_BINS)  # yield/fail + soft bins in tool coverage
    hypothesis_checks = n_wat + n_inline + n_sort + 4  # spatial, commonality, chain, inline_trace

    summary = (
        f"{n_wat} WAT parameters, {len(INLINE_METRICS)} inline metrics × {len(INLINE_STEPS)} steps, "
        f"sort yield and {len(SOFT_BINS)} soft-bin populations, across {n_lots} lots — "
        f"{prestep_cells:,} pre-step feature signals evaluated"
    )
    if flagged_lots:
        summary += (
            f", {len(flagged_lots)} lot(s) flagged above {STRONG_ANOMALY_THRESHOLD} "
            f"({', '.join(flagged_lots[:3])}{'…' if len(flagged_lots) > 3 else ''})"
        )
    else:
        summary += f", none flagged above {STRONG_ANOMALY_THRESHOLD}"

    if primary_lot:
        summary += (
            f". On primary lot {primary_lot}: {hypothesis_checks} targeted hypothesis checks "
            f"(WAT, inline grid, sort bins, spatial, chamber commonality, defect chain, inline trend)."
        )

    return {
        "summary": summary,
        "prestep_cells": prestep_cells,
        "flagged_count": len(flagged_lots),
        "hypothesis_checks": hypothesis_checks if primary_lot else 0,
    }


def _build_ruled_out(
    tables: dict[str, pd.DataFrame],
    primary_lot: str | None,
    findings: dict[str, Any],
) -> list[dict[str, str]]:
    if not primary_lot:
        return [
            {
                "hypothesis": "Process excursion",
                "tool": "pre-step screen",
                "evidence": "No lot exceeded strong anomaly threshold",
                "verdict": "No excursion flagged",
            }
        ]

    rows: list[dict[str, str]] = []
    wat_sigmas = {
        param: float(wat_profile(primary_lot, param, tables)["sigma_shift"])
        for param in WAT_PARAMS
    }
    max_wat = max(abs(v) for v in wat_sigmas.values())
    top_wat = max(wat_sigmas, key=lambda p: abs(wat_sigmas[p]))
    rows.append(
        {
            "hypothesis": "WAT parametric shift",
            "tool": "wat_profile (all 7 params)",
            "evidence": f"max |σ|={max_wat:.2f} on {top_wat} (all params in-family)",
            "verdict": "Ruled out" if max_wat < 2.0 else "Not ruled out",
        }
    )

    comm = commonality(primary_lot, tables, population="yield")
    chamber_ruled = abs(float(comm["strength"])) < 2.0 and float(comm["yield_gap"]) < 0.05
    rows.append(
        {
            "hypothesis": "Chamber commonality",
            "tool": f"commonality @ {COMMONALITY_STEP}",
            "evidence": (
                f"commons_to={comm['commons_to']}, yield gap={comm['yield_gap']:.3f}, "
                f"strength={comm['strength']:.2f}"
            ),
            "verdict": "Ruled out" if chamber_ruled else "Not ruled out",
        }
    )

    chain = chain_correlate(primary_lot, tables)
    inline_link = chain["links"][0]
    inline_df = tables["inline"]
    defect_sigmas = {
        step: abs(
            robust_shift(
                float(
                    inline_df[(inline_df["lot"] == primary_lot) & (inline_df["step"] == step)][
                        "defect_density"
                    ].mean()
                ),
                inline_df[inline_df["step"] == step].groupby("lot")["defect_density"].mean(),
                primary_lot,
            )
        )
        for step in INLINE_STEPS
    }
    max_defect = max(defect_sigmas.values())
    worst_step = max(defect_sigmas, key=defect_sigmas.get)
    defect_ruled = inline_link["status"] == "normal" and max_defect < 2.0
    rows.append(
        {
            "hypothesis": "Inline defect propagation",
            "tool": "chain_correlate + defect_density scan",
            "evidence": (
                f"origin {inline_link['step']} defect σ={inline_link['shift_sigma']:.2f} "
                f"({inline_link['status']}); max defect_density |σ|={max_defect:.2f} @ {worst_step}"
            ),
            "verdict": "Ruled out" if defect_ruled else "Not ruled out",
        }
    )

    spatial = spatial_signature(primary_lot, tables, population="fail")
    spatial_ruled = spatial["signature"] == "uniform"
    rows.append(
        {
            "hypothesis": "Spatial edge/center signature",
            "tool": "spatial_signature (fail)",
            "evidence": (
                f"signature={spatial['signature']}, edge fail={spatial['edge_fail_ratio']:.1%}, "
                f"center fail={spatial['center_fail_ratio']:.1%}"
            ),
            "verdict": "Ruled out" if spatial_ruled else "Not ruled out",
        }
    )

    yield_exc = excursion_confirm(primary_lot, "yield", tables)
    speed_exc = excursion_confirm(primary_lot, "speed", tables)
    sort_ruled = not yield_exc["out_of_control"] and not speed_exc["out_of_control"]
    lot_yield = float(tables["sort"][tables["sort"]["lot"] == primary_lot]["pass"].mean())
    rows.append(
        {
            "hypothesis": "Sort yield / speed bin excursion",
            "tool": "excursion_confirm",
            "evidence": (
                f"yield={lot_yield:.1%} (σ={yield_exc['sigma']:+.2f}, OOC={yield_exc['out_of_control']}); "
                f"speed σ={speed_exc['sigma']:+.2f}, OOC={speed_exc['out_of_control']}"
            ),
            "verdict": "Ruled out" if sort_ruled else "Not ruled out",
        }
    )

    if findings.get("type") != "correlation_break":
        gross_exc = excursion_confirm(primary_lot, "gross", tables)
        rows.append(
            {
                "hypothesis": "Gross bin test failure",
                "tool": "excursion_confirm (gross)",
                "evidence": f"σ={gross_exc['sigma']:+.2f}, OOC={gross_exc['out_of_control']}",
                "verdict": "Ruled out" if not gross_exc["out_of_control"] else "Not ruled out",
            }
        )

    return rows


def _build_monitoring_gap(
    tables: dict[str, pd.DataFrame],
    primary_lot: str | None,
    findings: dict[str, Any],
) -> dict[str, Any]:
    if not primary_lot or not findings.get("detected"):
        return {
            "headline": "Standard WAT/Sort disposition gates align with the no-excursion conclusion.",
            "bullets": ["Pre-step found no strong anomaly requiring fab hold."],
        }

    sort_df = tables["sort"]
    lot_yield = float(sort_df[sort_df["lot"] == primary_lot]["pass"].mean())
    yield_exc = excursion_confirm(primary_lot, "yield", tables)
    wat_sigmas = {
        param: abs(float(wat_profile(primary_lot, param, tables)["sigma_shift"]))
        for param in WAT_PARAMS
    }
    max_wat = max(wat_sigmas.values())

    bullets: list[str] = []
    if not yield_exc["out_of_control"]:
        bullets.append(
            f"Sort yield: {lot_yield:.1%} (σ={yield_exc['sigma']:+.2f}) — nominal; "
            "would pass standard yield disposition gate."
        )
    else:
        bullets.append(
            f"Sort yield: {lot_yield:.1%} — out of control (σ={yield_exc['sigma']:+.2f}); "
            "standard yield gate would flag this lot."
        )

    if max_wat < DEFAULT_SIGMA_THRESHOLD:
        bullets.append(
            f"WAT parametrics: in-family (max |σ|={max_wat:.2f} across 7 params) — "
            "would pass WAT SPC disposition."
        )
    else:
        top = max(wat_sigmas, key=wat_sigmas.get)
        bullets.append(
            f"WAT parametrics: {top} |σ|={max_wat:.2f} — WAT disposition would flag this lot."
        )

    excursion_type = findings.get("type")
    if excursion_type == "early_detection":
        trace = inline_trace(primary_lot, tables)
        bullets.append(
            f"Inline {trace['step']} {trace['metric']}: level σ={trace['level_sigma']:+.2f}, "
            f"trend slope={trace['trend_slope']:.3f}, sustained={trace['sustained']:.2f} — "
            "visible only as multi-lot inline drift, not on single-lot WAT/Sort gates."
        )
        headline = (
            "This excursion is invisible to standard WAT/Sort disposition gates; it is detectable "
            "only via multi-lot inline trend analysis (early detection window)."
        )
    elif excursion_type == "correlation_break":
        headline = (
            "WAT parametrics pass standard gates while Sort fails — a correlation break invisible "
            "to WAT-only disposition."
        )
    elif excursion_type == "edge_signature":
        headline = (
            "Lot-average yield may remain near nominal; the edge-ring spatial signature is missed "
            "by lot-mean yield gates without wafer-map review."
        )
    elif not yield_exc["out_of_control"] and max_wat < DEFAULT_SIGMA_THRESHOLD:
        headline = (
            "Standard lot-level WAT and yield gates would not have flagged this material; "
            "detection relied on deeper inline/spatial/commonality screening."
        )
    else:
        headline = (
            "Standard WAT and/or Sort disposition gates would have flagged this lot; "
            "root-cause typing required the full tool triage above."
        )

    return {"headline": headline, "bullets": bullets}


def _render_html(
    *,
    dataset_id: str,
    findings: dict[str, Any],
    primary_lot: str | None,
    affected_lots: list[str],
    charts: list[dict[str, Any]],
    evidence: list[dict[str, str]],
    severity: str,
    signals_scanned: dict[str, Any],
    ruled_out: list[dict[str, str]],
    monitoring_gap: dict[str, Any],
) -> str:
    excursion_type = findings.get("type") or "clean"
    chain = _CAUSAL_CHAINS.get(str(excursion_type), _CAUSAL_CHAINS["clean"])
    action = _ACTIONS.get(str(excursion_type), _ACTIONS["clean"])

    chart_blocks = ""
    for block in charts:
        layout = block.get("layout", "stack")
        if layout == "row":
            row_items = ""
            for item in block["items"]:
                caption = item.get("caption")
                cap_html = f'<p class="chart-caption">{_esc(caption)}</p>' if caption else ""
                row_items += f"""
                <div class="chart-card">
                  <h3>{_esc(item['title'])}</h3>
                  <img alt="{_esc(item['title'])}" src="data:image/png;base64,{item['b64']}" />
                  {cap_html}
                </div>
                """
            chart_blocks += f'<div class="charts-row">{row_items}</div>'
        else:
            caption = block.get("caption")
            cap_html = f'<p class="chart-caption">{_esc(caption)}</p>' if caption else ""
            chart_blocks += f"""
            <div class="chart-card">
              <h3>{_esc(block['title'])}</h3>
              <img alt="{_esc(block['title'])}" src="data:image/png;base64,{block['b64']}" />
              {cap_html}
            </div>
            """

    evidence_rows = ""
    for row in evidence:
        evidence_rows += f"""
        <tr>
          <td>{_esc(row['metric'])}</td>
          <td>{_esc(row['value'])}</td>
          <td>{_esc(row['baseline'])}</td>
          <td>{_esc(row['sigma'])}</td>
        </tr>
        """

    ruled_out_rows = ""
    for row in ruled_out:
        verdict_class = "verdict-out" if row["verdict"] == "Ruled out" else "verdict-open"
        ruled_out_rows += f"""
        <tr>
          <td>{_esc(row['hypothesis'])}</td>
          <td>{_esc(row['tool'])}</td>
          <td>{_esc(row['evidence'])}</td>
          <td class="{verdict_class}">{_esc(row['verdict'])}</td>
        </tr>
        """

    monitoring_bullets = "".join(f"<li>{_esc(line)}</li>" for line in monitoring_gap["bullets"])

    lot_display = primary_lot or "N/A (no excursion)"
    affected_display = ", ".join(affected_lots) if affected_lots else "—"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RCA Report — {_esc(dataset_id)} — {_esc(lot_display)}</title>
  <style>
    :root {{
      --bg: #f4f6f8; --card: #ffffff; --ink: #1a1a2e; --muted: #5c6b7a;
      --accent: #0b3d60; --line: #d9e2ec; --warn: #c0392b; --ok: #1e8449;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: var(--bg);
      color: var(--ink); line-height: 1.45;
    }}
    header {{
      background: linear-gradient(135deg, #0b3d60, #145a86); color: #fff;
      padding: 28px 36px 22px;
    }}
    header h1 {{ margin: 0 0 6px; font-size: 1.55rem; font-weight: 600; }}
    header p {{ margin: 0; opacity: 0.92; font-size: 0.95rem; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px 48px; }}
    section {{
      background: var(--card); border: 1px solid var(--line); border-radius: 8px;
      padding: 20px 22px; margin-bottom: 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    h2 {{
      margin: 0 0 12px; font-size: 1.05rem; color: var(--accent);
      border-bottom: 2px solid var(--line); padding-bottom: 8px;
    }}
    .summary-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;
    }}
    .summary-item {{
      background: #f8fafc; border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px;
    }}
    .summary-item label {{
      display: block; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
      color: var(--muted); margin-bottom: 4px;
    }}
    .summary-item span {{ font-size: 1rem; font-weight: 600; }}
    .severity-critical {{ color: var(--warn); }}
    .severity-high {{ color: #d35400; }}
    .severity-moderate {{ color: #b7950b; }}
    .chain {{
      background: #eef4fb; border-left: 4px solid var(--accent); padding: 12px 14px;
      border-radius: 4px; font-size: 0.98rem;
    }}
    .charts {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    .charts-row {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px;
    }}
    .chart-card img {{ max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 4px; }}
    .chart-card h3 {{ margin: 0 0 10px; font-size: 0.95rem; color: var(--muted); }}
    .chart-caption {{ margin: 8px 0 0; font-size: 0.85rem; color: var(--muted); font-style: italic; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 10px; text-align: left; }}
    th {{ background: #eef2f6; font-weight: 600; }}
    .action {{ font-size: 0.98rem; }}
    .scan-summary {{ font-size: 0.98rem; margin: 0; }}
    .callout {{
      background: #fff8e6; border: 1px solid #f0d78c; border-left: 4px solid #d4ac0d;
      border-radius: 4px; padding: 14px 16px; margin: 0;
    }}
    .callout strong {{ display: block; margin-bottom: 8px; color: #7d6608; }}
    .callout ul {{ margin: 8px 0 0 18px; padding: 0; }}
    .verdict-out {{ color: var(--ok); font-weight: 600; }}
    .verdict-open {{ color: #d35400; font-weight: 600; }}
    footer {{ text-align: center; color: var(--muted); font-size: 0.8rem; padding: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>Root Cause Analysis — Yield Excursion Report</h1>
    <p>Dataset {_esc(dataset_id)} · Primary lot {_esc(lot_display)} · Generated for fab disposition review</p>
  </header>
  <main>
    <section id="executive-summary">
      <h2>Executive Summary</h2>
      <div class="summary-grid">
        <div class="summary-item"><label>Primary lot</label><span>{_esc(lot_display)}</span></div>
        <div class="summary-item"><label>Affected lots</label><span>{_esc(affected_display)}</span></div>
        <div class="summary-item"><label>Excursion type</label><span>{_esc(excursion_type)}</span></div>
        <div class="summary-item"><label>Location</label><span>{_esc(findings.get('location'))}</span></div>
        <div class="summary-item"><label>Origin step</label><span>{_esc(findings.get('origin_step'))}</span></div>
        <div class="summary-item"><label>Affected parameter</label><span>{_esc(findings.get('affected_param'))}</span></div>
        <div class="summary-item"><label>Severity</label><span class="severity-{severity.lower()}">{_esc(severity)}</span></div>
        <div class="summary-item"><label>Confidence</label><span>{_esc(findings.get('confidence'))}</span></div>
        <div class="summary-item"><label>Detected</label><span>{'Yes' if findings.get('detected') else 'No'}</span></div>
      </div>
    </section>

    <section id="signals-scanned">
      <h2>Signals Scanned</h2>
      <p class="scan-summary">{_esc(signals_scanned['summary'])}</p>
    </section>

    <section id="root-cause">
      <h2>Root Cause Statement</h2>
      <p><strong>Diagnosis:</strong> {_esc(findings.get('cause'))}</p>
      <p class="chain"><strong>Causal chain:</strong> {_esc(chain)}</p>
    </section>

    <section id="ruled-out">
      <h2>Alternative Causes Ruled Out</h2>
      <table>
        <thead><tr><th>Hypothesis</th><th>Tool</th><th>Evidence</th><th>Verdict</th></tr></thead>
        <tbody>{ruled_out_rows}</tbody>
      </table>
    </section>

    <section id="monitoring-gap">
      <h2>Why Standard Monitoring Missed This</h2>
      <div class="callout">
        <strong>{_esc(monitoring_gap['headline'])}</strong>
        <ul>{monitoring_bullets}</ul>
      </div>
    </section>

    <section id="charts">
      <h2>Evidence Charts</h2>
      <div class="charts">{chart_blocks if chart_blocks else '<p>No charts for this excursion profile.</p>'}</div>
    </section>

    <section id="evidence">
      <h2>Evidence Tables</h2>
      <table>
        <thead><tr><th>Metric</th><th>Observed</th><th>Baseline</th><th>Shift</th></tr></thead>
        <tbody>{evidence_rows}</tbody>
      </table>
    </section>

    <section id="action">
      <h2>Recommended Action</h2>
      <p class="action">{_esc(action)}</p>
    </section>
  </main>
  <footer>Confidential — for internal foundry yield engineering disposition · yield-benchmark RCA generator</footer>
</body>
</html>
"""


def generate_report(
    findings: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    output_path: str | Path,
    *,
    dataset_id: str = "",
    affected_lots: list[str] | None = None,
    primary_lot: str | None = None,
) -> Path:
    """
    Build a standalone HTML RCA report from findings + table data.

    Returns the path written.
    """
    findings = _normalize_findings(findings)
    prestep = run_prestep(tables)
    primary, affected = _resolve_lots(
        findings,
        tables,
        affected_lots=affected_lots,
        primary_lot=primary_lot,
        prestep=prestep,
    )
    plan = _chart_plan(findings.get("type"))
    neg_plan = _negative_chart_plan(findings.get("type"))
    affected_set = set(affected)
    lots = ordered_lots(tables["sort"])
    charts: list[dict[str, Any]] = []

    def _chart_item(title: str, b64: str, *, caption: str | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {"title": title, "b64": b64}
        if caption:
            item["caption"] = caption
        return item

    inline_item: dict[str, Any] | None = None
    yield_item: dict[str, Any] | None = None

    if plan["spc_inline"] and primary:
        trace = inline_trace(primary, tables)
        step = findings.get("origin_step") or trace["step"]
        metric = trace["metric"]
        inline_series = (
            tables["inline"][tables["inline"]["step"] == step].groupby("lot")[metric].mean()
        )
        inline_item = _chart_item(
            f"Inline drift — {step} {metric} by lot (positive signal)",
            _plot_spc_series(
                lots,
                inline_series,
                affected_set,
                ylabel=f"{metric}",
                title=f"Inline {metric} @ {step} (multi-lot drift)",
                highlight_lot=primary,
            ),
        )

    if neg_plan.get("spc_yield") and primary:
        yield_series = tables["sort"].groupby("lot")["pass"].mean()
        yield_item = _chart_item(
            f"Sort yield SPC — 50-lot sequence (negative evidence)",
            _plot_spc_series(
                lots,
                yield_series,
                affected_set,
                ylabel="Sort yield",
                title="Lot sort yield with robust center + 3σ limits",
                highlight_lot=primary,
            ),
            caption=(
                "All lots inside control limits — standard Sort yield disposition gate would pass "
                "(official gate is green)."
            ),
        )

    if yield_item and inline_item:
        charts.append({"layout": "row", "items": [yield_item, inline_item]})
    elif yield_item:
        charts.append({"layout": "stack", **yield_item})
    elif inline_item:
        charts.append({"layout": "stack", **inline_item})

    if primary and plan["wafer_map"]:
        charts.append(
            {
                "layout": "stack",
                **_chart_item(
                    f"Wafer map — Sort pass/fail spatial pattern ({primary})",
                    _plot_wafer_map(
                        tables["sort"],
                        primary,
                        title=f"Die map {primary} (all wafers aggregated)",
                    ),
                ),
            }
        )

    if primary and neg_plan.get("wafer_map"):
        charts.append(
            {
                "layout": "stack",
                **_chart_item(
                    f"Wafer map — {primary} (negative evidence)",
                    _plot_wafer_map(
                        tables["sort"],
                        primary,
                        title=f"Die pass/fail — {primary} (uniform scatter)",
                    ),
                    caption=(
                        "No edge ring, center cluster, or scratch line — rules out spatial "
                        "edge/center/scratch signature."
                    ),
                ),
            }
        )

    if plan["spc_wat"]:
        param = findings.get("affected_param")
        if param not in WAT_PARAMS and primary:
            trace_lot = primary
            lot_means = {p: abs(robust_shift(float(tables["wat"].groupby("lot")[p].mean().get(trace_lot, 0)),
                                           tables["wat"].groupby("lot")[p].mean(), trace_lot))
                          for p in WAT_PARAMS if trace_lot in tables["wat"]["lot"].values}
            param = max(lot_means, key=lot_means.get) if lot_means else "rc_ohm"
        if param in WAT_PARAMS:
            series = tables["wat"].groupby("lot")[param].mean()
            charts.append(
                {
                    "layout": "stack",
                    **_chart_item(
                        f"SPC trend — WAT {param} by lot",
                        _plot_spc_series(
                            lots,
                            series,
                            affected_set,
                            ylabel=param,
                            title=f"Lot-level WAT {param} with robust control limits",
                            highlight_lot=primary,
                        ),
                    ),
                }
            )

    if primary and neg_plan.get("wat_sigma_panel"):
        charts.append(
            {
                "layout": "stack",
                **_chart_item(
                    f"WAT parametrics panel — {primary} (negative evidence)",
                    _plot_wat_sigma_bars(primary, tables),
                    caption=(
                        "All seven WAT parameters remain inside ±3σ vs population — "
                        "electrical tests still in-family."
                    ),
                ),
            }
        )

    if plan["commonality"] and primary:
        charts.append(
            {
                "layout": "stack",
                **_chart_item(
                    f"Chamber commonality — yield by chamber ({primary})",
                    _plot_commonality_bars(
                        tables["route"],
                        tables["sort"],
                        primary,
                        highlight_chamber=findings.get("location"),
                    ),
                ),
            }
        )

    if primary and neg_plan.get("chamber_yield"):
        charts.append(
            {
                "layout": "stack",
                **_chart_item(
                    f"Chamber yield @ {COMMONALITY_STEP} — {primary} (negative evidence)",
                    _plot_commonality_bars(
                        tables["route"],
                        tables["sort"],
                        primary,
                        highlight_chamber=None,
                        neutral=True,
                    ),
                    caption=(
                        f"Level sort yield across all chambers at {COMMONALITY_STEP} — "
                        "rules out chamber commonality."
                    ),
                ),
            }
        )

    evidence = _build_evidence(findings, tables, primary, affected)
    severity = _severity(_max_sigma(evidence)) if findings.get("detected") else "None"
    signals_scanned = _build_signals_scanned(tables, prestep, primary)
    ruled_out = _build_ruled_out(tables, primary, findings)
    monitoring_gap = _build_monitoring_gap(tables, primary, findings)

    html_doc = _render_html(
        dataset_id=dataset_id or "dataset",
        findings=findings,
        primary_lot=primary,
        affected_lots=affected,
        charts=charts,
        evidence=evidence,
        severity=severity,
        signals_scanned=signals_scanned,
        ruled_out=ruled_out,
        monitoring_gap=monitoring_gap,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    return out


def load_investigation(path: str | Path) -> dict[str, Any]:
    """Load a saved investigation bundle (findings + optional metadata)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "findings" not in data:
        raise ValueError("investigation JSON must contain a 'findings' object")
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a standalone HTML RCA report.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory (sort/wat/inline/route CSVs).",
    )
    parser.add_argument(
        "--investigation",
        help="JSON file with {findings, affected_lots?, primary_lot?, dataset_id?}.",
    )
    parser.add_argument(
        "--findings",
        help="JSON file with findings only (FINDINGS_CONTRACT fields).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output .html path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tables = load_tables(args.dataset)

    if args.investigation:
        bundle = load_investigation(args.investigation)
        findings = bundle["findings"]
        dataset_id = bundle.get("dataset_id") or Path(args.dataset).name
        affected_lots = bundle.get("affected_lots")
        primary_lot = bundle.get("primary_lot")
    elif args.findings:
        findings = _normalize_findings(
            json.loads(Path(args.findings).read_text(encoding="utf-8"))
        )
        dataset_id = Path(args.dataset).name
        affected_lots = None
        primary_lot = None
    else:
        raise SystemExit("Provide --investigation or --findings")

    out = generate_report(
        findings,
        tables,
        args.output,
        dataset_id=dataset_id,
        affected_lots=affected_lots,
        primary_lot=primary_lot,
    )
    print(f"Wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
