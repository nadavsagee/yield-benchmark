#!/usr/bin/env python3
"""
Repeated v0_baseline runs — measure non-deterministic agent stability.

Runs investigate() on all 8 _verify datasets (schema.ANOMALY_TYPES).

Requires ANTHROPIC_API_KEY.

Usage:
    python run_v0_repeated.py
    python run_v0_repeated.py --runs 5 --seed 42
    python run_v0_repeated.py --runs 3 --verbose-traces
    python run_v0_repeated.py --prestep-only
    python run_v0_repeated.py --types early_detection,clean,correlation_break
    python run_v0_repeated.py --version v1 --runs 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from agent.prestep import STRONG_ANOMALY_THRESHOLD, run_prestep
from agent.tools._load import load_tables
from schema import ANOMALY_TYPES
from scorer import load_ground_truth, score_batch, score_dataset

VERIFY_DIR = Path("datasets") / "_verify"
DETECTION_ORDER = ("TP", "FP", "TN", "FN")
AGENT_VERSIONS = ("v0", "v1")


def _load_investigate(version: str) -> Callable[..., dict[str, Any]]:
    if version == "v0":
        from agent.v0 import investigate

        return investigate
    if version == "v1":
        from agent.v1 import investigate

        return investigate
    raise ValueError(f"unknown agent version {version!r}")


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _detection_frequency(counts: Counter[str], n: int) -> str:
    parts = [f"{label} {counts[label]}/{n}" for label in DETECTION_ORDER if counts[label]]
    return ", ".join(parts) if parts else "none"


def _prestep_has_lead(prestep: dict[str, Any]) -> bool:
    top = prestep["suspects"][0]
    threshold = prestep.get("strong_threshold", STRONG_ANOMALY_THRESHOLD)
    return float(top["anomaly_score"]) >= float(threshold)


def _print_trace(trace: list[dict], indent: str = "      ") -> None:
    for i, step in enumerate(trace, 1):
        args = step.get("args") or {}
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"{indent}{i}. {step['tool']}({arg_str})")


def _run_once(
    dataset_path: Path,
    *,
    run_index: int,
    run_seed: int | None,
    investigate: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    out = investigate(dataset_path, run_seed=run_seed, run_index=run_index)
    gt = load_ground_truth(dataset_path)
    row = score_dataset(gt, out["findings"])
    prestep = out["prestep"]
    top = prestep["suspects"][0]
    return {
        "run_index": run_index,
        "run_seed": run_seed,
        "findings": out["findings"],
        "trace": out["trace"],
        "turns": out["iterations"],
        "tool_calls": len(out["trace"]),
        "detection": row["detection"],
        "diagnosis_accuracy": row["diagnosis"]["accuracy"]
        if row["diagnosis"]["scored"]
        else None,
        "prestep_has_lead": _prestep_has_lead(prestep),
        "prestep_top_lot": top["lot"],
        "prestep_top_score": float(top["anomaly_score"]),
    }


def _summarize_dataset(anomaly_type: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(runs)
    det_counts = Counter(r["detection"] for r in runs)
    detections = [r["detection"] for r in runs]
    inconsistent = len(set(detections)) > 1

    tp_diags = [r["diagnosis_accuracy"] for r in runs if r["diagnosis_accuracy"] is not None]
    turns = [r["turns"] for r in runs]
    tools = [r["tool_calls"] for r in runs]
    prestep_leads = sum(1 for r in runs if r["prestep_has_lead"])

    return {
        "anomaly_type": anomaly_type,
        "n_runs": n,
        "detection_counts": det_counts,
        "detection_frequency": _detection_frequency(det_counts, n),
        "inconsistent": inconsistent,
        "detection_labels_seen": sorted(set(detections)),
        "diagnosis_mean": statistics.mean(tp_diags) if tp_diags else None,
        "diagnosis_min": min(tp_diags) if tp_diags else None,
        "diagnosis_max": max(tp_diags) if tp_diags else None,
        "diagnosis_scored_runs": len(tp_diags),
        "turns_mean": statistics.mean(turns),
        "turns_min": min(turns),
        "turns_max": max(turns),
        "tool_calls_mean": statistics.mean(tools),
        "tool_calls_min": min(tools),
        "tool_calls_max": max(tools),
        "prestep_lead_runs": prestep_leads,
        "prestep_no_lead": prestep_leads == 0,
        "runs": runs,
    }


def _parse_types(types_arg: str | None) -> list[str]:
    if not types_arg:
        return list(ANOMALY_TYPES)
    selected = [t.strip() for t in types_arg.split(",") if t.strip()]
    if not selected:
        print("--types must list at least one anomaly type", file=sys.stderr)
        raise SystemExit(1)
    unknown = [t for t in selected if t not in ANOMALY_TYPES]
    if unknown:
        valid = ", ".join(ANOMALY_TYPES)
        print(
            f"Unknown type(s): {', '.join(unknown)}. Valid types: {valid}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return selected


def _batch_summary_for_run(
    run_index: int,
    ground_truths: dict[str, dict],
    per_dataset_findings: dict[str, dict],
    types: list[str],
) -> dict[str, Any]:
    pairs = [(ground_truths[t], per_dataset_findings[t]) for t in types]
    summary = score_batch(pairs)["summary"]
    return {
        "run_index": run_index,
        "recall": summary["detection"]["recall"],
        "precision": summary["detection"]["precision"],
        "diagnosis_accuracy": summary["diagnosis"]["accuracy"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v0_baseline on all 8 _verify datasets repeatedly."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of investigate() repetitions per dataset (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base seed for the run series. Run i uses run_seed=seed+i "
            "(logged via investigate; LLM remains stochastic)."
        ),
    )
    parser.add_argument(
        "--verbose-traces",
        action="store_true",
        help="Print full tool traces for every run (default: summarized only).",
    )
    parser.add_argument(
        "--prestep-only",
        action="store_true",
        help="Run deterministic pre-step on all 8 types (no LLM / API key).",
    )
    parser.add_argument(
        "--types",
        default=None,
        metavar="TYPE[,TYPE...]",
        help=(
            "Comma-separated anomaly types to run (default: all 8). "
            f"Valid: {', '.join(ANOMALY_TYPES)}"
        ),
    )
    parser.add_argument(
        "--version",
        choices=AGENT_VERSIONS,
        default="v0",
        help="Agent version to run: v0 (baseline) or v1 (+ inline_trace). Default: v0.",
    )
    return parser.parse_args(argv)


def _best_affected_score(
    result: dict[str, Any],
    affected_lots: list[str],
) -> tuple[str, float, int]:
    best_lot = ""
    best_score = -1.0
    best_rank = len(result["suspects"]) + 1
    for lot in affected_lots:
        row = next(r for r in result["suspects"] if r["lot"] == lot)
        score = float(row["anomaly_score"])
        if score > best_score:
            best_lot = lot
            best_score = score
            best_rank = int(row["rank"])
    return best_lot, best_score, best_rank


def run_prestep_only(types: list[str]) -> int:
    print(f"yield-benchmark pre-step only ({len(types)} type(s))")
    print(f"Strong threshold: {STRONG_ANOMALY_THRESHOLD}")
    print(f"Datasets: {VERIFY_DIR.resolve()}")
    print(f"Types: {', '.join(types)}")
    print()

    ok = True
    for anomaly_type in types:
        gt = load_ground_truth(VERIFY_DIR / f"verify_{anomaly_type}")
        result = run_prestep(load_tables(VERIFY_DIR / f"verify_{anomaly_type}"))
        top = result["suspects"][0]
        affected = gt.get("affected_lots") or []
        if affected:
            best_lot, best_score, best_rank = _best_affected_score(result, affected)
            hit = best_score >= STRONG_ANOMALY_THRESHOLD
            ok &= hit
            status = "PASS" if hit else "FAIL"
            print(
                f"  {anomaly_type:20} [{status}] "
                f"best_affected={best_lot}@{best_score:.2f} rank={best_rank} "
                f"top={top['lot']}@{top['anomaly_score']:.2f}"
            )
        else:
            hit = float(top["anomaly_score"]) < STRONG_ANOMALY_THRESHOLD
            ok &= hit
            status = "PASS" if hit else "FAIL"
            print(f"  {anomaly_type:20} [{status}] max={top['anomaly_score']:.2f}")

    print()
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_types = _parse_types(args.types)

    if args.runs < 1:
        print("--runs must be >= 1", file=sys.stderr)
        return 1

    if not VERIFY_DIR.exists():
        print(f"{VERIFY_DIR} not found — run verify_generation.py first.", file=sys.stderr)
        return 1

    missing = [
        t for t in selected_types if not (VERIFY_DIR / f"verify_{t}").exists()
    ]
    if missing:
        print(
            f"Missing verify datasets: {', '.join(missing)} "
            "(run verify_generation.py)",
            file=sys.stderr,
        )
        return 1

    if args.prestep_only:
        return run_prestep_only(selected_types)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 1

    investigate = _load_investigate(args.version)

    print(f"yield-benchmark {args.version} — repeated investigate + score")
    print(f"Datasets: {VERIFY_DIR.resolve()} ({len(selected_types)} type(s))")
    print(f"Types: {', '.join(selected_types)}")
    print(f"Runs per dataset: {args.runs}")
    if args.seed is not None:
        print(f"Seed base: {args.seed} (run i -> run_seed={args.seed}+i)")
    print()

    ground_truths = {
        t: load_ground_truth(VERIFY_DIR / f"verify_{t}") for t in selected_types
    }
    dataset_runs: dict[str, list[dict[str, Any]]] = {t: [] for t in selected_types}
    batch_run_summaries: list[dict[str, Any]] = []

    total_steps = args.runs * len(selected_types)
    step = 0
    for run_index in range(args.runs):
        run_seed = (args.seed + run_index) if args.seed is not None else None
        per_dataset_findings: dict[str, dict] = {}

        for anomaly_type in selected_types:
            step += 1
            dataset_path = VERIFY_DIR / f"verify_{anomaly_type}"
            print(
                f"[{step}/{total_steps}] run {run_index + 1}/{args.runs} "
                f"- {anomaly_type}...",
                flush=True,
            )
            run = _run_once(
                dataset_path,
                run_index=run_index,
                run_seed=run_seed,
                investigate=investigate,
            )
            dataset_runs[anomaly_type].append(run)
            per_dataset_findings[anomaly_type] = run["findings"]

        batch_run_summaries.append(
            _batch_summary_for_run(
                run_index, ground_truths, per_dataset_findings, selected_types
            )
        )

    print()
    inconsistent_types: list[str] = []
    no_prestep_lead_types: list[str] = []

    for anomaly_type in selected_types:
        summary = _summarize_dataset(anomaly_type, dataset_runs[anomaly_type])
        gt = ground_truths[anomaly_type]
        is_excursion = bool(gt["excursion"])

        print(f"=== {anomaly_type} ({gt['dataset_id']}) — {args.runs} runs ===")
        print(f"  detection frequency: {summary['detection_frequency']}")
        if summary["inconsistent"]:
            inconsistent_types.append(anomaly_type)
            labels = ", ".join(summary["detection_labels_seen"])
            print(f"  ** UNSTABLE DETECTION ** varied across runs: {labels}")

        lead_runs = summary["prestep_lead_runs"]
        print(f"  prestep lead: {lead_runs}/{args.runs} runs (score >= {STRONG_ANOMALY_THRESHOLD})")
        if summary["prestep_no_lead"]:
            if is_excursion:
                no_prestep_lead_types.append(anomaly_type)
                print("  ** NO PRESTEP LEAD ** agent had no strong suspect on any run")
            elif anomaly_type == "clean":
                print("  prestep: no strong lead (expected for clean)")

        if summary["diagnosis_scored_runs"]:
            print(
                f"  diagnosis accuracy (TP runs): "
                f"mean={_fmt_pct(summary['diagnosis_mean'])}, "
                f"min={_fmt_pct(summary['diagnosis_min'])}, "
                f"max={_fmt_pct(summary['diagnosis_max'])} "
                f"({summary['diagnosis_scored_runs']} scored)"
            )
        else:
            print("  diagnosis accuracy (TP runs): n/a (no TP runs)")

        print(
            f"  efficiency: turns mean={_fmt_float(summary['turns_mean'])}, "
            f"spread {summary['turns_min']}-{summary['turns_max']}; "
            f"tool calls mean={_fmt_float(summary['tool_calls_mean'])}, "
            f"spread {summary['tool_calls_min']}-{summary['tool_calls_max']}"
        )

        print(f"  per-run turns:   {[r['turns'] for r in summary['runs']]}")
        print(f"  per-run tools:   {[r['tool_calls'] for r in summary['runs']]}")
        print(
            "  per-run detect:  "
            + ", ".join(r["detection"] for r in summary["runs"])
        )
        print(
            "  per-run prestep: "
            + ", ".join(
                f"{'lead' if r['prestep_has_lead'] else 'none'}"
                f" ({r['prestep_top_lot']}@{r['prestep_top_score']:.1f})"
                for r in summary["runs"]
            )
        )

        if args.verbose_traces:
            for run in summary["runs"]:
                seed_note = (
                    f", run_seed={run['run_seed']}" if run["run_seed"] is not None else ""
                )
                print(
                    f"    trace run {run['run_index'] + 1}/{args.runs}{seed_note} "
                    f"({run['detection']}, turns={run['turns']}, tools={run['tool_calls']}):"
                )
                _print_trace(run["trace"])
        else:
            print("  traces: summarized (use --verbose-traces for full detail)")
        print()

    recalls = [b["recall"] for b in batch_run_summaries if b["recall"] is not None]
    precisions = [b["precision"] for b in batch_run_summaries if b["precision"] is not None]
    diags = [
        b["diagnosis_accuracy"]
        for b in batch_run_summaries
        if b["diagnosis_accuracy"] is not None
    ]

    print(f"=== BATCH SUMMARY (mean over run batches, {len(selected_types)} type(s)) ===")
    print(f"  batch runs: {len(batch_run_summaries)}")
    print(f"  mean recall:    {_fmt_pct(statistics.mean(recalls)) if recalls else 'n/a'}")
    print(
        f"  mean precision: {_fmt_pct(statistics.mean(precisions)) if precisions else 'n/a'}"
    )
    print(
        f"  mean diagnosis: {_fmt_pct(statistics.mean(diags)) if diags else 'n/a'}"
    )
    if recalls:
        print(
            f"  recall spread:    "
            f"{_fmt_pct(min(recalls))} .. {_fmt_pct(max(recalls))}"
        )
    if precisions:
        print(
            f"  precision spread: "
            f"{_fmt_pct(min(precisions))} .. {_fmt_pct(max(precisions))}"
        )
    if diags:
        print(
            f"  diagnosis spread: "
            f"{_fmt_pct(min(diags))} .. {_fmt_pct(max(diags))}"
        )

    if inconsistent_types:
        print(
            f"  unstable detection: {', '.join(inconsistent_types)}"
        )
    else:
        print("  unstable detection: none")

    if no_prestep_lead_types:
        print(
            f"  no prestep lead (excursion types): {', '.join(no_prestep_lead_types)}"
        )
    else:
        print("  no prestep lead (excursion types): none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
