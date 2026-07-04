#!/usr/bin/env python3
"""
Run v0_baseline agent on Phase 1 _verify datasets and score with Phase 2 scorer.

Requires ANTHROPIC_API_KEY.

Usage:
    python run_v0.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from agent.v0 import investigate
from scorer import load_ground_truth, score_batch, score_dataset

VERIFY_DIR = Path("datasets") / "_verify"
VERIFY_TYPES = ("edge_signature", "chamber_specific", "propagation", "clean")


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _print_trace(trace: list[dict]) -> None:
    calls = [f"{t['tool']}({', '.join(f'{k}={v!r}' for k, v in t.get('args', {}).items())})"
             for t in trace]
    for i, call in enumerate(calls, 1):
        print(f"    {i}. {call}")


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 1

    if not VERIFY_DIR.exists():
        print(f"{VERIFY_DIR} not found — run verify_generation.py first.", file=sys.stderr)
        return 1

    print("yield-benchmark v0_baseline — investigate + score")
    print(f"Datasets: {VERIFY_DIR.resolve()}\n")

    pairs: list[tuple[dict, dict]] = []
    results_meta: list[dict] = []

    for anomaly_type in VERIFY_TYPES:
        dataset_path = VERIFY_DIR / f"verify_{anomaly_type}"
        gt = load_ground_truth(dataset_path)
        print(f"=== {anomaly_type} ({gt['dataset_id']}) ===")

        out = investigate(dataset_path)
        findings = out["findings"]
        row = score_dataset(gt, findings)
        pairs.append((gt, findings))
        results_meta.append({"type": anomaly_type, "row": row, "trace": out["trace"]})

        det = row["detection"]
        diag = row["diagnosis"]
        print(f"  detection: {det}")
        if diag["scored"]:
            acc = diag["accuracy"]
            print(f"  diagnosis accuracy: {_fmt_pct(acc) if acc is not None else 'n/a'}")
        else:
            print(f"  diagnosis accuracy: n/a ({det} — not scored)")
        print(f"  findings: detected={findings['detected']}, type={findings['type']!r}, "
              f"param={findings['affected_param']!r}, location={findings['location']!r}")
        print(f"  model turns: {out['iterations']}, tool calls: {len(out['trace'])}")
        print("  tool trace:")
        _print_trace(out["trace"])
        print()

    batch = score_batch(pairs)
    summary = batch["summary"]
    det = summary["detection"]
    diag = summary["diagnosis"]

    print("=== BATCH SUMMARY ===")
    print(f"  detection counts: {det['counts']}")
    print(f"  recall:    {_fmt_pct(det['recall'])}")
    print(f"  precision: {_fmt_pct(det['precision'])}")
    print(f"  diagnosis accuracy: {_fmt_pct(diag['accuracy'])}")
    print(f"  diagnosis cases (TP): {diag['cases']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
