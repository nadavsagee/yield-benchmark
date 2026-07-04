#!/usr/bin/env python3
"""
Phase 3b verification — unsupervised pre-step on Phase 1 _verify datasets.

Ground truth is used ONLY to identify expected excursion lots and to assert
ranking behaviour. run_prestep() never reads ground_truth.json.

Usage:
    python verify_prestep.py
    python verify_prestep.py --prestep-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.prestep import STRONG_ANOMALY_THRESHOLD, run_prestep
from agent.tools._load import load_tables
from schema import ANOMALY_TYPES

VERIFY_DIR = Path("datasets") / "_verify"
EXCURSION_TYPES = [t for t in ANOMALY_TYPES if t != "clean"]
TOP_RANK_TOLERANCE = 3


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise AssertionError(msg)


def _ok(msg: str) -> None:
    print(f"  OK: {msg}")


def _load_gt(anomaly_type: str) -> dict:
    path = VERIFY_DIR / f"verify_{anomaly_type}" / "ground_truth.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _best_affected_score(result: dict, affected_lots: list[str]) -> tuple[str, float, int]:
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


def check_no_ground_truth_in_prestep() -> None:
    print("\n=== prestep does not reference ground_truth ===")
    src = Path("agent/prestep.py").read_text(encoding="utf-8")
    if "ground_truth" in src:
        _fail("agent/prestep.py mentions ground_truth")
    _ok("no ground_truth references in prestep module")


def check_clean() -> None:
    print("\n=== clean dataset - no strong anomalies ===")
    tables = load_tables(VERIFY_DIR / "verify_clean")
    result = run_prestep(tables)

    strong = [
        row for row in result["suspects"]
        if row["anomaly_score"] >= STRONG_ANOMALY_THRESHOLD
    ]
    if strong:
        top = strong[0]
        _fail(
            f"expected all scores below {STRONG_ANOMALY_THRESHOLD}, "
            f"but {len(strong)} lots flagged strongly "
            f"(top: {top['lot']} score={top['anomaly_score']})"
        )

    max_score = result["suspects"][0]["anomaly_score"]
    _ok(
        f"all {result['n_lots']} lots below strong threshold "
        f"(max anomaly_score={max_score:.3f})"
    )


def check_excursion_type(anomaly_type: str) -> None:
    print(f"\n=== {anomaly_type} - excursion lot near top ===")
    gt = _load_gt(anomaly_type)
    affected_lots = gt.get("affected_lots") or []
    if not affected_lots:
        _fail(f"{anomaly_type} missing affected_lots in ground truth")

    tables = load_tables(VERIFY_DIR / f"verify_{anomaly_type}")
    result = run_prestep(tables)

    best_lot, best_score, best_rank = _best_affected_score(result, affected_lots)
    top = result["suspects"][0]

    if best_score < STRONG_ANOMALY_THRESHOLD:
        _fail(
            f"expected best affected lot >= {STRONG_ANOMALY_THRESHOLD}, "
            f"got {best_lot} score={best_score:.3f}"
        )

    if best_rank > TOP_RANK_TOLERANCE:
        top3 = result["suspects"][:TOP_RANK_TOLERANCE]
        _fail(
            f"expected an affected lot in top {TOP_RANK_TOLERANCE}, "
            f"got best={best_lot} rank={best_rank} (score={best_score:.3f}). "
            f"Top: {[(r['lot'], r['anomaly_score']) for r in top3]}"
        )

    drivers = ", ".join(
        f"{d['feature']}({d['z_score']:+.1f})"
        for d in next(r for r in result["suspects"] if r["lot"] == best_lot)["top_features"][:3]
    )
    _ok(
        f"best affected {best_lot} rank={best_rank}/{result['n_lots']}, "
        f"score={best_score:.3f} (top={top['lot']} @ {top['anomaly_score']:.3f}), "
        f"drivers: {drivers}"
    )


def check_summary_shape() -> None:
    print("\n=== summary contract ===")
    tables = load_tables(VERIFY_DIR / "verify_propagation")
    result = run_prestep(tables)

    for key in ("n_lots", "suspects", "pca", "strong_threshold"):
        if key not in result:
            _fail(f"missing key {key!r} in prestep summary")

    row = result["suspects"][0]
    for key in ("lot", "rank", "anomaly_score", "top_features", "pca"):
        if key not in row:
            _fail(f"missing suspect key {key!r}")

    if not row["top_features"] or "feature" not in row["top_features"][0]:
        _fail("top_features entries malformed")

    if "explained_variance_ratio" not in result["pca"]:
        _fail("pca missing explained_variance_ratio")

    _ok(
        f"summary shape valid ({result['feature_count']} features, "
        f"PCA {result['pca']['n_components']} components)"
    )


def run_prestep_only_report(selected_types: list[str] | None = None) -> None:
    types = selected_types if selected_types else list(ANOMALY_TYPES)
    label = ", ".join(types) if selected_types else "all 8 types"
    print(f"\n=== prestep-only summary ({label}) ===")
    for anomaly_type in types:
        gt = _load_gt(anomaly_type)
        result = run_prestep(load_tables(VERIFY_DIR / f"verify_{anomaly_type}"))
        top = result["suspects"][0]
        affected = gt.get("affected_lots") or []
        if affected:
            best_lot, best_score, best_rank = _best_affected_score(result, affected)
            status = "PASS" if best_score >= STRONG_ANOMALY_THRESHOLD else "FAIL"
            print(
                f"  {anomaly_type:20} [{status}] "
                f"best_affected={best_lot}@{best_score:.2f} rank={best_rank} "
                f"top={top['lot']}@{top['anomaly_score']:.2f}"
            )
        else:
            status = "PASS" if top["anomaly_score"] < STRONG_ANOMALY_THRESHOLD else "FAIL"
            print(
                f"  {anomaly_type:20} [{status}] "
                f"max={top['anomaly_score']:.2f}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify unsupervised pre-step.")
    parser.add_argument(
        "--prestep-only",
        action="store_true",
        help="Print per-type prestep scores only (no assertion suite).",
    )
    parser.add_argument(
        "--types",
        default=None,
        help="Comma-separated anomaly types to run (default: all 8).",
    )
    return parser.parse_args(argv)


def _parse_types(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    selected = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [t for t in selected if t not in ANOMALY_TYPES]
    if unknown:
        raise SystemExit(f"Unknown type(s): {', '.join(unknown)}")
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("yield-benchmark Phase 3b pre-step verification")
    if not VERIFY_DIR.exists():
        _fail(f"{VERIFY_DIR} not found — run verify_generation.py first (Phase 1)")

    if args.prestep_only:
        run_prestep_only_report(_parse_types(args.types))
        return 0

    check_no_ground_truth_in_prestep()
    check_summary_shape()
    check_clean()
    for anomaly_type in EXCURSION_TYPES:
        check_excursion_type(anomaly_type)

    print("\n=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\nVerification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
