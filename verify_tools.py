#!/usr/bin/env python3
"""
Phase 3a verification — run core agent tools on Phase 1 _verify datasets.

Ground truth is used ONLY to pick which lot to test and to assert expected outcomes.
The tools themselves never read ground_truth.json.

Usage:
    python verify_tools.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent.tools._load import load_tables
from agent.tools.chain_correlate import chain_correlate
from agent.tools.commonality import commonality
from agent.tools.excursion_confirm import excursion_confirm
from agent.tools.inline_trace import inline_trace
from agent.tools.spatial_signature import spatial_signature
from agent.tools.wat_profile import wat_profile

VERIFY_DIR = Path("datasets") / "_verify"


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise AssertionError(msg)


def _ok(msg: str) -> None:
    print(f"  OK: {msg}")


def _load_gt(anomaly_type: str) -> dict:
    path = VERIFY_DIR / f"verify_{anomaly_type}" / "ground_truth.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _excursion_lot(gt: dict, fallback: str = "lot_010") -> str:
    lots = gt.get("affected_lots") or []
    return lots[0] if lots else fallback


def check_spatial_signature_edge() -> None:
    print("\n=== spatial_signature on edge_signature ===")
    gt = _load_gt("edge_signature")
    lot = _excursion_lot(gt)
    tables = load_tables(VERIFY_DIR / "verify_edge_signature")

    result = spatial_signature(lot, tables, population="fail")
    if result["signature"] != "edge_ring":
        _fail(f"expected edge_ring, got {result['signature']!r} ({result})")
    if result["edge_fail_ratio"] <= result["center_fail_ratio"]:
        _fail(
            f"edge_fail_ratio should exceed center: "
            f"{result['edge_fail_ratio']} vs {result['center_fail_ratio']}"
        )
    _ok(
        f"lot {lot} -> signature=edge_ring, "
        f"edge={result['edge_fail_ratio']:.1%}, center={result['center_fail_ratio']:.1%}"
    )


def check_commonality_chamber() -> None:
    print("\n=== commonality on chamber_specific ===")
    gt = _load_gt("chamber_specific")
    lot = _excursion_lot(gt)
    expected_chamber = gt["location"]
    tables = load_tables(VERIFY_DIR / "verify_chamber_specific")

    result = commonality(lot, tables, population="yield")
    if result["commons_to"] != expected_chamber:
        _fail(f"expected commons_to={expected_chamber!r}, got {result!r}")
    if result["type"] != "chamber":
        _fail(f"expected type='chamber', got {result['type']!r}")
    if result["yield_gap"] <= 0:
        _fail(f"expected positive yield_gap, got {result['yield_gap']}")
    _ok(
        f"lot {lot} -> commons_to={result['commons_to']}, "
        f"yield_gap={result['yield_gap']:.3f}, strength={result['strength']:.2f}"
    )


def check_wat_profile_propagation() -> None:
    print("\n=== wat_profile(rc_ohm) on propagation ===")
    gt = _load_gt("propagation")
    lot = _excursion_lot(gt)
    tables = load_tables(VERIFY_DIR / "verify_propagation")

    result = wat_profile(lot, "rc_ohm", tables)
    if result["sigma_shift"] <= 2.0:
        _fail(f"expected positive sigma_shift > 2, got {result['sigma_shift']}")
    if result["mean"] <= result["baseline_mean"]:
        _fail(
            f"lot mean should exceed baseline: "
            f"{result['mean']} vs {result['baseline_mean']}"
        )
    _ok(
        f"lot {lot} -> sigma_shift={result['sigma_shift']:.2f}, "
        f"mean={result['mean']:.1f} vs baseline {result['baseline_mean']:.1f}"
    )


def check_chain_correlate_propagation() -> None:
    print("\n=== chain_correlate on propagation ===")
    gt = _load_gt("propagation")
    lot = _excursion_lot(gt)
    tables = load_tables(VERIFY_DIR / "verify_propagation")

    result = chain_correlate(lot, tables)
    if not result["chain_intact"]:
        _fail(f"expected chain_intact=True, got {result}")
    if result["break_at"] is not None:
        _fail(f"expected break_at=None, got {result['break_at']!r}")
    inline_link = result["links"][0]
    if inline_link["status"] != "elevated":
        _fail(f"inline defect link should be elevated: {inline_link}")
    _ok(f"lot {lot} -> chain_intact=True, break_at=None, links elevated")


def check_chain_correlate_clean() -> None:
    print("\n=== chain_correlate on clean ===")
    tables = load_tables(VERIFY_DIR / "verify_clean")
    lot = "lot_010"

    result = chain_correlate(lot, tables)
    if result["break_at"] is not None:
        _fail(f"clean lot should have break_at=None, got {result['break_at']!r}")
    if not result["chain_intact"]:
        _fail(f"clean lot should have chain_intact=True, got {result}")
    _ok(f"lot {lot} -> chain_intact=True, no break flagged")


def check_excursion_confirm_clean() -> None:
    print("\n=== excursion_confirm on clean ===")
    tables = load_tables(VERIFY_DIR / "verify_clean")
    lot = "lot_010"

    result = excursion_confirm(lot, "rc_ohm", tables)
    if result["out_of_control"]:
        _fail(f"clean lot rc_ohm should be in control, got {result}")
    if abs(result["sigma"]) >= 3.0:
        _fail(f"clean lot sigma should be < 3, got {result['sigma']}")
    _ok(
        f"lot {lot} -> out_of_control=False, sigma={result['sigma']:.2f}, "
        f"pct_out={result['pct_out']:.1%}"
    )


def check_inline_trace_early_detection() -> None:
    print("\n=== inline_trace on early_detection ===")
    tables = load_tables(VERIFY_DIR / "verify_early_detection")
    lot = "lot_038"

    result = inline_trace(lot, tables)
    if result["step"] != "m1_litho":
        _fail(f"expected step='m1_litho', got {result!r}")
    if result["metric"] != "overlay_nm":
        _fail(f"expected metric='overlay_nm', got {result!r}")
    if result["level_sigma"] < 4.0:
        _fail(f"expected level_sigma ~4-5 (subtle drift), got {result['level_sigma']}")
    if result["level_sigma"] > 6.0:
        _fail(f"expected level_sigma ~4-5 (not a cliff), got {result['level_sigma']}")
    if not (0.25 <= result["trend_slope"] <= 0.40):
        _fail(f"expected gentle trend_slope ~0.30, got {result['trend_slope']}")
    if result["sustained"] < 1.15:
        _fail(f"expected sustained drift signal, got {result['sustained']}")
    if not result["out_of_control"]:
        _fail(f"expected out_of_control=True, got {result}")
    _ok(
        f"lot {lot} -> {result['step']}/{result['metric']}, "
        f"level_sigma={result['level_sigma']:.2f}, "
        f"trend_slope={result['trend_slope']:.3f}, "
        f"sustained={result['sustained']:.3f}, out_of_control=True"
    )


def check_no_ground_truth_in_tools() -> None:
    print("\n=== tools do not reference ground_truth ===")
    tool_files = [
        Path("agent/tools/spatial_signature.py"),
        Path("agent/tools/wat_profile.py"),
        Path("agent/tools/commonality.py"),
        Path("agent/tools/chain_correlate.py"),
        Path("agent/tools/excursion_confirm.py"),
        Path("agent/tools/inline_trace.py"),
        Path("agent/tools/_inline.py"),
        Path("agent/tools/_baseline.py"),
    ]
    for path in tool_files:
        src = path.read_text(encoding="utf-8")
        if "ground_truth" in src:
            _fail(f"{path} mentions ground_truth")
    _ok("no ground_truth references in tool modules")


def main() -> int:
    print("yield-benchmark Phase 3a tools verification")
    if not VERIFY_DIR.exists():
        _fail(f"{VERIFY_DIR} not found — run verify_generation.py first (Phase 1)")

    check_no_ground_truth_in_tools()
    check_spatial_signature_edge()
    check_commonality_chamber()
    check_wat_profile_propagation()
    check_chain_correlate_propagation()
    check_chain_correlate_clean()
    check_excursion_confirm_clean()
    check_inline_trace_early_detection()

    print("\n=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\nVerification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
