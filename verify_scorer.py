#!/usr/bin/env python3
"""
Phase 2 verification — confirm scorer metrics with hand-made findings.

Usage:
    python verify_scorer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from scorer import aggregate_scores, load_ground_truth, score_batch, score_dataset

VERIFY_DIR = Path("datasets") / "_verify"


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise AssertionError(msg)


def _ok(msg: str) -> None:
    print(f"  OK: {msg}")


def _approx(a: float | None, b: float, tol: float = 1e-9) -> bool:
    if a is None:
        return False
    return abs(a - b) <= tol


def _load_verify_ground_truths() -> dict[str, dict]:
    gts: dict[str, dict] = {}
    for anomaly_type in (
        "edge_signature",
        "chamber_specific",
        "propagation",
        "clean",
    ):
        path = VERIFY_DIR / f"verify_{anomaly_type}"
        gts[anomaly_type] = load_ground_truth(path)
    return gts


def _base_findings(**overrides) -> dict:
    base = {
        "detected": True,
        "type": None,
        "location": None,
        "origin_step": None,
        "affected_param": None,
        "cause": "test",
        "confidence": "high",
        "reasoning": "verification fixture",
    }
    base.update(overrides)
    return base


def perfect_findings(gt: dict) -> dict:
    """Mirror ground truth into the findings shape for a perfect agent."""
    if not gt["excursion"]:
        return _base_findings(
            detected=False,
            type=None,
            location=None,
            origin_step=None,
            affected_param=None,
            cause="No excursion detected",
        )
    return _base_findings(
        detected=True,
        type=gt["type"],
        location=gt["location"],
        origin_step=gt["origin_step"],
        affected_param=gt["affected_param"],
        cause="Matches injected excursion",
    )


def check_perfect_scores(gts: dict[str, dict]) -> None:
    print("\n=== perfect findings ===")
    pairs = [(gts[t], perfect_findings(gts[t])) for t in gts]
    out = score_batch(pairs)
    summary = out["summary"]

    det = summary["detection"]
    if det["counts"] != {"TP": 3, "FP": 0, "TN": 1, "FN": 0}:
        _fail(f"perfect detection counts: {det['counts']}")
    if not _approx(det["recall"], 1.0):
        _fail(f"perfect recall expected 1.0, got {det['recall']}")
    if not _approx(det["precision"], 1.0):
        _fail(f"perfect precision expected 1.0, got {det['precision']}")
    if not _approx(det["specificity"], 1.0):
        _fail(f"perfect specificity expected 1.0, got {det['specificity']}")

    diag = summary["diagnosis"]
    if diag["cases"] != 3:
        _fail(f"perfect diagnosis cases expected 3, got {diag['cases']}")
    if not _approx(diag["accuracy"], 1.0):
        _fail(f"perfect diagnosis accuracy expected 1.0, got {diag['accuracy']}")

    for field in ("type", "location", "origin_step", "affected_param"):
        acc = diag["by_field"][field]["accuracy"]
        if field in ("location", "origin_step"):
            # Only chamber_specific + propagation contribute those fields.
            if acc is None:
                continue
        if field == "location":
            if not _approx(acc, 1.0):
                _fail(f"perfect {field} accuracy expected 1.0, got {acc}")
        elif field == "origin_step":
            if not _approx(acc, 1.0):
                _fail(f"perfect {field} accuracy expected 1.0, got {acc}")
        else:
            if not _approx(acc, 1.0):
                _fail(f"perfect {field} accuracy expected 1.0, got {acc}")

    for t in ("edge_signature", "chamber_specific", "propagation"):
        type_diag = summary["diagnosis"]["by_type"][t]
        if not _approx(type_diag["accuracy"], 1.0):
            _fail(f"perfect per-type diagnosis for {t}: {type_diag['accuracy']}")

    _ok("perfect agent -> recall=1, precision=1, diagnosis=1, counts TP=3 TN=1")


def check_miss_all_excursions(gts: dict[str, dict]) -> None:
    print("\n=== miss all excursions (detected=False everywhere) ===")
    pairs = []
    for t, gt in gts.items():
        pairs.append((gt, _base_findings(detected=False, type=None, cause="nothing")))
    summary = score_batch(pairs)["summary"]
    det = summary["detection"]

    if det["counts"] != {"TP": 0, "FP": 0, "TN": 1, "FN": 3}:
        _fail(f"miss-all counts: {det['counts']}")
    if not _approx(det["recall"], 0.0):
        _fail(f"miss-all recall expected 0.0, got {det['recall']}")
    if det["precision"] is not None:
        _fail(f"miss-all precision expected None, got {det['precision']}")

    if summary["diagnosis"]["cases"] != 0:
        _fail("miss-all should have zero diagnosis cases")

    edge = summary["by_type"]["edge_signature"]
    if not _approx(edge["recall"], 0.0):
        _fail(f"edge_signature recall expected 0.0, got {edge['recall']}")

    _ok("miss-all -> recall=0, FN=3, TN=1, no diagnosis scored")


def check_false_alarm_on_clean(gts: dict[str, dict]) -> None:
    print("\n=== false alarm on clean + perfect on excursions ===")
    pairs = []
    for t, gt in gts.items():
        if t == "clean":
            findings = _base_findings(
                detected=True,
                type="edge_signature",
                affected_param="rc_ohm",
                cause="false alarm",
            )
        else:
            findings = perfect_findings(gt)
        pairs.append((gt, findings))

    summary = score_batch(pairs)["summary"]
    det = summary["detection"]
    if det["counts"] != {"TP": 3, "FP": 1, "TN": 0, "FN": 0}:
        _fail(f"false-alarm counts: {det['counts']}")
    if not _approx(det["recall"], 1.0):
        _fail(f"false-alarm recall expected 1.0, got {det['recall']}")
    if not _approx(det["precision"], 0.75):
        _fail(f"false-alarm precision expected 0.75, got {det['precision']}")

    clean = summary["by_type"]["clean"]
    if not _approx(clean["false_alarm_rate"], 1.0):
        _fail(f"clean false_alarm_rate expected 1.0, got {clean['false_alarm_rate']}")

    _ok("one FP on clean -> precision=0.75, clean false_alarm_rate=1.0")


def check_wrong_diagnosis(gts: dict[str, dict]) -> None:
    print("\n=== detected correctly but wrong diagnosis on propagation ===")
    pairs = [(gts[t], perfect_findings(gts[t])) for t in gts]
    # Replace propagation with correct detection but wrong origin_step + param.
    gt_prop = gts["propagation"]
    wrong_prop = _base_findings(
        detected=True,
        type="propagation",
        location=None,
        origin_step="gate_etch",
        affected_param="idsat_uA",
        cause="wrong chain",
    )
    pairs = [(p[0], p[1]) for p in pairs if p[0]["type"] != "propagation"]
    pairs.append((gt_prop, wrong_prop))

    result = score_dataset(gt_prop, wrong_prop)
    if result["detection"] != "TP":
        _fail(f"wrong-diagnosis should still be TP, got {result['detection']}")

    diag = result["diagnosis"]
    if not _approx(diag["accuracy"], 1 / 3):
        _fail(f"propagation wrong 2/3 fields -> accuracy 1/3, got {diag['accuracy']}")

    fields = diag["fields"]
    if not fields["type"]["correct"]:
        _fail("type should still be correct")
    if fields["origin_step"]["correct"]:
        _fail("origin_step should be wrong")
    if fields["affected_param"]["correct"]:
        _fail("affected_param should be wrong")

    summary = score_batch(pairs)["summary"]
    overall = summary["diagnosis"]["accuracy"]
    # 2 perfect TPs (acc=1) + 1 partial TP (acc=1/3) -> mean over 3 excursion cases
    expected = (1 + 1 + 1 / 3) / 3
    if not _approx(overall, expected):
        _fail(f"overall diagnosis accuracy expected {expected}, got {overall}")

    prop_type = summary["by_type"]["propagation"]
    if not _approx(prop_type["diagnosis_accuracy"], 1 / 3):
        _fail(
            f"propagation per-type diagnosis expected 1/3, "
            f"got {prop_type['diagnosis_accuracy']}"
        )

    _ok(
        "wrong propagation diagnosis -> type correct, origin/param wrong, "
        f"case accuracy=1/3, batch accuracy={expected:.4f}"
    )


def check_single_dataset_labels(gts: dict[str, dict]) -> None:
    print("\n=== per-dataset detection labels ===")
    expectations = {
        "edge_signature": "TP",
        "chamber_specific": "TP",
        "propagation": "TP",
        "clean": "TN",
    }
    for t, expected in expectations.items():
        row = score_dataset(gts[t], perfect_findings(gts[t]))
        if row["detection"] != expected:
            _fail(f"{t} perfect findings -> {expected}, got {row['detection']}")
    _ok("TP/TN labels correct for perfect findings on each verify dataset")


def main() -> int:
    print("yield-benchmark Phase 2 scorer verification")
    if not VERIFY_DIR.exists():
        _fail(
            f"{VERIFY_DIR} not found — run verify_generation.py first "
            "(Phase 1)"
        )

    gts = _load_verify_ground_truths()
    check_single_dataset_labels(gts)
    check_perfect_scores(gts)
    check_miss_all_excursions(gts)
    check_false_alarm_on_clean(gts)
    check_wrong_diagnosis(gts)

    print("\n=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\nVerification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
