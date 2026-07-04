"""Score agent findings against ground truth — detection + diagnosis layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schema import FINDINGS_CONTRACT, GROUND_TRUTH_CONTRACT

DETECTION_LABELS = ("TP", "FP", "TN", "FN")
DIAGNOSIS_FIELDS = ("type", "location", "origin_step", "affected_param")


def load_ground_truth(path: str | Path) -> dict[str, Any]:
    """Load ground_truth.json from a dataset directory or file path."""
    p = Path(path)
    if p.is_dir():
        p = p / "ground_truth.json"
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    for key in GROUND_TRUTH_CONTRACT:
        if key not in data:
            raise ValueError(f"ground_truth missing required key {key!r} in {p}")
    return data


def _validate_findings(findings: dict[str, Any]) -> None:
    for key in FINDINGS_CONTRACT:
        if key not in findings:
            raise ValueError(f"findings missing required key {key!r}")


def classify_detection(ground_truth: dict[str, Any], findings: dict[str, Any]) -> str:
    """Return TP, FP, TN, or FN from excursion vs detected."""
    has_excursion = bool(ground_truth["excursion"])
    detected = bool(findings["detected"])
    if has_excursion and detected:
        return "TP"
    if has_excursion and not detected:
        return "FN"
    if not has_excursion and detected:
        return "FP"
    return "TN"


def _field_applicable(ground_truth: dict[str, Any], field: str) -> bool:
    """A diagnosis field is scored only when ground truth carries a value."""
    if field == "type":
        return bool(ground_truth["excursion"])
    value = ground_truth.get(field)
    return value is not None


def _field_correct(
    ground_truth: dict[str, Any], findings: dict[str, Any], field: str
) -> bool:
    return findings.get(field) == ground_truth.get(field)


def score_diagnosis(
    ground_truth: dict[str, Any], findings: dict[str, Any]
) -> dict[str, Any]:
    """
    Score diagnosis fields for a correctly-detected excursion (TP case).

    Only fields with non-null ground-truth values are counted.
    """
    fields: dict[str, dict[str, Any]] = {}
    applicable = 0
    correct = 0

    for field in DIAGNOSIS_FIELDS:
        if not _field_applicable(ground_truth, field):
            fields[field] = {"applicable": False, "correct": None}
            continue
        is_correct = _field_correct(ground_truth, findings, field)
        fields[field] = {"applicable": True, "correct": is_correct}
        applicable += 1
        if is_correct:
            correct += 1

    accuracy = (correct / applicable) if applicable else None
    return {
        "scored": True,
        "fields": fields,
        "applicable_count": applicable,
        "correct_count": correct,
        "accuracy": accuracy,
    }


def score_dataset(
    ground_truth: dict[str, Any], findings: dict[str, Any]
) -> dict[str, Any]:
    """Score one dataset: detection label + diagnosis (TP only)."""
    _validate_findings(findings)
    detection = classify_detection(ground_truth, findings)

    if detection == "TP":
        diagnosis = score_diagnosis(ground_truth, findings)
    else:
        diagnosis = {
            "scored": False,
            "fields": {
                field: {"applicable": False, "correct": None}
                for field in DIAGNOSIS_FIELDS
            },
            "applicable_count": 0,
            "correct_count": 0,
            "accuracy": None,
        }

    return {
        "dataset_id": ground_truth.get("dataset_id"),
        "ground_truth_type": ground_truth.get("type"),
        "detection": detection,
        "diagnosis": diagnosis,
    }


def score_batch(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    """Score multiple datasets and return per-dataset results plus aggregates."""
    results = [score_dataset(gt, findings) for gt, findings in pairs]
    return {
        "datasets": results,
        "summary": aggregate_scores(results),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _count_labels(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in DETECTION_LABELS}
    for row in results:
        counts[row["detection"]] += 1
    return counts


def aggregate_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate detection recall/precision and diagnosis accuracy."""
    counts = _count_labels(results)
    tp, fp, tn, fn = (counts[k] for k in DETECTION_LABELS)

    detection = {
        "counts": counts,
        "recall": _ratio(tp, tp + fn),
        "precision": _ratio(tp, tp + fp),
        "specificity": _ratio(tn, tn + fp),
    }

    tp_results = [r for r in results if r["detection"] == "TP"]
    diagnosis_cases = len(tp_results)
    case_accuracies = [
        r["diagnosis"]["accuracy"]
        for r in tp_results
        if r["diagnosis"]["accuracy"] is not None
    ]
    overall_diagnosis_accuracy = (
        sum(case_accuracies) / len(case_accuracies) if case_accuracies else None
    )

    by_field: dict[str, dict[str, Any]] = {}
    for field in DIAGNOSIS_FIELDS:
        field_total = 0
        field_correct = 0
        for row in tp_results:
            info = row["diagnosis"]["fields"][field]
            if not info["applicable"]:
                continue
            field_total += 1
            if info["correct"]:
                field_correct += 1
        by_field[field] = {
            "correct": field_correct,
            "total": field_total,
            "accuracy": _ratio(field_correct, field_total),
        }

    by_type: dict[str, Any] = {}
    types = sorted({r["ground_truth_type"] for r in results})
    for anomaly_type in types:
        subset = [r for r in results if r["ground_truth_type"] == anomaly_type]
        type_counts = _count_labels(subset)
        t_tp, t_fp, t_tn, t_fn = (type_counts[k] for k in DETECTION_LABELS)
        type_tp_results = [r for r in subset if r["detection"] == "TP"]
        type_case_acc = [
            r["diagnosis"]["accuracy"]
            for r in type_tp_results
            if r["diagnosis"]["accuracy"] is not None
        ]

        type_entry: dict[str, Any] = {
            "counts": type_counts,
            "recall": _ratio(t_tp, t_tp + t_fn),
            "precision": _ratio(t_tp, t_tp + t_fp),
            "false_alarm_rate": _ratio(t_fp, t_fp + t_tn),
            "diagnosis_cases": len(type_tp_results),
            "diagnosis_accuracy": (
                sum(type_case_acc) / len(type_case_acc) if type_case_acc else None
            ),
        }

        type_by_field: dict[str, dict[str, Any]] = {}
        for field in DIAGNOSIS_FIELDS:
            f_total = 0
            f_correct = 0
            for row in type_tp_results:
                info = row["diagnosis"]["fields"][field]
                if not info["applicable"]:
                    continue
                f_total += 1
                if info["correct"]:
                    f_correct += 1
            type_by_field[field] = {
                "correct": f_correct,
                "total": f_total,
                "accuracy": _ratio(f_correct, f_total),
            }
        type_entry["diagnosis_by_field"] = type_by_field
        by_type[anomaly_type] = type_entry

    return {
        "detection": detection,
        "diagnosis": {
            "cases": diagnosis_cases,
            "accuracy": overall_diagnosis_accuracy,
            "by_field": by_field,
            "by_type": {
                t: {
                    "cases": by_type[t]["diagnosis_cases"],
                    "accuracy": by_type[t]["diagnosis_accuracy"],
                    "by_field": by_type[t]["diagnosis_by_field"],
                }
                for t in types
                if by_type[t]["diagnosis_cases"] > 0
            },
        },
        "by_type": by_type,
    }
