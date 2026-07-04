"""Orchestrate dataset generation: normal material + injector + ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from schema import ANOMALY_TYPES, DIFFICULTY_LEVELS, LOTS_PER_DATASET

from generator.injectors import (
    EARLY_DETECTION_SEQUENCE_LEN,
    INJECTORS,
    drift_sequence,
)
from generator.normal import build_normal, lot_id

_DIFFICULTY_NOISE = {
    "easy": 0.15,
    "medium": 0.30,
    "hard": 0.50,
}


def _pick_excursion_lot(rng: np.random.Generator) -> str:
    """Random lot index so agents cannot cheat by always checking the newest lot."""
    idx = int(rng.integers(5, LOTS_PER_DATASET - 4))
    return lot_id(idx)


def _pick_drift_start_lot(rng: np.random.Generator) -> str:
    """Start lot for early_detection drift sequences (room for consecutive lots)."""
    max_start = LOTS_PER_DATASET - EARLY_DETECTION_SEQUENCE_LEN - 4
    idx = int(rng.integers(10, max_start))
    return lot_id(idx)


def generate_benchmark(
    dataset_id: str,
    anomaly_type: str,
    difficulty: str = "medium",
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build one benchmark dataset: 4 CSVs + ground_truth.json.

    Parameters
    ----------
    dataset_id : e.g. 'ds_001'
    anomaly_type : one of ANOMALY_TYPES
    difficulty : easy | medium | hard
    seed : RNG seed (random if None)
    output_dir : write CSVs + ground_truth here (default: datasets/<dataset_id>)

    Returns
    -------
    ground_truth dict (also written to disk when output_dir is set)
    """
    if anomaly_type not in INJECTORS:
        raise ValueError(
            f"Unknown anomaly_type {anomaly_type!r}. "
            f"Supported: {ANOMALY_TYPES}"
        )
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(f"difficulty must be one of {DIFFICULTY_LEVELS}")

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))
    rng = np.random.default_rng(seed)
    noise_level = _DIFFICULTY_NOISE[difficulty]

    tables = build_normal(seed=seed, noise_level=noise_level)
    tables.pop("_meta", None)

    if anomaly_type == "clean":
        affected_lots: list[str] = []
        excursion = False
        inject_result = INJECTORS["clean"](tables, None, rng, noise_level)
    elif anomaly_type == "early_detection":
        start_lot = _pick_drift_start_lot(rng)
        affected_lots = drift_sequence(start_lot, EARLY_DETECTION_SEQUENCE_LEN)
        excursion = True
        inject_result = INJECTORS[anomaly_type](
            tables, start_lot, rng, noise_level
        )
        affected_lots = inject_result.get("affected_lots", affected_lots)
    else:
        affected_lot = _pick_excursion_lot(rng)
        affected_lots = [affected_lot]
        excursion = True
        inject_result = INJECTORS[anomaly_type](
            tables, affected_lot, rng, noise_level
        )

    tables = inject_result["tables"]

    ground_truth: dict[str, Any] = {
        "dataset_id": dataset_id,
        "difficulty": difficulty,
        "excursion": excursion,
        "type": anomaly_type,
        "location": inject_result["location"],
        "origin_step": inject_result["origin_step"],
        "affected_param": inject_result["affected_param"],
        "signature": inject_result["signature"],
        "causal_chain": inject_result["causal_chain"],
        "confounder": inject_result.get("confounder"),
        "affected_lots": affected_lots,
        "noise_level": noise_level,
        "seed": seed,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tables["sort"].to_csv(out / "sort.csv", index=False)
        tables["wat"].to_csv(out / "wat.csv", index=False)
        tables["inline"].to_csv(out / "inline.csv", index=False)
        tables["route"].to_csv(out / "route.csv", index=False)
        with open(out / "ground_truth.json", "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)

    return ground_truth
