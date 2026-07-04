"""Anomaly injectors for Phase 1 types."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from schema import BASELINE

from generator.normal import commonality_step, lot_id

# Edge ring threshold (matches spatial.py edge sites ~0.80)
_EDGE_RADIUS = 0.65
_EDGE_PARAM = "rc_ohm"

# Chamber excursion targets this step + chamber
_CHAMBER_STEP = commonality_step()
_CHAMBER_PARAM = "idsat_uA"

# Propagation chain: inline defect -> rc WAT -> speed sort fails
_PROPAGATION_ORIGIN = "contact_etch"
_PROPAGATION_PARAM = "rc_ohm"
_PROPAGATION_SOFT_BIN = "speed"

# mean_shift: lot-wide WAT param crosses upper spec
_MEAN_SHIFT_PARAM = "vtn_mV"
_MEAN_SHIFT_SPEC_SIGMA = 4.0

# early_detection: inline drift across consecutive lots, WAT/Sort still normal
_EARLY_DETECTION_ORIGIN = "m1_litho"
_EARLY_DETECTION_METRIC = "overlay_nm"
_EARLY_DETECTION_SEQUENCE_LEN = 4
# Total overlay ramp on the last lot (~baseline σ units). Gentle, not a cliff.
_EARLY_DETECTION_RAMP_SIGMA = 1.35
# Back-loaded fractions: same terminal offset, steeper trailing-window trend for pre-step
_EARLY_DETECTION_RAMP_FRACTIONS = [0.10, 0.30, 0.60, 1.00]
EARLY_DETECTION_SEQUENCE_LEN = _EARLY_DETECTION_SEQUENCE_LEN

# correlation_break: Sort fails with in-family WAT
_CORRELATION_BREAK_SOFT_BIN = "gross"

# confounding: correlated param pair, only one causal for Sort
_CONFOUNDING_PARAM = "rc_ohm"
_CONFOUNDING_CONFOUNDER = "rs_ohm"


def _lot_index(lot: str) -> int:
    return int(lot.split("_")[1])


def drift_sequence(start_lot: str, length: int = _EARLY_DETECTION_SEQUENCE_LEN) -> list[str]:
    """Consecutive lot ids beginning at start_lot."""
    start = _lot_index(start_lot)
    return [lot_id(start + i) for i in range(length)]


def inject_edge_signature(
    tables: dict[str, pd.DataFrame],
    affected_lot: str,
    rng: np.random.Generator,
    noise_level: float,
) -> dict[str, Any]:
    """Edge-concentrated WAT shift + sort fails on the wafer edge."""
    sort_df = tables["sort"].copy()
    wat_df = tables["wat"].copy()

    shift = BASELINE[_EDGE_PARAM]["sigma"] * (4.0 + noise_level * 2)

    edge_wat = (wat_df["lot"] == affected_lot) & (wat_df["radius"] >= _EDGE_RADIUS)
    wat_df.loc[edge_wat, _EDGE_PARAM] = (
        wat_df.loc[edge_wat, _EDGE_PARAM] + shift
    )

    edge_sort = (sort_df["lot"] == affected_lot) & (sort_df["radius"] >= _EDGE_RADIUS)
    fail_prob = 0.55 + noise_level * 0.15
    for idx in sort_df.index[edge_sort]:
        if rng.random() < fail_prob:
            sort_df.at[idx, "pass"] = False
            sort_df.at[idx, "hard_bin"] = int(rng.integers(2, 5))
            sort_df.at[idx, "soft_bin"] = rng.choice(
                ["speed", "leakage", "gross"], p=[0.5, 0.3, 0.2]
            )

    return {
        "tables": {**tables, "sort": sort_df, "wat": wat_df},
        "location": None,
        "origin_step": None,
        "affected_param": _EDGE_PARAM,
        "signature": "edge_ring",
        "causal_chain": None,
        "confounder": None,
    }


def inject_chamber_specific(
    tables: dict[str, pd.DataFrame],
    affected_lot: str,
    rng: np.random.Generator,
    noise_level: float,
) -> dict[str, Any]:
    """Excursion only on wafers processed in one chamber at the commonality step."""
    route_df = tables["route"].copy()
    wat_df = tables["wat"].copy()
    inline_df = tables["inline"].copy()
    sort_df = tables["sort"]

    lot_routes = route_df[route_df["lot"] == affected_lot]
    chambers_in_lot = lot_routes[lot_routes["step"] == _CHAMBER_STEP]["chamber"].unique()
    affected_chamber = str(rng.choice(chambers_in_lot))

    affected_wafers = lot_routes[
        (lot_routes["step"] == _CHAMBER_STEP) & (lot_routes["chamber"] == affected_chamber)
    ]["wafer"].unique()

    shift = BASELINE[_CHAMBER_PARAM]["sigma"] * (5.0 + noise_level * 2)
    wat_mask = (wat_df["lot"] == affected_lot) & (wat_df["wafer"].isin(affected_wafers))
    wat_df.loc[wat_mask, _CHAMBER_PARAM] = (
        wat_df.loc[wat_mask, _CHAMBER_PARAM] - shift
    )

    inline_mask = (
        (inline_df["lot"] == affected_lot)
        & (inline_df["wafer"].isin(affected_wafers))
        & (inline_df["step"] == _CHAMBER_STEP)
    )
    inline_df.loc[inline_mask, "cd_nm"] = (
        inline_df.loc[inline_mask, "cd_nm"]
        + BASELINE["cd_nm"]["sigma"] * (3.0 + noise_level)
    )

    fail_prob = 0.35 + noise_level * 0.1
    sort_df = sort_df.copy()
    sort_mask = (sort_df["lot"] == affected_lot) & (sort_df["wafer"].isin(affected_wafers))
    for idx in sort_df.index[sort_mask]:
        if rng.random() < fail_prob:
            sort_df.at[idx, "pass"] = False
            sort_df.at[idx, "hard_bin"] = int(rng.integers(2, 5))
            sort_df.at[idx, "soft_bin"] = "speed"

    return {
        "tables": {
            **tables,
            "sort": sort_df,
            "wat": wat_df,
            "inline": inline_df,
        },
        "location": affected_chamber,
        "origin_step": _CHAMBER_STEP,
        "affected_param": _CHAMBER_PARAM,
        "signature": None,
        "causal_chain": None,
        "confounder": None,
    }


def inject_mean_shift(
    tables: dict[str, pd.DataFrame],
    affected_lot: str,
    rng: np.random.Generator,
    noise_level: float,
) -> dict[str, Any]:
    """Lot-wide WAT mean shift that crosses the upper spec limit."""
    wat_df = tables["wat"].copy()
    sort_df = tables["sort"].copy()

    spec_limit = (
        BASELINE[_MEAN_SHIFT_PARAM]["mean"]
        + _MEAN_SHIFT_SPEC_SIGMA * BASELINE[_MEAN_SHIFT_PARAM]["sigma"]
    )
    lot_mean = float(wat_df.loc[wat_df["lot"] == affected_lot, _MEAN_SHIFT_PARAM].mean())
    shift = (spec_limit - lot_mean) + BASELINE[_MEAN_SHIFT_PARAM]["sigma"] * (
        1.5 + noise_level
    )

    wat_mask = wat_df["lot"] == affected_lot
    wat_df.loc[wat_mask, _MEAN_SHIFT_PARAM] = (
        wat_df.loc[wat_mask, _MEAN_SHIFT_PARAM] + shift
    )

    fail_prob = 0.25 + noise_level * 0.1
    for idx in sort_df.index[sort_df["lot"] == affected_lot]:
        if rng.random() < fail_prob:
            sort_df.at[idx, "pass"] = False
            sort_df.at[idx, "hard_bin"] = int(rng.integers(2, 5))
            sort_df.at[idx, "soft_bin"] = rng.choice(["speed", "leakage"], p=[0.6, 0.4])

    return {
        "tables": {**tables, "sort": sort_df, "wat": wat_df},
        "location": None,
        "origin_step": None,
        "affected_param": _MEAN_SHIFT_PARAM,
        "signature": None,
        "causal_chain": ["vt_shift", "speed_fail_up"],
        "confounder": None,
    }


def inject_propagation(
    tables: dict[str, pd.DataFrame],
    affected_lot: str,
    rng: np.random.Generator,
    noise_level: float,
) -> dict[str, Any]:
    """Inline defect at origin step -> WAT rc shift -> speed sort fails."""
    inline_df = tables["inline"].copy()
    wat_df = tables["wat"].copy()
    sort_df = tables["sort"].copy()

    lot_wafers = inline_df.loc[inline_df["lot"] == affected_lot, "wafer"].unique()

    defect_boost = BASELINE["defect_density"]["sigma"] * (8.0 + noise_level * 4)
    origin_mask = (inline_df["lot"] == affected_lot) & (
        inline_df["step"] == _PROPAGATION_ORIGIN
    )
    inline_df.loc[origin_mask, "defect_density"] = (
        inline_df.loc[origin_mask, "defect_density"] + defect_boost
    )
    inline_df.loc[origin_mask, "defect_count"] = (
        inline_df.loc[origin_mask, "defect_count"]
        + (8 + int(rng.integers(2, 6)))
    ).astype(int)

    rc_shift = BASELINE[_PROPAGATION_PARAM]["sigma"] * (5.0 + noise_level * 2)
    wat_mask = (wat_df["lot"] == affected_lot) & (wat_df["wafer"].isin(lot_wafers))
    wat_df.loc[wat_mask, _PROPAGATION_PARAM] = (
        wat_df.loc[wat_mask, _PROPAGATION_PARAM] + rc_shift
    )

    fail_prob = 0.40 + noise_level * 0.15
    sort_mask = (sort_df["lot"] == affected_lot) & (sort_df["wafer"].isin(lot_wafers))
    for idx in sort_df.index[sort_mask]:
        if rng.random() < fail_prob:
            sort_df.at[idx, "pass"] = False
            sort_df.at[idx, "hard_bin"] = int(rng.integers(2, 5))
            sort_df.at[idx, "soft_bin"] = _PROPAGATION_SOFT_BIN

    return {
        "tables": {
            **tables,
            "sort": sort_df,
            "wat": wat_df,
            "inline": inline_df,
        },
        "location": None,
        "origin_step": _PROPAGATION_ORIGIN,
        "affected_param": _PROPAGATION_PARAM,
        "signature": None,
        "causal_chain": ["defect_up", "rc_up", "speed_fail_up"],
        "confounder": None,
    }


def inject_early_detection(
    tables: dict[str, pd.DataFrame],
    affected_lot: str,
    rng: np.random.Generator,
    noise_level: float,
) -> dict[str, Any]:
    """Inline metric drifts across consecutive lots; WAT and Sort stay in-family."""
    inline_df = tables["inline"].copy()
    sequence = drift_sequence(affected_lot, _EARLY_DETECTION_SEQUENCE_LEN)
    ramp_total = BASELINE[_EARLY_DETECTION_METRIC]["sigma"] * (
        _EARLY_DETECTION_RAMP_SIGMA + noise_level * 0.25
    )
    fractions = _EARLY_DETECTION_RAMP_FRACTIONS
    if len(fractions) != len(sequence):
        raise ValueError(
            f"ramp fractions length {len(fractions)} != sequence length {len(sequence)}"
        )

    for seq_idx, lot in enumerate(sequence):
        lot_mask = (inline_df["lot"] == lot) & (
            inline_df["step"] == _EARLY_DETECTION_ORIGIN
        )
        inline_df.loc[lot_mask, _EARLY_DETECTION_METRIC] = (
            inline_df.loc[lot_mask, _EARLY_DETECTION_METRIC]
            + ramp_total * fractions[seq_idx]
        )

    return {
        "tables": {**tables, "inline": inline_df},
        "location": None,
        "origin_step": _EARLY_DETECTION_ORIGIN,
        "affected_param": None,
        "signature": None,
        "causal_chain": ["overlay_shift"],
        "confounder": None,
        "affected_lots": sequence,
    }


def inject_correlation_break(
    tables: dict[str, pd.DataFrame],
    affected_lot: str,
    rng: np.random.Generator,
    noise_level: float,
) -> dict[str, Any]:
    """Real Sort failure with WAT left in-family."""
    sort_df = tables["sort"].copy()
    fail_prob = 0.45 + noise_level * 0.15

    lot_mask = sort_df["lot"] == affected_lot
    for idx in sort_df.index[lot_mask]:
        if rng.random() < fail_prob:
            sort_df.at[idx, "pass"] = False
            sort_df.at[idx, "hard_bin"] = int(rng.integers(2, 5))
            sort_df.at[idx, "soft_bin"] = _CORRELATION_BREAK_SOFT_BIN

    return {
        "tables": {**tables, "sort": sort_df},
        "location": None,
        "origin_step": None,
        "affected_param": None,
        "signature": None,
        "causal_chain": ["wat_normal", "sort_fail_up"],
        "confounder": None,
    }


def inject_confounding(
    tables: dict[str, pd.DataFrame],
    affected_lot: str,
    rng: np.random.Generator,
    noise_level: float,
) -> dict[str, Any]:
    """Two WAT params move together; only the causal param drives Sort fails."""
    wat_df = tables["wat"].copy()
    sort_df = tables["sort"].copy()

    causal_shift = BASELINE[_CONFOUNDING_PARAM]["sigma"] * (5.0 + noise_level * 2)
    conf_shift = BASELINE[_CONFOUNDING_CONFOUNDER]["sigma"] * (5.0 + noise_level * 2)

    wat_mask = wat_df["lot"] == affected_lot
    wat_df.loc[wat_mask, _CONFOUNDING_PARAM] = (
        wat_df.loc[wat_mask, _CONFOUNDING_PARAM] + causal_shift
    )
    wat_df.loc[wat_mask, _CONFOUNDING_CONFOUNDER] = (
        wat_df.loc[wat_mask, _CONFOUNDING_CONFOUNDER] + conf_shift
    )

    fail_prob = 0.38 + noise_level * 0.12
    for idx in sort_df.index[sort_df["lot"] == affected_lot]:
        if rng.random() < fail_prob:
            sort_df.at[idx, "pass"] = False
            sort_df.at[idx, "hard_bin"] = int(rng.integers(2, 5))
            sort_df.at[idx, "soft_bin"] = "speed"

    return {
        "tables": {**tables, "sort": sort_df, "wat": wat_df},
        "location": None,
        "origin_step": None,
        "affected_param": _CONFOUNDING_PARAM,
        "signature": None,
        "causal_chain": ["rc_up", "speed_fail_up"],
        "confounder": _CONFOUNDING_CONFOUNDER,
    }


def inject_clean(
    tables: dict[str, pd.DataFrame],
    affected_lot: str | None,
    rng: np.random.Generator,
    noise_level: float,
) -> dict[str, Any]:
    """No excursion — false-positive control (tables unchanged)."""
    _ = (affected_lot, rng, noise_level)
    return {
        "tables": tables,
        "location": None,
        "origin_step": None,
        "affected_param": None,
        "signature": None,
        "causal_chain": None,
        "confounder": None,
    }


INJECTORS = {
    "edge_signature": inject_edge_signature,
    "chamber_specific": inject_chamber_specific,
    "propagation": inject_propagation,
    "clean": inject_clean,
    "mean_shift": inject_mean_shift,
    "early_detection": inject_early_detection,
    "correlation_break": inject_correlation_break,
    "confounding": inject_confounding,
}
