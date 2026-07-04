#!/usr/bin/env python3
"""
Phase 1 verification — generate one dataset per anomaly type and validate structure + signal.

Usage:
    python verify_generation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from schema import (
    ANOMALY_TYPES,
    BASELINE,
    DIE_PER_WAFER,
    GROUND_TRUTH_CONTRACT,
    INLINE_COLUMNS,
    INLINE_STEPS,
    LOTS_PER_DATASET,
    ROUTE_COLUMNS,
    SORT_COLUMNS,
    WAFERS_PER_LOT,
    WAT_COLUMNS,
    WAT_PARAMS,
    WAT_SITES,
)

from generator.benchmark import generate_benchmark
from generator.injectors import (
    EARLY_DETECTION_SEQUENCE_LEN,
    _CHAMBER_STEP,
    _CONFOUNDING_CONFOUNDER,
    _CONFOUNDING_PARAM,
    _CORRELATION_BREAK_SOFT_BIN,
    _EARLY_DETECTION_METRIC,
    _EARLY_DETECTION_ORIGIN,
    _EDGE_PARAM,
    _EDGE_RADIUS,
    _MEAN_SHIFT_PARAM,
    _MEAN_SHIFT_SPEC_SIGMA,
    _PROPAGATION_ORIGIN,
    _PROPAGATION_PARAM,
    _PROPAGATION_SOFT_BIN,
)
from generator.normal import commonality_step

DATASETS_DIR = Path("datasets")
VERIFY_DIR = DATASETS_DIR / "_verify"


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    raise AssertionError(msg)


def _ok(msg: str) -> None:
    print(f"  OK: {msg}")


def check_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    if list(df.columns) != expected:
        _fail(f"{name} columns mismatch: got {list(df.columns)}, expected {expected}")
    _ok(f"{name} columns match schema")


def check_row_counts(sort_df, wat_df, inline_df, route_df) -> None:
    n_sort = LOTS_PER_DATASET * WAFERS_PER_LOT * DIE_PER_WAFER
    n_wat = LOTS_PER_DATASET * WAFERS_PER_LOT * WAT_SITES
    n_inline = LOTS_PER_DATASET * WAFERS_PER_LOT * len(INLINE_STEPS)
    n_route = n_inline

    for name, df, expected in [
        ("sort", sort_df, n_sort),
        ("wat", wat_df, n_wat),
        ("inline", inline_df, n_inline),
        ("route", route_df, n_route),
    ]:
        if len(df) != expected:
            _fail(f"{name} row count {len(df)} != {expected}")
    _ok(f"row counts correct ({n_sort:,} sort / {n_wat:,} wat / {n_inline:,} inline+route)")


def check_ground_truth(gt: dict, anomaly_type: str) -> None:
    for key in GROUND_TRUTH_CONTRACT:
        if key not in gt:
            _fail(f"ground_truth missing key {key!r}")

    if gt["type"] != anomaly_type:
        _fail(f"type {gt['type']!r} != expected {anomaly_type!r}")

    if anomaly_type == "clean":
        if gt["excursion"] is not False:
            _fail("clean dataset must have excursion=False")
        if gt["affected_lots"]:
            _fail("clean dataset must have empty affected_lots")
    elif anomaly_type == "early_detection":
        if gt["excursion"] is not True:
            _fail("early_detection must have excursion=True")
        if len(gt["affected_lots"]) != EARLY_DETECTION_SEQUENCE_LEN:
            _fail(
                f"early_detection expects {EARLY_DETECTION_SEQUENCE_LEN} affected lots, "
                f"got {gt['affected_lots']}"
            )
        if gt["origin_step"] != _EARLY_DETECTION_ORIGIN:
            _fail(f"expected origin_step {_EARLY_DETECTION_ORIGIN!r}")
        if gt["confounder"] is not None:
            _fail("early_detection confounder must be null")
    elif anomaly_type == "confounding":
        if gt["excursion"] is not True:
            _fail("confounding must have excursion=True")
        if len(gt["affected_lots"]) != 1:
            _fail(f"confounding expects 1 affected lot, got {gt['affected_lots']}")
        if gt["confounder"] != _CONFOUNDING_CONFOUNDER:
            _fail(f"expected confounder {_CONFOUNDING_CONFOUNDER!r}, got {gt['confounder']!r}")
        if gt["affected_param"] != _CONFOUNDING_PARAM:
            _fail(f"expected affected_param {_CONFOUNDING_PARAM!r}")
    else:
        if gt["excursion"] is not True:
            _fail(f"{anomaly_type} must have excursion=True")
        if len(gt["affected_lots"]) != 1:
            _fail(f"{anomaly_type} expects exactly 1 affected lot, got {gt['affected_lots']}")

    _ok("ground_truth contract fields present and consistent")


def _baseline_lots(all_lots: list[str], affected: list[str]) -> list[str]:
    return [lot for lot in all_lots if lot not in affected]


def verify_edge_signature(
    sort_df: pd.DataFrame, wat_df: pd.DataFrame, gt: dict
) -> None:
    lot = gt["affected_lots"][0]
    baseline = _baseline_lots(sorted(sort_df["lot"].unique()), gt["affected_lots"])

    aff_edge = sort_df[(sort_df["lot"] == lot) & (sort_df["radius"] >= _EDGE_RADIUS)]
    base_edge = sort_df[
        (sort_df["lot"].isin(baseline)) & (sort_df["radius"] >= _EDGE_RADIUS)
    ]
    aff_fail = 1.0 - aff_edge["pass"].mean()
    base_fail = 1.0 - base_edge["pass"].mean()
    if aff_fail <= base_fail + 0.15:
        _fail(
            f"edge sort fail rate not elevated enough "
            f"(affected={aff_fail:.3f}, baseline={base_fail:.3f})"
        )

    aff_wat = wat_df[(wat_df["lot"] == lot) & (wat_df["radius"] >= _EDGE_RADIUS)][
        _EDGE_PARAM
    ].mean()
    base_wat = wat_df[
        (wat_df["lot"].isin(baseline)) & (wat_df["radius"] >= _EDGE_RADIUS)
    ][_EDGE_PARAM].mean()
    if aff_wat <= base_wat + 5:
        _fail(
            f"edge WAT {_EDGE_PARAM} not elevated "
            f"(affected={aff_wat:.1f}, baseline={base_wat:.1f})"
        )

    if gt["signature"] != "edge_ring":
        _fail(f"expected signature edge_ring, got {gt['signature']!r}")
    if gt["affected_param"] != _EDGE_PARAM:
        _fail(f"expected affected_param {_EDGE_PARAM}, got {gt['affected_param']!r}")

    _ok(
        f"edge_signature signal: edge fail {aff_fail:.1%} vs {base_fail:.1%}, "
        f"{_EDGE_PARAM} +{aff_wat - base_wat:.1f} mV-equiv at edge"
    )


def verify_chamber_specific(
    sort_df: pd.DataFrame,
    wat_df: pd.DataFrame,
    inline_df: pd.DataFrame,
    route_df: pd.DataFrame,
    gt: dict,
) -> None:
    lot = gt["affected_lots"][0]
    chamber = gt["location"]
    if chamber is None:
        _fail("chamber_specific must set location to affected chamber")

    step = commonality_step()
    aff_wafers = route_df[
        (route_df["lot"] == lot)
        & (route_df["step"] == step)
        & (route_df["chamber"] == chamber)
    ]["wafer"].unique()
    other_wafers = route_df[
        (route_df["lot"] == lot)
        & (route_df["step"] == step)
        & (route_df["chamber"] != chamber)
    ]["wafer"].unique()

    if len(aff_wafers) == 0 or len(other_wafers) == 0:
        _fail("need both affected-chamber and other-chamber wafers in the lot")

    param = gt["affected_param"]
    aff_val = wat_df[
        (wat_df["lot"] == lot) & (wat_df["wafer"].isin(aff_wafers))
    ][param].mean()
    other_val = wat_df[
        (wat_df["lot"] == lot) & (wat_df["wafer"].isin(other_wafers))
    ][param].mean()
    if abs(aff_val - other_val) < 10:
        _fail(
            f"chamber wafers not separated on {param} "
            f"(affected={aff_val:.1f}, other={other_val:.1f})"
        )

    if gt["origin_step"] != _CHAMBER_STEP:
        _fail(f"expected origin_step {_CHAMBER_STEP}, got {gt['origin_step']!r}")

    _ok(
        f"chamber_specific signal: {chamber} wafers {param} "
        f"{aff_val:.1f} vs others {other_val:.1f} at {step}"
    )


def verify_propagation(
    sort_df: pd.DataFrame, wat_df: pd.DataFrame, inline_df: pd.DataFrame, gt: dict
) -> None:
    lot = gt["affected_lots"][0]
    baseline = _baseline_lots(sorted(sort_df["lot"].unique()), gt["affected_lots"])

    aff_defect = inline_df[
        (inline_df["lot"] == lot) & (inline_df["step"] == _PROPAGATION_ORIGIN)
    ]["defect_density"].mean()
    base_defect = inline_df[
        (inline_df["lot"].isin(baseline))
        & (inline_df["step"] == _PROPAGATION_ORIGIN)
    ]["defect_density"].mean()
    if aff_defect <= base_defect + 0.05:
        _fail(
            f"defect_density not elevated at {_PROPAGATION_ORIGIN} "
            f"(affected={aff_defect:.3f}, baseline={base_defect:.3f})"
        )

    aff_rc = wat_df[(wat_df["lot"] == lot)][_PROPAGATION_PARAM].mean()
    base_rc = wat_df[(wat_df["lot"].isin(baseline))][_PROPAGATION_PARAM].mean()
    if aff_rc <= base_rc + 5:
        _fail(
            f"{_PROPAGATION_PARAM} not elevated "
            f"(affected={aff_rc:.1f}, baseline={base_rc:.1f})"
        )

    lot_sort = sort_df[sort_df["lot"] == lot]
    aff_speed_rate = (lot_sort["soft_bin"] == _PROPAGATION_SOFT_BIN).mean()
    base_speed_rate = (
        sort_df[sort_df["lot"].isin(baseline)]["soft_bin"] == _PROPAGATION_SOFT_BIN
    ).mean()
    if aff_speed_rate <= base_speed_rate + 0.05:
        _fail(
            f"{_PROPAGATION_SOFT_BIN} bin rate not elevated "
            f"(affected={aff_speed_rate:.3f}, baseline={base_speed_rate:.3f})"
        )

    if gt["causal_chain"] != ["defect_up", "rc_up", "speed_fail_up"]:
        _fail(f"unexpected causal_chain: {gt['causal_chain']!r}")
    if gt["origin_step"] != _PROPAGATION_ORIGIN:
        _fail(f"expected origin_step {_PROPAGATION_ORIGIN}")

    _ok(
        f"propagation chain: defect +{aff_defect - base_defect:.3f}, "
        f"rc +{aff_rc - base_rc:.1f}, {_PROPAGATION_SOFT_BIN} rate "
        f"{aff_speed_rate:.1%} vs {base_speed_rate:.1%}"
    )


def verify_clean(
    sort_df: pd.DataFrame, wat_df: pd.DataFrame, inline_df: pd.DataFrame, gt: dict
) -> None:
    lots = sorted(sort_df["lot"].unique())
    # No single lot should look like a strong excursion on yield
    lot_yields = sort_df.groupby("lot")["pass"].mean()
    global_mean = lot_yields.mean()
    global_std = lot_yields.std()
    outliers = lot_yields[
        (lot_yields < global_mean - 2.5 * global_std)
        | (lot_yields > global_mean + 2.5 * global_std)
    ]
    if len(outliers) > 2:
        _fail(f"clean dataset has {len(outliers)} lot yield outliers (>2 expected by chance)")

    _ok(
        f"clean control: no injected excursion, "
        f"yield mean={global_mean:.1%} std={global_std:.3f} across {len(lots)} lots"
    )


def verify_mean_shift(wat_df: pd.DataFrame, gt: dict) -> None:
    lot = gt["affected_lots"][0]
    param = gt["affected_param"]
    spec_limit = (
        BASELINE[param]["mean"] + _MEAN_SHIFT_SPEC_SIGMA * BASELINE[param]["sigma"]
    )
    lot_mean = float(wat_df.loc[wat_df["lot"] == lot, param].mean())
    if lot_mean <= spec_limit:
        _fail(f"{param} mean {lot_mean:.1f} did not cross spec {spec_limit:.1f}")
    if gt["affected_param"] != _MEAN_SHIFT_PARAM:
        _fail(f"expected affected_param {_MEAN_SHIFT_PARAM!r}")
    _ok(f"mean_shift: {param} lot mean {lot_mean:.1f} > spec {spec_limit:.1f}")


def verify_early_detection(
    sort_df: pd.DataFrame, wat_df: pd.DataFrame, inline_df: pd.DataFrame, gt: dict
) -> None:
    lots = gt["affected_lots"]
    baseline = _baseline_lots(sorted(sort_df["lot"].unique()), lots)

    drift_vals = [
        float(
            inline_df[
                (inline_df["lot"] == lot) & (inline_df["step"] == _EARLY_DETECTION_ORIGIN)
            ][_EARLY_DETECTION_METRIC].mean()
        )
        for lot in lots
    ]
    if not all(drift_vals[i] < drift_vals[i + 1] for i in range(len(drift_vals) - 1)):
        _fail(f"inline drift not monotonic across sequence: {drift_vals}")

    last_lot = lots[-1]
    for param in WAT_PARAMS:
        aff = float(wat_df.loc[wat_df["lot"] == last_lot, param].mean())
        pop = wat_df.loc[wat_df["lot"].isin(baseline), param].mean()
        sigma = BASELINE[param]["sigma"] * 2.5
        if abs(aff - pop) > sigma:
            _fail(
                f"early_detection last lot WAT {param} not in-family "
                f"(aff={aff:.2f}, pop={pop:.2f})"
            )

    last_fail = 1.0 - float(sort_df.loc[sort_df["lot"] == last_lot, "pass"].mean())
    base_fail = 1.0 - float(sort_df.loc[sort_df["lot"].isin(baseline), "pass"].mean())
    if last_fail > base_fail + 0.08:
        _fail(
            f"early_detection last lot sort fail rate too high "
            f"({last_fail:.3f} vs baseline {base_fail:.3f})"
        )

    _ok(
        f"early_detection: {_EARLY_DETECTION_METRIC} drift {drift_vals[0]:.2f}"
        f"->{drift_vals[-1]:.2f} over {len(lots)} lots; "
        f"WAT/Sort normal on {last_lot}"
    )


def verify_correlation_break(
    sort_df: pd.DataFrame, wat_df: pd.DataFrame, gt: dict
) -> None:
    lot = gt["affected_lots"][0]
    baseline = _baseline_lots(sorted(sort_df["lot"].unique()), [lot])

    aff_fail = 1.0 - float(sort_df.loc[sort_df["lot"] == lot, "pass"].mean())
    base_fail = 1.0 - float(sort_df.loc[sort_df["lot"].isin(baseline), "pass"].mean())
    if aff_fail <= base_fail + 0.15:
        _fail(f"sort fail rate not elevated (affected={aff_fail:.3f}, base={base_fail:.3f})")

    gross_rate = float(
        (sort_df.loc[sort_df["lot"] == lot, "soft_bin"] == _CORRELATION_BREAK_SOFT_BIN).mean()
    )
    if gross_rate < 0.10:
        _fail(f"gross bin rate too low on affected lot ({gross_rate:.3f})")

    for param in WAT_PARAMS:
        aff = float(wat_df.loc[wat_df["lot"] == lot, param].mean())
        pop_med = float(wat_df.loc[wat_df["lot"].isin(baseline), param].mean())
        tol = BASELINE[param]["sigma"] * 2.5
        if abs(aff - pop_med) > tol:
            _fail(
                f"WAT {param} not in-family on correlation_break lot "
                f"(aff={aff:.2f}, pop={pop_med:.2f}, tol={tol:.2f})"
            )

    if gt["causal_chain"] != ["wat_normal", "sort_fail_up"]:
        _fail(f"unexpected causal_chain: {gt['causal_chain']!r}")

    _ok(
        f"correlation_break: sort fail {aff_fail:.1%} vs {base_fail:.1%}, "
        f"WAT in-family, gross rate {gross_rate:.1%}"
    )


def verify_confounding(sort_df: pd.DataFrame, wat_df: pd.DataFrame, gt: dict) -> None:
    lot = gt["affected_lots"][0]
    baseline = _baseline_lots(sorted(sort_df["lot"].unique()), [lot])

    causal = float(wat_df.loc[wat_df["lot"] == lot, _CONFOUNDING_PARAM].mean())
    conf = float(wat_df.loc[wat_df["lot"] == lot, _CONFOUNDING_CONFOUNDER].mean())
    base_causal = float(wat_df.loc[wat_df["lot"].isin(baseline), _CONFOUNDING_PARAM].mean())
    base_conf = float(wat_df.loc[wat_df["lot"].isin(baseline), _CONFOUNDING_CONFOUNDER].mean())

    if causal <= base_causal + 5:
        _fail(f"causal param {_CONFOUNDING_PARAM} not elevated enough")
    if conf <= base_conf + 5:
        _fail(f"confounder {_CONFOUNDING_CONFOUNDER} not elevated enough")

    aff_fail = 1.0 - float(sort_df.loc[sort_df["lot"] == lot, "pass"].mean())
    base_fail = 1.0 - float(sort_df.loc[sort_df["lot"].isin(baseline), "pass"].mean())
    if aff_fail <= base_fail + 0.10:
        _fail(f"sort fail rate not elevated on confounding lot")

    _ok(
        f"confounding: {_CONFOUNDING_PARAM} +{causal - base_causal:.1f}, "
        f"{_CONFOUNDING_CONFOUNDER} +{conf - base_conf:.1f}, "
        f"sort fail {aff_fail:.1%} vs {base_fail:.1%}"
    )


def verify_dataset(anomaly_type: str, seed: int) -> None:
    dataset_id = f"verify_{anomaly_type}"
    out_dir = VERIFY_DIR / dataset_id
    print(f"\n=== {anomaly_type} (seed={seed}) ===")

    gt = generate_benchmark(
        dataset_id=dataset_id,
        anomaly_type=anomaly_type,
        difficulty="medium",
        seed=seed,
        output_dir=out_dir,
    )

    sort_df = pd.read_csv(out_dir / "sort.csv")
    wat_df = pd.read_csv(out_dir / "wat.csv")
    inline_df = pd.read_csv(out_dir / "inline.csv")
    route_df = pd.read_csv(out_dir / "route.csv")
    with open(out_dir / "ground_truth.json", encoding="utf-8") as f:
        gt_disk = json.load(f)

    if gt != gt_disk:
        _fail("ground_truth.json on disk differs from returned dict")

    check_columns(sort_df, SORT_COLUMNS, "sort")
    check_columns(wat_df, WAT_COLUMNS, "wat")
    check_columns(inline_df, INLINE_COLUMNS, "inline")
    check_columns(route_df, ROUTE_COLUMNS, "route")
    check_row_counts(sort_df, wat_df, inline_df, route_df)
    check_ground_truth(gt, anomaly_type)

    if anomaly_type == "edge_signature":
        verify_edge_signature(sort_df, wat_df, gt)
    elif anomaly_type == "chamber_specific":
        verify_chamber_specific(sort_df, wat_df, inline_df, route_df, gt)
    elif anomaly_type == "propagation":
        verify_propagation(sort_df, wat_df, inline_df, gt)
    elif anomaly_type == "clean":
        verify_clean(sort_df, wat_df, inline_df, gt)
    elif anomaly_type == "mean_shift":
        verify_mean_shift(wat_df, gt)
    elif anomaly_type == "early_detection":
        verify_early_detection(sort_df, wat_df, inline_df, gt)
    elif anomaly_type == "correlation_break":
        verify_correlation_break(sort_df, wat_df, gt)
    elif anomaly_type == "confounding":
        verify_confounding(sort_df, wat_df, gt)
    else:
        _fail(f"no verifier for anomaly type {anomaly_type!r}")


def main() -> int:
    print("yield-benchmark Phase 1 verification")
    print(f"Output: {VERIFY_DIR.resolve()}")

    seeds = {
        "edge_signature": 42,
        "chamber_specific": 101,
        "propagation": 202,
        "clean": 303,
        "mean_shift": 404,
        "early_detection": 505,
        "correlation_break": 606,
        "confounding": 707,
    }

    for anomaly_type in ANOMALY_TYPES:
        verify_dataset(anomaly_type, seeds[anomaly_type])

    print("\n=== ALL CHECKS PASSED ===")
    print(f"Generated {len(ANOMALY_TYPES)} datasets under {VERIFY_DIR}/")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\nVerification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
