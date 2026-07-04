"""Shared robust population baselines — no ground-truth access."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826
DEFAULT_SIGMA_THRESHOLD = 3.0

# Spatial zones (match generator/injectors edge threshold)
EDGE_RADIUS = 0.65
CENTER_RADIUS = 0.35

# Step used for chamber commonality (matches generator.normal)
COMMONALITY_STEP = "gate_etch"


def robust_center_spread(values: Iterable[float]) -> tuple[float, float]:
    """Return (median, MAD) for a numeric sample."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad


def robust_sigma(values: Iterable[float]) -> float:
    """MAD scaled to a pseudo-standard-deviation."""
    _, mad = robust_center_spread(values)
    return MAD_SCALE * mad if mad > 1e-12 else 1.0


def leave_one_out_values(series: pd.Series, lot: str) -> np.ndarray:
    """Population lot-level values excluding the lot under test."""
    return series.drop(index=lot, errors="ignore").to_numpy()


def robust_shift(lot_value: float, population: pd.Series, lot: str) -> float:
    """Sigma shift of lot_value vs leave-one-out population (median + MAD)."""
    ref = leave_one_out_values(population, lot)
    if ref.size == 0:
        return 0.0
    med, _ = robust_center_spread(ref)
    sigma = robust_sigma(ref)
    return (lot_value - med) / sigma


def robust_control_limits(
    population: pd.Series,
    lot: str,
    k: float = 3.0,
) -> tuple[float, float, float, float]:
    """
    Leave-one-out robust UCL/LCL.

    Returns (ucl, lcl, center, sigma).
    """
    ref = leave_one_out_values(population, lot)
    if ref.size == 0:
        return float("inf"), float("-inf"), lot_value_if_missing(population, lot), 1.0
    med, _ = robust_center_spread(ref)
    sigma = robust_sigma(ref)
    return med + k * sigma, med - k * sigma, med, sigma


def lot_value_if_missing(population: pd.Series, lot: str) -> float:
    if lot in population.index:
        return float(population[lot])
    return 0.0


def lot_sort_metric(
    sort_df: pd.DataFrame,
    population: str,
) -> pd.Series:
    """
    Per-lot sort metric keyed by population name.

    population='yield' -> pass rate; 'fail' -> fail rate;
    otherwise soft_bin rate (e.g. 'speed').
    """
    if population == "yield":
        return sort_df.groupby("lot")["pass"].mean()
    if population == "fail":
        return 1.0 - sort_df.groupby("lot")["pass"].mean()

    def _rate(group: pd.DataFrame) -> float:
        return float((group["soft_bin"] == population).mean())

    return sort_df.groupby("lot").apply(_rate, include_groups=False)


def wafer_sort_metric(
    sort_df: pd.DataFrame,
    lot: str,
    population: str,
) -> pd.Series:
    """Per-wafer sort metric within a lot."""
    lot_df = sort_df[sort_df["lot"] == lot]
    if population == "yield":
        return lot_df.groupby("wafer")["pass"].mean()
    if population == "fail":
        return 1.0 - lot_df.groupby("wafer")["pass"].mean()

    def _rate(group: pd.DataFrame) -> float:
        return float((group["soft_bin"] == population).mean())

    return lot_df.groupby("wafer").apply(_rate, include_groups=False)


def zone_fail_rate(
    sort_df: pd.DataFrame,
    lot: str,
    zone: str,
    population: str,
) -> float:
    """Fail/bin rate within edge or center zone for one lot."""
    lot_df = sort_df[sort_df["lot"] == lot]
    if zone == "edge":
        lot_df = lot_df[lot_df["radius"] >= EDGE_RADIUS]
    elif zone == "center":
        lot_df = lot_df[lot_df["radius"] <= CENTER_RADIUS]
    if lot_df.empty:
        return 0.0

    if population == "fail":
        return float((~lot_df["pass"]).mean())
    if population == "yield":
        return float(lot_df["pass"].mean())
    return float((lot_df["soft_bin"] == population).mean())


def lot_zone_rates(
    sort_df: pd.DataFrame,
    zone: str,
    population: str,
) -> pd.Series:
    """Per-lot zone metric across all lots."""
    lots = sort_df["lot"].unique()
    return pd.Series(
        {lot: zone_fail_rate(sort_df, lot, zone, population) for lot in lots}
    )
