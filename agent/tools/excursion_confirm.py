"""excursion_confirm — SPC-style out-of-control check for a lot metric."""

from __future__ import annotations

import numpy as np
import pandas as pd

from schema import WAT_PARAMS

from agent.tools._baseline import (
    DEFAULT_SIGMA_THRESHOLD,
    lot_sort_metric,
    robust_center_spread,
    robust_control_limits,
    robust_shift,
    robust_sigma,
    wafer_sort_metric,
)


def excursion_confirm(
    lot: str,
    param: str,
    tables: dict[str, pd.DataFrame],
    population: str = "yield",
) -> dict:
    """
    Confirm whether a lot is out of control vs population baseline.

    param: WAT param name, or 'yield'/'fail'/'speed'/soft_bin for sort population.
    population: used when param is a sort metric alias (default sort view).
    Returns {out_of_control, sigma, ucl, lcl, pct_out}.
    """
    lot_values, pop_series = _resolve_series(lot, param, population, tables)
    lot_mean = float(lot_values.mean()) if len(lot_values) else 0.0

    ucl, lcl, _, _ = robust_control_limits(pop_series, lot)
    sigma = robust_shift(lot_mean, pop_series, lot)

    pct_out = _pct_out_vs_wafer_limits(lot_values, lot, param, population, tables)

    out_of_control = bool(
        abs(sigma) >= DEFAULT_SIGMA_THRESHOLD or lot_mean > ucl or lot_mean < lcl
    )

    return {
        "out_of_control": out_of_control,
        "sigma": round(sigma, 3),
        "ucl": round(ucl, 3),
        "lcl": round(lcl, 3),
        "pct_out": round(pct_out, 4),
    }


def _sort_aliases() -> set[str]:
    from schema import SOFT_BINS
    return set(SOFT_BINS) | {"yield", "fail"}


def _resolve_series(
    lot: str,
    param: str,
    population: str,
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.Series, pd.Series]:
    wat_df = tables["wat"]
    sort_df = tables["sort"]

    sort_key = param if param in _sort_aliases() else population

    if param in WAT_PARAMS:
        pop = wat_df.groupby("lot")[param].mean()
        lot_vals = wat_df[wat_df["lot"] == lot].groupby("wafer")[param].mean()
        return lot_vals, pop

    pop = lot_sort_metric(sort_df, sort_key)
    lot_df = sort_df[sort_df["lot"] == lot]
    if sort_key == "yield":
        lot_vals = lot_df.groupby("wafer")["pass"].mean()
    elif sort_key == "fail":
        lot_vals = 1.0 - lot_df.groupby("wafer")["pass"].mean()
    else:
        lot_vals = lot_df.groupby("wafer").apply(
            lambda g: float((g["soft_bin"] == sort_key).mean()),
            include_groups=False,
        )
    return lot_vals, pop


def _wafer_population_values(
    lot: str,
    param: str,
    population: str,
    tables: dict[str, pd.DataFrame],
) -> np.ndarray:
    """Wafer-level reference sample: all wafers from lots other than `lot`."""
    wat_df = tables["wat"]
    sort_df = tables["sort"]
    sort_key = param if param in _sort_aliases() else population

    if param in WAT_PARAMS:
        wafer_means = wat_df.groupby(["lot", "wafer"])[param].mean()
        other = wafer_means[wafer_means.index.get_level_values(0) != lot]
        return other.to_numpy(dtype=float)

    parts: list[pd.Series] = []
    for other_lot in sort_df["lot"].unique():
        if other_lot == lot:
            continue
        parts.append(wafer_sort_metric(sort_df, other_lot, sort_key))
    if not parts:
        return np.array([], dtype=float)
    return pd.concat(parts).to_numpy(dtype=float)


def _limits_from_values(values: np.ndarray, k: float = DEFAULT_SIGMA_THRESHOLD) -> tuple[float, float]:
    if values.size == 0:
        return float("inf"), float("-inf")
    med, _ = robust_center_spread(values)
    sigma = robust_sigma(values)
    return med + k * sigma, med - k * sigma


def _pct_out_vs_wafer_limits(
    lot_values: pd.Series,
    lot: str,
    param: str,
    population: str,
    tables: dict[str, pd.DataFrame],
) -> float:
    """Share of this lot's wafers outside wafer-level UCL/LCL (same k as lot SPC)."""
    if len(lot_values) == 0:
        return 0.0
    ref = _wafer_population_values(lot, param, population, tables)
    wafer_ucl, wafer_lcl = _limits_from_values(ref)
    if len(lot_values) == 1:
        val = float(lot_values.iloc[0])
        return 1.0 if (val > wafer_ucl or val < wafer_lcl) else 0.0
    return float(((lot_values > wafer_ucl) | (lot_values < wafer_lcl)).mean())
