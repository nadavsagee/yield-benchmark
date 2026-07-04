"""wat_profile — WAT parametric profile for a lot vs population."""

from __future__ import annotations

import pandas as pd

from schema import WAT_PARAMS

from agent.tools._baseline import (
    CENTER_RADIUS,
    EDGE_RADIUS,
    robust_center_spread,
    robust_shift,
)


def wat_profile(
    lot: str,
    param: str,
    tables: dict[str, pd.DataFrame],
) -> dict:
    """
    WAT parametric summary for a lot vs leave-one-out population.

    Returns {sigma_shift, radial, bimodal, mean, baseline_mean}.
    """
    if param not in WAT_PARAMS:
        raise ValueError(f"param must be one of {WAT_PARAMS}, got {param!r}")

    wat_df = tables["wat"]
    lot_means = wat_df.groupby("lot")[param].mean()
    mean = float(lot_means[lot])
    baseline_mean = float(lot_means.drop(index=lot, errors="ignore").median())
    sigma_shift = robust_shift(mean, lot_means, lot)

    lot_df = wat_df[wat_df["lot"] == lot]
    edge_vals = lot_df[lot_df["radius"] >= EDGE_RADIUS][param]
    center_vals = lot_df[lot_df["radius"] <= CENTER_RADIUS][param]
    radial = float(edge_vals.mean() - center_vals.mean()) if len(edge_vals) and len(center_vals) else 0.0

    wafer_means = lot_df.groupby("wafer")[param].mean()
    bimodal = _detect_bimodal(wafer_means, wat_df, param, lot)

    return {
        "sigma_shift": round(sigma_shift, 3),
        "radial": round(radial, 3),
        "bimodal": bimodal,
        "mean": round(mean, 3),
        "baseline_mean": round(baseline_mean, 3),
    }


def _detect_bimodal(
    wafer_means: pd.Series,
    wat_df: pd.DataFrame,
    param: str,
    lot: str,
) -> bool:
    """Simple bimodality: lot wafer spread vs population wafer-spread baseline."""
    if len(wafer_means) < 4:
        return False

    lot_spread = float(wafer_means.std(ddof=0))
    pop_spreads = []
    for other in wat_df["lot"].unique():
        if other == lot:
            continue
        wm = wat_df[wat_df["lot"] == other].groupby("wafer")[param].mean()
        if len(wm) >= 4:
            pop_spreads.append(float(wm.std(ddof=0)))

    if not pop_spreads:
        return False

    med, mad = robust_center_spread(pop_spreads)
    threshold = med + 3 * (1.4826 * mad if mad > 1e-12 else 1.0)
    return lot_spread > threshold and lot_spread > med * 1.8
