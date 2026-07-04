"""inline_trace — inline level + multi-lot trend for a lot across steps/metrics."""

from __future__ import annotations

import pandas as pd

from schema import INLINE_METRICS, INLINE_STEPS

from agent.tools._baseline import DEFAULT_SIGMA_THRESHOLD
from agent.tools._inline import (
    inline_level_sigma,
    inline_lot_means,
    inline_trend,
    ordered_lots,
)


def inline_trace(
    lot: str,
    tables: dict[str, pd.DataFrame],
) -> dict:
    """
    Inspect inline metrics at each step for one lot.

    Reports single-lot level vs population (median/MAD sigma) and trailing-window
    trend (slope + sustained drift). Returns the step/metric with the strongest
    combined signal.
    """
    inline_df = tables["inline"]
    lots = ordered_lots(tables["sort"])
    if lot not in lots:
        raise ValueError(f"unknown lot {lot!r}")

    lot_index = lots.index(lot)
    inline_means = inline_lot_means(inline_df, lots)

    best: dict | None = None
    best_strength = -1.0

    for step in INLINE_STEPS:
        for metric in INLINE_METRICS:
            level_sigma = inline_level_sigma(lot, step, metric, inline_df)
            trend = inline_trend(lots, lot_index, step, metric, inline_means)
            strength = max(abs(level_sigma), float(trend["sustained"]))
            if strength > best_strength:
                best_strength = strength
                best = {
                    "step": step,
                    "metric": metric,
                    "level_sigma": level_sigma,
                    "trend_slope": float(trend["trend_slope"]),
                    "sustained": float(trend["sustained"]),
                    "monotonic": bool(trend["monotonic"]),
                }

    assert best is not None
    out_of_control = abs(best["level_sigma"]) >= DEFAULT_SIGMA_THRESHOLD

    return {
        "step": best["step"],
        "metric": best["metric"],
        "level_sigma": round(best["level_sigma"], 3),
        "trend_slope": round(best["trend_slope"], 4),
        "sustained": round(best["sustained"], 4),
        "out_of_control": out_of_control,
    }
