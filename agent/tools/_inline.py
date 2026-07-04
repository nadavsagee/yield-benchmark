"""Shared inline lot ordering, level shifts, and trailing-window trend logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from schema import INLINE_METRICS, INLINE_STEPS

from agent.tools._baseline import robust_shift

# Match agent.prestep trailing-window size for trend features.
TREND_WINDOW = 4


def lot_order(lot: str) -> int:
    return int(lot.split("_")[1])


def ordered_lots(sort_df: pd.DataFrame) -> list[str]:
    return sorted(sort_df["lot"].unique(), key=lot_order)


def inline_lot_means(
    inline_df: pd.DataFrame,
    lots: list[str],
) -> dict[tuple[str, str, str], float]:
    means: dict[tuple[str, str, str], float] = {}
    for lot in lots:
        lot_df = inline_df[inline_df["lot"] == lot]
        for step in INLINE_STEPS:
            step_df = lot_df[lot_df["step"] == step]
            for metric in INLINE_METRICS:
                means[(lot, step, metric)] = (
                    float(step_df[metric].mean()) if len(step_df) else 0.0
                )
    return means


def inline_step_metric_series(
    inline_df: pd.DataFrame,
    step: str,
    metric: str,
) -> pd.Series:
    step_df = inline_df[inline_df["step"] == step]
    return step_df.groupby("lot")[metric].mean()


def inline_level_sigma(
    lot: str,
    step: str,
    metric: str,
    inline_df: pd.DataFrame,
) -> float:
    series = inline_step_metric_series(inline_df, step, metric)
    if lot not in series.index:
        return 0.0
    return robust_shift(float(series[lot]), series, lot)


def inline_trend(
    lots: list[str],
    lot_index: int,
    step: str,
    metric: str,
    inline_means: dict[tuple[str, str, str], float],
    *,
    window: int = TREND_WINDOW,
) -> dict[str, float | int | bool]:
    """
    Trailing-window slope and sustained drift for one inline step/metric.

    Same logic as agent.prestep._inline_trend_features per pair.
    """
    start = max(0, lot_index - window + 1)
    window_lots = lots[start : lot_index + 1]

    if len(window_lots) < 3:
        return {
            "trend_slope": 0.0,
            "sustained": 0.0,
            "monotonic": False,
            "window_size": len(window_lots),
        }

    y = np.array(
        [inline_means[(lot, step, metric)] for lot in window_lots],
        dtype=float,
    )
    x = np.arange(len(window_lots), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])

    diffs = np.diff(y)
    if len(diffs) == 0:
        mono = 0.0
    elif slope >= 0:
        mono = float(np.mean(diffs >= 0))
    else:
        mono = float(np.mean(diffs <= 0))

    sustained = abs(slope) * mono * len(window_lots)
    return {
        "trend_slope": slope,
        "sustained": sustained,
        "monotonic": mono >= 0.75,
        "window_size": len(window_lots),
    }
