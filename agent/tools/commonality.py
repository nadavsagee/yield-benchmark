"""commonality — chamber/tool grouping within a lot."""

from __future__ import annotations

import pandas as pd

from agent.tools._baseline import (
    COMMONALITY_STEP,
    robust_shift,
    wafer_sort_metric,
)


def commonality(
    lot: str,
    tables: dict[str, pd.DataFrame],
    population: str = "yield",
) -> dict:
    """
    Find chamber commonality at the standard inline step for a lot.

    population: 'yield', 'fail', or a soft_bin name.
    Returns {commons_to, type, yield_gap, strength}.
    """
    route_df = tables["route"]
    sort_df = tables["sort"]

    lot_route = route_df[
        (route_df["lot"] == lot) & (route_df["step"] == COMMONALITY_STEP)
    ]
    if lot_route.empty:
        return {
            "commons_to": None,
            "type": None,
            "yield_gap": 0.0,
            "strength": 0.0,
        }

    wafer_metric = wafer_sort_metric(sort_df, lot, population)
    chamber_means: dict[str, float] = {}
    for chamber, group in lot_route.groupby("chamber"):
        wafers = group["wafer"].unique()
        vals = wafer_metric.reindex(wafers).dropna()
        if len(vals):
            chamber_means[str(chamber)] = float(vals.mean())

    if len(chamber_means) < 2:
        top = next(iter(chamber_means), None)
        return {
            "commons_to": top,
            "type": "chamber",
            "yield_gap": 0.0,
            "strength": 0.0,
        }

    lot_mean = float(wafer_metric.mean())
    commons_to = min(chamber_means, key=chamber_means.get)
    best_mean = chamber_means[commons_to]
    others = [v for c, v in chamber_means.items() if c != commons_to]
    other_mean = float(sum(others) / len(others))
    yield_gap = round(other_mean - best_mean, 4)

    pop_gaps = _population_chamber_gaps(route_df, sort_df, population)
    strength = robust_shift(yield_gap, pop_gaps, lot)

    return {
        "commons_to": commons_to,
        "type": "chamber",
        "yield_gap": yield_gap,
        "strength": round(strength, 3),
    }


def _population_chamber_gaps(
    route_df: pd.DataFrame,
    sort_df: pd.DataFrame,
    population: str,
) -> pd.Series:
    """Max chamber yield gap per lot (population reference for strength)."""
    gaps: dict[str, float] = {}
    for lot in route_df["lot"].unique():
        lot_route = route_df[
            (route_df["lot"] == lot) & (route_df["step"] == COMMONALITY_STEP)
        ]
        wafer_metric = wafer_sort_metric(sort_df, lot, population)
        chamber_means: dict[str, float] = {}
        for chamber, group in lot_route.groupby("chamber"):
            wafers = group["wafer"].unique()
            vals = wafer_metric.reindex(wafers).dropna()
            if len(vals):
                chamber_means[str(chamber)] = float(vals.mean())
        if len(chamber_means) >= 2:
            best = min(chamber_means.values())
            others = [v for v in chamber_means.values() if v != best or len(chamber_means) > 2]
            if len(chamber_means) == 2:
                others = [max(chamber_means.values())]
            other_mean = float(sum(others) / len(others))
            gaps[lot] = other_mean - best
        else:
            gaps[lot] = 0.0
    return pd.Series(gaps)
