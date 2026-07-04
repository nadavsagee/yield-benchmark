"""excursion_confirm — SPC-style out-of-control check for a lot metric."""

from __future__ import annotations

import pandas as pd

from schema import WAT_PARAMS

from agent.tools._baseline import (
    DEFAULT_SIGMA_THRESHOLD,
    lot_sort_metric,
    robust_control_limits,
    robust_shift,
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

    if len(lot_values) > 1 and param in WAT_PARAMS:
        pct_out = float(((lot_values > ucl) | (lot_values < lcl)).mean())
    elif len(lot_values) > 1:
        pct_out = float(((lot_values > ucl) | (lot_values < lcl)).mean())
    else:
        pct_out = 1.0 if (lot_mean > ucl or lot_mean < lcl) else 0.0

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
