"""chain_correlate — inline -> WAT -> Sort correlation chain for a lot."""

from __future__ import annotations

import pandas as pd

from schema import INLINE_STEPS, WAT_PARAMS

from agent.tools._baseline import (
    DEFAULT_SIGMA_THRESHOLD,
    lot_sort_metric,
    robust_shift,
)

LinkStatus = str  # "elevated" | "normal" | "misaligned"


def chain_correlate(
    lot: str,
    tables: dict[str, pd.DataFrame],
) -> dict:
    """
    Test inline defect -> WAT param -> sort fail chain for a lot.

    Returns {chain_intact, links, break_at}.
    """
    inline_df = tables["inline"]
    wat_df = tables["wat"]
    sort_df = tables["sort"]

    inline_shifts = _inline_defect_shifts(inline_df, lot)
    origin_step = max(inline_shifts, key=inline_shifts.get)
    defect_sigma = inline_shifts[origin_step]

    wat_shifts = _wat_param_shifts(wat_df, lot)
    top_wat_param = max(wat_shifts, key=lambda p: abs(wat_shifts[p]))
    wat_sigma = wat_shifts[top_wat_param]

    speed_sigma = robust_shift(
        float(lot_sort_metric(sort_df, "speed")[lot]),
        lot_sort_metric(sort_df, "speed"),
        lot,
    )

    inline_status = _shift_status(defect_sigma, direction="up")
    wat_status = _wat_link_status(top_wat_param, wat_sigma, inline_status)
    sort_status = _shift_status(speed_sigma, direction="up")

    links = [
        {
            "link": "inline_defect",
            "step": origin_step,
            "shift_sigma": round(defect_sigma, 3),
            "status": inline_status,
        },
        {
            "link": "wat_param",
            "param": top_wat_param,
            "shift_sigma": round(wat_sigma, 3),
            "status": wat_status,
        },
        {
            "link": "sort_speed",
            "population": "speed",
            "shift_sigma": round(speed_sigma, 3),
            "status": sort_status,
        },
    ]

    break_at = _find_break(links, inline_status, wat_status, sort_status, origin_step, top_wat_param)
    chain_intact = _chain_intact(inline_status, wat_status, sort_status, break_at)

    return {
        "chain_intact": chain_intact,
        "links": links,
        "break_at": break_at,
    }


def _shift_status(sigma: float, direction: str = "up") -> LinkStatus:
    if direction == "up":
        if sigma >= 2.0:
            return "elevated"
        if sigma <= -DEFAULT_SIGMA_THRESHOLD:
            return "misaligned"
    return "normal"


def _wat_link_status(param: str, wat_sigma: float, inline_status: LinkStatus) -> LinkStatus:
    if inline_status != "elevated":
        if abs(wat_sigma) >= DEFAULT_SIGMA_THRESHOLD:
            return "misaligned"
        return "normal"
    if param in ("rc_ohm", "rs_ohm"):
        return "elevated" if wat_sigma >= 2.0 else ("misaligned" if wat_sigma < -2.0 else "normal")
    if param == "idsat_uA":
        return "elevated" if wat_sigma <= -2.0 else ("misaligned" if wat_sigma >= 2.0 else "normal")
    return _shift_status(wat_sigma, direction="up")


def _find_break(
    links: list[dict],
    inline_status: LinkStatus,
    wat_status: LinkStatus,
    sort_status: LinkStatus,
    origin_step: str,
    wat_param: str,
) -> str | None:
    """Flag a break only when an elevated upstream link is not propagated downstream."""
    if inline_status == "elevated" and wat_status != "elevated":
        return f"wat:{wat_param}"
    if inline_status == "elevated" and wat_status == "elevated" and sort_status != "elevated":
        return "speed"
    if inline_status == "misaligned":
        return origin_step
    if wat_status == "misaligned":
        return f"wat:{wat_param}"
    if sort_status == "misaligned":
        return "speed"
    # Elevated sort without inline signal → correlation break (not flagged as break_at for clean)
    if sort_status == "elevated" and inline_status == "normal":
        return "speed"
    return None


def _chain_intact(
    inline_status: LinkStatus,
    wat_status: LinkStatus,
    sort_status: LinkStatus,
    break_at: str | None,
) -> bool:
    if break_at is not None:
        return False
    if inline_status == "elevated" and wat_status == "elevated" and sort_status == "elevated":
        return True
    if inline_status == "normal" and wat_status == "normal" and sort_status == "normal":
        return True
    return False


def _inline_defect_shifts(inline_df: pd.DataFrame, lot: str) -> dict[str, float]:
    shifts: dict[str, float] = {}
    for step in INLINE_STEPS:
        step_means = (
            inline_df[inline_df["step"] == step]
            .groupby("lot")["defect_density"]
            .mean()
        )
        if lot in step_means.index:
            shifts[step] = robust_shift(float(step_means[lot]), step_means, lot)
        else:
            shifts[step] = 0.0
    return shifts


def _wat_param_shifts(wat_df: pd.DataFrame, lot: str) -> dict[str, float]:
    shifts: dict[str, float] = {}
    for param in WAT_PARAMS:
        lot_means = wat_df.groupby("lot")[param].mean()
        if lot in lot_means.index:
            shifts[param] = robust_shift(float(lot_means[lot]), lot_means, lot)
    return shifts
