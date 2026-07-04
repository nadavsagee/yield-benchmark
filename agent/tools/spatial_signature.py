"""spatial_signature — edge/center fail pattern for a lot vs population."""

from __future__ import annotations

from agent.tools._baseline import (
    lot_zone_rates,
    robust_shift,
    zone_fail_rate,
)


def spatial_signature(
    lot: str,
    tables: dict[str, pd.DataFrame],
    population: str = "speed",
) -> dict:
    """
    Compare spatial fail/bin rates for a lot against the population.

    population: 'fail', 'yield', or a soft_bin name (e.g. 'speed').
    Returns {signature, edge_fail_ratio, center_fail_ratio}.
    """
    sort_df = tables["sort"]

    edge_fail_ratio = zone_fail_rate(sort_df, lot, "edge", population)
    center_fail_ratio = zone_fail_rate(sort_df, lot, "center", population)

    pop_edge = lot_zone_rates(sort_df, "edge", population)
    pop_center = lot_zone_rates(sort_df, "center", population)

    edge_shift = robust_shift(edge_fail_ratio, pop_edge, lot)
    center_shift = robust_shift(center_fail_ratio, pop_center, lot)
    enrichment = edge_fail_ratio / max(center_fail_ratio, 0.005)

    signature = _classify_signature(
        edge_fail_ratio,
        center_fail_ratio,
        enrichment,
        edge_shift,
        center_shift,
    )

    return {
        "signature": signature,
        "edge_fail_ratio": round(edge_fail_ratio, 4),
        "center_fail_ratio": round(center_fail_ratio, 4),
    }


def _classify_signature(
    edge_fail_ratio: float,
    center_fail_ratio: float,
    enrichment: float,
    edge_shift: float,
    center_shift: float,
) -> str | None:
    if edge_shift >= 2.5 and enrichment >= 3.0 and edge_fail_ratio > center_fail_ratio * 2:
        return "edge_ring"
    if center_shift >= 2.5 and center_fail_ratio > edge_fail_ratio * 2:
        return "center"
    if edge_shift >= 1.5 and center_shift >= 1.5 and abs(edge_fail_ratio - center_fail_ratio) > 0.05:
        return "gradient"
    if edge_fail_ratio < 0.02 and center_fail_ratio < 0.02:
        return None
    if enrichment < 1.5 and edge_shift < 2.0 and center_shift < 2.0:
        return "uniform"
    return None
