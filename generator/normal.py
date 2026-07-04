"""Build in-family (normal) foundry data for all four tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from schema import (
    BASELINE,
    CHAMBERS,
    DIE_PER_WAFER,
    INLINE_COLUMNS,
    INLINE_METRICS,
    INLINE_STEPS,
    LOTS_PER_DATASET,
    ROUTE_COLUMNS,
    SORT_COLUMNS,
    WAT_COLUMNS,
    WAT_PARAMS,
    WAFERS_PER_LOT,
)

from generator.spatial import make_die_grid, make_wat_sites

# Healthy sort yield and soft-bin mix (background noise adds variation).
_BASE_YIELD = 0.965
_SOFT_BIN_WEIGHTS = {
    "speed": 0.45,
    "leakage": 0.25,
    "gross": 0.15,
    "open_short": 0.15,
}

_STEP_TOOLS = {
    "gate_etch": "ETCH-G01",
    "contact_etch": "ETCH-C01",
    "m1_litho": "LITHO-M1",
    "m1_etch": "ETCH-M1",
    "cmp": "CMP-01",
}

_COMMONALITY_STEP = "gate_etch"


def lot_id(index: int) -> str:
    return f"lot_{index:03d}"


def wafer_id(index: int) -> str:
    return f"wafer_{index:02d}"


def _noise_sigma(param: str, noise_level: float) -> float:
    return BASELINE[param]["sigma"] * (1.0 + noise_level)


def _sample_wat_value(rng: np.random.Generator, param: str, noise_level: float) -> float:
    return float(
        rng.normal(BASELINE[param]["mean"], _noise_sigma(param, noise_level))
    )


def _assign_chamber(rng: np.random.Generator, lot: str, wafer: str, step: str) -> str:
    """Deterministic-but-varied chamber assignment per wafer/step."""
    key = f"{lot}|{wafer}|{step}"
    idx = rng.integers(0, len(CHAMBERS))
    # Mix hash-like spread with rng for reproducibility per dataset seed
    spread = sum(ord(c) for c in key) % len(CHAMBERS)
    return CHAMBERS[(idx + spread) % len(CHAMBERS)]


def build_normal(
    seed: int = 0,
    noise_level: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """
    Generate baseline material: LOTS_PER_DATASET lots with no injected excursion.

    Returns dict with keys sort, wat, inline, route (DataFrames).
    """
    rng = np.random.default_rng(seed)
    die_x, die_y, die_radius = make_die_grid()
    site_ids, wat_x, wat_y, wat_radius = make_wat_sites()

    lots = [lot_id(i + 1) for i in range(LOTS_PER_DATASET)]
    wafers = [wafer_id(i + 1) for i in range(WAFERS_PER_LOT)]

    sort_rows: list[dict] = []
    wat_rows: list[dict] = []
    inline_rows: list[dict] = []
    route_rows: list[dict] = []

    base_ts = pd.Timestamp("2025-01-01")

    for lot_idx, lot in enumerate(lots):
        for wafer_idx, wafer in enumerate(wafers):
            # --- route (needed before inline for chamber linkage) ---
            wafer_routes: dict[str, str] = {}
            for step_idx, step in enumerate(INLINE_STEPS):
                chamber = _assign_chamber(rng, lot, wafer, step)
                wafer_routes[step] = chamber
                route_rows.append(
                    {
                        "lot": lot,
                        "wafer": wafer,
                        "step": step,
                        "tool_id": _STEP_TOOLS[step],
                        "chamber": chamber,
                        "timestamp": base_ts + pd.Timedelta(
                            days=lot_idx, hours=wafer_idx * 2 + step_idx
                        ),
                    }
                )

            # --- inline ---
            for step in INLINE_STEPS:
                row: dict = {
                    "lot": lot,
                    "wafer": wafer,
                    "step": step,
                    "tool_id": _STEP_TOOLS[step],
                }
                for metric in INLINE_METRICS:
                    row[metric] = float(
                        rng.normal(
                            BASELINE[metric]["mean"],
                            _noise_sigma(metric, noise_level),
                        )
                    )
                # defect_count scales with wafer area proxy
                row["defect_count"] = max(
                    0,
                    int(round(row["defect_density"] * 100 + rng.normal(0, 0.5))),
                )
                inline_rows.append(row)

            # --- WAT ---
            for site_idx, site in enumerate(site_ids):
                wat_row: dict = {
                    "lot": lot,
                    "wafer": wafer,
                    "site": site,
                    "x": wat_x[site_idx],
                    "y": wat_y[site_idx],
                    "radius": wat_radius[site_idx],
                }
                for param in WAT_PARAMS:
                    wat_row[param] = _sample_wat_value(rng, param, noise_level)
                wat_rows.append(wat_row)

            # --- sort ---
            for d in range(DIE_PER_WAFER):
                passed = rng.random() < _BASE_YIELD
                if passed:
                    hard_bin, soft_bin = 1, "pass"
                else:
                    hard_bin = rng.integers(2, 6)
                    modes = list(_SOFT_BIN_WEIGHTS.keys())
                    weights = np.array([_SOFT_BIN_WEIGHTS[m] for m in modes])
                    soft_bin = rng.choice(modes, p=weights / weights.sum())
                sort_rows.append(
                    {
                        "lot": lot,
                        "wafer": wafer,
                        "die_x": die_x[d],
                        "die_y": die_y[d],
                        "radius": die_radius[d],
                        "hard_bin": hard_bin,
                        "soft_bin": soft_bin,
                        "pass": passed,
                    }
                )

    return {
        "sort": pd.DataFrame(sort_rows, columns=SORT_COLUMNS),
        "wat": pd.DataFrame(wat_rows, columns=WAT_COLUMNS),
        "inline": pd.DataFrame(inline_rows, columns=INLINE_COLUMNS),
        "route": pd.DataFrame(route_rows, columns=ROUTE_COLUMNS),
        "_meta": pd.DataFrame({"commonality_step": [_COMMONALITY_STEP]}),
    }


def commonality_step() -> str:
    """Inline step used for chamber commonality analysis."""
    return _COMMONALITY_STEP
