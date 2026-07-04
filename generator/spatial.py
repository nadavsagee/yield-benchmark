"""Spatial layouts for die grid and WAT sites."""

from __future__ import annotations

import numpy as np

from schema import DIE_PER_WAFER, WAT_SITES


def make_die_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return die_x, die_y, radius for DIE_PER_WAFER dies on a ~10x10 grid."""
    side = int(np.sqrt(DIE_PER_WAFER))
    xs, ys, rs = [], [], []
    for i in range(side):
        for j in range(side):
            x = (j - side / 2 + 0.5) / (side / 2) * 0.95
            y = (i - side / 2 + 0.5) / (side / 2) * 0.95
            r = float(np.sqrt(x * x + y * y) / 0.95)
            if r <= 1.0:
                xs.append(x)
                ys.append(y)
                rs.append(r)
    # Trim or pad to exactly DIE_PER_WAFER
    while len(xs) < DIE_PER_WAFER:
        xs.append(0.0)
        ys.append(0.0)
        rs.append(0.0)
    return (
        np.array(xs[:DIE_PER_WAFER]),
        np.array(ys[:DIE_PER_WAFER]),
        np.array(rs[:DIE_PER_WAFER]),
    )


def make_wat_sites() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Return site ids and x, y, radius for WAT_SITES locations.

    Layout: 1 center, 6 mid-ring (~0.45), 6 edge-ring (~0.80).
    """
    sites: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    rs: list[float] = []

    sites.append("S00")
    xs.append(0.0)
    ys.append(0.0)
    rs.append(0.05)

    for k in range(6):
        angle = 2 * np.pi * k / 6
        x = 0.45 * np.cos(angle)
        y = 0.45 * np.sin(angle)
        sites.append(f"S{len(sites):02d}")
        xs.append(float(x))
        ys.append(float(y))
        rs.append(0.45)

    for k in range(6):
        angle = 2 * np.pi * k / 6 + np.pi / 6
        x = 0.80 * np.cos(angle)
        y = 0.80 * np.sin(angle)
        sites.append(f"S{len(sites):02d}")
        xs.append(float(x))
        ys.append(float(y))
        rs.append(0.80)

    assert len(sites) == WAT_SITES
    return sites, np.array(xs), np.array(ys), np.array(rs)
