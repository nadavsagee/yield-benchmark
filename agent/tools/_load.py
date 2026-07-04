"""Load dataset CSVs into a tables dict (for tools and verification)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from schema import INLINE_COLUMNS, ROUTE_COLUMNS, SORT_COLUMNS, WAT_COLUMNS


def load_tables(dataset_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load sort, wat, inline, route from a dataset directory."""
    root = Path(dataset_dir)
    return {
        "sort": pd.read_csv(root / "sort.csv", usecols=SORT_COLUMNS),
        "wat": pd.read_csv(root / "wat.csv", usecols=WAT_COLUMNS),
        "inline": pd.read_csv(root / "inline.csv", usecols=INLINE_COLUMNS),
        "route": pd.read_csv(root / "route.csv", usecols=ROUTE_COLUMNS),
    }
