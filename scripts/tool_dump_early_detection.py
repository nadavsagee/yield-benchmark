#!/usr/bin/env python3
"""Deterministic tool dump for early_detection lot_038 (no LLM)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.dispatch import dispatch
from agent.tools._baseline import robust_shift
from agent.tools._load import load_tables
from schema import INLINE_STEPS, WAT_PARAMS

DATASET = Path("datasets/_verify/verify_early_detection")
LOT = "lot_038"
ORIGIN = "m1_litho"
METRIC = "overlay_nm"


def pp(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def inline_overlay_context(inline_df: pd.DataFrame) -> dict:
    def lot_order(l: str) -> int:
        return int(l.split("_")[1])

    lots = sorted(inline_df["lot"].unique(), key=lot_order)
    step_df = inline_df[inline_df["step"] == ORIGIN]
    lot_means = step_df.groupby("lot")[METRIC].mean()

    # trailing 4-lot window ending at lot_038
    idx = lots.index(LOT)
    window = lots[max(0, idx - 3) : idx + 1]
    y = [float(lot_means[lot]) for lot in window]
    x = np.arange(len(window), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(window) >= 2 else 0.0

    pop = lot_means.drop(index=LOT, errors="ignore")
    lot_val = float(lot_means[LOT])
    sigma = robust_shift(lot_val, lot_means, LOT)

    return {
        "m1_litho_overlay_nm_mean": round(lot_val, 4),
        "population_median_excl_lot": round(float(pop.median()), 4),
        "sigma_vs_all_lots": round(sigma, 3),
        "trailing_4lot_window": window,
        "trailing_4lot_overlay_means": [round(v, 4) for v in y],
        "trailing_4lot_slope": round(slope, 4),
        "note": "Computed from raw inline table — NOT exposed by any agent tool.",
    }


def main() -> None:
    tables = load_tables(DATASET)
    inline_df = tables["inline"]

    print(f"Dataset: {DATASET}")
    print(f"Lot: {LOT}  (ground truth: last lot in 4-lot overlay drift at {ORIGIN})")
    print(f"Affected lots: lot_035 .. lot_038")
    print()

    print("=== RAW INLINE CONTEXT (no agent tool returns this) ===")
    print(pp(inline_overlay_context(inline_df)))
    print()

    tool_calls: list[tuple[str, dict]] = [
        ("chain_correlate", {"lot": LOT}),
        ("spatial_signature", {"lot": LOT, "population": "fail"}),
        ("commonality", {"lot": LOT, "population": "yield"}),
    ]
    for param in WAT_PARAMS:
        tool_calls.append(("wat_profile", {"lot": LOT, "param": param}))
    for param in ["yield", "fail", "speed", "gross", "leakage"]:
        tool_calls.append(("excursion_confirm", {"lot": LOT, "param": param}))

    print("=== AGENT TOOL OUTPUTS ===")
    for name, args in tool_calls:
        result = dispatch(name, args, tables)
        print(f"\n--- {name}({', '.join(f'{k}={v!r}' for k, v in args.items())}) ---")
        print(pp(result))

    print("\n=== SUMMARY ===")
    chain = dispatch("chain_correlate", {"lot": LOT}, tables)
    inline_link = chain["links"][0]
    print(f"chain_correlate inline link: step={inline_link['step']!r}, "
          f"metric=defect_density (hardcoded), shift_sigma={inline_link['shift_sigma']}, "
          f"status={inline_link['status']!r}")
    print(f"chain_correlate chain_intact={chain['chain_intact']}, break_at={chain['break_at']!r}")
    print()
    print("Tool coverage for overlay drift:")
    print("  - chain_correlate: inline side uses defect_density ONLY — overlay_nm not read")
    print("  - excursion_confirm: WAT params + sort metrics only — no inline params")
    print("  - wat_profile / spatial_signature / commonality: WAT or Sort only")
    print("  => No current tool surfaces m1_litho overlay_nm drift or trailing-window trend.")


if __name__ == "__main__":
    main()
