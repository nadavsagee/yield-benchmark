"""
schema.py — the single source of truth for the yield-benchmark project.

Every component (generator, agent tools, scorer) imports from here so the four
tables and the ground-truth / findings contracts can never silently drift apart.

Data model (4 tables, joined on lot+wafer):
    sort.csv   : per-die   bin results + wafer coordinates
    wat.csv    : per-site  parametric e-test (WAT/PCM)
    inline.csv : per-wafer/step inline metrology (CD, overlay, film, defects)
    route.csv  : per-wafer/operation tool/chamber routing (the 'glue' for commonality)

Keys:
    lot + wafer            -> joins all four tables
    step + tool_id         -> commonality (inline <-> route)
    die_x, die_y, radius   -> spatial analysis (sort)
    x, y, radius           -> spatial analysis (wat, per site)
"""

# --------------------------------------------------------------------------- #
# Table column definitions
# --------------------------------------------------------------------------- #

SORT_COLUMNS = [
    "lot", "wafer", "die_x", "die_y", "radius",
    "hard_bin", "soft_bin", "pass",
]

WAT_COLUMNS = [
    "lot", "wafer", "site", "x", "y", "radius",
    "vtn_mV", "vtp_mV", "idsat_uA", "ioff_nA",
    "rc_ohm", "rs_ohm", "rosc_MHz",
]

INLINE_COLUMNS = [
    "lot", "wafer", "step", "tool_id",
    "cd_nm", "overlay_nm", "film_A",
    "defect_count", "defect_density",
]

ROUTE_COLUMNS = [
    "lot", "wafer", "step", "tool_id", "chamber", "timestamp",
]

# --------------------------------------------------------------------------- #
# Canonical baseline values
#   - used by the generator to build "normal" (in-family) material
#   - used by the tools as the reference distribution for SPC / anomaly checks
#   mean = center of the healthy distribution, sigma = normal variation
# --------------------------------------------------------------------------- #

BASELINE = {
    # WAT parametrics
    "vtn_mV":         {"mean": 300.0,  "sigma": 8.0},
    "vtp_mV":         {"mean": -298.0, "sigma": 8.0},
    "idsat_uA":       {"mean": 620.0,  "sigma": 15.0},
    "ioff_nA":        {"mean": 1.0,    "sigma": 0.15},
    "rc_ohm":         {"mean": 20.0,   "sigma": 1.2},
    "rs_ohm":         {"mean": 15.0,   "sigma": 1.0},
    "rosc_MHz":       {"mean": 2000.0, "sigma": 25.0},
    # inline metrology
    "cd_nm":          {"mean": 46.0,   "sigma": 0.8},
    "overlay_nm":     {"mean": 2.0,    "sigma": 0.5},
    "film_A":         {"mean": 512.0,  "sigma": 6.0},
    "defect_density": {"mean": 0.10,   "sigma": 0.03},
}

# WAT parameters only (subset of BASELINE) — convenience for the tools
WAT_PARAMS = ["vtn_mV", "vtp_mV", "idsat_uA", "ioff_nA",
              "rc_ohm", "rs_ohm", "rosc_MHz"]

INLINE_METRICS = ["cd_nm", "overlay_nm", "film_A", "defect_density"]

# --------------------------------------------------------------------------- #
# Process structure
# --------------------------------------------------------------------------- #

# Inline process steps (order matters: this is the FEOL->BEOL sequence)
INLINE_STEPS = ["gate_etch", "contact_etch", "m1_litho", "m1_etch", "cmp"]

# Etch/patterning chambers (the unit commonality is measured against)
CHAMBERS = ["CH-A", "CH-B", "CH-C", "CH-D"]

# Soft-bin fail modes seen at Sort
SOFT_BINS = ["pass", "speed", "gross", "leakage", "open_short"]

# --------------------------------------------------------------------------- #
# Generation scale (the agreed configuration)
# --------------------------------------------------------------------------- #

DIE_PER_WAFER    = 100     # ~10x10 usable grid inside the wafer circle
WAFERS_PER_LOT   = 20
LOTS_PER_DATASET = 50      # ~45 baseline; excursion in 1-2 lots (random) or a drift sequence
WAT_SITES        = 13      # 3 distinct radii (center, mid ~0.45, edge ~0.80)

# --------------------------------------------------------------------------- #
# The 8 anomaly types for the first benchmark
#   (start with the 4 marked PHASE1_START; add the rest afterwards)
# --------------------------------------------------------------------------- #

ANOMALY_TYPES = [
    "edge_signature",     # PHASE1_START — spatial: edge-concentrated fails + edge-high WAT param
    "chamber_specific",   # PHASE1_START — commonality: excursion only on one chamber's wafers
    "propagation",        # PHASE1_START — chain: inline defect -> WAT shift -> Sort fail
    "clean",              # PHASE1_START — control: no excursion (false-positive test)
    "mean_shift",         # WAT parameter shifts across a lot, crosses spec
    "early_detection",    # inline drift over a lot sequence, not yet in WAT/Sort
    "correlation_break",  # WAT normal but Sort fails (between-layers or test issue)
    "confounding",        # two params move together, only one is causal
]

# Subset to implement first (verify the whole pipeline before adding the rest)
PHASE1_START_TYPES = [
    "edge_signature", "chamber_specific", "propagation", "clean",
]

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]  # mostly controlled via noise_level

# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #

# ---- ground_truth.json : written BY the generator, read ONLY by the scorer ----
# Fields are null when they don't apply to a given anomaly type.
GROUND_TRUTH_CONTRACT = {
    "dataset_id":     "str   — e.g. 'ds_017'",
    "difficulty":     "str   — easy | medium | hard",
    "excursion":      "bool  — False only for 'clean'",
    "type":           "str   — one of ANOMALY_TYPES",
    "location":       "str|None — affected chamber/tool (e.g. 'CH-C')",
    "origin_step":    "str|None — inline step where it starts (e.g. 'contact_etch')",
    "affected_param": "str|None — primary WAT param (e.g. 'rc_ohm')",
    "signature":      "str|None — spatial pattern (edge_ring|center|gradient|None)",
    "causal_chain":   "list|None — ordered tokens, e.g. ['defect_up','rc_up','speed_fail_up']",
    "confounder":     "str|None — for 'confounding': the correlated-but-not-causal param",
    "affected_lots":  "list — lot ids carrying the excursion (empty for clean)",
    "noise_level":    "float — 0..1 background noise (difficulty)",
}

# ---- findings : produced BY the agent (submit_findings), read by the scorer ----
# Mirrors ground truth minus fields the agent can't know (noise_level, confounder-as-truth).
FINDINGS_CONTRACT = {
    "detected":       "bool  — is there an excursion at all",
    "type":           "str|None — predicted anomaly type",
    "location":       "str|None — predicted chamber/tool",
    "origin_step":    "str|None — predicted inline origin step",
    "affected_param": "str|None — predicted primary WAT param",
    "cause":          "str  — short physical explanation",
    "confidence":     "str  — high | low",
    "reasoning":      "str  — free text, NOT scored (shown in the demo/trace)",
}

# Canonical causal-chain tokens (keep simple for v1)
CHAIN_TOKENS = [
    "defect_up", "cd_shift", "overlay_shift",
    "rc_up", "rs_up", "vt_shift", "idsat_down", "rosc_down",
    "speed_fail_up", "gross_fail_up", "leakage_up",
    "wat_normal", "sort_fail_up",
]
