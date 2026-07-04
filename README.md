# yield-benchmark

A system that generates realistic synthetic semiconductor foundry data with
**known injected anomalies**, runs an LLM **agent** to investigate each dataset,
**scores** the diagnosis against ground truth, and ranks agent versions on a
**leaderboard**.

Built for a Senior IC Foundry Engineer interview (NVIDIA Networking). The point:
move from *detecting a known excursion* to *discovering the unknown* — and to
**measure** which agent design works best.

---

## Start here

1. Read **`docs/PLAN.md`** — the anchor document. It explains the data model, the
   8 anomaly types, the agent design, the scorer, the leaderboard, and the build
   order, plus *why* each decision was made.
2. Read **`schema.py`** — the concrete contract: the 4 tables, baseline values,
   anomaly types, and the ground-truth / findings shapes.

## Data model (4 tables, joined on `lot` + `wafer`)

| table | grain | role |
|-------|-------|------|
| `sort.csv`   | per-die   | bin results + wafer coords |
| `wat.csv`    | per-site  | parametric e-test (WAT/PCM) |
| `inline.csv` | per-wafer/step | CD, overlay, film, defect |
| `route.csv`  | per-wafer/op   | tool/chamber routing (commonality) |

The chain **inline -> WAT -> Sort** is the heart of the tool: the agent locates an
excursion on the chain, not just "yield is low."

## Build phases (stop and verify after each)

| phase | what | status |
|-------|------|--------|
| 0 | repo skeleton + `schema.py` + contracts + plan | ✅ done |
| 1 | benchmark generator + `verify_generation.py` | ⏳ next |
| 2 | scorer (detection + diagnosis) | ⬜ |
| 3 | baseline agent `v0` (unsupervised pre-step + LLM tool loop) | ⬜ |
| 4 | agent versions (v1..v5) | ⬜ |
| 5 | runner / leaderboard | ⬜ |

## How to continue in Cursor

Open this folder in Cursor and say:

> Read `README.md` and `docs/PLAN.md`, then build **Phase 1** (the benchmark
> generator). Start with the 4 anomaly types marked PHASE1_START in `schema.py`
> (`edge_signature`, `chamber_specific`, `propagation`, `clean`). Stop after
> Phase 1 so I can run `verify_generation.py` before we continue.

## Config (agreed)

50 lots/dataset · 20 wafers/lot · 100 die/wafer · 13 WAT sites · 5 inline steps ·
~30 datasets. See `schema.py` for exact values.

## Setup

```bash
pip install -r requirements.txt
```

The agent (Phase 3) uses the Anthropic API — set `ANTHROPIC_API_KEY` when you get
there. Phases 1–2 are pure numpy/pandas/scipy and need no API key.
