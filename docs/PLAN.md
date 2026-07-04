# yield-benchmark — Project Plan (anchor document)

**Re-read this file at the start of every phase.** It is the source of truth for
*why* the project is built the way it is. `schema.py` is the source of truth for
the concrete data contract.

---

## 1. Goal

Arrive at the interview with a system NVIDIA's foundry group would actually want
to use — something the successful candidate might work on day one.

The system:
1. Generates realistic synthetic foundry data with **known** injected anomalies.
2. Lets an **independent LLM agent** investigate each dataset and diagnose it.
3. **Scores** the agent against ground truth.
4. Ranks multiple **agent versions** on a **leaderboard** to find the best one.

This is the level-up from a previous demo that detected a single *known* excursion.
The point here is **discovery of the unknown** + **measurement**.

> Clean slate: do **not** reuse code from the previous project. Rebuild fresh.

---

## 2. The data (what NVIDIA sees from the foundry)

NVIDIA's foundry-interface role sees **results + metadata**, NOT TSMC's process
recipe (RF power, gas flows, etc. are proprietary). So we investigate *backwards
from results*. Four tables (see `schema.py` for exact columns):

| table | grain | role |
|-------|-------|------|
| `sort.csv`   | per-die   | bin results + wafer coords — closest to yield |
| `wat.csv`    | per-site  | parametric e-test — the early electrical fingerprint |
| `inline.csv` | per-wafer/step | CD, overlay, film, defect — process metrology |
| `route.csv`  | per-wafer/op   | tool/chamber routing — the glue for commonality |

**The correlation chain is the heart of the tool:**
```
inline (CD/overlay/defect)  ->  WAT (parametric)  ->  Sort (yield/bin)
   "what happened"              "electrical effect"     "what failed"
```
The agent's value is **locating an excursion on this chain** (which step, which
param, which failure), not just saying "yield is low."

### Scale (agreed)
- 100 die/wafer, 20 wafers/lot, **50 lots/dataset**, 13 WAT sites, 5 inline steps
- ~30 datasets total
- ~45 baseline (normal) lots per dataset establish the in-family reference
- single-lot excursions placed at a **random** lot index (agent can't cheat by
  always checking the newest lot)
- drift-type excursions develop across a **sequence** of lots

---

## 3. The 8 anomaly types

Grouped by where they live on the chain. (Start with the 4 marked ★, add rest after.)

**Spatial**
- ★ `edge_signature` — edge-concentrated fails + edge-high WAT param
- `mean_shift` — a WAT param shifts across a lot, crosses spec

**Commonality**
- ★ `chamber_specific` — excursion only on one chamber's wafers

**Chain (needs inline — the new capability)**
- ★ `propagation` — inline defect -> WAT shift -> Sort fail (locate the origin step)
- `early_detection` — inline drift over a lot sequence, not yet in WAT/Sort (proactive!)
- `correlation_break` — WAT normal but Sort fails (between-layers or test issue)

**Complexity / realism**
- `confounding` — two params move together, only one is causal
- ★ `clean` — no excursion (false-positive control; agent must say "nothing")

Every dataset also carries **background noise** so signals aren't trivially clean.

---

## 4. Architecture

```
[1] Benchmark Generator  -> ~30 datasets, each = 4 CSVs + ground_truth.json
[2] Agent (a version)    -> reads the 4 CSVs (NOT ground truth), returns findings
[3] Scorer               -> compares findings vs ground_truth -> scores
[4] Runner / Leaderboard -> runs each agent version over all datasets -> ranks
```

**Hard rule (keeps it honest):** `ground_truth.json` is written by the generator
and read ONLY by the scorer. The agent never sees it.

### The agent (how "investigate" works)
- A **fixed unsupervised pre-step (variant b)** runs first: isolation forest / PCA
  flags *"where it's weird"* WITHOUT being told what to look for. This is the
  answer to "find the unknown."
- Then the **agent loop** (Option 1: the **LLM decides**) investigates:
  `observe -> decide next tool -> act (run tool) -> read result -> decide ...`
  until it calls `submit_findings`.
- The agent **computes nothing**; deterministic **tools** do the math. The agent
  only orchestrates. This prevents hallucinated numbers.
- Quality of decide->act comes from 3 levers: **tool descriptions**, the
  **methodology system prompt**, and the **stop condition**. These are what the
  optimizer later tunes.

### Tools (deterministic functions over the 4 tables)
Baseline agent starts with ~5 core tools + `submit_findings`:
`spatial_signature`, `wat_profile`, `commonality`, `chain_correlate`,
`excursion_confirm`.
Add later as needed: `bin_pareto`, `inline_trace`, `wat_sort_correlation` (gated,
co-location != causation), `compare_candidates` (confounding breaker),
`baseline_check`.

Each hard type maps to a tool that unlocks it:
- propagation -> inline_trace + chain_correlate
- early_detection -> inline_trace (fires before WAT/Sort)
- correlation_break -> chain_correlate
- confounding -> compare_candidates
- clean -> baseline_check + excursion_confirm

### Scoring (two layers — we want both)
- **Detection:** recall (found real excursions) + precision (didn't cry wolf on clean).
- **Diagnosis:** of those detected, is type / location / origin_step / affected_param correct?
- Metrics in ML language: recall, precision, diagnosis accuracy. Report per-type
  so we see *where* an agent breaks.

### Leaderboard — who competes
Contestants are **agent versions** (same benchmark, judged independently), not
agents talking to each other. A version = a specific combo of levers:

| version | difference |
|---------|-----------|
| `v0_baseline`     | 5 core tools, simple prompt, pre-step (b), single agent |
| `v1_all_tools`    | all 10 tools |
| `v2_rich_prompt`  | detailed methodology prompt |
| `v3_tool_prestep` | unsupervised as a tool (variant a) |
| `v4_subagents`    | Subagents pattern (independent checks -> main agent) |
| `v5_deterministic`| fixed-policy oracle (no LLM) — the control |

"Optimize/improve the agent" = find the highest-scoring version. It is a **search
over hand-defined versions** (Adir's option A), NOT self-training / recursive
self-improvement. This distinction matters — never claim the agent trains itself.

**Multi-agent note:** if we ever split the agent, use the **Subagents** pattern
(independent spatial/commonality/parametric/chain checks reporting up to a main
agent — no peer-to-peer talk), NOT Agent Teams (there's no interdependence to
justify it). Best move: make `v4_subagents` a competitor and *measure* if it helps.

---

## 5. Build order (verify after every phase)

- **Phase 0** — repo skeleton + `schema.py` + contracts + this doc.  ✅ (provided)
- **Phase 1** — Benchmark Generator: `build_normal()`, injectors, `generate_benchmark()`,
  `verify_generation.py`. Start with the 4 ★ types, then add the rest.
- **Phase 2** — Scorer (detection + diagnosis).
- **Phase 3** — Baseline agent `v0`: unsupervised pre-step + LLM tool-use loop, 5 core tools.
- **Phase 4** — Add agent versions (v1..v5).
- **Phase 5** — Runner / Leaderboard.

Rationale: build the measurement infrastructure (benchmark, scorer) **before** the
thing being measured (agents), so every claim is backed by a number. This is the
SDLC discipline — a strong thing to narrate in the interview.

---

## 6. Guardrails for building with Cursor
- Re-read this file and `schema.py` at the start of each phase.
- **Stop after each phase** and let me verify before continuing.
- Tools return **summaries**, never raw rows — the agent's context must stay small
  even though datasets have ~100K sort rows.
- Everything imports column/baseline definitions from `schema.py` — no re-defining.
