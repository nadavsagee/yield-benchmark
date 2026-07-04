# yield-benchmark

**Interview demo — not production.** This repo proves an end-to-end architecture for
semiconductor yield root-cause analysis: synthetic fab data with known ground truth,
an unsupervised pre-step that surfaces suspects without labels, an LLM agent that
orchestrates deterministic investigation tools (never inventing numbers), automated
scoring against ground truth, and a self-contained HTML RCA report for disposition
review. It is built to show *how* you would measure and improve an agent design—not
to ship as a fab system.

---

## What it does

The benchmark generates realistic foundry datasets (Sort, WAT, inline metrology, and
tool routing tables) with one of eight injected anomaly types—or none at all—then
challenges an agent to investigate from CSVs alone. A fixed unsupervised pre-step
(Isolation Forest + PCA over per-lot features) ranks suspect lots without knowing
the excursion type. An LLM agent then runs a tool-use loop: each tool returns
deterministic statistics (SPC shifts, spatial signatures, chamber commonality, inline
drift, chain correlation). The agent submits structured findings; the scorer compares
them to `ground_truth.json` the agent never sees. A report generator turns findings
plus table data into an interview-ready RCA HTML with evidence charts and ruled-out
hypotheses.

---

## Architecture

```
Generator          →  datasets/_verify/verify_<type>/  (4 CSVs + ground_truth.json)
Unsupervised       →  prestep: Isolation Forest + PCA flags suspect lots (no labels)
LLM agent (v0/v1)  →  observe → pick tool → read result → … → submit_findings
Deterministic tools→  wat_profile, spatial_signature, commonality, chain_correlate,
                       excursion_confirm, inline_trace (v1)
Scorer             →  detection (TP/FP/TN/FN) + diagnosis (type, location, step, param)
RCA report         →  generate_report.py → standalone HTML (charts, tables, actions)
```

**Hard rule:** `ground_truth.json` is written by the generator and read only by the
scorer. The agent and report tools never load it during investigation.

Data joins on `lot` + `wafer` across four tables; the causal chain
**inline → WAT → Sort** is what separates “yield is low” from “here is where and why.”

| Table | Grain | Role |
|-------|-------|------|
| `sort.csv` | per-die | bin results + wafer coordinates |
| `wat.csv` | per-site | parametric e-test (7 params, 13 sites) |
| `inline.csv` | per-wafer/step | CD, overlay, film, defect metrics |
| `route.csv` | per-wafer/op | tool/chamber routing (commonality) |

Scale: 50 lots · 20 wafers/lot · 100 die/wafer · 5 inline steps. See `schema.py` for
contracts and baselines.

---

## Eight anomaly types

| Type | Where it lives | What the agent must find |
|------|----------------|--------------------------|
| `edge_signature` | Spatial (Sort + WAT) | Edge-concentrated fails + edge-high WAT param |
| `chamber_specific` | Commonality (`route`) | Yield/parametric split tied to one chamber |
| `propagation` | Chain (inline → WAT → Sort) | Inline defect origin → WAT shift → Sort fail |
| `early_detection` | Inline (pre-WAT/Sort) | Multi-lot inline drift; WAT and Sort still in-family |
| `mean_shift` | WAT | Lot-wide parametric shift crossing SPC |
| `correlation_break` | Sort vs WAT | WAT normal; Sort fails (test/layer disconnect) |
| `confounding` | WAT (causal vs correlated) | Two params move; only one drives the failure |
| `clean` | — | No excursion (false-positive control) |

Each injected dataset lives under `datasets/_verify/verify_<type>/`.

---

## Measured results (8-dataset `_verify` suite, 5 runs each)

Comparing agent versions on the same benchmark—agent versions compete; they do not
train each other.

### How to read this

**The suite.** Eight small committed fixtures under `datasets/_verify/`—one per
anomaly type (`edge_signature`, `chamber_specific`, …, `clean`). Each folder has
four CSVs plus `ground_truth.json`. The agent reads only the CSVs; the scorer reads
ground truth to label each run.

**Five runs each.** The LLM agent is non-deterministic, so `run_v0_repeated.py`
calls `investigate()` five times per dataset (`--runs 5`) and reports how often each
outcome appears. A score like `early_detection` **0/5** means zero true positives
across five attempts—the agent never set `detected=true` on that dataset. **5/5**
means it caught the excursion every time.

**Detection labels** (from `scorer/score.py`):

| Label | Meaning |
|-------|---------|
| **TP** | Real excursion present; agent said `detected=true` |
| **FN** | Real excursion present; agent missed it (`detected=false`) |
| **FP** | No excursion (`clean`); agent falsely flagged one |
| **TN** | No excursion; agent correctly said clean |

**Row highlights in the table below:**

- **`early_detection` detection** — the hardest case for v0: inline metrology drifts
  over several lots while WAT and Sort still pass standard gates. v0 had no
  `inline_trace` tool, so it missed the signal all five times. v1 adds that tool plus
  prompt rules for sustained inline drift and detected it every run.
- **`clean` false positives** — v0 cried wolf on the control dataset all five runs
  (five **FP**s). v1’s stricter “default to clean unless a driver clears three bars”
  rule eliminated those false alarms (five **TN**s).
- **Overall precision** — `TP / (TP + FP)` aggregated across all eight datasets in
  each batch run, then averaged over five batch runs. It answers: *when the agent
  says “excursion,” how often is it right?* v0’s clean-dataset FPs pulled precision
  to 75%; v1 reached 100% on this suite.

These numbers are from the `_verify` smoke suite (50-lot fixtures), not a full
~30-dataset production benchmark. They are meant to show architectural leverage—one
tool + prompt change moving a blind spot to reliable detection—not fab sign-off stats.

| Metric | v0 baseline | v1 (+ `inline_trace`, early-detection rules) |
|--------|-------------|-----------------------------------------------|
| `early_detection` detection | 0/5 (0%) | 5/5 (100%) |
| `clean` false positives | 5/5 | 0/5 |
| Overall precision | 75% | 100% |

v0 missed subtle inline drift because it lacked `inline_trace` and tended to over-call
excursions on clean material. v1 adds the inline tool and stricter pre-step gating for
`early_detection` while keeping the same deterministic tool stack otherwise.

Run the measurement yourself:

```bash
python run_v0_repeated.py --version v0 --runs 5
python run_v0_repeated.py --version v1 --runs 5
```

Pre-step only (no API, deterministic):

```bash
python verify_prestep.py --prestep-only
```

---

## Quickstart

```bash
pip install -r requirements.txt
```

**1. Verify the generator (free, no API)**

```bash
python verify_generation.py
python verify_tools.py
```

**2. Run the agent benchmark (requires API key)**

```bash
export ANTHROPIC_API_KEY=your_key_here   # Windows: set ANTHROPIC_API_KEY=...
python run_v0_repeated.py --version v1 --runs 5
```

**3. Generate an RCA report (free, no API)**

Example investigation fixture and rendered report:

- `report/examples/investigation_early_detection.json` — sample agent findings for
  the `early_detection` case (overlay drift at `m1_litho`, WAT/Sort still nominal).
- **`report/examples/early_detection_rca.html`** — committed demo report (open in a
  browser on GitHub or locally; no generation step required).
- Regenerate after changing findings or charts:

```bash
python report/generate_report.py \
  --dataset datasets/_verify/verify_early_detection \
  --investigation report/examples/investigation_early_detection.json \
  --output report/examples/early_detection_rca.html
```

Local scratch output can go to `report/output/` (gitignored). The report includes
positive inline-drift evidence, negative-evidence charts (yield SPC, wafer map, WAT
panel, chamber bars), ruled-out hypotheses, and a monitoring-gap callout— all from
real table data, no LLM at render time.

---

## Repo map

| Path | Purpose |
|------|---------|
| `schema.py` | Tables, baselines, anomaly types, findings contract |
| `generator/` | Synthetic data + injectors |
| `agent/prestep.py` | Unsupervised suspect ranking |
| `agent/v0.py`, `agent/v1.py` | LLM investigation loops |
| `agent/tools/` | Deterministic investigation functions |
| `scorer/` | Detection + diagnosis scoring |
| `report/generate_report.py` | Standalone HTML RCA |
| `report/examples/` | Demo investigation JSON + committed `early_detection_rca.html` |
| `datasets/_verify/` | Small committed fixtures (one per anomaly type) |
| `docs/PLAN.md` | Design rationale and build history |

---

## License / use

Built as an architecture and measurement demo for a senior IC foundry yield
engineering interview. Synthetic data only; not fab production code.
