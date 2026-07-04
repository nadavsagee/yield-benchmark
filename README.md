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
- Render the HTML:

```bash
python report/generate_report.py \
  --dataset datasets/_verify/verify_early_detection \
  --investigation report/examples/investigation_early_detection.json \
  --output report/output/early_detection_rca.html
```

Open `report/output/early_detection_rca.html` in a browser. The report includes
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
| `datasets/_verify/` | Small committed fixtures (one per anomaly type) |
| `docs/PLAN.md` | Design rationale and build history |

---

## License / use

Built as an architecture and measurement demo for a senior IC foundry yield
engineering interview. Synthetic data only; not fab production code.
