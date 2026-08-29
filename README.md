# SBA-XAI Ethiopia

Explainable Machine Learning with Actionable Counterfactuals for Skilled
Birth Attendance in Ethiopia — a national predictive model with
region-modified effects, built on EDHS 2024/25 data.

Implements Study Protocol v1.2 (Kassa, Kassahun, Alene — Woldia
University), targeting **BMC Medicine** (fallback: *International Journal
for Equity in Health*).

## Project layout

```
sba_xai_ethiopia/
├── config/
│   └── config.yaml          # single source of truth: paths, variables, filters,
│                             # model + XAI settings (mirrors protocol sections 3-7)
├── data/
│   └── raw/
|       └── 2024/                # put the unzipped DHS archive here (see below)
├── src/
│   ├── utils.py                          # config/logging/path helpers
│   ├── step01_data_loading.py            # §3.1  load BR / IR / GE recodes
│   ├── step02_data_merging.py            #       merge BR+IR, GE linkage check
│   ├── step03_eligibility_filter.py      # §4.1  b19<=35 delivery-module filter
│   ├── step04_outcome_derivation.py      # §4.2  SBA outcome (m3a-d)
│   ├── step05_feature_engineering.py     # §5    raw predictors + region_stratum
│   ├── step06_survey_design.py           # §6.1  weights, cluster CV, geo hold-out
│   ├── step07_model_training.py          # §6.3  XGBoost/LightGBM training
│   ├── step08_model_evaluation.py        # §6.3  AUC/calibration/equity on hold-out
│   ├── step09_shap_analysis.py           # §7.1  global + region-interaction SHAP
│   ├── step10_counterfactual_analysis.py # §7.2  DiCE constrained counterfactuals
│   └── step11_reporting.py               #       prevalence tables + run summary
├── main.py                  # pipeline combiner — runs the steps in order
├── requirements.txt
└── outputs/                 # models/ figures/ tables/ reports/ logs/ (generated)
```

Each `stepNN_*.py` is independently runnable (`python src/step05_feature_engineering.py`)
and reads/writes parquet files under `data/interim/` and `data/processed/`, so you
can inspect or re-run any single stage without repeating the whole pipeline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place the unzipped DHS 2024/25 archive under `data/raw/` — the loader
(`step01_data_loading.py`) locates files by glob pattern (`**/*BR8A*.dta`,
`**/*IR8A*.dta`, `**/*GE8A*.shp`), so it doesn't matter whether your folder
names are `ETBR8ADT/`, `ETBR8AFL/`, or similar inconsistent suffixes — it
just needs to find one Births Recode `.dta`, one Individual Recode `.dta`,
and (optionally) one GPS `.shp` somewhere under that tree.

Per protocol section 3.1, Household (HR), Children's (KR), Men's (MR), and
Household Member (PR) recodes are **not used** and are never loaded —
`utils.assert_no_excluded_datasets_used` fails loudly if a future edit
wires one in by mistake.

## Running the pipeline

```bash
python main.py                    # run all 11 steps end to end
python main.py --list             # show step ids and descriptions
python main.py --from step07      # resume from model training onward
python main.py --only step09      # re-run just the SHAP step
```

Outputs land under `outputs/`:
- `outputs/models/` — trained model bundle (`.joblib`)
- `outputs/figures/` — SHAP summary plot, calibration plot, PD-by-stratum plots
- `outputs/tables/` — CV/hold-out metrics, SHAP importance & interaction tables,
  equity-by-wealth-quintile table, DiCE counterfactuals + validity check
- `outputs/reports/` — consolidated `pipeline_run_summary.json`,
  counterfactual framing note
- `outputs/logs/` — one log file per step

## Known assumptions to verify against your actual extract

These are flagged inline in the code with `# verify` comments — check them
against the `.DO`/`.MAP` files that ship with your specific DHS extract
before trusting outputs for publication:

1. **`V024_REGION_LABELS`** (`step05_feature_engineering.py`) — a
   placeholder v024 code→region-name map. EDHS round-to-round code
   ordering can differ; confirm against `ETBR8AFL.DO`.
2. **WASH improved/unimproved codes** (`step05_feature_engineering.py`) —
   placeholder JMP-ladder category sets for v113/v116; confirm against the
   IR `.DO` file.
3. **Exact post-filter regional sample sizes** (protocol §6.2) — the
   config lists the five regions expected to fall below n=100 after the
   b19≤35 filter; `step05` logs the actual counts each run so you can
   confirm/update the list.
4. Model algorithm defaults to XGBoost (`config.model.algorithm`) — switch
   to `lightgbm` there if preferred; both paths are implemented.

## Design choices that map directly to the protocol

- **No PCA composite for WASH** — protocol explicitly rejects this in favor
  of the WHO/UNICEF JMP ladder (reporting-only; raw v113/v116 stay as
  model inputs).
- **Reporting recodes never enter the model matrix** — `step05` writes
  them under an `rr_` prefix; `step06`'s `get_model_feature_list()` only
  ever pulls from `config.predictors.*.raw`, so there's no code path by
  which a simplified recode can leak into the feature matrix.
- **Cluster-based CV + separate geographic hold-out** — folds and the
  hold-out split are both built from `v021` (PSU/cluster), never from
  individual rows, to prevent leakage between births in the same
  enumeration area (protocol §6.1, §6.3).
- **SHAP interaction on region_stratum, not 14 separate region models** —
  protocol §6.2's explicit alternative to underpowered per-region fits.
- **DiCE is supplementary, never primary** — `step10` enforces the
  mutability structure (modifiable / semi-modifiable / immutable) from
  protocol §7.2 and reports an in-distribution validity rate rather than
  presenting counterfactuals as causal.
- **Every report carries the causal disclaimer** (`config.provenance`) —
  SHAP and DiCE outputs are associational, not causal, given the
  cross-sectional design (protocol §9).
