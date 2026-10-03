"""
Step 12 — 2016-vs-2024/25 scenario comparison

Runs the SAME model (config.model.algorithm / config.model.params — held
fixed across every scenario on purpose, see note below) under five
train/test configurations that mix the two EDHS rounds differently, and
writes a comparison table so the effect of *which data the model sees* can
be read off directly.

  Scenario 1 — combined 2016+2024/25 data, cluster-level 80/20 split
  Scenario 2 — train on ALL 2024/25, test on ALL 2016 (temporal generalization,
               new -> old)
  Scenario 3 — train on ALL 2016, test on ALL 2024/25 (temporal generalization,
               old -> new)
  Scenario 4 — train on 80% of 2016 clusters, test on the remaining 20%
               (within-round; reuses the geo_holdout split step06 already
               assigned for the 2016 round)
  Scenario 5 — train on 80% of 2024/25 clusters, test on the remaining 20%
               (within-round; reuses the geo_holdout split step06 already
               assigned for the 2024/25 round — this is exactly the split
               step07/step08 already report under outputs/tables/, scenario
               5 just re-surfaces it alongside the other four for the table)

Prerequisite: BOTH rounds must already have been run through step06 (model
matrix assembly), i.e.:

    python main.py --config config/config_2016.yaml --through step06
    python main.py --config config/config.yaml      --through step06

This step reads each round's `analytic_full.parquet` (NOT model_matrix.parquet)
so it can re-derive a single, CANONICAL region_stratum encoding shared by
both rounds — see `_encode_region_stratum_canonical` docstring for why using
each round's own step06-assigned integer codes directly would silently
mislabel one of the two rounds whenever they don't happen to sort
identically.

Why the model itself is held fixed across scenarios: the point of this
comparison is to isolate the effect of the DATA (which round(s) trained on,
which round(s) tested on), not to also let hyperparameters vary scenario to
scenario — that would confound the two. If you want a hyperparameter-tuned
model per scenario, tune once on scenario 1 (the largest, most representative
pool) and keep those hyperparameters fixed for 2-5 too.

Causal disclaimer (protocol section 8/9, carried through from every other
step in this pipeline): all metrics below describe predictive performance
under a cross-sectional, observational design — a large train/test AUC drop
between a within-round scenario (4 or 5) and a cross-round scenario (2 or 3)
reflects DISTRIBUTION SHIFT between survey rounds (e.g. real underlying
trends in SBA + genuine question/coding changes across rounds), not a causal
claim about time.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from step06_survey_design import get_model_feature_list
from step07_model_training import fit_with_class_weight
from utils import PROJECT_ROOT, load_config, read_parquet, save_parquet, setup_logger

# (year, config path relative to project root, human label)
ROUND_SOURCES = [
    (2016, "config/config_2016.yaml", "EDHS 2016"),
    (2024, "config/config.yaml", "EDHS 2024/25"),
]

# Canonical, round-independent region_stratum encoding. Fixed here (not
# re-derived from whatever categories happen to be present in a given round
# or scenario slice) so the same code always means the same stratum on both
# sides of every scenario's train/test split — see module docstring.
CANONICAL_REGION_STRATUM = {
    "Agrarian Highlands": 0,
    "Pastoralist/Arid": 1,
    "Urban/Peri-urban": 2,
}


def _load_round(year: int, config_path: str, label: str, logger) -> dict:
    """Load one survey round's analytic_full.parquet and assemble X/y/groups
    using the canonical region_stratum encoding (not the per-round integer
    codes step06 already wrote into that round's model_matrix.parquet —
    those codes are internally consistent WITHIN a round but are not
    guaranteed to agree ACROSS rounds, since step06 assigns them via
    `sorted(categories present in that round)`: a round missing one of the
    3 strata entirely, or simply differing in how many distinct categories
    ended up in its post-filter sample, would sort to a different mapping.
    Re-deriving from the TEXT region_stratum in analytic_full.parquet here,
    once, against a fixed dict avoids that failure mode entirely."""

    cfg = load_config(PROJECT_ROOT / config_path)
    processed = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    path = processed / "analytic_full.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found for {label}. Run this round through step06 first:\n"
            f"    python main.py --config {config_path} --through step06"
        )

    df = read_parquet(path)
    logger.info("%s: loaded analytic_full.parquet — %s rows", label, len(df))

    feature_cols = get_model_feature_list(cfg)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label}: configured model features not found: {missing}")

    X = df[feature_cols].copy()
    if "region_stratum" in X.columns:
        unmapped = ~X["region_stratum"].isin(CANONICAL_REGION_STRATUM)
        if unmapped.any():
            logger.warning(
                "%s: %s rows have a region_stratum value outside the canonical "
                "3-category set %s — these rows are dropped from the scenario "
                "comparison (they would silently become NaN features otherwise).",
                label, int(unmapped.sum()), list(CANONICAL_REGION_STRATUM),
            )
            df = df.loc[~unmapped].reset_index(drop=True)
            X = X.loc[~unmapped].reset_index(drop=True)
        X["region_stratum"] = X["region_stratum"].map(CANONICAL_REGION_STRATUM)

    y = df[cfg["outcome"]["name"]].astype(int).reset_index(drop=True)

    cluster_var = cfg["merge"]["cluster_var"]
    # Prefix with the round so scenario 1's combined split never treats a
    # cluster id reused independently by both rounds as the same cluster.
    groups = (f"{year}_" + df[cluster_var].astype(str)).reset_index(drop=True)

    survey_year = (
        df["survey_year"] if "survey_year" in df.columns
        else pd.Series(year, index=df.index)
    ).reset_index(drop=True)

    geo_holdout = (
        df["geo_holdout"] if "geo_holdout" in df.columns
        else pd.Series(False, index=df.index)
    ).reset_index(drop=True)

    return {
        "year": year, "label": label, "cfg": cfg,
        "X": X, "y": y, "groups": groups,
        "survey_year": survey_year, "geo_holdout": geo_holdout,
        "feature_cols": feature_cols,
    }


def _fit_and_evaluate(model_cfg: dict, seed: int,
                       X_train, y_train, X_test, y_test, logger, scenario_name: str) -> dict:
    if y_train.nunique() < 2:
        raise ValueError(f"{scenario_name}: training outcome has a single class — cannot fit.")
    if y_test.nunique() < 2:
        logger.warning("%s: test outcome has a single class — AUC/AP will be NaN.", scenario_name)

    model = fit_with_class_weight(model_cfg, seed, X_train, y_train, logger)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    def _safe(fn, *a, **kw):
        try:
            return float(fn(*a, **kw))
        except ValueError:
            return float("nan")

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_prevalence": float(y_train.mean()),
        "test_prevalence": float(y_test.mean()),
        "roc_auc": _safe(roc_auc_score, y_test, proba),
        "average_precision": _safe(average_precision_score, y_test, proba),
        "precision": _safe(precision_score, y_test, pred, zero_division=0),
        "recall": _safe(recall_score, y_test, pred, zero_division=0),
        "f1": _safe(f1_score, y_test, pred, zero_division=0),
        "brier_score": _safe(brier_score_loss, y_test, proba),
    }
    for k, v in metrics.items():
        logger.info("%s: %s = %s", scenario_name, k, v)
    return metrics


def _write_scenario_output(reports_dir: Path, scenario_id: str, title: str,
                            description: str, metrics: dict, logger) -> None:
    out_dir = reports_dir / scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"title": title, "description": description, "metrics": metrics}, f, indent=2)

    lines = [
        f"# {scenario_id}: {title}", "", description, "", "## Metrics", "",
        "| metric | value |", "|---|---|",
    ]
    for k, v in metrics.items():
        v_str = f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else str(v)
        lines.append(f"| {k} | {v_str} |")
    with open(out_dir / "README.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Wrote %s -> %s", scenario_id, out_dir)


def run_scenarios(logger) -> pd.DataFrame:
    r2016 = _load_round(*ROUND_SOURCES[0], logger)
    r2024 = _load_round(*ROUND_SOURCES[1], logger)

    # Both rounds share the same model config (algorithm/params/seed) by
    # construction (config_2016.yaml's model: block is kept identical to
    # config.yaml's on purpose — see module docstring). Use the 2024/25
    # config's model block as the single source of truth for fitting.
    model_cfg = r2024["cfg"]
    seed = model_cfg["project"]["random_seed"]

    # Output location: this step's own reports dir, not tied to either
    # round's per-round outputs/by_round/<year>/reports/.
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # ---- Scenario 1: combined data, cluster-level 80/20 split -------------
    X_all = pd.concat([r2016["X"], r2024["X"]], ignore_index=True)
    y_all = pd.concat([r2016["y"], r2024["y"]], ignore_index=True)
    groups_all = pd.concat([r2016["groups"], r2024["groups"]], ignore_index=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups_all))

    m1 = _fit_and_evaluate(
        model_cfg, seed,
        X_all.iloc[train_idx], y_all.iloc[train_idx],
        X_all.iloc[test_idx], y_all.iloc[test_idx],
        logger, "scenario1",
    )
    _write_scenario_output(
        reports_dir, "scenario1", "Combined 2016+2024/25 — 80/20 split",
        "Both survey rounds pooled into a single analytic file, then split "
        "80/20 by cluster (never by row, to avoid leaking correlated births "
        "from the same enumeration area across the split). Train and test "
        "sets each contain a mix of both rounds.",
        m1, logger,
    )
    rows.append({"scenario": "scenario1", "description": "Combined 2016+2024/25, 80/20 cluster split", **m1})

    # ---- Scenario 2: train on 2024/25, test on 2016 ------------------------
    m2 = _fit_and_evaluate(
        model_cfg, seed,
        r2024["X"], r2024["y"], r2016["X"], r2016["y"],
        logger, "scenario2",
    )
    _write_scenario_output(
        reports_dir, "scenario2", "Train on 2024/25, test on 2016",
        "Model fit on the ENTIRE 2024/25 round, evaluated on the entire 2016 "
        "round as a fully out-of-round test set. Tests whether a model built "
        "on the most recent data generalizes backward to the earlier round.",
        m2, logger,
    )
    rows.append({"scenario": "scenario2", "description": "Train ALL 2024/25, test ALL 2016", **m2})

    # ---- Scenario 3: train on 2016, test on 2024/25 ------------------------
    m3 = _fit_and_evaluate(
        model_cfg, seed,
        r2016["X"], r2016["y"], r2024["X"], r2024["y"],
        logger, "scenario3",
    )
    _write_scenario_output(
        reports_dir, "scenario3", "Train on 2016, test on 2024/25",
        "Model fit on the ENTIRE 2016 round, evaluated on the entire 2024/25 "
        "round as a fully out-of-round test set. Tests whether a model built "
        "on the older round generalizes forward to the current round.",
        m3, logger,
    )
    rows.append({"scenario": "scenario3", "description": "Train ALL 2016, test ALL 2024/25", **m3})

    # ---- Scenario 4: within-2016, 80/20 (reuses step06's geo_holdout) -----
    tr_mask, te_mask = ~r2016["geo_holdout"], r2016["geo_holdout"]
    m4 = _fit_and_evaluate(
        model_cfg, seed,
        r2016["X"][tr_mask], r2016["y"][tr_mask],
        r2016["X"][te_mask], r2016["y"][te_mask],
        logger, "scenario4",
    )
    _write_scenario_output(
        reports_dir, "scenario4", "2016 only — train 80%, test 20%",
        "Trained and tested entirely within the 2016 round, using the "
        "cluster-level 80/20 geographic hold-out already assigned by "
        "step06_survey_design.py for the 2016 round (config_2016.yaml, "
        "validation.geographic_holdout_fraction=0.2).",
        m4, logger,
    )
    rows.append({"scenario": "scenario4", "description": "2016 only, 80% train / 20% test", **m4})

    # ---- Scenario 5: within-2024/25, 80/20 (reuses step06's geo_holdout) --
    tr_mask, te_mask = ~r2024["geo_holdout"], r2024["geo_holdout"]
    m5 = _fit_and_evaluate(
        model_cfg, seed,
        r2024["X"][tr_mask], r2024["y"][tr_mask],
        r2024["X"][te_mask], r2024["y"][te_mask],
        logger, "scenario5",
    )
    _write_scenario_output(
        reports_dir, "scenario5", "2024/25 only — train 80%, test 20%",
        "Trained and tested entirely within the 2024/25 round, using the "
        "cluster-level 80/20 geographic hold-out already assigned by "
        "step06_survey_design.py for the 2024/25 round (config.yaml, "
        "validation.geographic_holdout_fraction=0.2). Equivalent to the "
        "single-round result already reported in "
        "outputs/tables/holdout_metrics.json.",
        m5, logger,
    )
    rows.append({"scenario": "scenario5", "description": "2024/25 only, 80% train / 20% test", **m5})

    comparison = pd.DataFrame(rows)
    return comparison


def main():
    cfg = load_config()  # 2024/25 default config, used only for logging path
    logger = setup_logger("step12_scenario_comparison", cfg)

    logger.info("=== Step 12: 2016-vs-2024/25 scenario comparison ===")

    comparison = run_scenarios(logger)

    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "scenario_comparison.csv"
    comparison.to_csv(csv_path, index=False)
    logger.info("Wrote comparison table -> %s", csv_path)

    md_cols = ["scenario", "description", "n_train", "n_test",
               "train_prevalence", "test_prevalence",
               "roc_auc", "average_precision", "precision", "recall", "f1", "brier_score"]
    md_lines = ["# Scenario comparison: 2016 vs 2024/25 EDHS", "",
                "All 5 scenarios use the SAME model (algorithm + hyperparameters, "
                "config.model) so the comparison isolates the effect of which "
                "data the model trains/tests on. See each `outputs/reports/scenarioN/` "
                "folder for the full write-up.", "",
                "| " + " | ".join(md_cols) + " |",
                "|" + "---|" * len(md_cols)]
    for _, r in comparison.iterrows():
        vals = []
        for c in md_cols:
            v = r[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else str(v))
        md_lines.append("| " + " | ".join(vals) + " |")
    md_lines += [
        "", "## Caveats",
        "- All metrics are model-derived predictive-performance estimates from "
        "cross-sectional, observational data — not causal effect estimates "
        "(protocol section 9).",
        "- region_stratum is encoded with a single canonical mapping shared by "
        "both rounds (see step12_scenario_comparison.py docstring) — NOT the "
        "per-round codes in each round's own outputs/by_round/<year>/tables/"
        "region_stratum_code_map.json.",
        "- Scenarios 2 and 3 test generalization ACROSS survey rounds; a "
        "performance drop relative to scenarios 4/5 reflects genuine "
        "distribution shift between rounds (population change, question/"
        "coding differences), not model quality alone.",
        "- Verify config_2016.yaml's `region_labels` and the b19<=35 "
        "eligibility-window assumption against the actual 2016 .DO/.MAP "
        "files before treating these numbers as final (see that config's "
        "inline `# 2016:` / VERIFY comments).",
    ]
    md_path = reports_dir / "scenario_comparison.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    logger.info("Wrote comparison report -> %s", md_path)

    logger.info("Step 12 complete.")


if __name__ == "__main__":
    main()
