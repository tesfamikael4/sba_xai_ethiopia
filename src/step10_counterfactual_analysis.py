"""
Step 10 — Constrained counterfactual explanations, DiCE (protocol section 7.2)

SUPPLEMENTARY SENSITIVITY ANALYSIS, NOT a primary study contribution: DiCE
generates plausible in-distribution feature combinations, not causal effect
estimates. A small set of illustrative counterfactuals (3 per
region_stratum) is generated for the MoH policy brief, framed explicitly
as scenario exploration.

Mutability structure (protocol 7.2 table):
  - Modifiable (individual-level): ANC visits (m14), media exposure
    (v157-v159), insurance (v481) -> "what if this woman had X?" framing
  - Semi-modifiable (system-level): distance-is-a-problem (v467d) -> framed
    as service-deployment scenarios, NOT individual choices
  - Immutable: age, education, wealth, urban/rural, region -> held fixed as
    conditioning context in all counterfactual generation

A validity-check step reports the proportion of generated counterfactuals
falling within the observed empirical covariate distribution.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

from utils import get_path, load_config, read_parquet, setup_logger


def prep_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    if "region_stratum" in X.columns:
        X["region_stratum"] = X["region_stratum"].astype("category").cat.codes
    return X


def build_dice_data(train_pool: pd.DataFrame, feature_cols: list[str], outcome_name: str, cfg: dict):
    import dice_ml

    continuous = [f for f in feature_cols
                  if f not in ("region_stratum",) and train_pool[f].nunique() > 10]

    d = dice_ml.Data(
        dataframe=train_pool[feature_cols + [outcome_name]],
        continuous_features=continuous,
        outcome_name=outcome_name,
    )
    return d


def get_dice_explainer(model, dice_data, feature_cols):
    import dice_ml
    m = dice_ml.Model(model=model, backend="sklearn")
    return dice_ml.Dice(dice_data, m, method="random")


def counterfactuals_for_stratum(exp, query_instances: pd.DataFrame, cfg: dict, logger):
    mut = cfg["xai"]["counterfactuals"]["mutability"]
    n_per = cfg["xai"]["counterfactuals"]["n_per_stratum"]
    features_to_vary = mut["modifiable"] + mut["semi_modifiable"]

    try:
        cf = exp.generate_counterfactuals(
            query_instances,
            total_CFs=n_per,
            desired_class="opposite",
            features_to_vary=features_to_vary,
        )
        return cf
    except Exception as e:
        logger.warning("DiCE generation failed for this query batch: %s", e)
        return None


def validity_check(cf_df: pd.DataFrame, empirical: pd.DataFrame, feature_cols: list[str], logger) -> float:
    """Fraction of generated counterfactual rows whose values fall within
    the observed empirical min/max for every feature (a coarse but
    transparent in-distribution check, per protocol 7.2)."""
    if cf_df is None or len(cf_df) == 0:
        return float("nan")
    valid = np.ones(len(cf_df), dtype=bool)
    for f in feature_cols:
        if f not in cf_df.columns or f not in empirical.columns:
            continue
        lo, hi = empirical[f].min(), empirical[f].max()
        valid &= cf_df[f].between(lo, hi)
    rate = float(valid.mean())
    logger.info("Counterfactual validity (within empirical range): %.1f%%", 100 * rate)
    return rate


def main():
    cfg = load_config()
    logger = setup_logger("step10_counterfactual_analysis", cfg)
    processed = get_path(cfg, "processed_dir")
    models_dir = get_path(cfg, "models_dir")
    tables_dir = get_path(cfg, "tables_dir")
    reports_dir = get_path(cfg, "reports_dir")

    logger.info("=== Step 10: Counterfactual analysis (DiCE, supplementary) ===")
    df = read_parquet(processed / "model_matrix.parquet")

    model_path = models_dir / f"sba_{cfg['model']['algorithm']}.joblib"
    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    outcome_name = cfg["outcome"]["name"]

    train_pool = df[~df["geo_holdout"]].copy()
    X_full = prep_X(train_pool, feature_cols)
    train_pool_model = pd.concat(
        [X_full, train_pool[outcome_name].astype(int).reset_index(drop=True)], axis=1
    )

    dice_data = build_dice_data(train_pool_model, feature_cols, outcome_name, cfg)
    exp = get_dice_explainer(model, dice_data, feature_cols)

    all_cfs = []
    validity_rows = []
    strata = sorted(train_pool["region_stratum"].dropna().unique())

    for stratum in strata:
        stratum_rows = train_pool[
            (train_pool["region_stratum"] == stratum) & (train_pool[outcome_name] == 0)
        ]
        if stratum_rows.empty:
            logger.warning("No SBA=0 rows for stratum '%s' — skipping counterfactuals.", stratum)
            continue

        sample = stratum_rows.sample(
            n=min(3, len(stratum_rows)), random_state=cfg["project"]["random_seed"]
        )
        query = prep_X(sample, feature_cols)

        cf_result = counterfactuals_for_stratum(exp, query, cfg, logger)
        if cf_result is None:
            continue

        for cf_example in cf_result.cf_examples_list:
            if cf_example.final_cfs_df is not None:
                cf_df = cf_example.final_cfs_df.copy()
                cf_df["region_stratum_label"] = stratum
                all_cfs.append(cf_df)

    if all_cfs:
        combined = pd.concat(all_cfs, ignore_index=True)
        combined.to_csv(tables_dir / "dice_counterfactuals.csv", index=False)
        logger.info("Saved %s counterfactual rows -> dice_counterfactuals.csv", len(combined))

        rate = validity_check(combined, X_full, feature_cols, logger)
        with open(tables_dir / "dice_validity_check.json", "w") as f:
            json.dump({"in_distribution_rate": rate,
                       "n_counterfactuals": len(combined)}, f, indent=2)
    else:
        logger.warning("No counterfactuals were generated — check DiCE installation "
                        "and that SBA=0 rows exist per stratum.")

    with open(reports_dir / "counterfactual_framing_note.md", "w") as f:
        f.write(
            "# Counterfactual framing (protocol section 7.2)\n\n"
            "These counterfactuals are a **supplementary sensitivity analysis**, "
            "not a primary study contribution or a causal effect estimate.\n\n"
            "- Modifiable (individual-level): ANC visits, media exposure, "
            "insurance -> framed as \"what if this woman had X?\"\n"
            "- Semi-modifiable (system-level): distance-is-a-problem -> framed as "
            "service-deployment scenarios, not individual choices.\n"
            "- Immutable: age, education, wealth, urban/rural, region -> held "
            "fixed as conditioning context.\n\n"
            f"{cfg['provenance']['causal_disclaimer']}\n"
        )

    logger.info("Step 10 complete.")


if __name__ == "__main__":
    main()
