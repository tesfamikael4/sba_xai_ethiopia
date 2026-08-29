"""
Step 6 — Complex survey design + model matrix assembly (protocol section 6.1)

- Sample weights (v005), PSU/cluster (v021), strata (v022) are retained
  through data management but the model itself is fit UNWEIGHTED, per
  standard practice for tree-based ML classifiers. Weights are only used
  for population-level estimates (prevalence, weighted SHAP summaries).
- Cross-validation folds are built BY CLUSTER (v021), never by row, so
  correlated births in the same enumeration area never span train/test.
- A geographic hold-out (entirely unseen clusters) is carved out as the
  PRIMARY generalization test (protocol section 6.3), separate from the
  cluster-based k-fold CV used for hyperparameter tuning.

This step also assembles the final model feature matrix: ONLY the "raw"
predictors listed in config.predictors.*.raw plus region_stratum — the
rr_ reporting recodes from step05 are deliberately excluded here.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from utils import get_path, load_config, read_parquet, save_parquet, setup_logger


def get_model_feature_list(cfg: dict) -> list[str]:
    feats = []
    for domain, spec in cfg["predictors"].items():
        if domain == "excluded":
            continue
        feats.extend(spec.get("raw", []))
        feats.extend(spec.get("derived", []))
    # de-dup while preserving order (m14/v467d appear under health_system_access
    # and are sourced from BR/IR consistently, so no collision expected)
    seen = set()
    ordered = [f for f in feats if not (f in seen or seen.add(f))]
    return ordered


def attach_weight(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sd = cfg["survey_design"]
    df = df.copy()
    df["weight"] = df[sd["weight_var"]] / sd["weight_divisor"]
    return df


def assign_cv_folds(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    df = df.copy()
    group_var = cfg["validation"]["cv_group_var"]
    n_folds = cfg["validation"]["cv_folds"]

    gkf = GroupKFold(n_splits=n_folds)
    df["cv_fold"] = -1
    for fold_idx, (_, test_idx) in enumerate(gkf.split(df, groups=df[group_var])):
        df.iloc[test_idx, df.columns.get_loc("cv_fold")] = fold_idx

    logger.info("Assigned %s cluster-based CV folds (grouped by %s).", n_folds, group_var)
    return df


def assign_geographic_holdout(df: pd.DataFrame, cfg: dict, logger, seed: int) -> pd.DataFrame:
    df = df.copy()
    cluster_var = cfg["validation"]["cv_group_var"]
    frac = cfg["validation"]["geographic_holdout_fraction"]

    clusters = df[cluster_var].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    n_holdout = max(1, int(round(len(clusters) * frac)))
    holdout_clusters = set(clusters[:n_holdout])

    df["geo_holdout"] = df[cluster_var].isin(holdout_clusters)
    logger.info("Geographic hold-out: %s/%s clusters (%.1f%%), %s rows held out.",
                n_holdout, len(clusters), 100 * frac, df["geo_holdout"].sum())
    return df


def main():
    cfg = load_config()
    logger = setup_logger("step06_survey_design", cfg)
    interim = get_path(cfg, "interim_dir")
    processed = get_path(cfg, "processed_dir")

    logger.info("=== Step 6: Survey design + model matrix assembly ===")
    df = read_parquet(interim / "featured.parquet")

    df = attach_weight(df, cfg)
    df = assign_cv_folds(df, cfg, logger)
    df = assign_geographic_holdout(df, cfg, logger, seed=cfg["project"]["random_seed"])

    feature_list = get_model_feature_list(cfg)
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        raise KeyError(f"Configured model features not found in data: {missing}")

    id_cols = ["caseid", "bidx", cfg["merge"]["cluster_var"], "weight",
               "cv_fold", "geo_holdout", cfg["outcome"]["name"]]
    id_cols = [c for c in id_cols if c in df.columns]

    model_df = df[id_cols + feature_list].copy()
    logger.info("Assembled model matrix: %s rows x %s features "
                "(reporting-only rr_ columns excluded by construction).",
                len(model_df), len(feature_list))

    save_parquet(model_df, processed / "model_matrix.parquet", logger)
    # Keep the full featured frame (incl. rr_ recodes) around for reporting/step09
    save_parquet(df, processed / "analytic_full.parquet", logger)

    logger.info("Step 6 complete. Model features: %s", feature_list)


if __name__ == "__main__":
    main()
