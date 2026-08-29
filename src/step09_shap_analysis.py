"""
Step 9 — Explainable AI: SHAP (protocol section 7.1)

Global and interaction SHAP values are computed on the final national
model. region_stratum x modifiable-predictor interaction values are the
STUDY'S PRIMARY EXPLANATORY CONTRIBUTION (protocol section 6.2) — they
characterize regional effect modification without fitting statistically
underpowered per-region models. Population-level SHAP summaries are
weighted using v005 (protocol section 6.1).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_path, load_config, read_parquet, setup_logger


def prep_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    if "region_stratum" in X.columns:
        X["region_stratum"] = X["region_stratum"].astype("category").cat.codes
    return X


def compute_global_shap(model, X: pd.DataFrame, logger):
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    logger.info("Computed SHAP values for %s rows x %s features.", *X.shape)
    return shap_values


def weighted_mean_abs_shap(shap_values, weights: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    abs_vals = np.abs(shap_values.values)
    w = weights.reshape(-1, 1)
    weighted_importance = (abs_vals * w).sum(axis=0) / w.sum()
    return (
        pd.DataFrame({"feature": feature_names, "weighted_mean_abs_shap": weighted_importance})
        .sort_values("weighted_mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def plot_global_summary(shap_values, X, figures_dir, logger):
    import shap
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, show=False)
    out = figures_dir / "shap_global_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved global SHAP summary plot -> %s", out)


def region_interaction_values(model, X: pd.DataFrame, cfg: dict, logger) -> dict[str, pd.DataFrame]:
    """
    For each region_stratum x predictor pair configured under
    xai.shap.interaction_pairs, split SHAP values by stratum and summarize
    the modifiable predictor's contribution within each — this is the
    lightweight, non-underpowered alternative to per-region models.
    """
    import shap
    explainer = shap.TreeExplainer(model)

    results = {}
    region_col_idx = list(X.columns).index("region_stratum") if "region_stratum" in X.columns else None

    interaction_values = None
    try:
        interaction_values = explainer.shap_interaction_values(X)
    except Exception as e:  # not all model/explainer combos support this cheaply
        logger.warning("shap_interaction_values unavailable/expensive for this "
                        "model (%s) — falling back to stratified SHAP summaries.", e)

    for stratum_var, predictor in cfg["xai"]["shap"]["interaction_pairs"]:
        if stratum_var not in X.columns or predictor not in X.columns:
            logger.warning("Skipping interaction pair (%s, %s) — column not in model matrix.",
                            stratum_var, predictor)
            continue

        pred_idx = list(X.columns).index(predictor)
        stratum_idx = list(X.columns).index(stratum_var)

        if interaction_values is not None:
            pairwise = interaction_values[:, pred_idx, stratum_idx]
            df = pd.DataFrame({stratum_var: X[stratum_var].values, "shap_interaction": pairwise})
            summary = df.groupby(stratum_var)["shap_interaction"].agg(["mean", "std", "count"])
        else:
            sv = explainer(X)
            df = pd.DataFrame({stratum_var: X[stratum_var].values,
                                "shap_value": sv.values[:, pred_idx]})
            summary = df.groupby(stratum_var)["shap_value"].agg(["mean", "std", "count"])

        results[f"{stratum_var}_x_{predictor}"] = summary
        logger.info("Region interaction summary for %s x %s:\n%s", stratum_var, predictor, summary)

    return results


def partial_dependence_by_stratum(model, X: pd.DataFrame, cfg: dict, figures_dir, logger):
    """PD profiles for ANC (m14) and distance (v467d) shown separately per
    region_stratum, as specified in protocol section 6.2."""
    from sklearn.inspection import PartialDependenceDisplay

    if "region_stratum" not in X.columns:
        return
    pd_vars = cfg["xai"]["shap"]["partial_dependence_vars"]
    for var in pd_vars:
        if var not in X.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for stratum_code in sorted(X["region_stratum"].unique()):
            mask = X["region_stratum"] == stratum_code
            if mask.sum() < 30:
                continue
            try:
                PartialDependenceDisplay.from_estimator(
                    model, X[mask], [var], ax=ax, kind="average",
                    line_kw={"label": f"stratum={stratum_code}"},
                )
            except Exception as e:
                logger.warning("PD plot failed for %s stratum=%s: %s", var, stratum_code, e)
        ax.set_title(f"Partial dependence: {var} by region_stratum")
        ax.legend(fontsize=7)
        fig.tight_layout()
        out = figures_dir / f"pd_{var}_by_stratum.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Saved PD plot -> %s", out)


def main():
    cfg = load_config()
    logger = setup_logger("step09_shap_analysis", cfg)
    processed = get_path(cfg, "processed_dir")
    models_dir = get_path(cfg, "models_dir")
    figures_dir = get_path(cfg, "figures_dir")
    tables_dir = get_path(cfg, "tables_dir")

    logger.info("=== Step 9: SHAP analysis ===")
    df = read_parquet(processed / "model_matrix.parquet")

    model_path = models_dir / f"sba_{cfg['model']['algorithm']}.joblib"
    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    # Use the full non-holdout pool for explanation, consistent with the
    # model's training population.
    train_pool = df[~df["geo_holdout"]]
    X = prep_X(train_pool, feature_cols)
    weights = train_pool["weight"].values

    shap_values = compute_global_shap(model, X, logger)

    importance_df = weighted_mean_abs_shap(shap_values, weights, feature_cols)
    importance_df.to_csv(tables_dir / "shap_weighted_global_importance.csv", index=False)
    logger.info("Weighted global SHAP importance (top 10):\n%s", importance_df.head(10))

    plot_global_summary(shap_values, X, figures_dir, logger)

    interactions = region_interaction_values(model, X, cfg, logger)
    for name, summary in interactions.items():
        summary.to_csv(tables_dir / f"shap_interaction_{name}.csv")

    partial_dependence_by_stratum(model, X, cfg, figures_dir, logger)

    logger.info("Step 9 complete.")


if __name__ == "__main__":
    main()
