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

import json
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
from variable_labels import get_variable_label, label_axis_list


def cast_model_matrix_to_float(X: pd.DataFrame) -> pd.DataFrame:
    """sklearn PartialDependenceDisplay requires float dtypes, not int."""
    X = X.copy()
    for col in X.columns:
        if X[col].dtype.kind in 'iub':  # integer, unsigned int, boolean
            X[col] = X[col].astype('float64')
    return X


def prep_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # region_stratum arrives already integer-coded from step06 (single
    # source of truth) — do NOT re-derive codes on this subset.
    return df[feature_cols].copy()


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
    fig = plt.figure(figsize=(8, 8))
    # feature_names overrides only the y-axis labels shown; shap_values/X
    # themselves are untouched, so this can't affect what's computed —
    # display-only, same as everywhere else labels are applied in this step.
    print("Plotting global SHAP summary (beeswarm) — this can take a while...", X.columns.tolist())
    shap.summary_plot(shap_values, X, feature_names=label_axis_list(list(X.columns)), show=False)
    out = figures_dir / "shap_global_summary.tiff"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved global SHAP summary plot -> %s", out)


def plot_shap_bar_importance(importance_df: pd.DataFrame, figures_dir, logger, top_n: int = 15):
    """Horizontal bar chart of weighted mean |SHAP| — a plainer, easier-to-
    read companion to the beeswarm summary plot above. The beeswarm shows
    distribution + direction per feature; this shows magnitude ranking at a
    glance, which is what most readers actually scan for first."""
    top = importance_df.head(top_n).iloc[::-1]  # reverse so #1 plots at top
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(top))))
    ax.barh(label_axis_list(top["feature"].tolist()), top["weighted_mean_abs_shap"], color="#2b6cb0")
    ax.set_xlabel("Weighted mean |SHAP value|")
    ax.set_title(f"Global feature importance — top {len(top)} (SHAP)")
    fig.tight_layout()
    out = figures_dir / "shap_bar_importance.tiff"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved SHAP bar importance plot -> %s", out)


def plot_native_feature_importance(model, feature_cols: list[str], figures_dir, logger, top_n: int = 15):
    """Model-native (gain-based) feature importance, as a cross-check
    against the SHAP ranking above — SHAP and gain-based importance can
    legitimately disagree (gain rewards features that produce large, rare
    splits; SHAP rewards features that shift predictions consistently
    across many rows), so showing both lets a reader see whether the two
    metrics agree on what matters, rather than reporting only one and
    implying it's the only lens available."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        logger.warning("Model has no feature_importances_ — skipping native importance plot.")
        return

    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(imp_df))))
    ax.barh(label_axis_list(imp_df["feature"].tolist()), imp_df["importance"], color="#c05621")
    ax.set_xlabel("Model-native importance (gain)")
    ax.set_title(f"Global feature importance — top {len(imp_df)} (XGBoost gain)")
    fig.tight_layout()
    out = figures_dir / "native_feature_importance.tiff"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved native (gain-based) feature importance plot -> %s", out)


def region_interaction_values(model, X: pd.DataFrame, cfg: dict, logger,
                               stratum_labels: dict[int, str] | None = None) -> dict[str, pd.DataFrame]:
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

        if stratum_labels and stratum_var == "region_stratum":
            summary = summary.rename(index=stratum_labels)
            summary.index.name = stratum_var

        results[f"{stratum_var}_x_{predictor}"] = summary
        logger.info("Region interaction summary for %s x %s:\n%s", stratum_var, predictor, summary)

    return results


def partial_dependence_by_stratum(model, X: pd.DataFrame, cfg: dict, figures_dir, logger,
                                   stratum_labels: dict[int, str] | None = None):
    """PD profiles for ANC (m14) and distance (v467d) shown separately per
    region_stratum, as specified in protocol section 6.2.

    FIXES APPLIED:
    - Cast all integer features to float64 before sklearn PDP call
    - Create fresh figure/axes per feature to avoid axis reuse bugs
    - Handle per-stratum plotting robustly with try/except per stratum
    """
    from sklearn.inspection import PartialDependenceDisplay

    if "region_stratum" not in X.columns:
        return

    stratum_labels = stratum_labels or {}

    # CRITICAL FIX: Cast to float before ANY sklearn call
    X = cast_model_matrix_to_float(X.copy())

    pd_vars = cfg["xai"]["shap"]["partial_dependence_vars"]
    strata = sorted(X["region_stratum"].dropna().unique())
    n_strata = len(strata)

    for var in pd_vars:
        if var not in X.columns:
            continue

        # CRITICAL FIX: Create one figure with one subplot per stratum
        fig, axes = plt.subplots(1, n_strata, figsize=(5 * n_strata, 4), sharey=True)
        if n_strata == 1:
            axes = [axes]

        for ax, stratum_code in zip(axes, strata):
            mask = X["region_stratum"] == stratum_code
            n_sub = mask.sum()
            label = stratum_labels.get(int(stratum_code), f"Stratum {stratum_code}")
            if n_sub < 30:
                ax.text(0.5, 0.5, f"{label}\nn={n_sub} (too small)",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(label)
                continue

            try:
                # CRITICAL FIX: Pass the specific ax, never reuse across strata
                PartialDependenceDisplay.from_estimator(
                    model,
                    X[mask],
                    [var],
                    kind="average",
                    ax=ax,
                    line_kw={"linewidth": 2},
                )
                ax.set_title(f"{label} (n={n_sub})")
                ax.set_xlabel(label_axis_list([var])[0])
            except Exception as e:
                ax.text(0.5, 0.5, f"PDP failed\n{str(e)[:50]}",
                        ha='center', va='center', transform=ax.transAxes)
                logger.warning("PD plot failed for %s stratum=%s: %s", var, stratum_code, e)

        axes[0].set_ylabel("Partial dependence")
        fig.suptitle(f"Partial Dependence: {label_axis_list([var])[0]} by Region Stratum", y=1.02)
        fig.tight_layout()
        out = figures_dir / f"pd_{var}_by_stratum.tiff"
        fig.savefig(out, dpi=300, bbox_inches="")
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

    stratum_labels = None
    code_map_path = tables_dir / "region_stratum_code_map.json"
    if code_map_path.exists():
        with open(code_map_path) as f:
            label_to_code = json.load(f)
        stratum_labels = {code: label for label, code in label_to_code.items()}

    shap_values = compute_global_shap(model, X, logger)

    importance_df = weighted_mean_abs_shap(shap_values, weights, feature_cols)
    # Codes are shown once, in outputs/.../tables/variable_codebook.csv
    # (written by step06) — every other table/figure, including this one,
    # shows the label only. plot_shap_bar_importance below still receives
    # the CODE-keyed importance_df (it converts to labels itself, purely
    # for the plot) — only the CSV written here is label-only.
    importance_display = importance_df.copy()
    importance_display["feature"] = importance_display["feature"].map(get_variable_label)
    importance_display.to_csv(tables_dir / "shap_weighted_global_importance.csv", index=False)
    logger.info("Weighted global SHAP importance (top 10):\n%s", importance_df.head(10))

    plot_global_summary(shap_values, X, figures_dir, logger)
    plot_shap_bar_importance(importance_df, figures_dir, logger)
    plot_native_feature_importance(model, feature_cols, figures_dir, logger)

    interactions = region_interaction_values(model, X, cfg, logger, stratum_labels)
    for name, summary in interactions.items():
        summary.to_csv(tables_dir / f"shap_interaction_{name}.csv")

    partial_dependence_by_stratum(model, X, cfg, figures_dir, logger, stratum_labels)

    logger.info("Step 9 complete.")


if __name__ == "__main__":
    main()
