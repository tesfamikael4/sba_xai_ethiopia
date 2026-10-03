"""
Step 8 — Model validation on the geographic hold-out (protocol section 6.3)

Reports AUC-ROC, calibration (calibration plot + Brier score), precision,
recall, F1 on the held-out-cluster test set, PLUS an equity check: model
performance and predicted-probability distributions compared across wealth
quintiles (v190) to screen for differential performance.
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils import get_path, load_config, read_parquet, setup_logger


def prep_xy(df: pd.DataFrame, cfg: dict, feature_cols: list[str]):
    X = df[feature_cols].copy()
    # region_stratum arrives already integer-coded from step06 (single
    # source of truth) — do NOT re-derive codes on this subset.
    y = df[cfg["outcome"]["name"]].astype(int)
    return X, y


def evaluate_holdout(model, df: pd.DataFrame, cfg: dict, feature_cols: list[str], logger) -> dict:
    holdout = df[df["geo_holdout"]]
    X, y = prep_xy(holdout, cfg, feature_cols)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "n_holdout": int(len(holdout)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, proba)),
    }
    for k, v in metrics.items():
        logger.info("Geographic hold-out %s: %s", k, v)
    return metrics, proba, y


def plot_calibration(y, proba, figures_dir, logger):
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed SBA fraction")
    ax.set_title("Calibration — geographic hold-out")
    ax.legend()
    fig.tight_layout()
    out = figures_dir / "Fig 1.tiff"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved calibration plot -> %s", out)


def plot_confusion_matrix(y, proba, figures_dir, logger, threshold: float = 0.5):
    """Confusion matrix on the geographic hold-out at the same 0.5 threshold
    used for precision/recall/F1 elsewhere in this step, shown both as raw
    counts and row-normalized (recall-per-class) percentages side by side —
    counts alone hide how the class-imbalanced hold-out affects the
    minority class's recall, and percentages alone hide how small that
    class's n actually is."""
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ["No SBA (0)", "SBA (1)"]

    disp1 = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp1.plot(ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title(f"Counts (threshold={threshold})")

    disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=labels)
    disp2.plot(ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f")
    axes[1].set_title("Row-normalized (recall per class)")

    fig.suptitle("Confusion matrix — geographic hold-out (2024/25)")
    fig.tight_layout()
    out = figures_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved confusion matrix -> %s", out)


def plot_roc_curve(y, proba, figures_dir, logger):
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — geographic hold-out")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = figures_dir / "roc_curve.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved ROC curve -> %s", out)


def plot_pr_curve(y, proba, figures_dir, logger):
    precision, recall, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    prevalence = float(np.mean(y))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, linewidth=2, label=f"Model (AP = {ap:.3f})")
    ax.axhline(prevalence, linestyle="--", color="gray",
               label=f"No-skill baseline (prevalence = {prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve — geographic hold-out")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = figures_dir / "precision_recall_curve.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved precision-recall curve -> %s", out)


def equity_check(df_holdout: pd.DataFrame, proba: np.ndarray, y: pd.Series,
                  cfg: dict, tables_dir, logger, group_var: str, group_label: str) -> pd.DataFrame:
    """Performance + predicted-probability distribution by subgroup, plus
    TPR/FPR at the 0.5 threshold — a lightweight step toward an
    equalized-odds-style fairness check (large TPR or FPR gaps between
    subgroups indicate the model errs differently by group, not just that
    overall accuracy differs). This is a screening diagnostic on the
    geographic hold-out only (already small — see each subgroup's own n
    before reading too much into any single group's AUC/TPR/FPR)."""
    tmp = df_holdout.reset_index(drop=True).copy()
    tmp["_proba"] = proba
    tmp["_y"] = y.reset_index(drop=True)
    tmp["_pred"] = (tmp["_proba"] >= 0.5).astype(int)

    rows = []
    for q, grp in tmp.groupby(group_var):
        if grp["_y"].nunique() < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(grp["_y"], grp["_proba"])

        pos = grp[grp["_y"] == 1]
        neg = grp[grp["_y"] == 0]
        tpr = pos["_pred"].mean() if len(pos) else np.nan  # sensitivity / recall
        fpr = neg["_pred"].mean() if len(neg) else np.nan  # 1 - specificity

        rows.append({
            group_label: q,
            "n": len(grp),
            "observed_sba_rate": grp["_y"].mean(),
            "mean_predicted_proba": grp["_proba"].mean(),
            "auc": auc,
            "tpr": tpr,
            "fpr": fpr,
        })
    equity_df = pd.DataFrame(rows).sort_values(group_label)

    if equity_df["tpr"].notna().sum() > 1:
        tpr_gap = equity_df["tpr"].max() - equity_df["tpr"].min()
        fpr_gap = equity_df["fpr"].max() - equity_df["fpr"].min()
        logger.info(
            "%s equity check: TPR gap = %.3f, FPR gap = %.3f across groups "
            "(large gaps = model errs differently by %s, an equalized-odds "
            "concern, not just an overall-accuracy difference; interpret "
            "cautiously given small per-group n on the hold-out).",
            group_label, tpr_gap, fpr_gap, group_label,
        )

    out = tables_dir / f"equity_check_by_{group_label}.csv"
    equity_df.to_csv(out, index=False)
    logger.info("Saved equity check table -> %s", out)
    return equity_df


def main():
    cfg = load_config()
    logger = setup_logger("step08_model_evaluation", cfg)
    processed = get_path(cfg, "processed_dir")
    models_dir = get_path(cfg, "models_dir")
    figures_dir = get_path(cfg, "figures_dir")
    tables_dir = get_path(cfg, "tables_dir")

    logger.info("=== Step 8: Model evaluation ===")
    df = read_parquet(processed / "model_matrix.parquet")

    model_path = models_dir / f"sba_{cfg['model']['algorithm']}.joblib"
    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    metrics, proba, y = evaluate_holdout(model, df, cfg, feature_cols, logger)

    with open(tables_dir / "holdout_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_calibration(y, proba, figures_dir, logger)
    plot_confusion_matrix(y, proba, figures_dir, logger)
    plot_roc_curve(y, proba, figures_dir, logger)
    plot_pr_curve(y, proba, figures_dir, logger)

    holdout = df[df["geo_holdout"]]
    wealth_var = cfg["validation"]["equity_check_var"]
    equity_check(holdout, proba, y, cfg, tables_dir, logger, wealth_var, "wealth_quintile")
    # Rural/urban (v025) as a second equity dimension, per protocol section
    # 6.3's equity-check requirement — wealth alone can mask a residence
    # effect that isn't fully captured by wealth quintile (e.g. a rural
    # richest-quintile household still facing rural facility-access
    # barriers a same-quintile urban household does not).
    equity_check(holdout, proba, y, cfg, tables_dir, logger, "v025", "residence")

    logger.info("Step 8 complete.")


if __name__ == "__main__":
    main()
