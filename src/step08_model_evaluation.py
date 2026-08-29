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
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils import get_path, load_config, read_parquet, setup_logger


def prep_xy(df: pd.DataFrame, cfg: dict, feature_cols: list[str]):
    X = df[feature_cols].copy()
    if "region_stratum" in X.columns:
        X["region_stratum"] = X["region_stratum"].astype("category").cat.codes
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
    out = figures_dir / "calibration_plot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved calibration plot -> %s", out)


def equity_check(df_holdout: pd.DataFrame, proba: np.ndarray, y: pd.Series,
                  cfg: dict, tables_dir, logger) -> pd.DataFrame:
    """Performance + predicted-probability distribution by wealth quintile (v190)."""
    wealth_var = cfg["validation"]["equity_check_var"]
    tmp = df_holdout.reset_index(drop=True).copy()
    tmp["_proba"] = proba
    tmp["_y"] = y.reset_index(drop=True)

    rows = []
    for q, grp in tmp.groupby(wealth_var):
        if grp["_y"].nunique() < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(grp["_y"], grp["_proba"])
        rows.append({
            "wealth_quintile": q,
            "n": len(grp),
            "observed_sba_rate": grp["_y"].mean(),
            "mean_predicted_proba": grp["_proba"].mean(),
            "auc": auc,
        })
    equity_df = pd.DataFrame(rows).sort_values("wealth_quintile")
    out = tables_dir / "equity_check_by_wealth.csv"
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

    holdout = df[df["geo_holdout"]]
    equity_check(holdout, proba, y, cfg, tables_dir, logger)

    logger.info("Step 8 complete.")


if __name__ == "__main__":
    main()
