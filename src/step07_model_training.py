"""
Step 7 — Model development (protocol section 6.3)

Algorithm: gradient-boosted trees (XGBoost or LightGBM, config-selectable).
Validation:
  - cluster-based k-fold CV (folds from step06) for hyperparameter-stable
    performance estimation
  - a held-out geographic split (train on a subset of clusters, test on
    entirely unseen clusters) as the PRIMARY generalization test
The final model is refit on all non-geo-holdout data and saved for use by
the SHAP (step08) and DiCE (step09) modules.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import get_path, load_config, read_parquet, setup_logger


def get_model(cfg: dict, seed: int):
    algo = cfg["model"]["algorithm"]
    params = dict(cfg["model"]["params"])
    if algo == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(**params, random_state=seed, use_label_encoder=False)
    elif algo == "lightgbm":
        import lightgbm as lgb
        params.pop("eval_metric", None)
        return lgb.LGBMClassifier(**params, random_state=seed)
    else:
        raise ValueError(f"Unknown model.algorithm '{algo}' — expected 'xgboost' or 'lightgbm'.")


def prep_xy(df: pd.DataFrame, cfg: dict, feature_cols: list[str]):
    X = df[feature_cols].copy()
    # Tree ensembles handle NaN natively in xgboost/lightgbm; categoricals
    # coming through as DHS integer codes are left as numeric on purpose
    # (raw, finely-grained codes per protocol section 5 rationale) except
    # region_stratum, which is genuinely categorical text.
    if "region_stratum" in X.columns:
        X["region_stratum"] = X["region_stratum"].astype("category").cat.codes
    y = df[cfg["outcome"]["name"]].astype(int)
    return X, y


def cluster_cv_scores(df: pd.DataFrame, cfg: dict, feature_cols: list[str], logger) -> dict:
    seed = cfg["project"]["random_seed"]
    n_folds = cfg["validation"]["cv_folds"]
    aucs = []
    for fold in range(n_folds):
        train = df[df["cv_fold"] != fold]
        test = df[df["cv_fold"] == fold]
        Xtr, ytr = prep_xy(train, cfg, feature_cols)
        Xte, yte = prep_xy(test, cfg, feature_cols)

        model = get_model(cfg, seed)
        model.fit(Xtr, ytr)
        preds = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, preds)
        aucs.append(auc)
        logger.info("Cluster-CV fold %s/%s: AUC = %.4f (n_test=%s)", fold + 1, n_folds, auc, len(test))

    return {"cluster_cv_auc_mean": float(np.mean(aucs)), "cluster_cv_auc_folds": [float(a) for a in aucs]}


def fit_final_model(df: pd.DataFrame, cfg: dict, feature_cols: list[str], logger):
    """Fit on everything EXCEPT the geographic hold-out (train pool)."""
    seed = cfg["project"]["random_seed"]
    train_pool = df[~df["geo_holdout"]]
    X, y = prep_xy(train_pool, cfg, feature_cols)

    model = get_model(cfg, seed)
    model.fit(X, y)
    logger.info("Fit final model on train pool: %s rows (geographic hold-out excluded).", len(train_pool))
    return model


def main():
    cfg = load_config()
    logger = setup_logger("step07_model_training", cfg)
    processed = get_path(cfg, "processed_dir")
    models_dir = get_path(cfg, "models_dir")
    tables_dir = get_path(cfg, "tables_dir")

    logger.info("=== Step 7: Model training ===")
    df = read_parquet(processed / "model_matrix.parquet")

    from step06_survey_design import get_model_feature_list
    feature_cols = get_model_feature_list(cfg)

    cv_results = cluster_cv_scores(df, cfg, feature_cols, logger)
    logger.info("Cluster-based CV mean AUC: %.4f", cv_results["cluster_cv_auc_mean"])

    model = fit_final_model(df, cfg, feature_cols, logger)

    import joblib
    model_path = models_dir / f"sba_{cfg['model']['algorithm']}.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    logger.info("Saved trained model -> %s", model_path)

    with open(tables_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    logger.info("Step 7 complete.")


if __name__ == "__main__":
    main()
