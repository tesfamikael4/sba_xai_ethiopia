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


def _smote_resample(cfg: dict, X: pd.DataFrame, y: pd.Series, seed: int, logger):
    """SMOTENC (not plain SMOTE) oversampling of the minority class, fit on
    the TRAINING data being passed in only — call sites here always pass a
    single fold's or the final train pool's data, never test/hold-out rows,
    so this can't leak.

    SMOTENC, not SMOTE: roughly half this feature set is categorical or an
    explicit sentinel code (region_stratum, v106, v025, v157-9, v481,
    v701/705/511 skip-pattern sentinels, v467a-d, v501, m10, v024) rather
    than continuous. Plain SMOTE linearly interpolates ALL features between
    neighbors — interpolating a categorical code or a -1 "not applicable"
    sentinel with a real category produces a fractional value (e.g. -0.3)
    that means nothing and actively corrupts the skip-pattern encoding this
    pipeline otherwise takes care to preserve (see step01's
    _recode_skip_pattern_na). SMOTENC keeps categorical columns at their
    nearest neighbor's actual category instead of interpolating them, and
    only interpolates the genuinely continuous columns listed in
    class_imbalance.smote.continuous_features."""
    try:
        from imblearn.over_sampling import SMOTENC
    except ImportError as e:
        raise ImportError(
            "class_imbalance.method='smote' requires imbalanced-learn, "
            "which isn't installed. Install with: pip install imbalanced-learn"
        ) from e

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        logger.warning("SMOTE requested but y has a single class present — skipping, fitting on raw data.")
        return X, y

    smote_cfg = cfg.get("class_imbalance", {}).get("smote", {})
    continuous = set(smote_cfg.get("continuous_features", []))
    unknown_continuous = continuous - set(X.columns)
    if unknown_continuous:
        logger.warning(
            "class_imbalance.smote.continuous_features lists column(s) not "
            "in the feature set: %s — ignoring them.", unknown_continuous,
        )
    categorical_idx = [i for i, c in enumerate(X.columns) if c not in continuous]

    k_configured = smote_cfg.get("k_neighbors", 5)
    n_minority = int(class_counts.min())
    k = min(k_configured, n_minority - 1)
    if k < 1:
        logger.warning(
            "Minority class too small for SMOTE (n=%s, need >1) — "
            "skipping, fitting on raw data.", n_minority,
        )
        return X, y
    if k < k_configured:
        logger.warning(
            "class_imbalance.smote.k_neighbors=%s but minority class only "
            "has %s rows in this fold/pool — using k=%s instead.",
            k_configured, n_minority, k,
        )

    sm = SMOTENC(categorical_features=categorical_idx, k_neighbors=k, random_state=seed)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(
        "SMOTENC: %s -> %s rows (class balance %s -> %s). Categorical/"
        "sentinel columns (%s of %s features) preserved as real categories, "
        "not interpolated — only %s continuous column(s) interpolated.",
        len(X), len(X_res), dict(y.value_counts()), dict(pd.Series(y_res).value_counts()),
        len(categorical_idx), len(X.columns), len(X.columns) - len(categorical_idx),
    )
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def fit_with_class_weight(cfg: dict, seed: int, X: pd.DataFrame, y: pd.Series, logger=None):
    """Dispatches on config.class_imbalance.method (default "class_weight"
    if the key is absent, for backward compatibility with configs written
    before this option existed):

    - "class_weight" (default): cost-sensitive scale_pos_weight, computed
      from THIS call's y only (train-fold-only, no leakage). No synthetic
      rows, so calibration is affected in a known, monotone way — this is
      why it was the original default (see git history / prior review).
    - "smote": SMOTENC oversampling before fitting (see _smote_resample).
      Available as an explicit, opt-in alternative per the protocol's own
      suggestion to "evaluate its impact on model calibration" — it is
      NOT the default, because on this specific feature set (majority
      categorical/sentinel-coded) plain SMOTE would be actively harmful,
      and even SMOTENC's synthetic rows still complicate calibration
      interpretation. If you switch to this, compare
      outputs/figures/calibration_plot.png and the Brier score against a
      class_weight run on the same data before trusting either one.
    - "none": no imbalance handling at all — useful as a baseline to
      compare the other two against.
    """
    import logging
    logger = logger or logging.getLogger(__name__)
    method = cfg.get("class_imbalance", {}).get("method", "class_weight")

    if method == "smote":
        X, y = _smote_resample(cfg, X, y, seed, logger)
        model = get_model(cfg, seed)
        model.fit(X, y)
        return model

    model = get_model(cfg, seed)
    if method == "class_weight":
        if y.nunique() == 2 and "scale_pos_weight" in model.get_params():
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            if n_pos > 0:
                model.set_params(scale_pos_weight=n_neg / n_pos)
    elif method != "none":
        raise ValueError(f"Unknown class_imbalance.method '{method}' — expected "
                          f"'class_weight', 'smote', or 'none'.")
    model.fit(X, y)
    return model


def prep_xy(df: pd.DataFrame, cfg: dict, feature_cols: list[str]):
    X = df[feature_cols].copy()
    # region_stratum arrives already integer-coded from step06
    # (encode_region_stratum) — a single, fixed mapping shared by every
    # downstream step. Do NOT re-derive codes here (see step06 docstring
    # for why independently re-deriving them per-subset was a bug).
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

        model = fit_with_class_weight(cfg, seed, Xtr, ytr, logger)
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

    model = fit_with_class_weight(cfg, seed, X, y, logger)
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

    # BUG FIXED: this used to run cluster_cv_scores on the full `df`
    # (including geo_holdout rows), while fit_final_model correctly trains
    # only on the train pool. That meant the reported "cluster-based CV"
    # AUC was not actually independent of the clusters reserved as the
    # PRIMARY generalization test (protocol 6.1/6.3) — some folds' train
    # and test splits both drew on hold-out clusters. Restrict CV to the
    # same train pool used for the final fit so the two validation numbers
    # (cluster-CV, geographic hold-out) are properly separate.
    train_pool = df[~df["geo_holdout"]]
    cv_results = cluster_cv_scores(train_pool, cfg, feature_cols, logger)
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
