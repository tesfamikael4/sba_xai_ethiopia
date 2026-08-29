"""
Step 11 — Reporting (protocol sections 2.2, 6.2, 9, 11)

Produces the descriptive tables a reviewer / MoH audience expects:
  - weighted SBA prevalence by all 14 individual regions (transparency,
    even though formal interaction testing uses the 3-category stratum)
  - weighted SBA prevalence by region_stratum
  - a compiled run summary tying together CV/holdout metrics, SHAP top
    features, and the DiCE validity rate, with the protocol's causal
    disclaimer attached to every output.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from utils import get_path, load_config, read_parquet, setup_logger


def weighted_prevalence(df: pd.DataFrame, group_col: str, outcome: str, weight_col: str) -> pd.DataFrame:
    def _wavg(g):
        w = g[weight_col]
        return pd.Series({
            "n": len(g),
            "weighted_sba_prevalence": (g[outcome] * w).sum() / w.sum(),
            "unweighted_sba_prevalence": g[outcome].mean(),
        })
    return df.groupby(group_col).apply(_wavg, include_groups=False).reset_index()


def main():
    cfg = load_config()
    logger = setup_logger("step11_reporting", cfg)
    processed = get_path(cfg, "processed_dir")
    tables_dir = get_path(cfg, "tables_dir")
    reports_dir = get_path(cfg, "reports_dir")

    logger.info("=== Step 11: Reporting ===")
    full = read_parquet(processed / "analytic_full.parquet")
    outcome = cfg["outcome"]["name"]
    weight_col = "weight"

    by_region = weighted_prevalence(full, "region_name", outcome, weight_col)
    by_region.to_csv(tables_dir / "sba_prevalence_by_region.csv", index=False)
    logger.info("Weighted SBA prevalence by region:\n%s", by_region)

    by_stratum = weighted_prevalence(full.dropna(subset=["region_stratum"]),
                                      "region_stratum", outcome, weight_col)
    by_stratum.to_csv(tables_dir / "sba_prevalence_by_region_stratum.csv", index=False)
    logger.info("Weighted SBA prevalence by region_stratum:\n%s", by_stratum)

    # Pull together whatever earlier-step artifacts exist into one summary.
    summary = {"provenance": cfg["provenance"]}
    for name, fname in [
        ("cv_results", "cv_results.json"),
        ("holdout_metrics", "holdout_metrics.json"),
        ("dice_validity_check", "dice_validity_check.json"),
    ]:
        fpath = tables_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                summary[name] = json.load(f)

    shap_path = tables_dir / "shap_weighted_global_importance.csv"
    if shap_path.exists():
        top10 = pd.read_csv(shap_path).head(10).to_dict(orient="records")
        summary["shap_top10_features"] = top10

    with open(reports_dir / "pipeline_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote consolidated run summary -> %s", reports_dir / "pipeline_run_summary.json")

    logger.info("Step 11 complete.")


if __name__ == "__main__":
    main()
