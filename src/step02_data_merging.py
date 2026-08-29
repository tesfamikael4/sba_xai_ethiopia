"""
Step 2 — Data merging

BR (primary, birth-level) is left-joined with IR (secondary, woman-level)
on `caseid` — every birth inherits its mother's demographic/SES/WASH/media
predictors. The GPS file is joined only to sanity-check that each cluster's
v024-derived region matches its ADM1 label in the GE file; it never
contributes an actual model feature (protocol section 6.4).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from utils import get_path, load_config, read_parquet, save_parquet, setup_logger


def merge_br_ir(br: pd.DataFrame, ir: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    woman_key = cfg["merge"]["woman_id_var"]

    # v024/v001/v005/v021/v022 exist on both sides (BR carries its own copy);
    # keep BR's copies as the row's authoritative sampling metadata and drop
    # IR's duplicates before merging to avoid _x/_y suffix churn.
    dup_cols = [c for c in ["v001", "v005", "v021", "v022", "v024"] if c in ir.columns]
    ir_slim = ir.drop(columns=dup_cols)

    before = len(br)
    merged = br.merge(ir_slim, on=woman_key, how="left", validate="m:1")
    logger.info("BR (%s rows) + IR -> merged (%s rows)", before, len(merged))

    unmatched = merged[ir_slim.columns.difference([woman_key])].isna().all(axis=1).sum()
    if unmatched:
        logger.warning("%s birth records had no matching IR woman record (dropped predictors, kept row).", unmatched)

    return merged


def check_cluster_region_linkage(merged: pd.DataFrame, ge: pd.DataFrame | None,
                                  cfg: dict, logger) -> None:
    """Confirmatory check only (protocol 6.4) — logs a warning, never raises,
    since a linkage mismatch shouldn't silently corrupt the analytic file."""
    if ge is None or "DHSCLUST" not in getattr(ge, "columns", []):
        logger.info("Skipping cluster-region linkage check (no usable GPS file).")
        return

    cluster_var = cfg["merge"]["cluster_var"]
    check = merged[[cluster_var, "v024"]].drop_duplicates().merge(
        ge, left_on=cluster_var, right_on="DHSCLUST", how="left"
    )
    if "ADM1NAME" in check.columns:
        mismatch_rate = (check["v024"].astype(str) != check["ADM1NAME"].astype(str)).mean()
        logger.info("Cluster-to-region linkage check: %.1f%% of clusters show a "
                     "v024/ADM1NAME label mismatch (expected to be near 0; "
                     "differences may just reflect label-string formatting).",
                     100 * mismatch_rate)


def main():
    cfg = load_config()
    logger = setup_logger("step02_data_merging", cfg)
    interim = get_path(cfg, "interim_dir")

    logger.info("=== Step 2: Data merging ===")
    br = read_parquet(interim / "births_recode.parquet")
    ir = read_parquet(interim / "individual_recode.parquet")

    ge = None
    ge_path = interim / "geographic_data.parquet"
    if ge_path.exists():
        ge = read_parquet(ge_path)

    merged = merge_br_ir(br, ir, cfg, logger)
    check_cluster_region_linkage(merged, ge, cfg, logger)

    save_parquet(merged, interim / "merged.parquet", logger)
    logger.info("Step 2 complete.")


if __name__ == "__main__":
    main()
