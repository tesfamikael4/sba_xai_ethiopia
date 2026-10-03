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
    merged = br.merge(ir_slim, on=woman_key, how="inner", validate="m:1")
    logger.info("BR (%s rows) + IR -> merged (%s rows)", before, len(merged))

    unmatched = merged[ir_slim.columns.difference([woman_key])].isna().all(axis=1).sum()
    if unmatched:
        logger.warning("%s birth records had no matching IR woman record (dropped predictors, kept row).", unmatched)

    return merged


def check_cluster_region_linkage(merged: pd.DataFrame, ge: pd.DataFrame | None,
                                  cfg: dict, tables_dir, logger) -> None:
    """Confirmatory check (protocol 6.4) — logs a warning, never raises,
    since a linkage mismatch shouldn't silently corrupt the analytic file.
    ALSO writes a v024->true-region crosswalk table (tables_dir /
    "v024_region_crosswalk.csv") derived from the GE file's real ADM1NAME
    for each v024 code, rather than the placeholder region_labels config —
    added after the 2016 round showed a 78.5% mismatch rate, confirming
    that table's guessed mapping was wrong for a large share of clusters.
    Read the crosswalk's `adm1name_majority` column directly into
    config_2016.yaml's `region_labels` instead of guessing again.

    IMPORTANT: v024 is a numeric region *code* (1, 2, 3 ...), while the GE
    file's ADM1NAME is a text region *name*. Comparing them directly (an
    earlier version of this function did `str(v024) != str(ADM1NAME)`)
    guarantees a mismatch on every row regardless of whether the linkage is
    actually correct — "1" will never equal "Tigray". v024 must be mapped
    to its region-name label first.
    """
    if ge is None or "DHSCLUST" not in getattr(ge, "columns", []):
        logger.info("Skipping cluster-region linkage check (no usable GPS file).")
        return

    # Local import to avoid a hard module-level dependency between steps.
    from step05_feature_engineering import get_region_labels

    cluster_var = cfg["merge"]["cluster_var"]
    check = merged[[cluster_var, "v024"]].drop_duplicates().merge(
        ge, left_on=cluster_var, right_on="DHSCLUST", how="left"
    )
    if "ADM1NAME" not in check.columns:
        logger.info("Skipping cluster-region linkage check (GE file has no ADM1NAME column).")
        return

    configured_labels = get_region_labels(cfg)
    region_from_v024 = check["v024"].map(configured_labels)
    n_unmapped = region_from_v024.isna().sum()

    def _norm(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.lower()

    mismatch_mask = _norm(region_from_v024) != _norm(check["ADM1NAME"])
    mismatch_rate = mismatch_mask.mean()

    logger.info(
        "Cluster-to-region linkage check: %.1f%% of clusters show a "
        "v024-label/ADM1NAME mismatch (%s of %s clusters had no v024->name "
        "mapping and are counted as mismatches).",
        100 * mismatch_rate, n_unmapped, len(check),
    )

    # --- crosswalk table: the TRUE v024 -> region-name mapping, read off
    # the GE file's ADM1NAME directly, one row per v024 code -----------------
    rows = []
    for code, grp in check.groupby("v024"):
        adm1_counts = grp["ADM1NAME"].value_counts(dropna=False)
        majority_name = adm1_counts.index[0] if len(adm1_counts) else None
        majority_n = int(adm1_counts.iloc[0]) if len(adm1_counts) else 0
        rows.append({
            "v024_code": code,
            "n_clusters": len(grp),
            "adm1name_majority": majority_name,
            "adm1name_majority_share": round(majority_n / len(grp), 3) if len(grp) else None,
            "adm1name_all_seen": ", ".join(
                f"{name!s} ({n})" for name, n in adm1_counts.items()
            ),
            "config_region_labels_says": configured_labels.get(code, "(not in config.region_labels)"),
            "config_matches_majority": (
                str(configured_labels.get(code, "")).strip().lower()
                == str(majority_name).strip().lower()
            ),
        })
    crosswalk = pd.DataFrame(rows).sort_values("v024_code")

    tables_dir = tables_dir if hasattr(tables_dir, "mkdir") else None
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        out_path = tables_dir / "v024_region_crosswalk.csv"
        crosswalk.to_csv(out_path, index=False)
        logger.info("Wrote v024->region crosswalk (read this to fix config.region_labels) -> %s", out_path)

    n_wrong = int((~crosswalk["config_matches_majority"]).sum())
    if n_wrong:
        logger.warning(
            "%s of %s v024 codes have config.region_labels != the GE file's "
            "majority ADM1NAME — see the crosswalk table for the corrected "
            "mapping to paste into config.yaml / config_2016.yaml.",
            n_wrong, len(crosswalk),
        )

    if mismatch_rate > 0.05:
        logger.warning(
            "Linkage mismatch rate is %.1f%% — see "
            "outputs/.../tables/v024_region_crosswalk.csv for the exact "
            "corrected v024->region mapping (adm1name_majority column) "
            "instead of re-guessing region_labels.",
            100 * mismatch_rate,
        )
    else:
        logger.info("Linkage check passed: v024-derived region names agree with "
                     "the GE file's ADM1NAME for %.1f%% of clusters.", 100 * (1 - mismatch_rate))


def main():
    cfg = load_config()
    logger = setup_logger("step02_data_merging", cfg)
    interim = get_path(cfg, "interim_dir")
    tables_dir = get_path(cfg, "tables_dir")

    logger.info("=== Step 2: Data merging ===")
    br = read_parquet(interim / "births_recode.parquet")
    ir = read_parquet(interim / "individual_recode.parquet")

    ge = None
    ge_path = interim / "geographic_data.parquet"
    if ge_path.exists():
        ge = read_parquet(ge_path)

    merged = merge_br_ir(br, ir, cfg, logger)
    check_cluster_region_linkage(merged, ge, cfg, tables_dir, logger)

    save_parquet(merged, interim / "merged.parquet", logger)
    logger.info("Step 2 complete.")


if __name__ == "__main__":
    main()