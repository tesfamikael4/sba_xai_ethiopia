"""
Step 5 — Predictor variables (protocol section 5)

General principle: continuous / finely-grained categorical variables are
kept AS-IS as model inputs, letting the tree ensemble and SHAP surface
nonlinear/threshold effects directly. Simplified recodes of the same
variables are derived into a *separate* namespace (`rr_` prefix) for
descriptive reporting / the MoH policy brief only — they are never
substituted into the model feature matrix. This module enforces that
separation structurally, not just by convention.

Also derives `region_stratum` (protocol section 6.2): a 3-category
collapse of v024 used for formal SHAP interaction testing, built as a
clean union of whole regions.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from utils import get_path, load_config, read_parquet, save_parquet, setup_logger

# v024 in the DHS recode is typically an integer code with attached value
# labels. Because we read with apply_value_formats=False in step01 (to keep
# merges numeric-safe), region names may arrive as codes. This map is a
# placeholder aligned to typical EDHS 2024/25 region ordering and MUST be
# verified against the actual .DO/.MAP file for the extract in hand —
# see README "Known assumptions to verify".
V024_REGION_LABELS = {
    1: "Tigray", 2: "Afar", 3: "Amhara", 4: "Oromia", 5: "Somali",
    6: "Benishangul-Gumuz", 7: "South West Ethiopia", 8: "South Ethiopia",
    9: "Central Ethiopia", 10: "Sidama", 11: "Gambella", 12: "Harari",
    13: "Addis Ababa", 14: "Dire Dawa",
}


def build_region_stratum_lookup(cfg: dict) -> dict[str, str]:
    lookup = {}
    for stratum, regions in cfg["region_stratum"].items():
        if stratum == "small_sample_regions_flagged":
            continue
        for region in regions:
            lookup[region] = stratum
    return lookup


def derive_region_stratum(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    df = df.copy()

    if df["v024"].dtype.kind in "iuf":
        region_name = df["v024"].map(V024_REGION_LABELS)
        if region_name.isna().any():
            logger.warning("Some v024 codes did not map to a region name via the "
                            "placeholder label table — verify V024_REGION_LABELS "
                            "against this extract's .DO/.MAP file.")
    else:
        region_name = df["v024"].astype(str)

    df["region_name"] = region_name
    stratum_lookup = build_region_stratum_lookup(cfg)
    df["region_stratum"] = region_name.map(stratum_lookup)

    unmapped = df["region_stratum"].isna().sum()
    if unmapped:
        logger.warning("%s rows have a region_name not found in the region_stratum "
                        "mapping (check V024_REGION_LABELS / config region_stratum).",
                        unmapped)

    flagged = set(cfg["region_stratum"]["small_sample_regions_flagged"])
    counts = df["region_name"].value_counts()
    small = {r: int(counts.get(r, 0)) for r in flagged}
    logger.info("Small-sample region counts post-filter (flag if <100 per protocol 6.2): %s", small)

    return df


def derive_reporting_recodes(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    """Descriptive/policy-brief-only recodes, kept under an `rr_` prefix so
    they can never be silently swept into the model matrix (see step06)."""
    df = df.copy()
    rr = cfg["reporting_recodes"]

    age_bins = rr["age_group_5yr"]["bins"]
    # df["rr_age_group"] = pd.cut(df["v012"], bins=age_bins, right=False)
    df["rr_age_group"] = pd.cut(df["v012"], bins=age_bins, right=False).astype(str)

    edu_map = {int(k): v for k, v in rr["education_simple"]["map"].items()}
    df["rr_education_simple"] = df["v106"].map(edu_map)

    par = rr["parity_group"]
    df["rr_parity_group"] = pd.cut(df["v218"], bins=par["bins"], labels=par["labels"])

    anc = rr["anc_group"]
    df["rr_anc_group"] = pd.cut(df["m14"], bins=anc["bins"], labels=anc["labels"])

    media_cols = rr["media_any_exposure"]["source"]
    df["rr_media_any_exposure"] = (df[media_cols].fillna(0) > 0).any(axis=1)

    # WHO/UNICEF JMP improved/unimproved ladder — simplified binary flag.
    # v113 (water) and v116 (toilet) codes follow the standard DHS recode
    # scheme; verify categories 10-51 (improved) vs 30-72 (unimproved) etc.
    # against this round's .DO file before trusting the flag for reporting.
    improved_water_codes = {10, 11, 12, 13, 20, 21, 30, 31, 41, 51}
    improved_sanitation_codes = {10, 11, 12, 13, 14, 15, 20, 21, 22, 23}
    df["rr_wash_improved"] = (
        df["v113"].isin(improved_water_codes) & df["v116"].isin(improved_sanitation_codes)
    )

    logger.info("Derived %s reporting-only recodes (rr_ prefix).",
                sum(c.startswith("rr_") for c in df.columns))
    return df


def main():
    cfg = load_config()
    logger = setup_logger("step05_feature_engineering", cfg)
    interim = get_path(cfg, "interim_dir")

    logger.info("=== Step 5: Feature engineering ===")
    df = read_parquet(interim / "with_outcome.parquet")

    df = derive_region_stratum(df, cfg, logger)
    df = derive_reporting_recodes(df, cfg, logger)

    save_parquet(df, interim / "featured.parquet", logger)
    logger.info("Step 5 complete.")


if __name__ == "__main__":
    main()
