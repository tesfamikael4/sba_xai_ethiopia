"""
Step 1 — Data loading (protocol section 3.1)

Loads the three in-scope DHS 2024/25 recodes:
  - Births Recode      (ETBR8A) — primary: outcome + birth-level predictors
  - Individual Recode  (ETIR8A) — secondary: demographic/SES/WASH/media/access
  - Geographic Data    (ETGE8A) — cluster-to-region linkage check ONLY

Household (HR), Children's (KR), Men's (MR) and Household Member (PR)
recodes are intentionally NOT loaded (protocol section 3.1 — verified
redundant / wrong unit of analysis / out of scope).

Output: one interim parquet per recode, subset to only the variables this
study actually uses (full DHS recodes are wide — hundreds of columns —
so we trim early rather than carrying everything through the pipeline).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from utils import (
    assert_no_excluded_datasets_used,
    find_recode_file,
    get_path,
    load_config,
    save_parquet,
    setup_logger,
)

BR_COLS = [
    "caseid", "bidx", "v001", "v005", "v021", "v022", "v024",
    "b19", "b6", "b7",
    "m3a", "m3b", "m3c", "m3d",
    "m14", "v467d",
]

IR_COLS = [
    "caseid", "v001", "v005", "v021", "v022", "v024",
    "v012", "v106", "v511", "v218",
    "v190", "v025",
    "v481",
    "v157", "v158", "v159",
    "v113", "v116",
    "v701", "v705",
]


def _read_dta(path, usecols: list[str], logger) -> pd.DataFrame:
    try:
        import pyreadstat
        df, meta = pyreadstat.read_dta(str(path), usecols=usecols, apply_value_formats=False)
        logger.info("Read %s via pyreadstat: %s rows", path.name, len(df))
        return df
    except ImportError:
        logger.warning("pyreadstat not installed — falling back to pandas.read_stata "
                        "(loses DHS value-label metadata; fine for numeric codes).")
        df = pd.read_stata(path, columns=usecols, convert_categoricals=False)
        return df


def load_births_recode(cfg, logger) -> pd.DataFrame:
    path = find_recode_file(cfg, "births_recode")
    cols = [c for c in BR_COLS if c is not None]
    df = _read_dta(path, cols, logger)
    return df


def load_individual_recode(cfg, logger) -> pd.DataFrame:
    path = find_recode_file(cfg, "individual_recode")
    cols = [c for c in IR_COLS if c is not None]
    df = _read_dta(path, cols, logger)
    return df


def load_geographic_data(cfg, logger) -> pd.DataFrame | None:
    """
    Loaded ONLY to confirm cluster -> region/zone linkage (protocol 6.4).
    Never used for facility-distance calculation — DHS GPS points are
    randomly displaced (up to 2km urban / 5km rural / 10km for 1% of rural
    clusters) specifically to preserve confidentiality.
    """
    try:
        path = find_recode_file(cfg, "geographic_data")
    except FileNotFoundError:
        logger.warning("Geographic (GPS) file not found — skipping cluster-linkage "
                        "check. Region comes from v024 in BR/IR regardless.")
        return None
    try:
        import geopandas as gpd
        gdf = gpd.read_file(path)
        logger.info("Read %s: %s clusters", path.name, len(gdf))
        # Keep only cluster id + admin fields needed for the linkage check;
        # deliberately DROP the displaced lat/lon so nothing downstream is
        # tempted to compute a facility distance from it.
        keep = [c for c in gdf.columns if c.upper() in
                ("DHSCLUST", "ADM1NAME", "ADM1DHS", "URBAN_RURA")]
        return pd.DataFrame(gdf[keep])
    except ImportError:
        logger.warning("geopandas not installed — skipping GPS cluster-linkage "
                        "check (non-fatal; it is a confirmatory step only).")
        return None


def main():
    cfg = load_config()
    logger = setup_logger("step01_data_loading", cfg)
    interim = get_path(cfg, "interim_dir")

    logger.info("=== Step 1: Data loading ===")
    assert_no_excluded_datasets_used(cfg, ["births_recode", "individual_recode", "geographic_data"])

    br = load_births_recode(cfg, logger)
    save_parquet(br, interim / "births_recode.parquet", logger)

    ir = load_individual_recode(cfg, logger)
    save_parquet(ir, interim / "individual_recode.parquet", logger)

    ge = load_geographic_data(cfg, logger)
    if ge is not None:
        save_parquet(ge, interim / "geographic_data.parquet", logger)

    logger.info("Step 1 complete.")


if __name__ == "__main__":
    main()
