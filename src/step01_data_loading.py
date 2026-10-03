"""
Step 1 — Data loading (protocol section 3.1)

Loads:
  - Births Recode (ETBR8AFL)
  - Individual Recode (ETIR8AFL)
  - Geographic Data (ETGE8AFL), linkage check only

Complete-case policy:
  Rows with missing values in required analytical variables are removed
  immediately after loading. Identifiers such as caseid/bidx are not used
  for complete-case filtering.

Output used by every later step:
  interim/births_recode.parquet is what step02 (merging) reads, and every
  step from 02 through 11 chains off that merge. The m14-implausible-value
  filter below (m14 > 20, which catches the 98 "Don't know" sentinel) runs
  BEFORE this file is written, so the filtered, m14<=20 population is what
  the entire rest of the pipeline — model training, SHAP, PD plots, DiCE,
  reporting — is built on. There is no separate "use this for steps 2-11"
  step; it's the same file, already filtered.
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
    "m14", "v467b", "v467c", "v467d", "v467f",
    "bord",  # true birth order for THIS birth (added alongside v218's
             # "total living children as of interview", which mixes in
             # births that happened after this one for a BR-level model)
    "m10",   # wantedness of this pregnancy (1=then, 2=later, 3=no more) —
             # asked in the same births-recode module as m3a-d/m14, so it
             # shares that module's b19<=35 scoping; retrospective intent
             # at conception, not a post-outcome variable
]

IR_COLS = [
    "caseid", "v001", "v005", "v021", "v022", "v024",
    "v012", "v106", "v511", "v218",
    "v190", "v025",
    "v481",
    "v157", "v158", "v159",
    "v113", "v116",
    "v701", "v705",
    "v501",  # current marital/union status — DHS documents this as never
             # missing ("Marital status is not allowed to have missing
             # values" — Guide to DHS Statistics), unlike v701/v705/v511
             # which are skip-pattern conditional ON v501; safe to require
             # non-missing directly, no sentinel needed
    "v212",  # age at first birth — defined for any woman with v201>=1
             # (at least one birth), which every respondent in our
             # BR-merged sample satisfies by construction
]

# BUG FIXED (2nd incident) — RESOLVED, root cause confirmed, not just
# worked around: the study team uploaded the actual ETBR8AFL.DO/ETIR8AFL.DO
# setup files. v467a's own `label variable` line reads literally
# "NA - Getting medical help for self: know where to go" — the DHS Program
# itself marks this a legacy placeholder that was NOT fielded in Ethiopia's
# EDHS 2024/25 questionnaire (several other v467 letters — e, g, h, i, j,
# k, l, m — carry the same "NA -" prefix for the same reason). This was
# never a data-loading or merge bug; the column is structurally,
# permanently empty in this round. It's dropped entirely now rather than
# kept as a constant, zero-information sentinel-filled feature.
#
# The .DO file also showed v467b and v467c were NEVER actually the
# problem — they don't even appear in the "top 5 missingness" diagnostic
# on a real run (see step01 log from the incident) — only v467a was.
# They're back to fully required, same as v467d always was.
#
# v467f ("Getting medical help for self: not wanting to go alone") is a
# REAL, fielded item with the same 0/1/2 ("No problem"/"Big problem"/"Not
# a big problem") coding as v467b/c/d, confirmed via `label define V467F`
# in the same .DO file. It was missing from the original access-barrier
# battery entirely (only v467d was in the pipeline pre-review) — this is
# the 4th real barrier item, added now.
#
# CAVEAT: confirmed against the 2024/25 .DO file only. Not yet verified
# against a 2016 .DO file (not uploaded) — the 2016 run's log already
# showed v467a at 100% missing there too (consistent with this being a
# genuine cross-round DHS placeholder, not extract-specific), but v467f's
# behavior in the 2016 file is unverified. If it's an issue, step01's
# missingness diagnostic (added after the 1st incident) will name it
# directly on the next 2016 run rather than silently corrupting anything.
#
# BUG FIXED (1st incident): b6 and b7 ("age at death", months/imputed) are
# DHS skip-pattern variables — they are populated ONLY for births where the
# child later died, and are correctly, structurally blank for surviving
# children. They were previously listed as required-non-missing, which
# meant the step01 complete-case filter was silently keeping almost only
# births that ended in child death (e.g. 52,434 -> 237 rows on the
# 2024/25 extract) instead of the true ~7,563-row delivery-eligible
# population the protocol verified (section 4.1). Nothing downstream
# (step03's eligibility filter uses ONLY b19) ever reads b6/b7, so there
# was no reason for them to be required here. They stay in BR_COLS
# (retained, not required) in case a future extension implements the
# b6/b7 fallback recency logic the protocol's
# `eligibility.fallback_recency_vars` config key anticipates for children
# who died — that logic is not yet implemented anywhere in this pipeline.
BR_REQUIRED_COLS = [c for c in BR_COLS if c not in {"caseid", "bidx", "b6", "b7"}]

# BUG FIXED: v511 (age at first cohabitation), v701 (partner's education),
# and v705 (partner's occupation) are DHS skip-pattern variables — asked
# only of currently- or formerly-partnered women (v501 != "never in
# union"). Requiring them non-missing silently dropped every never-married
# respondent from the analytic sample before eligibility filtering even
# ran: a real selection-bias risk (recent adolescent/unpartnered mothers
# are plausibly at DIFFERENT SBA risk, not just a subgroup safe to drop),
# not merely a sample-size cost. They are recoded to an explicit
# "not applicable / no partner" sentinel in _recode_skip_pattern_na below
# instead, so the row is kept and the model can learn from "no partner" as
# its own category rather than the row silently vanishing. This is a
# substantive analytic decision, not just a bug fix — confirm it's the
# framing you want reported in the manuscript's methods section (a
# reasonable alternative would be a separate "currently partnered"
# indicator interacted with v701/v705, which the sentinel-category
# approach approximates but doesn't formally test).
IR_SKIP_PATTERN_COLS = ["v511", "v701", "v705"]
IR_SKIP_PATTERN_SENTINEL = -1

IR_REQUIRED_COLS = [
    c for c in IR_COLS if c != "caseid" and c not in IR_SKIP_PATTERN_COLS
]

# Upper bound for a plausible m14 (ANC-visit count) value. Per ETBR8AFL.DO,
# m14's only defined special codes are 0 ("No antenatal visits") and 98
# ("Don't know") — nothing between 21 and 97 is a documented code, so 20
# is a generous ceiling for a real visit count. It also matches the upper
# bound already used for m14 in the DiCE counterfactual permitted-range
# config, so model-fitting and counterfactual search now agree on what
# counts as a plausible value instead of using two different bounds.
M14_MAX_PLAUSIBLE = 20


def _recode_skip_pattern_na(
    df: pd.DataFrame, cols: list[str], sentinel: int, name: str, logger
) -> pd.DataFrame:
    """Recode structurally-missing (DHS skip-pattern) values to an explicit
    sentinel category instead of leaving them NaN (which would either force
    a complete-case drop, or route inconsistently through downstream tools
    that don't all handle NaN the same way — sklearn preprocessing, SHAP,
    DiCE). See IR_SKIP_PATTERN_COLS docstring above for which variables and
    why."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        n_missing = df[c].isna().sum()
        if n_missing:
            logger.info(
                "%s: recoded %s/%s missing '%s' values to sentinel %s "
                "(skip-pattern — not applicable, not a data-quality gap).",
                name, n_missing, len(df), c, sentinel,
            )
            df[c] = df[c].fillna(sentinel)
    return df


def _filter_m14_implausible(
    df: pd.DataFrame,
    logger,
    tables_dir=None,
    max_plausible: int = M14_MAX_PLAUSIBLE,
) -> pd.DataFrame:
    """Drop rows where m14 exceeds a plausible ANC-visit count.

    Rows with m14=98 ("Don't know", per ETBR8AFL.DO) were previously
    entering the model matrix as a literal count of 98 ANC visits — the
    flat partial-dependence tail out to ~98-100 seen in Fig. 3 for the
    Agrarian Highlands and Urban/Peri-urban strata was the visible
    signature of this. Rather than sentinel-coding 98 (the treatment used
    for v511/v701/v705, which are skip-pattern "not applicable" cases with
    a real not-a-number meaning), these rows are dropped outright:
    "don't know" is a genuine missing-data response for a variable with no
    other special/skip-pattern coding, not a structural category the model
    should learn as its own level.

    The result of this function is what gets complete-case filtered and
    then written to interim/births_recode.parquet — i.e. every step from
    02 onward (merging, model training, SHAP, PD plots, DiCE, reporting)
    only ever sees rows with m14 <= max_plausible.

    If tables_dir is given, the dropped rows (caseid/bidx/v024/m14) are
    also written to outputs/tables/m14_dropped_dontknow.csv for manual
    inspection — e.g. to confirm they really are all m14=98, or to spot-
    check whether any are a different, non-98 implausible value.
    """
    before = len(df)
    implausible = df["m14"] > max_plausible
    n_dropped = int(implausible.sum())

    if n_dropped:
        logger.info(
            "Births Recode: dropped %s/%s rows with m14 > %s "
            "(includes m14=98 'Don't know' sentinel per ETBR8AFL.DO; "
            "no valid ANC-visit count is defined above this range).",
            n_dropped, before, max_plausible,
        )
        if tables_dir is not None:
            id_cols = [c for c in ("caseid", "bidx", "v024") if c in df.columns]
            dropped_path = tables_dir / "m14_dropped_dontknow.csv"
            df.loc[implausible, id_cols + ["m14"]].to_csv(dropped_path, index=False)
            logger.info("Wrote %s dropped m14 rows -> %s", n_dropped, dropped_path)

    kept = df.loc[~implausible].copy()
    logger.info(
        "Births Recode: m14 plausibility filter kept %s/%s rows "
        "(m14 <= %s) — this is the population steps 02-11 will use.",
        len(kept), before, max_plausible,
    )
    return kept


def _read_dta(path, usecols: list[str], logger) -> pd.DataFrame:
    try:
        import pyreadstat

        df, meta = pyreadstat.read_dta(
            str(path), usecols=usecols, apply_value_formats=False,
        )
        logger.info("Read %s via pyreadstat: %s rows", path.name, len(df))
        return df

    except ImportError:
        logger.warning("pyreadstat not installed — using pandas.read_stata.")
        return pd.read_stata(path, columns=usecols, convert_categoricals=False)


def _remove_missing(
    df: pd.DataFrame, required_cols: list[str], name: str, logger,
) -> pd.DataFrame:
    before = len(df)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"{name}: required columns not found: {missing_cols}")

    # Diagnostic ALWAYS logged (not just on failure): per-column missingness
    # among the required set. Added after a real incident where adding new
    # required columns silently collapsed a round to 0 rows and the ONLY
    # signal was a downstream GroupKFold crash in step06, several steps and
    # one full pipeline run away from the actual cause. Reading this table
    # first should make the culprit visible in seconds instead of another
    # guess-and-rerun cycle.
    miss_pct = (df[required_cols].isna().mean() * 100).round(1).sort_values(ascending=False)
    logger.info(
        "%s missingness among required columns (top 5): %s",
        name, {k: f"{v}%" for k, v in miss_pct.head(5).items()},
    )

    fully_missing = miss_pct[miss_pct == 100.0]
    if len(fully_missing):
        logger.error(
            "%s: column(s) %s are 100%% missing in this extract — almost "
            "certainly not actually populated here (wrong variable name for "
            "this DHS phase, or not merged into this file at all). This WILL "
            "collapse the complete-case filter to 0 rows. Fix required_cols "
            "before proceeding rather than letting this run finish.",
            name, list(fully_missing.index),
        )

    df = df.dropna(subset=required_cols).copy()
    removed = before - len(df)
    logger.info(
        "%s complete-case filtering: %d -> %d rows (removed %d)",
        name, before, len(df), removed,
    )

    if before > 0 and len(df) == 0:
        raise ValueError(
            f"{name}: complete-case filtering removed ALL {before} rows. "
            f"See the missingness diagnostic logged just above for the "
            f"likely culprit column(s) — fix required_cols/config before "
            f"rerunning rather than letting this propagate to a confusing "
            f"crash several steps later."
        )

    return df


def load_births_recode(cfg, logger) -> pd.DataFrame:
    path = find_recode_file(cfg, "births_recode")
    df = _read_dta(path, BR_COLS, logger)

    df = _filter_m14_implausible(df, logger, tables_dir=get_path(cfg, "tables_dir"))

    return _remove_missing(df, BR_REQUIRED_COLS, "Births Recode", logger)


def load_individual_recode(cfg, logger) -> pd.DataFrame:
    path = find_recode_file(cfg, "individual_recode")
    df = _read_dta(path, IR_COLS, logger)

    df = _recode_skip_pattern_na(
        df, IR_SKIP_PATTERN_COLS, IR_SKIP_PATTERN_SENTINEL, "Individual Recode", logger,
    )

    return _remove_missing(df, IR_REQUIRED_COLS, "Individual Recode", logger)


def load_geographic_data(cfg, logger) -> pd.DataFrame | None:
    """
    Used only for cluster -> region/zone linkage confirmation.

    GPS coordinates are deliberately not retained for downstream
    distance calculations.
    """
    try:
        path = find_recode_file(cfg, "geographic_data")
    except FileNotFoundError:
        logger.warning("Geographic (GPS) file not found — skipping linkage check.")
        return None

    try:
        import geopandas as gpd

        gdf = gpd.read_file(path)
        logger.info("Read %s: %s clusters", path.name, len(gdf))

        keep = [
            c for c in gdf.columns
            if c.upper() in ("DHSCLUST", "ADM1NAME", "ADM1DHS", "URBAN_RURA")
        ]
        return pd.DataFrame(gdf[keep])

    except ImportError:
        logger.warning("geopandas not installed — skipping GPS linkage check.")
        return None


def main():
    cfg = load_config()
    logger = setup_logger("step01_data_loading", cfg)
    interim = get_path(cfg, "interim_dir")

    logger.info("=== Step 1: Data loading ===")

    assert_no_excluded_datasets_used(
        cfg, ["births_recode", "individual_recode", "geographic_data"],
    )

    # ------------------------------------------------------------------
    # Births Recode
    # ------------------------------------------------------------------
    br = load_births_recode(cfg, logger)
    save_parquet(br, interim / "births_recode.parquet", logger)
    br.to_csv(interim / "births_recode.csv", index=False)

    # ------------------------------------------------------------------
    # Individual Recode
    # ------------------------------------------------------------------
    ir = load_individual_recode(cfg, logger)
    save_parquet(ir, interim / "individual_recode.parquet", logger)
    ir.to_csv(interim / "individual_recode.csv", index=False)

    # ------------------------------------------------------------------
    # Geographic data
    # ------------------------------------------------------------------
    ge = load_geographic_data(cfg, logger)
    if ge is not None:
        save_parquet(ge, interim / "geographic_data.parquet", logger)
        ge.to_csv(interim / "geographic_data.csv", index=False)

    logger.info("Step 1 complete.")


if __name__ == "__main__":
    main()