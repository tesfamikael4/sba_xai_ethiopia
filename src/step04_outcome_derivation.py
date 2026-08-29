"""
Step 4 — Outcome derivation: Skilled Birth Attendance (protocol section 4.2)

SBA = 1 if delivery was assisted by a doctor (m3a), nurse (m3b), midwife
(m3c), or health officer (m3d); SBA = 0 otherwise (TBA, relative, other, or
no assistance).

Health officers (m3d) are counted as skilled attendants, consistent with
their role as mid-level clinical providers in the Ethiopian health system —
this is a protocol design decision, not a DHS-standard default, so it's
kept explicit and configurable rather than hardcoded silently.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from utils import get_path, load_config, read_parquet, save_parquet, setup_logger


def derive_sba(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    out_cfg = cfg["outcome"]
    vars_ = out_cfg["skilled_attendant_vars"]
    target = out_cfg["positive_if_any_equals"]
    outcome_name = out_cfg["name"]

    missing_cols = [v for v in vars_ if v not in df.columns]
    if missing_cols:
        raise KeyError(f"Outcome variables missing from merged data: {missing_cols}")

    df = df.copy()
    any_skilled = (df[vars_] == target).any(axis=1)
    all_missing = df[vars_].isna().all(axis=1)

    df[outcome_name] = np.where(all_missing, np.nan, any_skilled.astype(float))

    n_valid = df[outcome_name].notna().sum()
    prevalence = df[outcome_name].mean(skipna=True)
    logger.info("Derived outcome '%s': %s valid rows, unweighted prevalence = %.1f%%",
                outcome_name, n_valid, 100 * prevalence)

    n_dropped = df[outcome_name].isna().sum()
    if n_dropped:
        logger.warning("%s rows have no delivery-assistance info at all "
                        "(all of m3a-d missing) after the eligibility filter — "
                        "these will be dropped before modeling.", n_dropped)

    return df


def main():
    cfg = load_config()
    logger = setup_logger("step04_outcome_derivation", cfg)
    interim = get_path(cfg, "interim_dir")

    logger.info("=== Step 4: Outcome derivation ===")
    eligible = read_parquet(interim / "eligible.parquet")
    with_outcome = derive_sba(eligible, cfg, logger)

    with_outcome = with_outcome.dropna(subset=[cfg["outcome"]["name"]]).reset_index(drop=True)
    save_parquet(with_outcome, interim / "with_outcome.parquet", logger)
    logger.info("Step 4 complete.")


if __name__ == "__main__":
    main()
