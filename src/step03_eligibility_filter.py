"""
Step 3 — Eligibility filter (protocol section 4.1)

The EDHS delivery-care module (m3a-d, m14) is administered only for births
within 35 months of interview (b19 <= 35), NOT the full 5-year (59mo)
birth-recall window. This was verified against the ETBR8AFL frequency
distribution: n=7,563 non-missing-delivery-assistance births correspond
exactly to b19 in [0,35].

This filter supersedes any "most recent birth per woman" rule based on a
birth-order index — no such variable is populated as expected in this
survey round.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from utils import get_path, load_config, read_parquet, save_parquet, setup_logger


def apply_eligibility_filter(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    elig = cfg["eligibility"]
    recency_var = elig["recency_var"]
    max_months = elig["recency_max_months"]

    before = len(df)
    mask = df[recency_var] <= max_months
    out = df.loc[mask].copy()
    logger.info("Eligibility filter %s<=%s: %s -> %s rows (%s dropped)",
                recency_var, max_months, before, len(out), before - len(out))

    # Sanity check against the protocol's verified count of non-missing
    # delivery-assistance records; logged, not enforced, since exact counts
    # depend on the specific extract pulled.
    skilled_vars = cfg["outcome"]["skilled_attendant_vars"]
    non_missing_delivery = df[skilled_vars].notna().any(axis=1)
    n_non_missing = non_missing_delivery.sum()
    n_in_window = mask.sum()
    if n_non_missing != n_in_window:
        logger.warning(
            "b19<=%s selects %s rows but %s rows have non-missing delivery-"
            "assistance data — protocol expects these to match exactly. "
            "Investigate before proceeding (see protocol section 4.1).",
            max_months, n_in_window, n_non_missing,
        )
    else:
        logger.info("Confirmed: b19<=%s selection matches non-missing "
                     "delivery-assistance rows exactly, per protocol.", max_months)

    return out


def main():
    cfg = load_config()
    logger = setup_logger("step03_eligibility_filter", cfg)
    interim = get_path(cfg, "interim_dir")

    logger.info("=== Step 3: Eligibility filter ===")
    merged = read_parquet(interim / "merged.parquet")
    eligible = apply_eligibility_filter(merged, cfg, logger)

    save_parquet(eligible, interim / "eligible.parquet", logger)
    logger.info("Step 3 complete.")


if __name__ == "__main__":
    main()
