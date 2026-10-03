#!/usr/bin/env python3
"""
run_scenarios.py — orchestrates the 2016-vs-2024/25 scenario comparison
(protocol extension: see outputs/reports/scenario_comparison.md).

Runs, in order:
  1. 2016 round through step06 (model matrix assembly) via
     `python main.py --config config/config_2016.yaml --through step06`
  2. 2024/25 round through step06 via
     `python main.py --config config/config.yaml --through step06`
  3. step12_scenario_comparison.py — the 5 train/test scenarios

Steps 1-2 are skipped automatically if that round's
data/processed/<year>/analytic_full.parquet already exists, so re-running
this script after fixing one round's raw-data path doesn't force a
redundant re-run of the other round. Use --force to always re-run both.

Requires the raw DHS archives to already be unzipped under:
  data/raw/2016/  (ETBR71FL.dta, ETIR71FL.dta, optionally ETGE71FL.shp)
  data/raw/2024/  (ETBR8AFL.dta, ETIR8AFL.dta, optionally ETGE8AFL.shp)

Usage
-----
  python run_scenarios.py                 # skip a round if already processed
  python run_scenarios.py --force          # re-run both rounds through step06
  python run_scenarios.py --scenarios-only # skip straight to step12 (assumes
                                            # both rounds already processed)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


ROUNDS = [
    (2016, "config/config_2016.yaml", "data/processed/2016/analytic_full.parquet"),
    (2024, "config/config.yaml", "data/processed/2024/analytic_full.parquet"),
]


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}"
        )
    print(f"  done in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                         help="Re-run both rounds through step06 even if already processed.")
    parser.add_argument("--scenarios-only", action="store_true",
                         help="Skip step 1-2 entirely; just run step12 "
                              "(both rounds must already be processed).")
    args = parser.parse_args()

    if not args.scenarios_only:
        for year, config_path, marker_rel in ROUNDS:
            marker = PROJECT_ROOT / marker_rel
            if marker.exists() and not args.force:
                print(f"[{year}] {marker_rel} already exists — skipping "
                      f"(use --force to re-run).")
                continue
            data_root = PROJECT_ROOT / f"data/raw/{year}"
            if not data_root.exists() or not any(data_root.rglob("*.dta")):
                raise SystemExit(
                    f"[{year}] No .dta files found under {data_root}. Place the "
                    f"unzipped EDHS {year} archive there before running this "
                    f"script (see this script's module docstring)."
                )
            _run([sys.executable, "main.py", "--config", config_path, "--through", "step06"])

    _run([sys.executable, "src/step12_scenario_comparison.py"])

    print("\nScenario comparison complete.")
    print("  Per-scenario write-ups : outputs/reports/scenario1 .. scenario5/")
    print("  Comparison table (csv) : outputs/tables/scenario_comparison.csv")
    print("  Comparison report (md) : outputs/reports/scenario_comparison.md")


if __name__ == "__main__":
    main()
