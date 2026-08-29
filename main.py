#!/usr/bin/env python3
"""
main.py — pipeline combiner for the SBA-XAI Ethiopia study
(Explainable ML with Actionable Counterfactuals for Skilled Birth
Attendance, protocol v1.2, Aug 2026).

Runs every step script in src/ in order, each reading/writing
config-driven paths so steps stay independently runnable and testable.

Usage
-----
  python main.py                      # run the full pipeline
  python main.py --from step05        # resume from a given step
  python main.py --only step09        # run a single step
  python main.py --list                # show all steps and exit
  python main.py --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

# Ordered pipeline. Each tuple is (step_id, module_name, human description).
PIPELINE = [
    ("step01", "step01_data_loading", "Load raw DHS recodes (BR, IR, GE)"),
    ("step02", "step02_data_merging", "Merge BR + IR, check cluster-region linkage"),
    ("step03", "step03_eligibility_filter", "Apply b19<=35 delivery-module eligibility filter"),
    ("step04", "step04_outcome_derivation", "Derive SBA outcome from m3a-d"),
    ("step05", "step05_feature_engineering", "Derive region_stratum + reporting-only recodes"),
    ("step06", "step06_survey_design", "Attach weights, cluster CV folds, geo hold-out, model matrix"),
    ("step07", "step07_model_training", "Train gradient-boosted model (cluster CV + geo hold-out)"),
    ("step08", "step08_model_evaluation", "Evaluate on geo hold-out: AUC/calibration/equity"),
    ("step09", "step09_shap_analysis", "SHAP global values + region_stratum interactions"),
    ("step10", "step10_counterfactual_analysis", "DiCE constrained counterfactuals (supplementary)"),
    ("step11", "step11_reporting", "Weighted prevalence tables + consolidated run summary"),
]


def run_step(step_id: str, module_name: str, desc: str) -> None:
    print(f"\n{'=' * 70}\n>>> {step_id}: {desc}\n{'=' * 70}")
    t0 = time.time()
    module = importlib.import_module(module_name)
    module.main()
    print(f">>> {step_id} finished in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="SBA-XAI Ethiopia pipeline combiner")
    parser.add_argument("--from", dest="from_step", default=None,
                         help="Resume from this step id (e.g. step05) through the end.")
    parser.add_argument("--only", dest="only_step", default=None,
                         help="Run only this single step id (e.g. step09).")
    parser.add_argument("--list", action="store_true", help="List all steps and exit.")
    parser.add_argument("--config", default=None,
                         help="Path to an alternate config.yaml (default: config/config.yaml).")
    args = parser.parse_args()

    if args.list:
        for step_id, module_name, desc in PIPELINE:
            print(f"{step_id:8s} {module_name:32s} {desc}")
        return

    if args.config:
        # Steps call utils.load_config() with its own default; simplest override
        # is an env var the utils module could read — kept minimal here since
        # the common case is running with the shipped config.
        import os
        os.environ["SBA_XAI_CONFIG_PATH"] = args.config

    steps_to_run = PIPELINE
    if args.only_step:
        steps_to_run = [s for s in PIPELINE if s[0] == args.only_step]
        if not steps_to_run:
            parser.error(f"Unknown step id '{args.only_step}'. Use --list to see valid ids.")
    elif args.from_step:
        ids = [s[0] for s in PIPELINE]
        if args.from_step not in ids:
            parser.error(f"Unknown step id '{args.from_step}'. Use --list to see valid ids.")
        steps_to_run = PIPELINE[ids.index(args.from_step):]

    t_start = time.time()
    for step_id, module_name, desc in steps_to_run:
        run_step(step_id, module_name, desc)

    print(f"\nPipeline complete: {len(steps_to_run)} step(s) in {time.time() - t_start:.1f}s")
    print("See outputs/reports/pipeline_run_summary.json for the consolidated results.")


if __name__ == "__main__":
    main()
