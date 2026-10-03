"""
Step 13 — class-imbalance method comparison (protocol's own request:
"evaluate SMOTE's impact on model calibration" rather than assume a
default is best without checking).

Runs THREE class_imbalance.method settings — "class_weight" (current
default), "smote" (SMOTENC), and "none" (no handling at all, as an honest
baseline) — on the SAME processed data, SAME cluster-CV folds, SAME
geographic hold-out split, and SAME model hyperparameters, so the only
thing that differs between them is imbalance handling. Reuses step07's
cluster_cv_scores/fit_with_class_weight and step08's evaluate_holdout —
this is deliberately NOT a reimplementation, so a bug fixed in either step
is automatically inherited here too.

Prerequisite: the round has already been run through step06 (model matrix
assembly) — same prerequisite as step12_scenario_comparison.py.

Outputs:
  outputs/tables/imbalance_comparison.csv        — the 3-row comparison table
  outputs/reports/imbalance_comparison.md        — table + written verdict
  outputs/figures/imbalance_calibration_overlay.png — all 3 calibration
                                                       curves on one plot
  outputs/figures/imbalance_<method>_calibration.png — per-method calibration
                                                         (same plot step08 already
                                                         makes for the default method)
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from step06_survey_design import get_model_feature_list
from step07_model_training import cluster_cv_scores, fit_final_model
from step08_model_evaluation import evaluate_holdout
from utils import PROJECT_ROOT, get_path, load_config, read_parquet, setup_logger

METHODS = ["class_weight", "smote", "none"]

METHOD_LABELS = {
    "class_weight": "Cost-sensitive weighting (scale_pos_weight)",
    "smote": "SMOTENC oversampling",
    "none": "No imbalance handling (baseline)",
}

# Distinct, colorblind-reasonable colors per method, used consistently
# across the overlay plot and any per-method output.
METHOD_COLORS = {"class_weight": "#2b6cb0", "smote": "#c05621", "none": "#718096"}


def run_one_method(method: str, cfg: dict, df: pd.DataFrame, feature_cols: list[str], logger) -> dict:
    """Runs cluster-CV + final-fit + hold-out evaluation for one
    class_imbalance.method setting, WITHOUT mutating the caller's cfg
    (deepcopy'd per method so a run can't leak its setting into the next
    one via a shared dict reference)."""
    method_cfg = copy.deepcopy(cfg)
    method_cfg.setdefault("class_imbalance", {})["method"] = method

    logger.info("--- method=%s (%s) ---", method, METHOD_LABELS[method])

    train_pool = df[~df["geo_holdout"]]
    cv = cluster_cv_scores(train_pool, method_cfg, feature_cols, logger)

    model = fit_final_model(df, method_cfg, feature_cols, logger)
    holdout_metrics, proba, y = evaluate_holdout(model, df, method_cfg, feature_cols, logger)

    return {
        "method": method,
        "label": METHOD_LABELS[method],
        "cluster_cv_auc_mean": cv["cluster_cv_auc_mean"],
        "cluster_cv_auc_folds": cv["cluster_cv_auc_folds"],
        **holdout_metrics,
        "_proba": proba,   # kept for the overlay plot, stripped before JSON/CSV export
        "_y": y,
    }


def plot_calibration_overlay(results: list[dict], figures_dir: Path, logger) -> None:
    """All three methods' calibration curves on one plot — the direct,
    visual version of the protocol's "evaluate impact on calibration"
    request. A method whose curve deviates further from the diagonal is
    worse-calibrated at that predicted-probability range, regardless of
    what its Brier score alone would suggest (Brier is a single scalar
    that can hide WHERE the miscalibration is)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for r in results:
        frac_pos, mean_pred = calibration_curve(r["_y"], r["_proba"], n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{r['method']} (Brier={r['brier_score']:.3f})",
                 color=METHOD_COLORS[r["method"]])
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed SBA fraction")
    ax.set_title("Calibration by class-imbalance method — geographic hold-out")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = figures_dir / "imbalance_calibration_overlay.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration overlay -> %s", out)


def write_verdict(results: list[dict], reports_dir: Path, logger) -> None:
    """A data-driven verdict, not a pre-decided one — picks the method with
    the best hold-out AUC and separately flags the best-calibrated method
    (lowest Brier), since these can legitimately disagree (a method can
    discriminate slightly better while being worse-calibrated), and says so
    explicitly rather than picking one winner and hiding the tradeoff."""
    by_auc = sorted(results, key=lambda r: r["roc_auc"], reverse=True)
    by_brier = sorted(results, key=lambda r: r["brier_score"])
    by_f1 = sorted(results, key=lambda r: r["f1"], reverse=True)

    best_auc, best_brier, best_f1 = by_auc[0], by_brier[0], by_f1[0]

    lines = ["# Class-imbalance method comparison", "",
             "Same processed data, same cluster-CV folds, same geographic "
             "hold-out, same model hyperparameters — only `class_imbalance."
             "method` differs between rows. See "
             "`outputs/figures/imbalance_calibration_overlay.png` for the "
             "visual calibration comparison alongside the Brier scores below.",
             "", "## Results", "",
             "| method | cluster-CV AUC | hold-out AUC | avg precision | "
             "precision | recall | F1 | Brier score |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['method']} | {r['cluster_cv_auc_mean']:.4f} | {r['roc_auc']:.4f} | "
            f"{r['average_precision']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1']:.4f} | {r['brier_score']:.4f} |"
        )

    lines += ["", "## Verdict", "",
        f"- **Best discrimination (hold-out AUC):** `{best_auc['method']}` "
        f"({best_auc['roc_auc']:.4f}).",
        f"- **Best calibration (lowest Brier score):** `{best_brier['method']}` "
        f"({best_brier['brier_score']:.4f}).",
        f"- **Best F1 at the 0.5 threshold:** `{best_f1['method']}` "
        f"({best_f1['f1']:.4f}).",
        "",
    ]
    if best_auc["method"] == best_brier["method"]:
        lines.append(
            f"`{best_auc['method']}` wins on both discrimination and "
            f"calibration here — the more clear-cut case. Still confirm this "
            f"holds up against your actual class balance (see the class "
            f"counts each method's log line reports) before treating it as "
            f"final for a small or heavily imbalanced hold-out."
        )
    else:
        lines.append(
            f"**No single method wins on every metric** — `{best_auc['method']}` "
            f"discriminates best, but `{best_brier['method']}` is best-calibrated. "
            f"Which matters more depends on how the model's output will be used: "
            f"if downstream decisions threshold the predicted probability directly "
            f"(e.g. \"flag if predicted risk > 0.3\"), calibration matters more than "
            f"raw AUC; if only the ranking of predictions matters (e.g. \"visit the "
            f"highest-risk decile first\"), AUC matters more than calibration. "
            f"Look at `imbalance_calibration_overlay.png` directly — a Brier-score "
            f"difference in the third decimal place can still hide a calibration "
            f"curve that's badly off in one specific probability range."
        )
    if best_f1["method"] != best_auc["method"]:
        lines.append(
            f"\nSeparately: `{best_f1['method']}` has the best F1 at the fixed "
            f"0.5 threshold, even though it isn't the AUC/calibration winner. "
            f"This is a known SMOTE effect, not necessarily a real advantage — "
            f"SMOTE changes the TRAINING class balance (see each fold's logged "
            f"before/after counts above), which shifts where the model's "
            f"predicted probabilities sit relative to a fixed 0.5 cutoff. A "
            f"class_weight or no-handling model compared at ITS OWN "
            f"optimal threshold (not a blanket 0.5) might close some or all of "
            f"this F1 gap — this comparison doesn't do that threshold search, "
            f"so don't read the F1 column as SMOTE's genuine edge over the "
            f"other two methods without checking that first."
        )
    lines += ["",
        "## Caveat",
        "- This compares three fitted models on ONE geographic hold-out split "
        "and ONE set of cluster-CV folds — not repeated resampling. Treat "
        "small differences (e.g. third-decimal AUC/F1 gaps) as noise rather "
        "than a confident ranking; only trust a gap you'd also expect to see "
        "if you reran with a different random seed.",
        "- SMOTE'S synthetic rows only exist during that method's own model "
        "fit — the geographic hold-out used to evaluate ALL THREE methods "
        "is the same real, non-synthetic data throughout, so this comparison "
        "is not confounded by synthetic data leaking into evaluation.",
    ]

    out = reports_dir / "imbalance_comparison.md"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote verdict report -> %s", out)


def main():
    cfg = load_config()
    logger = setup_logger("step13_imbalance_comparison", cfg)
    logger.info("=== Step 13: class-imbalance method comparison ===")

    processed = get_path(cfg, "processed_dir")
    model_matrix_path = processed / "model_matrix.parquet"
    if not model_matrix_path.exists():
        raise FileNotFoundError(
            f"{model_matrix_path} not found. Run this round through step06 first:\n"
            f"    python main.py --through step06"
        )
    df = read_parquet(model_matrix_path)
    feature_cols = get_model_feature_list(cfg)

    results = [run_one_method(m, cfg, df, feature_cols, logger) for m in METHODS]

    figures_dir = get_path(cfg, "figures_dir")
    tables_dir = get_path(cfg, "tables_dir")
    reports_dir = get_path(cfg, "reports_dir")

    plot_calibration_overlay(results, figures_dir, logger)

    # Per-method calibration plot too (same style step08 already produces
    # for whichever method config.yaml has set as default) — useful if you
    # want one method's curve full-size rather than squeezed into the
    # 3-way overlay.
    for r in results:
        frac_pos, mean_pred = calibration_curve(r["_y"], r["_proba"], n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(mean_pred, frac_pos, marker="o", color=METHOD_COLORS[r["method"]])
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed SBA fraction")
        ax.set_title(f"Calibration \u2014 {r['method']}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"imbalance_{r['method']}_calibration.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    # Strip the non-serializable proba/y arrays before exporting table/JSON.
    export_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]

    comparison_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "cluster_cv_auc_folds"} for r in export_rows
    ])
    csv_path = tables_dir / "imbalance_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info("Wrote comparison table -> %s", csv_path)

    with open(tables_dir / "imbalance_comparison_full.json", "w") as f:
        json.dump(export_rows, f, indent=2)

    write_verdict(results, reports_dir, logger)

    logger.info("Step 13 complete.")


if __name__ == "__main__":
    main()
