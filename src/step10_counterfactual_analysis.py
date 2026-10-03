"""
Step 10 — Constrained counterfactual explanations, DiCE
(protocol section 7.2)

Supplementary sensitivity analysis.

DiCE counterfactuals are scenario explorations, NOT causal effect estimates.

Mutability:
  Modifiable:
    - m14
    - v157
    - v158
    - v159
    - v481

  Semi-modifiable:
    - v467d

  Immutable:
    - age
    - education
    - wealth
    - urban/rural
    - region_stratum

Missing-value policy:
  Complete cases are created upstream. Step 10 performs NO imputation and
  creates NO missing-value categories.
"""

from __future__ import annotations

import os
import sys
import json

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__)),
)

import joblib
import numpy as np
import pandas as pd

from variable_labels import get_variable_label
from utils import (
    get_path,
    load_config,
    read_parquet,
    setup_logger,
)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prep_X(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:

    return df[feature_cols].copy()


def canonical_category(value) -> str:
    """
    Convert DHS categorical values to one stable DiCE representation.

    Missing values are not converted into categories.
    """

    if pd.isna(value):
        raise ValueError(
            "Missing value reached DiCE. "
            "Complete-case filtering must occur upstream."
        )

    if isinstance(value, str):

        value = value.strip()

        if value in {
            "",
            "nan",
            "NaN",
            "NAN",
            "None",
            "none",
            "<NA>",
            "NA",
        }:
            raise ValueError(
                f"Missing-value token reached DiCE: {value!r}"
            )

        try:
            return f"{float(value):.1f}"
        except ValueError:
            return value

    if isinstance(
        value,
        (int, np.integer),
    ):
        return f"{float(value):.1f}"

    if isinstance(
        value,
        (float, np.floating),
    ):
        if np.isnan(value):
            raise ValueError(
                "NaN reached DiCE."
            )

        return f"{float(value):.1f}"

    return str(value)


def to_dice_dtypes(
    df: pd.DataFrame,
    feature_cols: list[str],
    continuous_features: set[str],
) -> pd.DataFrame:

    df = df[
        feature_cols
    ].copy()

    for col in feature_cols:

        if col in continuous_features:

            df[col] = pd.to_numeric(
                df[col],
                errors="raise",
            )

        else:

            df[col] = df[col].map(
                canonical_category
            )

    return df


# ============================================================================
# CONTINUOUS FEATURES
# ============================================================================

def determine_continuous_features(
    feature_cols: list[str],
    X: pd.DataFrame,
) -> set[str]:

    return {
        f
        for f in feature_cols
        if f != "region_stratum"
        and X[f].nunique() > 10
    }


# ============================================================================
# DICE MODEL WRAPPER
# ============================================================================

class DiceModelWrapper:
    """
    DiCE uses categorical strings while the trained XGBoost model expects
    numeric values. Convert DiCE's strings back to numeric model input.
    """

    def __init__(
        self,
        model,
        feature_cols,
    ):
        self.model = model
        self.feature_cols = feature_cols

    def _numeric(self, X):

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            X = pd.DataFrame(
                X,
                columns=self.feature_cols,
            )

        X = X[
            self.feature_cols
        ].copy()

        for col in X.columns:

            if (
                X[col].dtype == object
                or pd.api.types.is_string_dtype(
                    X[col]
                )
            ):
                X[col] = pd.to_numeric(
                    X[col],
                    errors="raise",
                )

        if X.isna().any().any():
            raise ValueError(
                "DiCE generated missing values."
            )

        return X.astype(float)

    def predict(self, X):
        return self.model.predict(
            self._numeric(X)
        )

    def predict_proba(self, X):
        return self.model.predict_proba(
            self._numeric(X)
        )


def get_dice_explainer(
    model,
    dice_data,
    feature_cols,
):

    import dice_ml

    wrapped = DiceModelWrapper(
        model,
        feature_cols,
    )

    dice_model = dice_ml.Model(
        model=wrapped,
        backend="sklearn",
    )

    return dice_ml.Dice(
        dice_data,
        dice_model,
        method="random",
    )


# ============================================================================
# OOD FILTERING
# ============================================================================

def filter_in_distribution(
    query,
    train,
    feature_cols,
    continuous_features,
    logger,
    label,
):

    keep = np.ones(
        len(query),
        dtype=bool,
    )

    for col in feature_cols:

        if col in continuous_features:

            train_values = pd.to_numeric(
                train[col],
                errors="raise",
            )

            query_values = pd.to_numeric(
                query[col],
                errors="raise",
            )

            valid = query_values.between(
                train_values.min(),
                train_values.max(),
            )

        else:

            allowed = set(
                train[col]
                .astype(str)
                .unique()
            )

            valid = (
                query[col]
                .astype(str)
                .isin(allowed)
            )

        invalid = int(
            (~valid).sum()
        )

        if invalid:

            logger.warning(
                "%s: feature '%s' rejected "
                "%d/%d query rows",
                label,
                col,
                invalid,
                len(query),
            )

        keep &= valid.to_numpy()

    return query.loc[
        keep
    ].copy()


# ============================================================================
# PERMITTED RANGES
# ============================================================================

def build_permitted_range(
    feature_cols,
    mutable_features,
    continuous_features,
    train,
):

    ranges = {}

    for feature in mutable_features:

        if feature not in feature_cols:
            continue

        if feature in continuous_features:

            if feature == "m14":

                ranges[feature] = [
                    0.0,
                    20.0,
                ]

            else:

                ranges[feature] = [
                    float(
                        train[feature].min()
                    ),
                    float(
                        train[feature].max()
                    ),
                ]

        else:

            ranges[feature] = sorted(
                train[feature]
                .astype(str)
                .unique()
                .tolist()
            )

    return ranges


# ============================================================================
# MAIN
# ============================================================================

def main():

    cfg = load_config()

    logger = setup_logger(
        "step10_counterfactual_analysis",
        cfg,
    )

    processed = get_path(
        cfg,
        "processed_dir",
    )

    models_dir = get_path(
        cfg,
        "models_dir",
    )

    tables_dir = get_path(
        cfg,
        "tables_dir",
    )

    reports_dir = get_path(
        cfg,
        "reports_dir",
    )

    logger.info(
        "=== Step 10: Counterfactual analysis (DiCE, supplementary) ==="
    )

    # ========================================================================
    # LOAD MODEL MATRIX
    # ========================================================================

    df = read_parquet(
        processed / "model_matrix.parquet"
    )

    # ========================================================================
    # LOAD MODEL
    # ========================================================================

    model_path = (
        models_dir
        / f"sba_{cfg['model']['algorithm']}.joblib"
    )

    bundle = joblib.load(
        model_path
    )

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    outcome_name = cfg[
        "outcome"
    ][
        "name"
    ]

    logger.info(
        "Loaded model matrix: %d rows × %d columns",
        len(df),
        len(df.columns),
    )

    logger.info(
        "Loaded model: %s",
        model_path,
    )

    logger.info(
        "Model feature count: %d",
        len(feature_cols),
    )

    # ========================================================================
    # TRAINING POOL
    # ========================================================================

    train_pool = df[
        ~df["geo_holdout"]
    ].copy()

    X_train = prep_X(
        train_pool,
        feature_cols,
    )

    logger.info(
        "Training pool: %d rows",
        len(train_pool),
    )

    # ========================================================================
    # COMPLETE-CASE CHECK
    # ========================================================================

    missing = (
        X_train
        .isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ].to_dict()

    if missing:

        raise ValueError(
            "model_matrix.parquet still contains missing values: "
            f"{missing}. "
            "The BR/IR merge is reintroducing unmatched records. "
            "Use an inner merge and rebuild the pipeline from Step 1."
        )

    logger.info(
        "Complete-case validation passed."
    )

    # ========================================================================
    # DICE DATA
    # ========================================================================

    continuous_features = (
        determine_continuous_features(
            feature_cols,
            X_train,
        )
    )

    train_dice = to_dice_dtypes(
        X_train,
        feature_cols,
        continuous_features,
    )

    train_dice[
        outcome_name
    ] = (
        train_pool[
            outcome_name
        ]
        .astype(int)
        .to_numpy()
    )

    import dice_ml

    dice_data = dice_ml.Data(
        dataframe=train_dice[
            feature_cols + [
                outcome_name
            ]
        ],
        continuous_features=list(
            continuous_features
        ),
        outcome_name=outcome_name,
    )

    exp = get_dice_explainer(
        model,
        dice_data,
        feature_cols,
    )

    logger.info(
        "DiCE data prepared: %d rows, %d features, "
        "%d continuous features",
        len(train_dice),
        len(feature_cols),
        len(continuous_features),
    )

    # ========================================================================
    # MUTABILITY / PERMITTED RANGES
    # ========================================================================

    mut = cfg[
        "xai"
    ][
        "counterfactuals"
    ][
        "mutability"
    ]

    mutable_features = (
        mut["modifiable"]
        + mut["semi_modifiable"]
    )

    permitted_range = build_permitted_range(
        feature_cols,
        mutable_features,
        continuous_features,
        train_dice,
    )

    logger.info(
        "Permitted ranges: %s",
        permitted_range,
    )

    # ========================================================================
    # STRATA
    # ========================================================================

    strata = sorted(
        train_pool[
            "region_stratum"
        ]
        .dropna()
        .unique()
    )

    all_cfs = []

    # ========================================================================
    # COUNTERFACTUAL GENERATION
    # ========================================================================

    for stratum in strata:

        rows = train_pool[
            (
                train_pool[
                    "region_stratum"
                ]
                == stratum
            )
            &
            (
                train_pool[
                    outcome_name
                ]
                == 0
            )
        ]

        if rows.empty:

            logger.warning(
                "No SBA=0 rows for stratum '%s' — skipping.",
                stratum,
            )

            continue

        sample = rows.sample(
            n=min(
                3,
                len(rows),
            ),
            random_state=cfg[
                "project"
            ][
                "random_seed"
            ],
        )

        query = prep_X(
            sample,
            feature_cols,
        )

        if query.isna().any().any():

            logger.warning(
                "Stratum '%s' contains missing query values — skipping.",
                stratum,
            )

            continue

        query_dice = to_dice_dtypes(
            query,
            feature_cols,
            continuous_features,
        )

        query_dice = filter_in_distribution(
            query_dice,
            train_dice,
            feature_cols,
            continuous_features,
            logger,
            f"Stratum '{stratum}'",
        )

        if query_dice.empty:

            logger.warning(
                "Stratum '%s': no valid queries after OOD filtering.",
                stratum,
            )

            continue

        # ------------------------------------------------------------
        # Diagnostic retained for v106.
        # ------------------------------------------------------------

        if (
            "v106" in query_dice.columns
            and "v106" in train_dice.columns
        ):

            logger.info(
                "Stratum '%s': v106 query=%s | "
                "training categories=%s | "
                "query_dtype=%s | train_dtype=%s",
                stratum,
                sorted(
                    query_dice[
                        "v106"
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                ),
                sorted(
                    train_dice[
                        "v106"
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                ),
                query_dice[
                    "v106"
                ].dtype,
                train_dice[
                    "v106"
                ].dtype,
            )

        try:

            result = exp.generate_counterfactuals(
                query_dice,
                total_CFs=cfg[
                    "xai"
                ][
                    "counterfactuals"
                ][
                    "n_per_stratum"
                ],
                desired_class="opposite",
                features_to_vary=mutable_features,
                permitted_range=permitted_range,
            )

        except Exception as exc:

            logger.warning(
                "DiCE failed for stratum '%s': %s",
                stratum,
                exc,
            )

            continue

        # ------------------------------------------------------------
        # Collect results.
        # ------------------------------------------------------------

        for example in result.cf_examples_list:

            if example.final_cfs_df is None:
                continue

            cf = (
                example
                .final_cfs_df
                .copy()
            )

            if cf.isna().any().any():

                logger.warning(
                    "Generated CF contains missing values "
                    "for stratum '%s' — discarded.",
                    stratum,
                )

                continue

            cf[
                "region_stratum_label"
            ] = stratum

            all_cfs.append(
                cf
            )

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    if all_cfs:

        combined = pd.concat(
            all_cfs,
            ignore_index=True,
        )

        output_path = (
            tables_dir
            / "dice_counterfactuals.csv"
        )

        combined.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Saved %d counterfactual rows -> %s",
            len(combined),
            output_path,
        )

        # Human-readable companion (column headers only — same rows/values,
        # DHS codes replaced with their .DO-file labels) for anyone reading
        # this table who isn't working from the codebook. The code-keyed
        # file above stays the canonical one; nothing downstream reads
        # either back in, so adding this can't break anything. Labels
        # only, no code prefix — the code<->label cross-reference lives in
        # outputs/.../tables/variable_codebook.csv (written once by
        # step06), not repeated in every downstream table.
        readable = combined.rename(columns=lambda c: get_variable_label(c) if c in combined.columns else c)
        readable_path = tables_dir / "dice_counterfactuals_readable.csv"
        readable.to_csv(readable_path, index=False)
        logger.info("Saved human-readable column headers -> %s", readable_path)

        validity_path = (
            tables_dir
            / "dice_validity_check.json"
        )

        with open(
            validity_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "n_counterfactuals": len(
                        combined
                    ),
                    "missing_value_policy":
                        "complete_case_no_imputation",
                },
                f,
                indent=2,
            )

    else:

        logger.warning(
            "No counterfactuals were generated."
        )

    # ========================================================================
    # FRAMING NOTE
    # ========================================================================

    framing_path = (
        reports_dir
        / "counterfactual_framing_note.md"
    )

    with open(
        framing_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Counterfactual framing "
            "(protocol section 7.2)\n\n"

            "These counterfactuals are a **supplementary sensitivity "
            "analysis**, not a primary study contribution or a causal "
            "effect estimate.\n\n"

            "- **Modifiable:** ANC visits, media exposure, insurance.\n"
            "- **Semi-modifiable:** distance-is-a-problem, framed as "
            "service-deployment scenarios.\n"
            "- **Immutable:** age, education, wealth, urban/rural, "
            "region.\n\n"

            "The analysis uses complete-case observations. No missing-value "
            "imputation or missing-value category is introduced.\n\n"

            f"{cfg['provenance']['causal_disclaimer']}\n"
        )

    logger.info(
        "Step 10 complete."
    )


if __name__ == "__main__":
    main()