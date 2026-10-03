"""
Human-readable variable labels for display purposes only (figures, tables,
SHAP plots) — the model itself always trains/predicts on the raw DHS codes
(v012, v106, ...), never on these labels. Keeping the two separate means
renaming a label here can never silently change what the model matrix
columns actually are.

SOURCE: label variable statements copied verbatim from the DHS Program's
own Stata setup files for the EDHS 2024/25 extract —
  ETBR8AFL.DO (Births Recode) and ETIR8AFL.DO (Individual Recode)
uploaded directly by the study team, not guessed or inferred. Where a
variable appears in both files with the same label (the common case), one
entry covers both.

CAVEAT — 2016 round: this label set has only been verified against the
2024/25 .DO file. DHS core variable labels are generally stable across
rounds, but this has NOT been confirmed against ETBR71FL.DO/ETIR71FL.DO
(EDHS 2016) specifically. If a 2016 .DO file becomes available, diff it
against this dict before assuming these labels also describe the 2016
round's columns.
"""

VARIABLE_LABELS: dict[str, str] = {
    "v012": "Respondent's current age",
    "v106": "Highest educational level",
    "v511": "Age at first cohabitation",
    "v218": "Number of living children",
    "bord": "Birth order number",
    "v212": "Age of respondent at 1st birth",
    "v190": "Wealth index combined",
    "v025": "Type of place of residence",
    "m14": "Number of antenatal visits during pregnancy",
    "v467a": "NA - Getting medical help for self: know where to go",  # legacy, unused (see step01)
    "v467b": "Getting medical help: permission to go",
    "v467c": "Getting medical help: money needed for treatment",
    "v467d": "Getting medical help: distance to health facility",
    "v467f": "Getting medical help: not wanting to go alone",
    "v481": "Covered by health insurance",
    "v157": "Frequency of reading newspaper/magazine",
    "v158": "Frequency of listening to radio",
    "v159": "Frequency of watching television",
    "m10": "Wanted pregnancy when became pregnant",
    "v113": "Source of drinking water",
    "v116": "Type of toilet facility",
    "v701": "Husband/partner's education level",
    "v705": "Husband/partner's occupation (grouped)",
    "v501": "Current marital status",
    "v024": "Region",
    "b19": "Current age of child in months",
    # Derived (not from any .DO file — constructed in step05):
    "region_stratum": "Region stratum (Agrarian Highlands / Pastoralist-Arid / Urban-Peri-urban)",
    "sba": "Skilled birth attendance (outcome)",
}


def get_variable_label(code: str) -> str:
    """Human-readable label for a variable code, falling back to the raw
    code itself (never raises) so an unlabeled variable degrades to
    exactly what you'd see today rather than breaking a plot."""
    return VARIABLE_LABELS.get(code, code)


def label_axis_list(codes: list[str]) -> list[str]:
    """Pure human-readable labels for plot axes/tables — codes are
    intentionally NOT shown alongside the label here. The one and only
    place a DHS code and its label appear together is the codebook table
    (write_variable_codebook, below), written once by step06 right after
    the model feature list is assembled. Every plot and table downstream
    of that point should show the label only."""
    return [get_variable_label(c) for c in codes]


def write_variable_codebook(feature_cols: list[str], tables_dir, logger) -> None:
    """Writes the ONE table where a DHS code and its label appear
    side by side — outputs/.../tables/variable_codebook.csv. Called once,
    from step06 right after the model feature list is finalized (the
    point at which these raw codes are "recoded" into the model matrix).
    Every other figure/table in this pipeline shows the label only and
    relies on this file as the cross-reference back to the codebook."""
    import pandas as pd
    labels = [get_variable_label(c) for c in feature_cols]

    # Two different codes mapping to the same label text would silently
    # collide when any downstream table renames columns code->label (e.g.
    # step10's dice_counterfactuals_readable.csv) — catch it here, once,
    # rather than a reader noticing two identically-named columns later.
    seen: dict[str, str] = {}
    for code, label in zip(feature_cols, labels):
        if label in seen and seen[label] != code:
            logger.warning(
                "Label collision: both '%s' and '%s' map to the label "
                "'%s' — any table that renames columns to labels only "
                "(step09/step10) will merge these into one column. Give "
                "one of them a more specific label in variable_labels.py.",
                seen[label], code, label,
            )
        seen[label] = code

    codebook = pd.DataFrame({"code": feature_cols, "label": labels})
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "variable_codebook.csv"
    codebook.to_csv(out, index=False)
    logger.info("Wrote variable codebook (code<->label, the only place both "
                "appear together) -> %s", out)
