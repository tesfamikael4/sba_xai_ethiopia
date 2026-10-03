"""
Generates S2 Table (model settings), S3 Table (full SHAP ranking) and the
cluster-bootstrap 95% CIs (AUC, Brier, AP, F1, equity gaps) for the SBA paper.
Run from anywhere:  python supporting_tables_and_ci.py
Output folder: <ROOT>/outputs/supporting/
"""
import sys, platform, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, f1_score

ROOT   = Path(r"D:\Research\Health\DHS\sba_xai_ethiopia")      # <- change if needed
MODEL  = ROOT / "outputs" / "models" / "sba_xgboost.joblib"
MATRIX = ROOT / "data" / "processed" / "model_matrix.parquet"
OUT    = ROOT / "outputs" / "supporting"; OUT.mkdir(parents=True, exist_ok=True)
Y, WEIGHT, CLUSTER = "sba", "v005", "v021"      # outcome, DHS weight, cluster id
WEALTH, RESID = "v190", "v025"                  # wealth quintile, residence
THRESHOLD = 0.5                                 # classification threshold used in step 8
B, SEED = 1000, 42                              # bootstrap replicates, seed

model = joblib.load(MODEL)
df = pd.read_parquet(MATRIX)
try:
    feats = list(model.feature_names_in_)
except AttributeError:
    feats = list(model.get_booster().feature_names)

hold_cols = [c for c in df.columns if "hold" in c.lower()]
if not hold_cols or Y not in df.columns:
    print("Could not find the hold-out flag or outcome column. Columns are:\n", list(df.columns))
    sys.exit(1)
HOLD = hold_cols[0]
is_hold = df[HOLD].astype(bool)
tr, ho = df[~is_hold], df[is_hold]
print(f"Hold-out flag column: {HOLD} | training {len(tr)} | hold-out {len(ho)}")

# ---------- S2 Table: settings ----------
params = {k: v for k, v in model.get_params().items() if v is not None}
import xgboost, sklearn
try:
    import shap, dice_ml
    shap_v, dice_v = shap.__version__, dice_ml.__version__
except Exception:
    shap_v = dice_v = "n/a"
s2 = pd.DataFrame(
    [(f"XGBoost: {k}", str(v)) for k, v in params.items()] +
    [("Classification threshold", str(THRESHOLD)),
     ("Cross-validation", "5-fold, grouped by DHS cluster (v021)"),
     ("Geographic hold-out", "20% of clusters"),
     ("Bootstrap", f"{B} cluster-bootstrap replicates, seed {SEED}"),
     ("Python", platform.python_version()), ("xgboost", xgboost.__version__),
     ("scikit-learn", sklearn.__version__), ("shap", shap_v), ("dice-ml", dice_v),
     ("numpy", np.__version__), ("pandas", pd.__version__)],
    columns=["Setting", "Value"])

# ---------- S3 Table: full SHAP ranking (training pool) ----------
import shap
Xtr = tr[feats]
sv = shap.TreeExplainer(model).shap_values(Xtr)
sv = sv[1] if isinstance(sv, list) else sv
w = tr[WEIGHT].astype(float).values; w = w / w.sum()
s3 = pd.DataFrame({"Feature": feats,
                   "Weighted mean |SHAP|": (np.abs(sv) * w[:, None]).sum(0),
                   "Unweighted mean |SHAP|": np.abs(sv).mean(0)})
s3 = s3.sort_values("Weighted mean |SHAP|", ascending=False).reset_index(drop=True)
s3.insert(0, "Rank", s3.index + 1)
try:
    cb = pd.read_csv(ROOT / "outputs" / "tables" / "variable_codebook.csv")
    print("Codebook columns (add labels to S3 manually if wanted):", list(cb.columns))
except Exception:
    pass

# ---------- Bootstrap CIs on the hold-out ----------
Xh, yh = ho[feats], ho[Y].astype(int).values
p = model.predict_proba(Xh)[:, 1]
cl = ho[CLUSTER].values; wq = ho[WEALTH].values; rs = ho[RESID].values

def gap(y, yhat, grp):
    tpr, fpr = [], []
    for g in np.unique(grp):
        m = grp == g
        pos, neg = (y[m] == 1).sum(), (y[m] == 0).sum()
        if pos >= 5 and neg >= 5:
            tpr.append(((yhat[m] == 1) & (y[m] == 1)).sum() / pos)
            fpr.append(((yhat[m] == 1) & (y[m] == 0)).sum() / neg)
    return (max(tpr) - min(tpr), max(fpr) - min(fpr)) if len(tpr) > 1 else (np.nan, np.nan)

def stats(idx):
    y, pp = yh[idx], p[idx]; yhat = (pp >= THRESHOLD).astype(int)
    out = [roc_auc_score(y, pp), average_precision_score(y, pp), brier_score_loss(y, pp), f1_score(y, yhat)]
    out += list(gap(y, yhat, wq[idx])) + list(gap(y, yhat, rs[idx]))
    return out

names = ["AUC", "Average precision", "Brier score", "F1",
         "Wealth TPR gap", "Wealth FPR gap", "Residence TPR gap", "Residence FPR gap"]
point = stats(np.arange(len(yh)))
rng = np.random.default_rng(SEED)
groups = {c: np.where(cl == c)[0] for c in np.unique(cl)}
keys = list(groups)
reps = []
for _ in range(B):
    pick = rng.choice(len(keys), len(keys), replace=True)
    idx = np.concatenate([groups[keys[i]] for i in pick])
    if len(np.unique(yh[idx])) < 2: continue
    reps.append(stats(idx))
reps = np.array(reps, dtype=float)
ci = pd.DataFrame({"Metric": names, "Estimate": point,
                   "95% CI lower": np.nanpercentile(reps, 2.5, axis=0),
                   "95% CI upper": np.nanpercentile(reps, 97.5, axis=0)}).round(3)

with pd.ExcelWriter(OUT / "Supporting_Tables_S2_S3_CI.xlsx") as xw:
    s2.to_excel(xw, sheet_name="S2 Table", index=False)
    s3.round(4).to_excel(xw, sheet_name="S3 Table", index=False)
    ci.to_excel(xw, sheet_name="Bootstrap CIs", index=False)
s2.to_csv(OUT / "S2_Table.csv", index=False); s3.to_csv(OUT / "S3_Table.csv", index=False)
ci.to_csv(OUT / "bootstrap_CIs.csv", index=False)
print(ci.to_string(index=False))
print("\nSaved to", OUT)
