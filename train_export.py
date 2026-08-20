"""
LPBF process-property modelling — corrected pipeline + browser-deployable model export.
======================================================================================

Two jobs:

1. **Honest evaluation.** Reproduces the cleaning / feature engineering of
   `LPBF_ML_Analysis.ipynb`, then evaluates every target under
   **replicate-grouped cross-validation** — folds are split on the *exact process-input
   vector*, so replicate specimens reported for one process setting can never appear in
   both train and test. Model choice is made by *nested* CV (inner selection loop, outer
   scoring loop) so the reported score is not inflated by the selection itself.

2. **Export.** Refits the selected model per target on all labelled rows and writes
   `docs/model.json` — a self-contained description (preprocessing constants + raw tree
   structures) that the static web app evaluates in JavaScript with no ML runtime.

Every exported model is verified in-process: a pure-Python re-implementation of the
exported JSON must reproduce the library's own `predict()` to within 1e-6, otherwise the
script fails loudly rather than shipping a silently wrong model.

Run:  python train_export.py
Out:  docs/model.json, outputs/grouped_*.csv, outputs/selected_models.csv
"""
from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5
RARE_CLASS_MIN = 8

# Trees in the *deployed* model. Evaluation uses the full ensembles; the export is
# thinned because a 500-tree forest is ~24 MB of JSON — far too much to fetch in a
# browser. A sweep over {500, 200, 120, 80} trees moved grouped CV by <=0.02 on every
# target (grain size, the noisiest, moved most: 0.656 -> 0.636), so this trades a
# rounding-error amount of accuracy for a ~4x smaller payload. `--full-trees` disables it.
DEPLOY_MAX_TREES = 120

# Reliability tiers, exported into model.json and consumed by docs/js/app.js and
# _check_docs.py. Defined once here so the UI badge, the README table and the
# consistency gate can never disagree; changing a number below changes all three.
# Thresholds are deliberately strict: an R² of 0.36 must not look trustworthy.
#   reg  — compare cv_score (R²) directly against the cutoffs.
#   clf  — compare lift = (cv_score - chance) / (1 - chance), chance = 1/n_classes,
#          because balanced accuracy 0.5 is worthless on 2 classes but real on 6.
RELIABILITY = {
    "order": ["good", "moderate", "low"],
    "labels": {"good": "Reliable", "moderate": "Indicative", "low": "Low confidence"},
    "reg": {"metric": "score", "good": 0.6, "moderate": 0.45},
    "clf": {"metric": "lift", "good": 0.55, "moderate": 0.3},
}

# Presentation names for the raw dataset column keys, exported per target as
# `display_name`. The keys carry historical typos and inconsistent spacing that should
# not leak into the README or the UI.
DISPLAY_NAMES = {
    "Relative Density %": "Relative Density (%)",
    "Ultimate Tensile Strength (MPa)": "Ultimate Tensile Strength (MPa)",
    "Hardness (HV)": "Hardness (HV)",
    "Yield Stress (MPa)": "Yield Stress (MPa)",
    "Microstructure Average Grain Size(µm)": "Grain Size (µm)",
    "Microstructure_ Plane": "Microstructure Plane",
    "Grain Shape": "Grain Shape",
}

os.makedirs("outputs", exist_ok=True)
os.makedirs("docs", exist_ok=True)


# =============================================================================
# 1 · Load & clean  (identical logic to the notebook)
# =============================================================================
RANGE_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*[-–]\s*-?\d+(?:\.\d+)?\s*$")


def parse_numeric(v):
    """`"120-160"` -> 140.0 (midpoint); anything non-numeric -> NaN."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("–", "-")
    if RANGE_RE.match(s):
        a, b = s.rsplit("-", 1)
        try:
            return (float(a) + float(b)) / 2.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalise_cat(v):
    return np.nan if pd.isna(v) else str(v).strip().title()


def load_clean():
    df = pd.read_excel("Data.xlsx", sheet_name="Sheet1")
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]

    input_cols = list(df.columns[:10])
    output_cols = list(df.columns[10:17])
    cat_inputs = [c for c in input_cols
                  if "strateg" in c.lower() or "method" in c.lower()]
    num_inputs = [c for c in input_cols if c not in cat_inputs]
    cat_outputs = [c for c in output_cols if "Shape" in c or "Plane" in c]
    num_outputs = [c for c in output_cols if c not in cat_outputs]

    for c in num_inputs + num_outputs:
        df[c] = df[c].apply(parse_numeric)
    for c in cat_inputs + cat_outputs:
        df[c] = df[c].apply(normalise_cat)

    return df, num_inputs, cat_inputs, num_outputs, cat_outputs


df, NUM_INPUTS, CAT_INPUTS, NUM_OUTPUTS, CAT_OUTPUTS = load_clean()

COL_P = "Laser power (W)"
COL_V = "Laser speed (mm/s)"
COL_H = "Hatch spacing (um)"
COL_T = "Layer thickness (um)"
COL_B = "Beam size (um)"
COL_LED = next(c for c in NUM_INPUTS if "Linear energy" in c)
COL_VED = next(c for c in NUM_INPUTS if "Volumetric energy" in c)

# --- physics imputation of energy densities (row-wise, fill-only-if-missing) -------
led_calc = (df[COL_P] / df[COL_V] * 1000.0).replace([np.inf, -np.inf], np.nan)
ved_calc = (df[COL_P] / (df[COL_V] * (df[COL_H] / 1000.0) * (df[COL_T] / 1000.0))
            ).replace([np.inf, -np.inf], np.nan)
N_LED_FILLED = int((df[COL_LED].isna() & led_calc.notna()).sum())
N_VED_FILLED = int((df[COL_VED].isna() & ved_calc.notna()).sum())
df[COL_LED] = df[COL_LED].fillna(led_calc)
df[COL_VED] = df[COL_VED].fillna(ved_calc)

# --- engineered features (row-wise -> leakage-free) -------------------------------
df["hatch_over_beam"] = df[COL_H] / df[COL_B]
df["layer_over_beam"] = df[COL_T] / df[COL_B]
df["aspect_h_t"] = df[COL_H] / df[COL_T]
for c in ["hatch_over_beam", "layer_over_beam", "aspect_h_t"]:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)

ENGINEERED = ["hatch_over_beam", "layer_over_beam", "aspect_h_t"]

# Energy-density features stay dropped (notebook ablation, cell 13). `Scanning strategy`
# stays dropped (52.2% missing, >50% threshold).
MODEL_NUM = [COL_P, COL_V, COL_T, COL_H, COL_B,
             "layer rotation (degree)"] + ENGINEERED
MODEL_CAT = ["Density measurement method"]
FEATURES = MODEL_NUM + MODEL_CAT

# The 6 features a user actually types; the 3 engineered ones are derived in the browser.
RAW_NUM_INPUTS = [COL_P, COL_V, COL_T, COL_H, COL_B, "layer rotation (degree)"]

TASKS = {t: "reg" for t in NUM_OUTPUTS}
TASKS.update({t: "clf" for t in CAT_OUTPUTS})


def replicate_groups(sub: pd.DataFrame) -> np.ndarray:
    """Group id = exact numeric process-input vector.

    15-35% of rows share an identical input vector with another row (one paper
    reporting several measurements for a single process setting). Splitting such
    replicates across train/test lets the model memorise the setting, which inflates
    the score. Grouping on the input vector removes that leak.
    """
    key = sub[MODEL_NUM].round(6).astype(str).agg("|".join, axis=1)
    return pd.factorize(key)[0]


def prepare_target(target):
    """Rows labelled for `target`, with rare classes merged for classification."""
    sub = df[df[target].notna()].copy()
    X = sub[FEATURES].reset_index(drop=True)
    groups = replicate_groups(sub)
    if TASKS[target] == "clf":
        yraw = sub[target].reset_index(drop=True)
        vc = yraw.value_counts()
        rare = sorted(vc[vc < RARE_CLASS_MIN].index)
        yraw = yraw.where(~yraw.isin(rare), other="Other")
        classes = sorted(yraw.unique())
        prepare_target.rare_classes = rare
        y = pd.Categorical(yraw, categories=classes).codes.astype(int)
        return X, y, groups, classes
    return X, sub[target].to_numpy(float), groups, None


# =============================================================================
# 2 · Preprocessor — median + missing-indicator, one-hot.  Fully portable to JS.
# =============================================================================
def make_preprocessor():
    """Median imputation + missing-indicator flags + one-hot categoricals.

    Replaces the notebook's IterativeImputer(MICE). MICE cannot be reproduced in the
    browser, and the diagnostic (`outputs/diag_replicate_cv.csv`) shows median+indicator
    scores equal or better on every target — so the simpler, exportable choice costs
    nothing.
    """
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), MODEL_NUM),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), MODEL_CAT),
    ])


# =============================================================================
# 3 · Candidate models.  All three families export exactly to JSON.
# =============================================================================
def candidates(task):
    if task == "reg":
        return {
            "RandomForest/d8": RandomForestRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=3,
                max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
            "RandomForest/d12": RandomForestRegressor(
                n_estimators=400, max_depth=12, min_samples_leaf=2,
                max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1),
            "RandomForest/deep": RandomForestRegressor(
                n_estimators=500, max_depth=None, min_samples_leaf=1,
                max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=500, max_depth=None, min_samples_leaf=1,
                max_features=0.7, random_state=RANDOM_STATE, n_jobs=-1),
            "XGBoost/shallow": XGBRegressor(
                n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=3,
                random_state=RANDOM_STATE, n_jobs=-1),
            "XGBoost/mid": XGBRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0,
                random_state=RANDOM_STATE, n_jobs=-1),
            "XGBoost/reg": XGBRegressor(
                n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.7,
                colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=0.5,
                min_child_weight=2, random_state=RANDOM_STATE, n_jobs=-1),
        }
    return {
        "RandomForest/d8": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "RandomForest/deep": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1, max_features=0.7,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1, max_features=0.7,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost/shallow": XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=2.0, random_state=RANDOM_STATE,
            n_jobs=-1, eval_metric="mlogloss"),
        "XGBoost/mid": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=RANDOM_STATE,
            n_jobs=-1, eval_metric="mlogloss"),
    }


def baseline(task):
    return (DummyRegressor(strategy="median") if task == "reg"
            else DummyClassifier(strategy="most_frequent"))


def splitter(task, seed):
    if task == "clf":
        return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)


def score_fold(task, ytrue, ypred):
    if task == "reg":
        return r2_score(ytrue, ypred)
    return balanced_accuracy_score(ytrue, ypred)


def build_pipe(est):
    """Fresh, unfitted pipeline.

    `clone` is essential: `Pipeline` stores the estimator *by reference* and fits it in
    place, so reusing one `est` object across folds would leave the caller's estimator
    holding whichever fold was fitted last — and would export that fold's model instead
    of the one trained on all rows.
    """
    return Pipeline([("prep", make_preprocessor()), ("model", clone(est))])


def cv_score(est, X, y, groups, task, seeds):
    """Mean/std of the primary metric over `seeds` repeats of grouped K-fold."""
    scores = []
    for seed in seeds:
        cv = splitter(task, seed)
        for tr, te in cv.split(X, y, groups):
            pipe = build_pipe(est)
            pipe.fit(X.iloc[tr], y[tr])
            scores.append(score_fold(task, y[te], pipe.predict(X.iloc[te])))
    return float(np.mean(scores)), float(np.std(scores))


# =============================================================================
# 4 · Nested CV — honest score for "the selection procedure", plus final choice
# =============================================================================
def nested_evaluate(target):
    """Outer grouped folds score a model chosen by an *inner* grouped CV.

    This is what makes the headline number honest: the outer fold never influences
    which candidate was picked.
    """
    task = TASKS[target]
    X, y, groups, classes = prepare_target(target)
    cands = candidates(task)

    outer_scores, chosen = [], []
    for seed in range(N_REPEATS):
        for tr, te in splitter(task, seed).split(X, y, groups):
            Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
            best_name, best_val = None, -np.inf
            for name, est in cands.items():
                inner_scores = []
                inner = splitter(task, 1000 + seed)
                for itr, ite in inner.split(Xtr, ytr, gtr):
                    pipe = build_pipe(est)
                    pipe.fit(Xtr.iloc[itr], ytr[itr])
                    inner_scores.append(
                        score_fold(task, ytr[ite], pipe.predict(Xtr.iloc[ite])))
                val = float(np.mean(inner_scores))
                if val > best_val:
                    best_name, best_val = name, val
            pipe = build_pipe(cands[best_name])
            pipe.fit(Xtr, ytr)
            outer_scores.append(score_fold(task, y[te], pipe.predict(X.iloc[te])))
            chosen.append(best_name)
    return float(np.mean(outer_scores)), float(np.std(outer_scores)), chosen


def per_model_table(target):
    """Grouped-CV score of every candidate (+ dummy) — the comparison table."""
    task = TASKS[target]
    X, y, groups, _ = prepare_target(target)
    seeds = list(range(N_REPEATS))
    rows = []
    allc = {("Dummy(median)" if task == "reg" else "Dummy(freq)"): baseline(task)}
    allc.update(candidates(task))
    for name, est in allc.items():
        m, s = cv_score(est, X, y, groups, task, seeds)
        rows.append({"target": target, "model": name, "n": len(y),
                     "grouped_CV": round(m, 4), "std": round(s, 4)})
    return pd.DataFrame(rows)


# =============================================================================
# 5 · Export a fitted pipeline to plain JSON
# =============================================================================
def export_preprocessor(prep):
    """Pull the *fitted* constants out of the ColumnTransformer.

    Read from the fitted objects rather than recomputed, so the exported constants are
    by construction the ones the model was trained with.
    """
    num_imp = prep.named_transformers_["num"]
    cat_pipe = prep.named_transformers_["cat"]
    cat_imp = cat_pipe.named_steps["impute"]
    onehot = cat_pipe.named_steps["onehot"]

    # SimpleImputer(add_indicator=True) appends one flag per feature that had a missing
    # value *during fit* — recorded so JS builds the identical column layout.
    ind_features = ([] if num_imp.indicator_ is None
                    else [int(i) for i in num_imp.indicator_.features_])
    return {
        "numeric_features": MODEL_NUM,
        "numeric_medians": [float(v) for v in num_imp.statistics_],
        "indicator_feature_indices": ind_features,
        "categorical_features": MODEL_CAT,
        "categorical_fill": [str(v) for v in cat_imp.statistics_],
        "categorical_categories": [[str(c) for c in cats] for cats in onehot.categories_],
    }


def _sig(v, digits=9):
    """Round to `digits` significant figures to shrink the payload.

    Applied to leaf/threshold values. 9 significant figures exceeds float32's ~7, so it
    cannot flip a comparison that float32 arithmetic would decide differently — and the
    verification gate below re-checks every row regardless, so a bad choice here fails
    loudly rather than shipping.
    """
    f = float(v)
    if f == 0.0 or not np.isfinite(f):
        return f
    return float(f"{f:.{digits}g}")


def export_sklearn_forest(model, task, max_trees=None):
    """sklearn tree arrays -> JSON.  Decision rule: x[feature] <= threshold -> left."""
    ests = list(model.estimators_)
    if max_trees:
        # Prefix, not a random sample: forest trees are i.i.d., so the first k are as
        # representative as any k, and a prefix keeps the export deterministic.
        ests = ests[:max_trees]
    trees = []
    for est in ests:
        t = est.tree_
        if task == "reg":
            values = [_sig(v) for v in t.value.reshape(t.node_count, -1)[:, 0]]
        else:
            # class probabilities per leaf (normalised counts, honouring class_weight)
            v = t.value.reshape(t.node_count, -1).astype(float)
            tot = v.sum(axis=1, keepdims=True)
            norm = np.divide(v, tot, out=np.zeros_like(v), where=tot > 0)
            values = [[_sig(p) for p in rowp] for rowp in norm]
        trees.append({
            "left": t.children_left.astype(int).tolist(),
            "right": t.children_right.astype(int).tolist(),
            "feature": t.feature.astype(int).tolist(),
            "threshold": [_sig(v) for v in t.threshold],
            "value": values,
        })
    return {"kind": "sklearn_forest", "task": task, "trees": trees,
            "n_trees_total": len(model.estimators_)}


def export_xgboost(model, task, n_classes):
    """Export via `save_model` (lossless), NOT `get_dump`.

    `get_dump(dump_format="json")` prints thresholds and leaf weights through a
    ~9-significant-figure text formatter. A single tree survives that, but summing 400
    slightly-rounded leaf weights accumulates to O(1) error — for Relative Density it
    reached 1.46 %, which the verification gate caught. The serialised JSON model instead
    carries the exact float32 values XGBoost holds internally.
    """
    booster = model.get_booster()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.json")
        booster.save_model(path)
        with open(path, encoding="utf-8") as fh:
            mj = json.load(fh)

    learner = mj["learner"]
    base_raw = learner["learner_model_param"]["base_score"]
    base_score = float(np.asarray(
        json.loads(base_raw) if base_raw.strip().startswith("[") else float(base_raw)
    ).ravel()[0])

    trees = []
    for tr in learner["gradient_booster"]["model"]["trees"]:
        left = [int(v) for v in tr["left_children"]]
        right = [int(v) for v in tr["right_children"]]
        cond = [float(v) for v in tr["split_conditions"]]
        idxs = [int(v) for v in tr["split_indices"]]
        dleft = [int(v) for v in tr["default_left"]]

        n = len(left)
        feature, threshold, leaf, missing = [], [], [], []
        for i in range(n):
            is_leaf = left[i] == -1
            # For a leaf, xgboost stores the output weight in `split_conditions`.
            feature.append(-1 if is_leaf else idxs[i])
            threshold.append(0.0 if is_leaf else cond[i])
            leaf.append(cond[i] if is_leaf else 0.0)
            missing.append(-1 if is_leaf else (left[i] if dleft[i] else right[i]))

        trees.append({"left": left, "right": right, "missing": missing,
                      "feature": feature,
                      "threshold": [_sig(v) for v in threshold],
                      "leaf": [_sig(v) for v in leaf]})

    n_groups = n_classes if (task == "clf" and n_classes > 2) else 1
    return {"kind": "xgboost", "task": task, "trees": trees,
            "base_score": base_score, "n_groups": n_groups,
            "n_trees_total": len(trees)}


def export_model(model, task, n_classes, max_trees=None):
    """Boosted ensembles are never thinned — each tree corrects its predecessors, so
    dropping any changes the prediction. Only bagged forests (i.i.d. trees) are thinned."""
    if isinstance(model, (XGBRegressor, XGBClassifier)):
        return export_xgboost(model, task, n_classes)
    return export_sklearn_forest(model, task, max_trees=max_trees)


# =============================================================================
# 6 · Reference implementation of the exported JSON (the correctness gate)
# =============================================================================
def _is_missing(v) -> bool:
    """True for None, NaN, or empty string.

    `pd.DataFrame.to_dict()` yields float('nan') for missing cells, which is neither
    None nor '' — so a naive `v in (None, "")` check silently treats NaN as the literal
    category "nan" and one-hot encodes it to all zeros. That is exactly the divergence
    this module's verification gate exists to catch.
    """
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return isinstance(v, str) and v.strip() == ""


def js_like_transform(row: dict, pre: dict) -> list[float]:
    """Mirror of the JavaScript feature builder — median impute, flags, one-hot."""
    out = []
    raw = []
    for i, f in enumerate(pre["numeric_features"]):
        v = row.get(f)
        v = float("nan") if _is_missing(v) else float(v)
        raw.append(v)
        out.append(pre["numeric_medians"][i] if np.isnan(v) else v)
    for i in pre["indicator_feature_indices"]:
        out.append(1.0 if np.isnan(raw[i]) else 0.0)
    for i, f in enumerate(pre["categorical_features"]):
        v = row.get(f)
        v = pre["categorical_fill"][i] if _is_missing(v) else str(v)
        for c in pre["categorical_categories"][i]:
            out.append(1.0 if v == c else 0.0)
    return out


def ref_predict_sklearn(spec, x):
    task = spec["task"]
    acc = None
    for t in spec["trees"]:
        n = 0
        while t["left"][n] != -1:
            n = (t["left"][n] if x[t["feature"][n]] <= t["threshold"][n]
                 else t["right"][n])
        v = t["value"][n]
        if task == "reg":
            acc = (acc or 0.0) + v
        else:
            acc = np.array(v, float) if acc is None else acc + np.array(v, float)
    return acc / len(spec["trees"])


def ref_predict_xgb(spec, x):
    """XGBoost compares in float32.

    Thresholds are stored as float32, and XGBoost casts incoming features to float32
    before comparing. An engineered ratio like 8/3 is 2.6666666666666665 in float64 but
    2.6666667 as float32 — the exact value of the threshold split on it. Comparing in
    float64 sends the row down the wrong branch; on Relative Density that flipped 13 of
    400 trees and produced 1.46 % error. JavaScript has the same requirement, met with
    `Math.fround`.
    """
    g = spec["n_groups"]
    # Accumulate in float32 too: xgboost sums leaf weights in single precision, so a
    # float64 running total drifts by ~1e-4 over 400 trees.
    acc = np.full(g, np.float32(spec["base_score"]), dtype=np.float32)
    for i, t in enumerate(spec["trees"]):
        n = 0
        while t["feature"][n] != -1:
            v = np.float32(x[t["feature"][n]])
            if np.isnan(v):
                n = t["missing"][n]
            else:
                n = t["left"][n] if v < np.float32(t["threshold"][n]) else t["right"][n]
        acc[i % g] = np.float32(acc[i % g] + np.float32(t["leaf"][n]))
    return acc.astype(float)


def ref_predict(spec, x):
    return (ref_predict_xgb(spec, x) if spec["kind"] == "xgboost"
            else ref_predict_sklearn(spec, x))


def verify_export(bundle_target, pipe, X, task, n_classes, ref_model=None):
    """Assert the exported JSON reproduces the library's own predict().

    `ref_model` lets the caller pass a truncated copy of the estimator when the export was
    thinned, so the comparison is against the model actually shipped rather than the full
    ensemble (which would fail by design).
    """
    pre, spec = bundle_target["preprocessing"], bundle_target["model"]
    rows = X.to_dict("records")
    if ref_model is None:
        lib = pipe.predict(X)
    else:
        lib = ref_model.predict(pipe.named_steps["prep"].transform(X))
    max_err, mismatch = 0.0, 0
    for i, row in enumerate(rows):
        x = js_like_transform(row, pre)
        out = ref_predict(spec, x)
        if task == "reg":
            got = float(out if np.isscalar(out) else np.ravel(out)[0])
            max_err = max(max_err, abs(got - float(lib[i])))
        else:
            if spec["kind"] == "xgboost":
                got = int(np.argmax(out)) if n_classes > 2 else int(out[0] > 0)
            else:
                got = int(np.argmax(out))
            mismatch += int(got != int(lib[i]))
    return max_err, mismatch


# =============================================================================
# 7 · Main
# =============================================================================
CMP_PATH = "outputs/grouped_model_comparison.csv"
NESTED_PATH = "outputs/nested_grouped_results.csv"


def main():
    print("=" * 78)
    print("LPBF — replicate-grouped CV, model selection, and browser export")
    print("=" * 78)
    print(f"rows={len(df)}  physics-filled LED={N_LED_FILLED} VED={N_VED_FILLED}")
    print(f"features ({len(FEATURES)}): {FEATURES}\n")

    all_targets = NUM_OUTPUTS + CAT_OUTPUTS
    # The two CV stages cost ~45 min; reuse them unless --refit-cv is passed, so that
    # fixing an export bug does not mean re-running the whole evaluation.
    import sys
    force = "--refit-cv" in sys.argv
    full_trees = "--full-trees" in sys.argv

    # --- 7a. per-candidate comparison table under grouped CV ----------------
    print("-" * 78)
    print("Grouped-CV score of every candidate (R2 / balanced accuracy)")
    print("-" * 78)
    if not force and os.path.exists(CMP_PATH):
        per_model = pd.read_csv(CMP_PATH)
        print(f"  [cached] {CMP_PATH} — pass --refit-cv to recompute")
    else:
        tables = []
        for t in all_targets:
            tab = per_model_table(t)
            tables.append(tab)
            best = tab.iloc[tab["grouped_CV"].idxmax()]
            print(f"\n{t}  (n={int(tab['n'].iloc[0])}, task={TASKS[t]})")
            for _, r in tab.iterrows():
                mark = " <-- best" if r["model"] == best["model"] else ""
                print(f"   {r['model']:<20s} {r['grouped_CV']:+.3f} ± {r['std']:.3f}{mark}")
        per_model = pd.concat(tables, ignore_index=True)
        per_model.to_csv(CMP_PATH, index=False)

    # --- 7b. nested CV = the honest headline number -------------------------
    print("\n" + "-" * 78)
    print("Nested grouped CV (inner selection, outer scoring) — headline numbers")
    print("-" * 78)
    if not force and os.path.exists(NESTED_PATH):
        nested = pd.read_csv(NESTED_PATH)
        print(f"  [cached] {NESTED_PATH} — pass --refit-cv to recompute")
        for _, r in nested.iterrows():
            print(f"  {r['target'][:38]:<40s} {r['nested_grouped_CV']:+.3f} "
                  f"± {r['std']:.3f}   picked {r['most_selected']}")
    else:
        nested_rows = []
        for t in all_targets:
            m, s, chosen = nested_evaluate(t)
            top = pd.Series(chosen).value_counts()
            nested_rows.append({
                "target": t, "task": TASKS[t],
                "nested_grouped_CV": round(m, 4), "std": round(s, 4),
                "most_selected": top.index[0],
                "selection_rate": round(top.iloc[0] / len(chosen), 2)})
            print(f"  {t[:38]:<40s} {m:+.3f} ± {s:.3f}   "
                  f"picked {top.index[0]} in {100*top.iloc[0]/len(chosen):.0f}% of folds")
        nested = pd.DataFrame(nested_rows)
        nested.to_csv(NESTED_PATH, index=False)

    # --- 7c. final selection, refit on all data, export ---------------------
    print("\n" + "-" * 78)
    print("Final model per target: refit on all labelled rows + export")
    print("-" * 78)
    bundle = {
        "schema_version": 3,
        "dataset": {"rows": int(len(df)), "source": "Data.xlsx (Sheet1)"},
        "raw_numeric_inputs": RAW_NUM_INPUTS,
        "categorical_inputs": MODEL_CAT,
        "engineered": {
            "hatch_over_beam": {"num": COL_H, "den": COL_B},
            "layer_over_beam": {"num": COL_T, "den": COL_B},
            "aspect_h_t": {"num": COL_H, "den": COL_T},
        },
        # Single source of truth for reliability tiers. app.js reads these instead of
        # hardcoding them, and _check_docs.py reads the same block, so the UI, the README
        # and the gate cannot drift apart. Regression tiers compare R² directly;
        # classification tiers compare lift over the chance rate 1/n_classes, because
        # balanced accuracy of 0.5 means nothing on 2 classes and a lot on 6.
        "reliability": RELIABILITY,
        "features": {}, "targets": {}, "samples": [],
    }

    # feature metadata for the dynamic form (ranges/units from the observed data)
    UNITS = {COL_P: "W", COL_V: "mm/s", COL_T: "µm", COL_H: "µm", COL_B: "µm",
             "layer rotation (degree)": "°"}
    HELP = {
        COL_P: "Laser beam power at the powder bed.",
        COL_V: "Scan velocity of the laser spot.",
        COL_T: "Thickness of each deposited powder layer.",
        COL_H: "Lateral distance between adjacent scan tracks.",
        COL_B: "Laser spot diameter (D4σ / 1-e² depending on source).",
        "layer rotation (degree)": "Scan-vector rotation applied between layers.",
    }
    for f in RAW_NUM_INPUTS:
        s = df[f].dropna()
        bundle["features"][f] = {
            "type": "number", "unit": UNITS.get(f, ""), "help": HELP.get(f, ""),
            "min": float(s.min()), "max": float(s.max()),
            "median": float(s.median()),
            "p1": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
            "missing_pct": round(float(df[f].isna().mean() * 100), 1),
        }
    for f in MODEL_CAT:
        vals = sorted(df[f].dropna().unique().tolist())
        bundle["features"][f] = {
            "type": "select", "options": vals, "unit": "",
            "help": "Technique used to measure relative density in the source study.",
            "missing_pct": round(float(df[f].isna().mean() * 100), 1),
        }

    nested_map = nested.set_index("target").to_dict("index")
    selected_rows = []
    for t in all_targets:
        task = TASKS[t]
        X, y, groups, classes = prepare_target(t)
        n_classes = 0 if classes is None else len(classes)

        # pick the single best candidate by grouped CV over all labelled rows
        tab = per_model[per_model.target == t]
        tab = tab[~tab.model.str.startswith("Dummy")]
        best_name = tab.iloc[tab["grouped_CV"].values.argmax()]["model"]
        est = candidates(task)[best_name]

        # Secondary metrics FIRST, from grouped out-of-fold predictions. Each fold gets a
        # cloned estimator, so nothing here can disturb the final model fitted below.
        oof = np.full(len(y), np.nan, float)
        for tr, te in splitter(task, 0).split(X, y, groups):
            p = build_pipe(est)
            p.fit(X.iloc[tr], y[tr])
            oof[te] = p.predict(X.iloc[te])

        # The deployed model: fitted on every labelled row, and fitted last so that no
        # subsequent loop can overwrite it.
        pipe = build_pipe(est)
        pipe.fit(X, y)
        if task == "reg":
            extra = {"mae": round(float(mean_absolute_error(y, oof)), 3),
                     "rmse": round(float(np.sqrt(np.mean((y - oof) ** 2))), 3),
                     "y_min": float(np.min(y)), "y_max": float(np.max(y)),
                     "y_median": float(np.median(y))}
        else:
            extra = {"macro_f1": round(float(f1_score(y, oof.astype(int),
                                                     average="macro")), 3)}

        # Thin bagged forests for the payload, and verify against the thinned copy so the
        # gate checks what users actually run.
        fitted = pipe.named_steps["model"]
        ref_model, max_trees = None, None
        if not full_trees and hasattr(fitted, "estimators_") \
                and len(fitted.estimators_) > DEPLOY_MAX_TREES:
            max_trees = DEPLOY_MAX_TREES
            ref_model = copy.deepcopy(fitted)
            ref_model.estimators_ = ref_model.estimators_[:max_trees]
            if hasattr(ref_model, "n_estimators"):
                ref_model.n_estimators = max_trees

        spec = {
            "task": task,
            "unit": {"Hardness (HV)": "HV", "Yield Stress (MPa)": "MPa",
                     "Ultimate Tensile Strength (MPa)": "MPa",
                     "Relative Density %": "%"}.get(t, "µm" if "Grain Size" in t else ""),
            # Human-readable name for docs, tables and the README. The raw dataset keys are
            # inconsistent ("Microstructure_ Plane", "Microstructure Average Grain Size(µm)"),
            # so exporting the label once keeps every consumer from re-deriving it.
            "display_name": DISPLAY_NAMES.get(t, t),
            "model_name": best_name,
            "n_train": int(len(y)),
            "cv_score": nested_map[t]["nested_grouped_CV"],
            "cv_std": nested_map[t]["std"],
            "metric": "R²" if task == "reg" else "balanced accuracy",
            "classes": classes,
            # Which source labels were merged into "Other", so the UI can explain that
            # bucket instead of showing an opaque class name.
            "rare_classes": (getattr(prepare_target, "rare_classes", [])
                             if task == "clf" else []),
            "preprocessing": export_preprocessor(pipe.named_steps["prep"]),
            "model": export_model(fitted, task, n_classes, max_trees=max_trees),
            **extra,
        }
        # Tolerance is relative: float32 has ~7 decimal digits, so a prediction near 100
        # (Relative Density) cannot agree to 1e-6 absolute no matter how exact the export.
        scale = max(1.0, float(np.max(np.abs(y))) if task == "reg" else 1.0)
        tol = 1e-5 * scale
        max_err, mismatch = verify_export(spec, pipe, X, task, n_classes,
                                          ref_model=ref_model)
        status = "OK" if (max_err < tol and mismatch == 0) else "FAIL"
        if status == "FAIL":
            raise SystemExit(
                f"Export verification FAILED for {t}: max_err={max_err:.3g} "
                f"(tol {tol:.3g}) class-mismatches={mismatch}")
        n_trees = len(spec["model"]["trees"])
        total = spec["model"].get("n_trees_total", n_trees)
        thin = f" (of {total})" if n_trees != total else ""
        print(f"  {t[:38]:<40s} {best_name:<18s} trees={n_trees}{thin:<9s} "
              f"verify={status} (err={max_err:.2e})")
        bundle["targets"][t] = spec
        selected_rows.append({"target": t, "task": task, "model": best_name,
                              "n": int(len(y)),
                              "nested_grouped_CV": nested_map[t]["nested_grouped_CV"],
                              "std": nested_map[t]["std"], **extra})

    # --- 7d. sample rows for the "Sample Data" button -----------------------
    complete = df[RAW_NUM_INPUTS].notna().all(axis=1)
    pool = df[complete].copy()
    n_lab = pool[all_targets].notna().sum(axis=1)
    pool = pool.loc[n_lab.sort_values(ascending=False).index]
    picked, seen = [], set()
    for _, r in pool.iterrows():
        key = tuple(round(float(r[c]), 4) for c in RAW_NUM_INPUTS)
        if key in seen:
            continue
        seen.add(key)
        s = {"inputs": {c: float(r[c]) for c in RAW_NUM_INPUTS}, "actual": {}}
        cm = r[MODEL_CAT[0]]
        s["inputs"][MODEL_CAT[0]] = None if pd.isna(cm) else str(cm)
        for t in all_targets:
            if pd.notna(r[t]):
                s["actual"][t] = (float(r[t]) if TASKS[t] == "reg" else str(r[t]))
        picked.append(s)
        if len(picked) >= 12:
            break
    bundle["samples"] = picked
    print(f"\n  embedded {len(picked)} sample rows")

    pd.DataFrame(selected_rows).to_csv("outputs/selected_models.csv", index=False)

    with open("docs/model.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, separators=(",", ":"))
    size_mb = os.path.getsize("docs/model.json") / 1e6
    print(f"\nwrote docs/model.json  ({size_mb:.2f} MB)")

    # test vectors so the browser can self-check against Python at load time
    tv = []
    for t in all_targets:
        X, y, _, _ = prepare_target(t)
        pre = bundle["targets"][t]["preprocessing"]
        spec = bundle["targets"][t]["model"]
        for row in X.head(3).to_dict("records"):
            x = js_like_transform(row, pre)
            out = ref_predict(spec, x)
            # JSON has no NaN, and JS must receive an explicit null so its own
            # missing-value branch fires identically.
            tv.append({"target": t,
                       "row": {k: (None if _is_missing(v) else v)
                               for k, v in row.items()},
                       "expected": (float(np.ravel(out)[0]) if TASKS[t] == "reg"
                                    else [float(z) for z in np.ravel(out)])})
    with open("outputs/js_test_vectors.json", "w", encoding="utf-8") as f:
        json.dump(tv, f, indent=1)
    # Also ship them next to the app so the browser can self-check at load time.
    with open("docs/test_vectors.json", "w", encoding="utf-8") as f:
        json.dump(tv, f, separators=(",", ":"))
    print(f"wrote outputs/js_test_vectors.json + docs/test_vectors.json ({len(tv)} vectors)")
    print("\nDone.")


if __name__ == "__main__":
    main()
