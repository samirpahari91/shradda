"""Diagnostic: does the presence of identical process-input vectors in both train and
test folds inflate the standard RepeatedKFold estimate?

Compares, using the SAME model and SAME preprocessor:
  (a) RepeatedKFold  (current reported protocol)
  (b) GroupKFold where the group = the exact process-input vector, so replicate
      specimens sharing identical inputs can never be split across train/test.
Also compares MICE vs median+indicator imputation, to see whether a
browser-portable imputer costs anything.
"""
import re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesRegressor)
from sklearn.model_selection import (RepeatedKFold, RepeatedStratifiedKFold,
                                     GroupKFold, StratifiedGroupKFold,
                                     cross_val_score)
from xgboost import XGBRegressor, XGBClassifier

RS = 42

# ---------------------------------------------------------------- load & clean
df = pd.read_excel("Data.xlsx", sheet_name="Sheet1")
df.columns = [c.replace("\n", " ").strip() for c in df.columns]
INPUT_COLS = list(df.columns[:10])
OUTPUT_COLS = list(df.columns[10:17])
CAT_INPUTS = [c for c in INPUT_COLS if "strateg" in c.lower() or "method" in c.lower()]
NUM_INPUTS = [c for c in INPUT_COLS if c not in CAT_INPUTS]
CAT_OUTPUTS = [c for c in OUTPUT_COLS if "Shape" in c or "Plane" in c]
NUM_OUTPUTS = [c for c in OUTPUT_COLS if c not in CAT_OUTPUTS]

RANGE_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*[-–]\s*-?\d+(?:\.\d+)?\s*$")


def parse_numeric(v):
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


for c in NUM_INPUTS + NUM_OUTPUTS:
    df[c] = df[c].apply(parse_numeric)
for c in CAT_INPUTS + CAT_OUTPUTS:
    df[c] = df[c].apply(lambda v: np.nan if pd.isna(v) else str(v).strip().title())

P, V = "Laser power (W)", "Laser speed (mm/s)"
H, T, B = "Hatch spacing (um)", "Layer thickness (um)", "Beam size (um)"
df["hatch_over_beam"] = df[H] / df[B]
df["layer_over_beam"] = df[T] / df[B]
df["aspect_h_t"] = df[H] / df[T]
for c in ["hatch_over_beam", "layer_over_beam", "aspect_h_t"]:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)

MODEL_NUM = [P, V, T, H, B, "layer rotation (degree)",
             "hatch_over_beam", "layer_over_beam", "aspect_h_t"]
MODEL_CAT = ["Density measurement method"]
FEATURES = MODEL_NUM + MODEL_CAT


def prep_mice():
    return ColumnTransformer([
        ("num", Pipeline([("i", IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=50, random_state=RS),
            max_iter=10, random_state=RS, add_indicator=True))]), MODEL_NUM),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), MODEL_CAT),
    ])


def prep_median():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), MODEL_NUM),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), MODEL_CAT),
    ])


def models(task):
    if task == "reg":
        return {
            "RandomForest": RandomForestRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=3,
                max_features="sqrt", random_state=RS, n_jobs=-1),
            "XGBoost": XGBRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=RS, n_jobs=-1),
        }
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=RS, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=RS, n_jobs=-1,
            eval_metric="mlogloss"),
    }


def dup_groups(sub):
    """Group id = exact numeric process-input vector (replicate specimens)."""
    key = sub[MODEL_NUM].round(6).astype(str).agg("|".join, axis=1)
    return pd.factorize(key)[0]


print("=" * 92)
print("REGRESSION — R2 under RepeatedKFold vs replicate-grouped KFold")
print("=" * 92)
print(f"{'target':<40s}{'model':<14s}{'MICE KF':>9s}{'MED KF':>9s}{'MICE Grp':>10s}{'MED Grp':>9s}")
rows = []
for t in NUM_OUTPUTS:
    sub = df[df[t].notna()].copy()
    X, y = sub[FEATURES], sub[t]
    g = dup_groups(sub)
    for name, est in models("reg").items():
        r = {}
        for pname, pf in (("MICE", prep_mice), ("MED", prep_median)):
            pipe = Pipeline([("p", pf()), ("m", est)])
            r[pname + "_kf"] = cross_val_score(
                pipe, X, y, scoring="r2",
                cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=RS),
                n_jobs=-1).mean()
            r[pname + "_grp"] = cross_val_score(
                pipe, X, y, scoring="r2", cv=GroupKFold(n_splits=5),
                groups=g, n_jobs=-1).mean()
        print(f"{t[:39]:<40s}{name:<14s}{r['MICE_kf']:9.3f}{r['MED_kf']:9.3f}"
              f"{r['MICE_grp']:10.3f}{r['MED_grp']:9.3f}")
        rows.append(dict(target=t, model=name, n=len(y), **{k: round(v, 3) for k, v in r.items()}))

print()
print("=" * 92)
print("CLASSIFICATION — balanced accuracy, same comparison")
print("=" * 92)
print(f"{'target':<40s}{'model':<14s}{'MICE KF':>9s}{'MED KF':>9s}{'MICE Grp':>10s}{'MED Grp':>9s}")
for t in CAT_OUTPUTS:
    sub = df[df[t].notna()].copy()
    X = sub[FEATURES]
    yraw = sub[t]
    vc = yraw.value_counts()
    yraw = yraw.where(~yraw.isin(vc[vc < 8].index), other="Other")
    y = pd.factorize(yraw)[0]
    g = dup_groups(sub)
    for name, est in models("clf").items():
        r = {}
        for pname, pf in (("MICE", prep_mice), ("MED", prep_median)):
            pipe = Pipeline([("p", pf()), ("m", est)])
            r[pname + "_kf"] = cross_val_score(
                pipe, X, y, scoring="balanced_accuracy",
                cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RS),
                n_jobs=-1).mean()
            r[pname + "_grp"] = cross_val_score(
                pipe, X, y, scoring="balanced_accuracy",
                cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS),
                groups=g, n_jobs=-1).mean()
        print(f"{t[:39]:<40s}{name:<14s}{r['MICE_kf']:9.3f}{r['MED_kf']:9.3f}"
              f"{r['MICE_grp']:10.3f}{r['MED_grp']:9.3f}")
        rows.append(dict(target=t, model=name, n=len(y), **{k: round(v, 3) for k, v in r.items()}))

pd.DataFrame(rows).to_csv("outputs/diag_replicate_cv.csv", index=False)
print("\nsaved outputs/diag_replicate_cv.csv")
