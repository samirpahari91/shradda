# Methodology — Machine Learning on the LPBF Process–Property Dataset

**Dataset:** `Data.xlsx` (Sheet1) — 674 rows × 18 columns from laser powder-bed-fusion (LPBF)
literature. 10 process **inputs**, 7 property **outputs**, and 1 `References` column that is
**bibliographic only** and is **not used anywhere in modelling**.

This document explains, and critically justifies, every modelling decision. The companion
notebook `LPBF_ML_Analysis.ipynb` implements it as self-contained cell blocks.

> **v2 update (data-scientist pass).** Three changes were made after the first version:
> (1) **Group-aware CV was removed** — paper-`References` grouping is not physically meaningful
> for this dataset (median 4 rows/paper, one paper with 64), so it added noise, not insight.
> (2) **Physics-based imputation** now recovers missing energy-density values exactly from other
> inputs in the same row (§4a). (3) **Domain feature engineering** and **iterative (MICE)
> imputation with missing-indicator flags** were added (§4b, §4c). All new steps are row-wise or
> fit strictly inside CV folds, so none introduce leakage or overfitting.

> ## ⚠️ v3 update — two corrections that change the reported numbers
>
> **(A) Replicate leakage was found and fixed (§7a).** `RepeatedKFold` assumes independent rows.
> 129 of 674 rows share an **exactly identical process-input vector** with another row (one study
> reporting several measurements at one laser setting; the same build measured on two planes), so
> 15–35 % of rows per target have a twin. Random K-Fold splits those twins across train and test,
> scoring memorisation as generalisation. Folds are now grouped on the exact input vector.
> **Grain Size falls 0.64 → 0.36 and Yield Stress 0.69 → 0.46**; Relative Density, Hardness and
> Microstructure Plane are unaffected. Adding nested model selection lowers these a little further
> to the reported **0.35 and 0.43** (§7a table, rightmost column).
> **§7a supersedes the R² values quoted in §6a.**
> This is *not* the paper-level grouping rejected in §7 — it is a mechanical de-duplication of
> identical rows, with none of that instability.
>
> **(B) MICE was replaced by median + missing-indicator (§4b).** Benchmarked head-to-head under
> an identical protocol, median imputation scored **equal or better on every target** — MICE's
> extra machinery was not earning its complexity on this dataset. The simpler imputer is also
> exactly reproducible in JavaScript, which is what makes the browser deployment possible at all.
>
> **(C) Deployment added.** `train_export.py` refits the selected model per target and exports it
> to `docs/model.json`; a static web app (`docs/`) evaluates the trees in plain JavaScript. Both
> the Python exporter and a Node harness (`verify_js.mjs`) assert the browser reproduces
> scikit-learn's / XGBoost's own `predict()` to within 1e-6 before anything ships.

---

## 1. Why the naïve approach fails

A first look at the data reveals four problems that make "drop the missing rows and fit a model"
both statistically invalid and practically impossible:

| Problem | Evidence in the data | Consequence |
|---|---|---|
| **Ranges stored as text** | `Laser power = "120-160"`, `Grain size = "3.4-8.3"` | Columns load as `object`, not numeric; silently break any model |
| **Pervasive missingness** | Inputs 1.6–52% missing; outputs 34–74% missing | Only **226 / 674** rows have all 8 numeric inputs; **listwise deletion destroys the dataset** |
| **Dirty categoricals** | `"Equiaxed "` vs `"Equiaxed"`, `"vertical"` vs `"Vertical"` | Same class counted as several; inflated cardinality |
| **Two targets are categorical** | Grain Shape, Microstructure Plane | Regression is meaningless for them → classification |

If we required every input **and** every output to be present, essentially no usable rows remain
for the harder targets. The methodology below is designed specifically to **use all the data that
exists for each target**, without fabricating information and without leaking test data into training.

---

## 2. Column roles

**Inputs (10):**
`Laser power (W)`, `Laser speed (mm/s)`, `Layer thickness (µm)`, `Hatch spacing (µm)`,
`Beam size (µm)`, `Scanning strategy` *(categorical)*, `layer rotation (degree)`,
`Linear energy density (J/m)`, `Volumetric energy density (J/mm³)`,
`Density measurement method` *(categorical)*.

> ⚠️ **Collinearity note.** Linear and volumetric energy density are *derived* from power/speed/
> layer thickness/hatch spacing. They are kept because tree models are robust to collinearity and
> they often carry predictive signal, but this dependence is documented and checked with
> correlation + permutation importance. `Density measurement method` describes *how the output was
> measured*, not a process setting — it is retained as a nuisance/metadata feature and its
> importance is inspected; it can be dropped in a sensitivity run.

**Outputs (7) — modelled independently, one model per target:**

| Target | Task |
|---|---|
| Hardness (HV) | Regression |
| Yield Stress (MPa) | Regression |
| Ultimate Tensile Strength (MPa) | Regression |
| Relative Density % | Regression |
| Grain Size (µm) | Regression |
| Grain Shape | Classification |
| Microstructure Plane | Classification |

**`References`** — bibliographic citation. **Not used anywhere** in modelling (no longer a CV
group either — see §6).

---

## 3. Cleaning & numeric representation

1. **Range → midpoint.** Any `"a-b"` string is parsed to its arithmetic midpoint `(a+b)/2`.
   This is the standard, defensible choice when only a range is reported: it is the unbiased
   point estimate under a uniform assumption and keeps the row usable. (An alternative — treating
   min/max as extra features — was rejected as over-engineering for this sample size.)
2. **Type coercion.** After range parsing, all numeric columns are coerced with
   `pd.to_numeric(errors='coerce')`; anything still non-numeric becomes `NaN` (then imputed).
3. **Categorical normalisation.** Trim whitespace and unify case so `"Equiaxed "`, `"Equiaxed"`
   collapse to one class; `"vertical"`/`"Vertical"` collapse to one class.
4. **Rare-class guard (classification).** Classes with fewer than 8 samples are grouped into
   `"Other"` so stratified CV is well-defined.

---

## 4. Missing-data strategy — a three-tier plan

We do **not** drop rows for missing inputs (that would leave ~226 usable rows and bias toward
fully-reported studies). Instead we fill missingness with the *highest-quality method available for
each column*, in three tiers — from deterministic physics to statistical modelling.

### 4a. Tier 1 — Physics-based imputation (deterministic, exact)

Two inputs are **exactly derivable from other inputs in the same row**, so we recover them from
first principles rather than guessing:

```
Linear energy density  LED [J/m]    = (P / v) × 1000
Volumetric energy dens VED [J/mm³]  = P / (v · h · t)      (h, t converted µm → mm)
```

We validated these against the reported values: **median ratio reported/calculated = 1.000**, with
~96% (LED) and ~92% (VED) of rows matching the formula within 5%. Per the chosen policy we **only
fill missing cells and never overwrite a value a paper reported** (so genuine non-standard
definitions are preserved). This filled 12 missing LED and 30 missing VED values. Because each fill
uses only that row's own numbers, it is **impossible for it to leak** across the train/test split —
this is the single best imputation available and carries zero overfitting risk.

### 4b. Tier 2 — statistical imputation, *inside* the CV fold

> **v3 revision.** The notebook uses `IterativeImputer` (MICE), described below. The deployed
> pipeline (`train_export.py`) uses **median imputation + missing-indicator flags** instead,
> for two reasons: (1) benchmarked head-to-head under an identical protocol, median scored
> **equal or better on every one of the seven targets** — MICE's cross-feature modelling was not
> earning its complexity on a dataset this small; (2) median imputation is a handful of constants,
> so it is **exactly reproducible in JavaScript**, which is what makes client-side deployment
> possible. Evidence: `outputs/diag_replicate_cv.csv`. The MICE description below is retained
> because the notebook still demonstrates it.

For the remaining moderately-missing numeric inputs we use scikit-learn's `IterativeImputer`
(MICE), which models each missing feature as a function of the *other* features (e.g. beam size
predicted from power/speed/hatch) using an `ExtraTrees` estimator. This is materially better than a
blind column median because it respects cross-feature structure. Crucially it is fit **only on the
training fold** of each CV split — no test information leaks in. `add_indicator=True` appends binary
**missing-indicator flags**, letting the model react to *"this value was absent"*, which is itself
often informative in literature data. Categorical inputs use most-frequent imputation + one-hot
encoding (with `handle_unknown='ignore'`), also in-fold.

### 4c. Tier 3 — Drop what cannot be trusted

Inputs still **>50% missing after Tier 1** are **dropped from the feature set** and the drop is
reported, not silent. Imputing a majority-missing column mostly injects the imputer's own guess,
which invites overfitting and gives a false sense of information; honest exclusion is safer.

> **The anti-leakage rule.** Tiers 2–3 operate strictly inside each CV fold via a scikit-learn
> `Pipeline`. Imputing on the full dataset before splitting would leak test statistics into training
> and inflate scores — we never do this.

**Missing *targets*:** rows with a missing value for the target being modelled are dropped **for
that target only** (you cannot learn from or score an unknown label). Because modelling is
per-target, each target uses its own maximal set of labelled rows (Hardness ≈173, UTS ≈291,
Relative Density ≈447, etc.). XGBoost additionally handles `NaN` inputs natively; we still impute
so RF and XGB see identical inputs and results are comparable.

---

## 5. Feature engineering (physically-motivated, row-wise → leakage-free)

Raw process parameters only *imply* the physics that governs LPBF outcomes. We add features that
make that physics explicit. Each is computed from a **single row** (no cross-row statistics, so no
leakage) and left `NaN` where its inputs are missing (imputed in-fold like any other feature):

| Engineered feature | Formula | Physical meaning |
|---|---|---|
| `hatch_over_beam` | hatch / beam size | Track **overlap**: <1 overlapping tracks (denser), >1 gaps/porosity |
| `layer_over_beam` | layer thick. / beam size | Penetration vs spot size — melt-pool coupling |
| `aspect_h_t` | hatch / layer thickness | Melt-pool geometry proxy |
| `P_over_v` | power / speed | Linear heat input (J/mm) |
| `log_VED`, `log_LED` | log1p(energy density) | Energy densities are right-skewed and act multiplicatively; log linearises the response |

Engineered features are **added to**, not substituted for, the raw inputs — the tree models select
whatever is most predictive, and permutation importance (§9-equivalent cell) reports whether the new
features earn their place. This is deliberately restrained: only ratios/transforms with a clear
metallurgical rationale, no automated polynomial explosion (which would overfit a small dataset).

### 5a. Energy-density features are dropped globally (evidence-based)

An **ablation study** (notebook cell 13) re-adds vs removes the derived energy features
(LED, VED, `P_over_v`, `log_LED`, `log_VED`) and measures ΔR² per target. Result: for the four
mechanical/density targets the change is within ±0.04 R² (noise) — unsurprising, since energy
density is *computed from* power/speed/hatch/layer thickness, so it carries no independent
information for a tree model. For **Grain Size, removing them improved R² by ~0.10**. We therefore
**drop all energy-density features from every model** (cell 6) for parsimony and reduced overfitting.
This is a data-driven decision documented by the ablation, not an assumption.

### 5b. Grain Size uses a log-transformed target

Grain Size is heavily right-skewed, so a few large values dominate the squared-error loss and give a
noisy estimate (raw CV R² ≈ 0.54 ± 0.42). Fitting in log-space via `TransformedTargetRegressor`
(`log1p`/`expm1`, R² reported back on the µm scale) plus the energy drop **stacks** to a materially
better and more stable model (cell 14 tests all four combinations and selects the best).

---

## 6. Models

- **Random Forest** (`RandomForestRegressor` / `Classifier`) — strong, low-variance baseline,
  minimal tuning, handles nonlinearities and mixed feature scales.
- **XGBoost** (`XGBRegressor` / `Classifier`) — gradient boosting, usually best-in-class on
  tabular data, native missing-value handling.
- **LightGBM** and **CatBoost** — additional gradient-boosting families included in the benchmark
  (§6a). CatBoost is notable for native categorical/missing handling on small mixed-type data.
- **Dummy baseline** (`DummyRegressor(median)` / `DummyClassifier(most_frequent)`) — the honesty
  check. A model that cannot beat this is worthless; reporting it prevents over-claiming.

### 6a. Model benchmark — is a fancier algorithm the answer? (No.)

A diagnostic one-way ANOVA of each target against paper identity showed **51–100 % of every
target's variance is *between-study*** (η²: Grain Size 1.00, Hardness 0.71, UTS 0.70, Relative
Density 0.51) — i.e. driven by alloy, machine, powder and heat-treatment that **are not columns in
this dataset**. This predicts the ~0.65 R² ceiling is a *data* limit, not a *model* limit.

Cell 15 tests this by giving **RandomForest, XGBoost, LightGBM and CatBoost** their best shot via
**Optuna (TPE) Bayesian tuning under nested CV**, all under an identical protocol. Result
(tuned CV R², Δ vs the untuned-XGBoost baseline):

| Target | RF | XGB | LightGBM | CatBoost | Best | Δ |
|---|---|---|---|---|---|---|
| UTS | **0.710** | 0.705 | 0.703 | 0.700 | RF | +0.032 |
| Yield | 0.654 | 0.683 | **0.688** | 0.676 | LightGBM | +0.001 |
| Rel. Density | 0.603 | **0.684** | 0.546 | 0.652 | XGB | +0.009 |
| Hardness | 0.578 | **0.579** | 0.559 | 0.552 | XGB | **+0.167** |

**Conclusions:** (1) All four families cluster within ~0.03 R² on every target — the hallmark of a
data-limited problem; LightGBM/CatBoost do **not** beat XGBoost/RF. (2) The only large gain came
from **tuning** (Hardness +0.167), not from a new algorithm. (3) The route to materially higher R²
is more *features* (alloy/material), not more *models* — now demonstrated, not assumed.

---

## 7. Cross-validation — built for a *small* dataset

With only tens-to-low-hundreds of labelled rows per target, a single train/test split is noisy and
unreliable. We use **repeated resampling CV**:

- **Repeated K-Fold (regression) / Repeated Stratified K-Fold (classification).**
  `RepeatedKFold(n_splits=5, n_repeats=5)` = 25 fits per model. Repeating with different shuffles
  shrinks the variance of the performance estimate — essential when folds are small. Stratification
  keeps class proportions stable for classification.
- **A fixed hold-out test set was rejected** — ~30 rows would be too small to trust; repeated CV
  uses every row for both training and validation across repeats.
- **Nested CV is available for tuned models.** If hyper-parameters are searched, tuning runs in an
  **inner** loop and scoring in an **outer** loop, giving an unbiased estimate of tuned performance
  (the standard defense against "tuning on the test set").

> **Why no group-aware / `References` CV?** An earlier version reported a `GroupKFold`-by-paper
> score. We removed it: with a **median of 4 rows per reference** (and one paper contributing 64),
> paper grouping is statistically unstable and not physically meaningful — folds become wildly
> uneven and the metric mostly reflects which large paper landed in the test fold, not
> generalisation.

### 7a. Replicate-grouped CV — the correction that supersedes §6a

Repeated K-Fold is only valid if **rows are independent**. In this dataset they are not, and the
consequence is a materially inflated score.

**The finding.** 129 of 674 rows share an **exactly identical numeric process-input vector** with
at least one other row. Two mechanisms produce them:

1. One publication reports **several measurements at a single process setting** — e.g. three
   grain-size values (11.0, 19.0, 17.5 µm) for identical `P=100 W, v=111 mm/s`.
2. The **same build is characterised on two planes**, giving two rows that differ only in the
   `Microstructure_ Plane` output.

Per target, the affected fraction is large:

| Target | n | replicate rows | % |
|---|---|---|---|
| Yield Stress | 249 | 86 | 35 % |
| UTS | 291 | 95 | 33 % |
| Hardness | 173 | 33 | 19 % |
| Relative Density | 447 | 77 | 17 % |
| Grain Size | 202 | 31 | 15 % |

**Why it inflates.** `RepeatedKFold` shuffles rows, so it routinely places one twin in the
**training** fold and its twin in the **test** fold. The model is then scored on a process setting
whose answer it has already memorised — a near-nearest-neighbour lookup, not generalisation. The
effect is strongest for targets where replicates carry most of the within-setting variance.

**The fix.** Group on the exact input vector and use `GroupKFold` /
`StratifiedGroupKFold`, so replicates are always on the same side of the split.

The first two columns are single-loop CV, isolating the effect of grouping. The final column adds
nested selection (see the note below) and is what the README headline and the web app report:

| Target | RepeatedKFold | Replicate-grouped | Inflation | Nested grouped (reported) |
|---|---|---|---|---|
| Grain Size | 0.636 | 0.356 (± 1.71) | −0.28 | **0.352** (± 1.71) |
| Yield Stress | 0.685 | 0.463 | −0.22 | **0.426** (± 0.32) |
| UTS | 0.692 | 0.646 | −0.05 | **0.616** (± 0.19) |
| Grain Shape (bal. acc.) | 0.702 | 0.665 | −0.04 | **0.656** (± 0.08) |
| Hardness | 0.554 | 0.519 | −0.03 | **0.478** (± 0.23) |
| Relative Density | 0.657 | 0.678 | +0.02 | **0.661** (± 0.13) |
| Microstructure Plane (bal. acc.) | 0.799 | 0.813 | +0.01 | **0.805** (± 0.04) |

**Interpretation.** Relative Density, Hardness and Microstructure Plane never leaned on replicate
leakage — their scores stand as reported. **Grain Size collapses**: at R² ≈ 0.36 with a fold
standard deviation of ±1.71 it is **not meaningfully predictable** from process parameters alone.
That is fully consistent with the η² = 1.00 between-study variance measured for grain size in
§6a — grain size is governed by alloy and thermal history, which this dataset does not encode.
The §4.5 "grain-size stabilisation" result (R² 0.51 → 0.64) is therefore an artefact of the same
leakage and should not be quoted.

**This is not the paper-level grouping rejected above.** Paper grouping pools 4–64 physically
*different* experiments per fold and destabilises the metric. Grouping identical **rows** is a
narrow, mechanical de-duplication: it removes exact duplicates from opposite sides of the split
and nothing else.

> **Model selection is also nested.** Because we now choose among several candidate models per
> target, the choice itself can overfit. `train_export.py` therefore wraps selection in an
> **inner** grouped CV and scores in an **outer** grouped loop, so the reported number reflects
> "the score of this selection procedure", not "the score of the luckiest candidate". The cost is
> a further −0.004 to −0.041 (rightmost column above), largest where candidates are closest and the
> per-fold winner is least stable — Hardness selects its winner in only 40 % of folds.
> `outputs/nested_grouped_results.csv` is the headline source; `grouped_model_comparison.csv` is
> for comparing candidates, not for quoting.

---

## 8. Guarding against overfitting

Overfitting is the central risk with small, noisy, literature-aggregated data. Controls applied:

1. **Regularised, shallow models** — capped tree depth, minimum-samples-per-leaf, moderate
   estimator counts, subsampling/column-sampling for XGBoost.
2. **Train-vs-CV gap reporting** — every table prints *both* the training-fold score and the
   held-out CV score. A large gap = memorisation, and is called out.
3. **Repeated (and optionally nested) CV** (§7) so reported numbers are honest, not cherry-picked.
4. **Beat-the-baseline test** (§6) — models must beat the Dummy predictor.
5. **Restrained feature engineering** (§5) — only physically-motivated features, no polynomial
   explosion, so we add signal without adding noise dimensions.
6. **Drop >50%-missing columns** (§4c) — avoids models leaning on mostly-imputed guesses.
7. **Permutation importance** — computed on held-out data (not impurity-based, which is biased
   toward high-cardinality features) to understand *what* drives predictions and sanity-check that
   the model relies on physically meaningful process parameters.

---

## 9. Metrics

- **Regression:** R² (primary), RMSE and MAE (original units). Reported as mean ± std across CV.
- **Classification:** balanced accuracy (robust to class imbalance) and macro-F1, plus a confusion
  matrix on out-of-fold predictions.

---

## 10. What we deliberately did *not* do

- **No listwise deletion** of rows for missing inputs (would leave ~226 rows and bias toward
  fully-reported studies).
- **No pre-split imputation / scaling** (leakage).
- **No group-aware CV by `References`** — statistically unstable given ~4 rows/paper (§7).
- **No SMOTE / synthetic oversampling** on this small, heterogeneous set — it invents data; we
  prefer honest reporting of imbalance.
- **No deep neural nets** — unjustified for ≤ a few hundred rows of tabular data; RF/XGB dominate.
- **No automated polynomial/interaction blow-up** — only physically-motivated engineered features.
- **No overwriting reported energy-density values** — physics imputation fills blanks only.
- **No treating range midpoints as ground truth without noting the assumption.**

---

## 11. How to run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl
jupyter notebook LPBF_ML_Analysis.ipynb   # run cells top-to-bottom
```

Each cell block is one logical stage (load → clean → physics-impute → feature-engineer → EDA →
per-target modelling → importance → summary), so results can be inspected incrementally. Outputs
(metric tables, figures) are written to `./outputs/`.

---

## 12. Reproducibility & limitations

- All randomness is seeded (`RANDOM_STATE = 42`).
- **Limitations to state in the thesis:** (a) range→midpoint discards within-range uncertainty;
  (b) iterative imputation of moderately-missing inputs still injects modelled estimates — treat
  those features' importances cautiously; (c) literature aggregation mixes alloys/machines/powders
  not captured as features, so absolute accuracy will not transfer to a single new machine without
  recalibration; (d) small per-target sample sizes mean wide confidence intervals — report the
  ± std, not just the mean.
