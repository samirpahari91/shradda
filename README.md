# LPBF Process–Property Machine Learning

Predict laser powder-bed-fusion (LPBF) part properties — hardness, yield stress, ultimate
tensile strength, relative density, grain size, grain shape and microstructure plane — from
process parameters, using 674 specimens compiled from 63 peer-reviewed publications.

The repository contains both halves of the work:

| | |
|---|---|
| **Analysis** | `LPBF_ML_Analysis.ipynb` — the full thesis pipeline (cleaning → imputation → feature engineering → per-target modelling → ablations → benchmarks → leakage correction) |
| **Deployment** | `docs/` — a dependency-free static web app that runs the selected models entirely in the browser |

**[▶ Live demo](https://YOUR-USERNAME.github.io/YOUR-REPO/)** — replace with your GitHub Pages URL
after following [Deployment](#deployment).

---

## Headline results

All scores are **nested replicate-grouped cross-validation** — the honest estimate, and the exact
numbers the web app displays. The inner loop selects the model and the outer loop scores it, so
the headline is not inflated by the selection itself. See
[Why grouped CV](#why-grouped-cv-matters) for why the more familiar `RepeatedKFold` numbers are
inflated, and [Nested vs flat](#nested-vs-flat-grouped-cv) for why these sit ~0.02–0.04 below the
single-loop grouped figures in `outputs/grouped_model_comparison.csv`.

Reliability labels below are the exact badges the app renders, computed from the same thresholds it
ships (see [Reliability tiers](#reliability-tiers)).

| Property | Task | n | Score | Reliability |
|---|---|---|---|---|
| Relative Density (%) | regression | 447 | R² 0.661 (±0.134) | Reliable |
| Ultimate Tensile Strength (MPa) | regression | 291 | R² 0.616 (±0.190) | Reliable |
| Hardness (HV) | regression | 173 | R² 0.478 (±0.228) | Indicative |
| Yield Stress (MPa) | regression | 249 | R² 0.426 (±0.325) | Low confidence — below the 0.45 cutoff |
| Grain Size (µm) | regression | 202 | R² 0.352 (±1.709) | Low confidence — see caveat |
| Microstructure Plane | classification | 315 | bal. acc. 0.805 (±0.041) | Reliable |
| Grain Shape | classification | 311 | bal. acc. 0.656 (±0.082) | Reliable |

> Exact figures are written to `outputs/nested_grouped_results.csv` and
> `outputs/selected_models.csv` when you run `train_export.py`; the web app reads them from
> `docs/model.json` so the UI can never disagree with the model it ships. The table above is
> transcribed from those same nested figures — rounded to two decimals — and the reliability tiers
> are the ones `app.js` computes from the shipped `cv_score`, so the README cannot drift from the
> UI either.

**The accuracy ceiling is a property of the data, not the models.** A one-way ANOVA against
publication identity attributes **51–100 % of each property's variance to between-study
factors** — alloy, machine, powder and heat treatment — none of which are columns in this
dataset. A four-way Optuna-tuned benchmark (RF / XGBoost / LightGBM / CatBoost) confirms all
families agree within ~0.03 R². The highest-value next step is acquiring compositional and
machine metadata, not a more complex model.

### Why grouped CV matters

129 of 674 rows share an **exactly identical process-input vector** with another row — a study
reporting three grain-size measurements at one laser setting, or the same build measured on two
planes. Per target, 15–35 % of rows have such a twin. Ordinary `RepeatedKFold` shuffles rows at
random and routinely puts one twin in *train* and the other in *test*, so the model is scored on
settings whose answers it has memorised.

Grouping folds on the exact input vector removes the leak, and the correction is large. Both
columns below are single-loop (flat) CV, so the comparison isolates the effect of grouping alone;
see [Nested vs flat](#nested-vs-flat-grouped-cv) for the further small drop to the headline
figures:

| Target | RepeatedKFold | Grouped (honest) | Inflation |
|---|---|---|---|
| Grain Size | 0.64 | 0.36 | **−0.28** |
| Yield Stress | 0.69 | 0.46 | **−0.23** |
| UTS | 0.69 | 0.65 | −0.04 |
| Grain Shape | 0.70 | 0.66 | −0.04 |
| Hardness | 0.55 | 0.52 | −0.03 |
| Relative Density | 0.66 | 0.68 | +0.02 |
| Microstructure Plane | 0.81 | 0.81 | 0.00 |

Relative Density, Hardness and Microstructure Plane were never leaning on replicates and their
scores stand. **Grain Size collapses** — with a fold standard deviation of ±1.7 it is not
meaningfully predictable from process parameters alone, consistent with its η² = 1.00
between-study variance. The web app labels it *Low confidence* rather than hiding it.

This is *not* grouping by publication. Paper-level `GroupKFold` was tried and rejected: with a
median of 4 rows per paper (one contributing 64), folds become wildly uneven and the metric
mostly reflects which large study landed in the test fold. Grouping duplicate **rows** is a
narrow, mechanical de-duplication with none of that instability.

### Reliability tiers

The badge on each result card is derived from the cross-validated score, not hand-assigned:

| Tier | Regression (R²) | Classification (lift over chance) |
|---|---|---|
| **Reliable** | ≥ 0.60 | ≥ 0.55 |
| **Indicative** | ≥ 0.45 | ≥ 0.30 |
| **Low confidence** | below 0.45 | below 0.30 |

Classification uses *lift* — `(score − chance) / (1 − chance)` with `chance = 1/n_classes` — because
balanced accuracy of 0.5 is worthless on a 2-class target and respectable on a 6-class one. Grain
Shape's 0.656 over six classes is a lift of 0.59, which is why it earns *Reliable* despite a lower
raw number than Relative Density.

These thresholds live in one place: `RELIABILITY` in `train_export.py`, exported into
`docs/model.json` as a `reliability` block. `app.js` reads it to render the badge and
`_check_docs.py` reads it to validate this table, so changing a cutoff updates the app, the gate and
this documentation together rather than leaving three copies to drift.

### Nested vs flat grouped CV

Grouping fixes the leak, but a single grouped loop still has a subtler bias: the winning model per
target is *chosen* on the same folds that score it, so the reported number inherits the luck of
that choice. Nested CV separates the two — an inner loop picks the model, an outer loop scores the
pick on data the selection never saw. The headline table quotes the nested figures:

| Target | Flat grouped | Nested grouped | Selection bias | Most-selected model (rate) |
|---|---|---|---|---|
| Relative Density | 0.678 | **0.661** | −0.017 | XGBoost/mid (60 %) |
| UTS | 0.646 | **0.616** | −0.030 | ExtraTrees (52 %) |
| Hardness | 0.519 | **0.478** | −0.041 | RandomForest/deep (40 %) |
| Yield Stress | 0.463 | **0.426** | −0.037 | ExtraTrees (52 %) |
| Grain Size | 0.356 | **0.352** | −0.004 | ExtraTrees (60 %) |
| Microstructure Plane | 0.813 | **0.805** | −0.008 | RandomForest/d8 (40 %) |
| Grain Shape | 0.665 | **0.656** | −0.009 | ExtraTrees (76 %) |

The penalty is small (≤0.04) and largest exactly where the candidate models are closest together
and selection rates are lowest — Hardness picks its winner in only 40 % of folds, so "the best
model" there is barely distinguishable from its runners-up. Since no selection rate reaches 100 %,
the model refit for deployment is the overall winner, which need not equal the per-fold favourite:
Relative Density ships `XGBoost/reg` although `XGBoost/mid` won most inner folds. That is expected,
not a bug — but it is why the flat number is the wrong one to advertise.

---

## Repository layout

```
.
├── Data.xlsx                     # source dataset (674 × 18)
├── LPBF_ML_Analysis.ipynb        # full analysis notebook
├── build_notebook.py             # regenerates the notebook from source
├── _run_nb.py                    # executes the notebook (nbconvert-config workaround)
├── train_export.py               # honest evaluation + browser model export
├── verify_js.mjs                 # proves the JS engine matches Python
├── _check_docs.py                # asserts docs quote the shipped nested scores
├── METHODOLOGY_README.md         # every modelling decision, justified
├── result_explaination.txt       # paper-style results report
├── outputs/                      # metric tables (CSV) and figures (PNG)
└── docs/                         # ← the deployable web app (GitHub Pages root)
    ├── index.html
    ├── model.json                # trees + preprocessing constants + samples
    ├── css/styles.css
    └── js/
        ├── app.js                # UI: form generation, validation, rendering
        └── inference.js          # tree-ensemble evaluation (no ML runtime)
```

---

## The web app

A **pure static site** — no backend, no build step, no ML runtime. `model.json` carries the raw
decision-tree structures plus the fitted preprocessing constants, and `inference.js` walks them
in plain JavaScript.

- **All seven properties predicted at once** in one pass, each with its model name, training
  size, cross-validated score and a reliability tier derived from that score.
- **Dynamic form** generated from the `features` metadata in `model.json` — add a feature in
  Python and the form grows automatically.
- **Missing inputs are legitimate.** Leave any field blank and it is median-imputed exactly as
  during training, with the same missing-indicator flag the model was trained to read.
- **Derived features shown live** — `hatch/beam`, `layer/beam`, `hatch/layer` update as you
  type, so the physics the model actually sees is visible.
- **Inline validation** with extrapolation warnings: values outside the training envelope are
  flagged, because tree ensembles cannot extrapolate and silently clamp to their edge leaf.
- **Sample Data** cycles through 12 real specimens embedded in the metadata and shows the source
  study's measured values next to the predictions, with per-property deltas.
- **Accessibility**: semantic landmarks, labelled inputs, `aria-describedby` hints, live regions
  announcing results, visible focus rings, full keyboard operation, honoured
  `prefers-reduced-motion` and `prefers-contrast`, and AA-contrast light *and* dark themes.
- **Responsive** mobile-first layout, single column on phones and a sticky two-pane layout on
  desktop.
- **Graceful failure**: if `model.json` cannot be fetched the app explains why — including the
  common `file://` mistake — instead of showing a blank page.

### Uncertainty, honestly

The models are tree ensembles with no dropout, so there is no Monte-Carlo dropout interval to
report. Rather than invent one, each regression card shows the **cross-validated MAE** — a real
out-of-sample error estimate. `inference.js` also exposes `treeSpread()`, which measures
disagreement across a random forest's trees; it is deliberately *not* presented as a confidence
interval, because ensemble spread systematically understates true predictive uncertainty.

### Run it locally

`fetch()` is blocked on `file://`, so the folder must be served:

```bash
cd docs
python -m http.server 8000
# then open http://localhost:8000
```

---

## Reproducing the analysis

### Requirements

Python 3.10+, then:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna \
            matplotlib seaborn openpyxl nbformat
```

Pinned versions used for the reported results: scikit-learn 1.7.2, XGBoost 3.2.0,
LightGBM 4.7.0, CatBoost 1.2.10, Optuna 4.9.0. Random seed 42 throughout.

### Retraining and re-exporting the models

```bash
python train_export.py      # evaluate, select, refit, export -> docs/model.json
node verify_js.mjs          # confirm the browser engine matches Python
```

`train_export.py`:

1. Cleans the data (`"120-160"` → 140 midpoints, categorical normalisation).
2. Physics-imputes the energy densities from `P`, `v`, `h`, `t` — row-wise, fill-only.
3. Adds the engineered ratio features.
4. Scores **every candidate model** for **every target** under replicate-grouped CV
   (`outputs/grouped_model_comparison.csv`).
5. Runs **nested** grouped CV — inner loop selects, outer loop scores — so the headline number
   is not inflated by the selection itself (`outputs/nested_grouped_results.csv`).
6. Refits the winner per target on all labelled rows and exports to `docs/model.json`.
7. **Verifies the export**: a Python re-implementation of the exported JSON must reproduce
   scikit-learn's / XGBoost's own `predict()` to within 1e-6, or the script aborts rather than
   shipping a silently wrong model.

`verify_js.mjs` then replays Python-generated test vectors through the *actual* shipped
`inference.js` and exits non-zero on any divergence above 1e-6. Both gates must pass before
deploying.

A third gate, `python _check_docs.py`, asserts that the scores quoted in this README,
`METHODOLOGY_README.md` and `result_explaination.txt` are the nested figures actually shipped in
`docs/model.json`, and that each reliability label matches the tier `app.js` computes from that
score. It exists because the two CSVs are easy to confuse — `nested_grouped_results.csv` is the
citable headline, while `grouped_model_comparison.csv` only ranks candidates and reads ~0.02–0.04
higher.

### Regenerating the notebook

`LPBF_ML_Analysis.ipynb` is generated, not hand-edited:

```bash
python build_notebook.py                     # writes the .ipynb
python _run_nb.py                            # executes it in place (~25 min)
```

Edit `build_notebook.py`, not the notebook, or your changes will be overwritten.

`_run_nb.py` drives the same nbclient engine as `jupyter nbconvert --execute --inplace`, which is
the documented invocation but fails on some machines: a user-level `~/.jupyter` config can register
a `jupyter_contrib_nbextensions` preprocessor that, when the package is absent from the active
environment, aborts nbconvert before the first cell runs (`ModuleNotFoundError`). The wrapper skips
that config layer. Use nbconvert directly if your environment is clean.

---

## Deployment

The app is self-contained in `docs/`, which GitHub Pages can serve directly.

1. Push this repository to GitHub.
2. **Settings → Pages**.
3. Under *Build and deployment*, set **Source** = `Deploy from a branch`,
   **Branch** = `main`, **Folder** = `/docs`. Save.
4. Wait for the Pages action, then open `https://<user>.github.io/<repo>/`.
5. Update the demo link at the top of this README.

No build step, bundler or CI is required — every asset is committed and served as-is. To host at
a different path, note that `docs/index.html` references `model.json`, `css/`, and `js/`
relatively, so any subdirectory works unchanged.

---

## Limitations

Please read these before using a prediction for anything consequential.

- **Screening tool, not a qualification tool.** Predictions are alloy-agnostic averages across
  many materials, machines and powders. Expect a real offset for any single machine until you
  recalibrate against your own measurements.
- **Grain Size is not reliably predictable here** (nested grouped R² ≈ 0.35, fold s.d. ±1.71). It
  is surfaced for completeness and labelled *Low confidence*.
- **Yield Stress is also below the reliability bar** (nested grouped R² ≈ 0.43 against a 0.45
  threshold, fold s.d. ±0.32). The app labels it *Low confidence* too; treat its predictions as
  directional only.
- **No extrapolation.** Tree ensembles clamp to their edge prediction outside the training
  envelope. The app warns; it cannot fix it.
- **Range→midpoint parsing discards within-range uncertainty** — `"120-160"` becomes 140 with no
  record of the spread.
- **Median imputation injects the dataset's central tendency.** Leaving many fields blank pulls
  predictions toward the corpus average; the missing-indicator flags let the model react to
  absence, but they cannot recover the missing information.
- **`Scanning strategy` was dropped** (52.2 % missing, above the 50 % threshold), so the models
  are blind to scan pattern.
- **`Density measurement method` is metadata, not a process setting.** It is retained because it
  carries predictive signal about how a study measured density, but it is not something you
  "set" on a machine.
- **Small per-target samples** (173–447 rows) mean wide confidence intervals. Always quote the
  ± standard deviation, not the mean alone.

## Documentation

- `SOP.txt` — step-by-step operating procedure from a clean machine to a live GitHub Pages
  deployment, with a verification gate after every stage and the environment traps that have
  actually bitten this project. Start here if you are reproducing or re-deploying the work.
- `METHODOLOGY_README.md` — every modelling decision and its justification, including the
  choices deliberately *not* made.
- `result_explaination.txt` — paper-style report (abstract → methods → results → discussion).
- `outputs/` — all metric tables and figures backing the numbers quoted above.
