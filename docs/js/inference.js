/**
 * inference.js — pure-JavaScript tree-ensemble inference for the LPBF models.
 *
 * No ML runtime, no WASM, no network calls beyond fetching model.json. The Python
 * exporter (`train_export.py`) writes raw tree structures plus the fitted preprocessing
 * constants; this file walks those trees.
 *
 * The two decision rules differ between families and MUST NOT be unified:
 *   - sklearn : go left when  x[f] <= threshold   (NaN never reaches a tree — imputed)
 *   - xgboost : go left when  x[f] <  threshold ; NaN follows the stored `missing` branch
 * `train_export.py` verifies this file's Python twin reproduces each library's own
 * predict() to <1e-6 before shipping, and `verifyAgainstPython()` re-checks in-browser.
 */

/** Median-impute, append missing-indicator flags, one-hot encode. Mirrors sklearn. */
export function buildFeatureVector(row, pre) {
  const out = [];
  const raw = [];

  for (let i = 0; i < pre.numeric_features.length; i++) {
    const v = row[pre.numeric_features[i]];
    const num = v === null || v === undefined || v === '' ? NaN : Number(v);
    raw.push(num);
    out.push(Number.isNaN(num) ? pre.numeric_medians[i] : num);
  }

  // SimpleImputer(add_indicator=True) appends one flag per feature that had a missing
  // value during fit — in the recorded order, so the column layout matches training.
  for (const idx of pre.indicator_feature_indices) {
    out.push(Number.isNaN(raw[idx]) ? 1 : 0);
  }

  for (let i = 0; i < pre.categorical_features.length; i++) {
    const v = row[pre.categorical_features[i]];
    const val = v === null || v === undefined || v === '' ? pre.categorical_fill[i] : String(v);
    for (const cat of pre.categorical_categories[i]) {
      out.push(val === cat ? 1 : 0);
    }
  }
  return out;
}

/** Average over sklearn trees. Regression -> scalar; classification -> prob vector. */
function predictSklearnForest(spec, x) {
  const nTrees = spec.trees.length;
  if (spec.task === 'reg') {
    let sum = 0;
    for (const t of spec.trees) {
      let n = 0;
      while (t.left[n] !== -1) {
        n = x[t.feature[n]] <= t.threshold[n] ? t.left[n] : t.right[n];
      }
      sum += t.value[n];
    }
    return sum / nTrees;
  }
  let acc = null;
  for (const t of spec.trees) {
    let n = 0;
    while (t.left[n] !== -1) {
      n = x[t.feature[n]] <= t.threshold[n] ? t.left[n] : t.right[n];
    }
    const v = t.value[n];
    if (acc === null) acc = v.slice();
    else for (let k = 0; k < v.length; k++) acc[k] += v[k];
  }
  return acc.map((v) => v / nTrees);
}

/**
 * Sum xgboost leaf weights onto base_score; trees round-robin across output groups.
 *
 * `Math.fround` is load-bearing, not defensive. XGBoost stores thresholds as float32 and
 * casts features to float32 before comparing. An engineered ratio like 8/3 is
 * 2.6666666666666665 in double precision but 2.6666667 as float32 — exactly equal to a
 * threshold split on it. Comparing in double precision takes the wrong branch; on
 * Relative Density that flipped 13 of 400 trees and shifted the prediction by 1.46 %.
 */
function predictXgboost(spec, x) {
  const g = spec.n_groups;
  // Accumulate in float32 as well — xgboost sums leaf weights in single precision, so a
  // double-precision running total drifts by ~1e-4 over 400 trees.
  const acc = new Array(g).fill(Math.fround(spec.base_score));
  for (let i = 0; i < spec.trees.length; i++) {
    const t = spec.trees[i];
    let n = 0;
    while (t.feature[n] !== -1) {
      const v = Math.fround(x[t.feature[n]]);
      if (Number.isNaN(v)) n = t.missing[n];
      else n = v < Math.fround(t.threshold[n]) ? t.left[n] : t.right[n];
    }
    acc[i % g] = Math.fround(acc[i % g] + Math.fround(t.leaf[n]));
  }
  return acc;
}

/**
 * Raw ensemble output for one row, before any sigmoid/softmax/argmax.
 *
 * The single entry point for tree traversal, so verification harnesses exercise exactly
 * the code path users get instead of a copy that can drift from it.
 */
export function rawEnsembleOutput(targetSpec, row) {
  const x = buildFeatureVector(row, targetSpec.preprocessing);
  const spec = targetSpec.model;
  const out = spec.kind === 'xgboost'
    ? predictXgboost(spec, x)
    : predictSklearnForest(spec, x);
  return Array.isArray(out) ? out : [out];
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/**
 * Predict one target.
 * @returns {{value:number, unit:string}} for regression, or
 *          {{label:string, probabilities:Array<{label:string,p:number}>}} for classification.
 */
export function predictTarget(targetSpec, row) {
  const spec = targetSpec.model;
  const isXgb = spec.kind === 'xgboost';
  const rawOut = rawEnsembleOutput(targetSpec, row);

  if (targetSpec.task === 'reg') {
    return { value: rawOut[0], unit: targetSpec.unit || '' };
  }

  const classes = targetSpec.classes || [];
  let probs;
  if (isXgb) {
    // Binary xgboost emits a single logit; multiclass emits one score per class.
    probs = spec.n_groups === 1
      ? (() => { const p = sigmoid(rawOut[0]); return [1 - p, p]; })()
      : softmax(rawOut);
  } else {
    probs = rawOut;
  }

  const pairs = classes
    .map((label, i) => ({ label, p: probs[i] ?? 0 }))
    .sort((a, b) => b.p - a.p);
  return { label: pairs[0]?.label ?? '—', probabilities: pairs };
}

/** Derive the engineered ratio features from the raw user inputs (row-wise, as in training). */
export function withEngineered(rawRow, bundle) {
  const row = { ...rawRow };
  for (const [name, def] of Object.entries(bundle.engineered)) {
    const a = row[def.num];
    const b = row[def.den];
    const na = a === null || a === undefined || a === '' ? NaN : Number(a);
    const nb = b === null || b === undefined || b === '' ? NaN : Number(b);
    const r = na / nb;
    row[name] = Number.isFinite(r) ? r : null; // matches the ±inf -> NaN rule in training
  }
  return row;
}

/** Predict every target at once. */
export function predictAll(bundle, rawRow) {
  const row = withEngineered(rawRow, bundle);
  const results = {};
  for (const [name, spec] of Object.entries(bundle.targets)) {
    results[name] = { ...predictTarget(spec, row), spec };
  }
  return results;
}

/**
 * Monte-Carlo style uncertainty band.
 *
 * These ensembles have no dropout, so instead of faking Bayesian intervals we report the
 * *spread across the ensemble's own trees* — an honest depiction of model disagreement.
 * Reported alongside, never instead of, the cross-validated error.
 */
export function treeSpread(targetSpec, row) {
  if (targetSpec.task !== 'reg') return null;
  const x = buildFeatureVector(row, targetSpec.preprocessing);
  const spec = targetSpec.model;
  if (spec.kind !== 'sklearn_forest') return null; // boosted trees are additive, not i.i.d.

  const preds = spec.trees.map((t) => {
    let n = 0;
    while (t.left[n] !== -1) {
      n = x[t.feature[n]] <= t.threshold[n] ? t.left[n] : t.right[n];
    }
    return t.value[n];
  });
  const mean = preds.reduce((a, b) => a + b, 0) / preds.length;
  const sd = Math.sqrt(preds.reduce((a, b) => a + (b - mean) ** 2, 0) / preds.length);
  return { mean, sd };
}

/**
 * Self-check: replay vectors produced by Python and confirm this engine agrees.
 * Surfaces a silent export/inference divergence instead of showing wrong numbers.
 */
export function verifyAgainstPython(bundle, vectors, tol = 1e-4) {
  let maxErr = 0;
  let checked = 0;
  for (const tv of vectors) {
    const spec = bundle.targets[tv.target];
    if (!spec) continue;
    const got = rawEnsembleOutput(spec, tv.row);
    const want = Array.isArray(tv.expected) ? tv.expected : [tv.expected];
    for (let i = 0; i < want.length; i++) {
      maxErr = Math.max(maxErr, Math.abs(got[i] - want[i]));
    }
    checked++;
  }
  return { ok: maxErr <= tol, maxErr, checked };
}
