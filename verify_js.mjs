/**
 * verify_js.mjs — proves the browser engine reproduces Python's predictions.
 *
 * `train_export.py` writes `outputs/js_test_vectors.json`: input rows plus the raw model
 * output computed in Python from the exported JSON. This script feeds the same rows
 * through the *actual* `docs/js/inference.js` that ships to users and compares.
 *
 * Run:  node verify_js.mjs
 * Exits non-zero if any prediction diverges by more than 1e-6.
 */
import { readFileSync } from 'node:fs';
import { predictAll, rawEnsembleOutput, verifyAgainstPython } from './docs/js/inference.js';

const bundle = JSON.parse(readFileSync('docs/model.json', 'utf8'));
const vectors = JSON.parse(readFileSync('outputs/js_test_vectors.json', 'utf8'));

// Relative tolerance: float32 carries ~7 decimal digits, so a prediction near 100
// (Relative Density) cannot agree to 1e-6 absolute however exact the export is.
const TOL = 1e-4;
let maxErr = 0;
let failures = 0;
let checked = 0;

// --- 1. raw model output must match Python ---
// Calls the SHIPPED traversal via rawEnsembleOutput rather than re-implementing it. An
// earlier version of this file duplicated the loop, omitted Math.fround, and reported a
// 0.31 error that existed only in the copy — a test asserting against its own bug.
for (const tv of vectors) {
  const spec = bundle.targets[tv.target];
  if (!spec) {
    console.error(`missing target in bundle: ${tv.target}`);
    failures++;
    continue;
  }
  const out = rawEnsembleOutput(spec, tv.row);
  const got = Array.isArray(out) ? out : [out];
  const want = Array.isArray(tv.expected) ? tv.expected : [tv.expected];

  for (let i = 0; i < want.length; i++) {
    const err = Math.abs(got[i] - want[i]);
    if (err > maxErr) maxErr = err;
    if (err > TOL) {
      failures++;
      console.error(`MISMATCH ${tv.target}[${i}]: js=${got[i]} py=${want[i]} err=${err}`);
    }
  }
  checked++;
}

console.log(`raw-output check: ${checked} vectors, max error ${maxErr.toExponential(3)} ` +
            `(tol ${TOL})`);

// --- 1b. the in-browser self-check must agree with this script ---
const selfCheck = verifyAgainstPython(bundle, vectors, TOL);
console.log(`in-browser self-check: ${selfCheck.ok ? 'ok' : 'FAILED'} on ` +
            `${selfCheck.checked} vectors (max error ${selfCheck.maxErr.toExponential(3)})`);
if (!selfCheck.ok) failures++;

// --- 2. end-to-end sanity on the embedded samples ---
console.log('\nend-to-end predictions on embedded sample rows:');
let sanityFail = 0;
for (const [i, s] of (bundle.samples || []).slice(0, 4).entries()) {
  const results = predictAll(bundle, s.inputs);
  const parts = [];
  for (const [name, r] of Object.entries(results)) {
    if (r.spec.task === 'reg') {
      if (!Number.isFinite(r.value)) {
        console.error(`  NON-FINITE prediction for ${name}`);
        sanityFail++;
      }
      parts.push(`${name.slice(0, 18)}=${r.value.toFixed(2)}`);
    } else {
      const tot = r.probabilities.reduce((a, b) => a + b.p, 0);
      if (Math.abs(tot - 1) > 1e-6) {
        console.error(`  probabilities for ${name} sum to ${tot}, expected 1`);
        sanityFail++;
      }
      parts.push(`${name.slice(0, 18)}=${r.label}`);
    }
  }
  console.log(`  sample ${i + 1}: ${parts.join('  ')}`);
}

// --- 3. missing inputs must not crash and must still produce finite output ---
console.log('\nall-blank input (pure median imputation):');
const blank = {};
for (const f of bundle.raw_numeric_inputs) blank[f] = null;
for (const f of bundle.categorical_inputs) blank[f] = null;
const blankRes = predictAll(bundle, blank);
for (const [name, r] of Object.entries(blankRes)) {
  const shown = r.spec.task === 'reg' ? r.value.toFixed(3) : r.label;
  if (r.spec.task === 'reg' && !Number.isFinite(r.value)) {
    console.error(`  NON-FINITE for ${name}`);
    sanityFail++;
  }
  console.log(`  ${name.slice(0, 40).padEnd(42)} ${shown}`);
}

const ok = failures === 0 && sanityFail === 0;
console.log(`\n${ok ? 'PASS' : 'FAIL'} — ${failures} mismatches, ${sanityFail} sanity failures, ` +
            `max error ${maxErr.toExponential(3)} (tol ${TOL})`);
process.exit(ok ? 0 : 1);
