/**
 * app.js — UI controller for the LPBF property predictor.
 *
 * Responsibilities: load model.json, build the form from its `features` metadata,
 * validate input, run inference, and render results. All prediction maths lives in
 * inference.js.
 */
import { predictAll, treeSpread, withEngineered, verifyAgainstPython } from './inference.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Slug for DOM ids — feature names contain spaces, parens and unicode. */
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

const state = {
  bundle: null,
  values: {},           // featureName -> raw string from the input
  lastResults: null,
  sampleActual: null,
  sampleIndex: -1,
  autoPredict: false,   // enabled after the first successful manual predict
};

/* ============================ number formatting ============================ */

function fmt(v, digits = null) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  const d = digits ?? (a >= 1000 ? 0 : a >= 100 ? 1 : a >= 10 ? 2 : a >= 1 ? 2 : 3);
  return v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}

/** Sensible display precision per property — density needs decimals, MPa does not. */
function fmtTarget(v, targetName) {
  if (/Relative Density/i.test(targetName)) return fmt(v, 2);
  if (/MPa|Hardness/i.test(targetName)) return fmt(v, 1);
  if (/Grain Size/i.test(targetName)) return fmt(v, 2);
  return fmt(v);
}

/**
 * Reliability tiers come from model.json (`reliability`), exported by train_export.py,
 * so this UI, the README table and _check_docs.py all read one definition. The fallback
 * matches schema_version 2 bundles, which predate the block.
 */
const RELIABILITY_FALLBACK = {
  order: ['good', 'moderate', 'low'],
  labels: { good: 'Reliable', moderate: 'Indicative', low: 'Low confidence' },
  reg: { metric: 'score', good: 0.6, moderate: 0.45 },
  clf: { metric: 'lift', good: 0.55, moderate: 0.3 },
};

const reliabilityCfg = () => state.bundle?.reliability ?? RELIABILITY_FALLBACK;

/**
 * Reliability tier from the honest grouped-CV score. Regression compares R² directly;
 * classification compares lift over the chance rate, because balanced accuracy of 0.5
 * is worthless on 2 classes and meaningful on 6.
 */
function reliability(spec) {
  const cfg = reliabilityCfg();
  const rule = cfg[spec.task] ?? RELIABILITY_FALLBACK[spec.task];
  let v = spec.cv_score;
  if (rule.metric === 'lift') {
    const chance = 1 / Math.max((spec.classes || []).length, 2);
    v = (v - chance) / (1 - chance);
  }
  // Walk tiers best-first; the last tier is the implicit floor and needs no threshold.
  const order = cfg.order ?? RELIABILITY_FALLBACK.order;
  for (const tier of order) {
    const cut = rule[tier];
    if (cut === undefined) return tier;
    if (v >= cut) return tier;
  }
  return order[order.length - 1];
}

const reliabilityLabel = (tier) =>
  (reliabilityCfg().labels ?? RELIABILITY_FALLBACK.labels)[tier] ?? tier;

/**
 * "Other" is the bucket that classes with fewer than 8 training samples were merged into.
 * Shown verbatim it reads like a real microstructure, so name it for what it is.
 */
function classLabel(label, spec) {
  if (label !== 'Other') return label;
  const n = (spec.rare_classes || []).length;
  return n ? `Other (${n} rare types)` : 'Other';
}

function classTitle(label, spec) {
  if (label !== 'Other') return label;
  const rare = spec.rare_classes || [];
  return rare.length
    ? `Rare classes merged for training (fewer than 8 samples each): ${rare.join(', ')}`
    : 'Rare classes merged during training';
}

/* ============================== form building ============================== */

function buildForm() {
  const { features } = state.bundle;
  const container = $('#fields');
  container.innerHTML = '';

  const order = [...state.bundle.raw_numeric_inputs, ...state.bundle.categorical_inputs];

  for (const name of order) {
    const meta = features[name];
    if (!meta) continue;

    const id = `f-${slug(name)}`;
    const hintId = `${id}-hint`;
    const field = document.createElement('div');
    field.className = 'field';
    field.dataset.feature = name;

    const label = document.createElement('label');
    label.className = 'field-label';
    label.setAttribute('for', id);
    label.innerHTML = `<span>${escapeHtml(stripUnit(name))}</span>` +
      (meta.unit ? `<span class="field-unit">${escapeHtml(meta.unit)}</span>` : '');
    field.append(label);

    const wrap = document.createElement('div');
    wrap.className = 'field-input-wrap';

    let input;
    if (meta.type === 'select') {
      input = document.createElement('select');
      input.innerHTML = `<option value="">Not specified (dataset mode)</option>` +
        meta.options.map((o) => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`).join('');
    } else {
      input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      input.inputMode = 'decimal';
      input.placeholder = `median ${fmt(meta.median)}`;
      // No hard min/max: users may legitimately explore beyond the corpus, and we warn
      // rather than block. Bounds are advisory, surfaced in the hint text.
      input.setAttribute('aria-describedby', hintId);
    }
    input.id = id;
    input.name = name;
    if (meta.type === 'select') input.setAttribute('aria-describedby', hintId);

    input.addEventListener('input', () => onFieldChange(name, input.value, field));
    input.addEventListener('blur', () => validateField(name, input.value, field, true));
    wrap.append(input);
    field.append(wrap);

    const hint = document.createElement('p');
    hint.className = 'field-hint';
    hint.id = hintId;
    hint.textContent = meta.type === 'select'
      ? `${meta.options.length} options · ${meta.missing_pct}% missing in dataset`
      : `Typical ${fmt(meta.p1)} – ${fmt(meta.p99)}`;
    field.append(hint);

    container.append(field);
  }

  $('#derived-panel').hidden = false;
  updateDerived();
}

const stripUnit = (n) => n.replace(/\s*\([^)]*\)\s*$/, '').replace(/\s*%$/, '').trim();

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ============================== validation ============================== */

/** @returns {boolean} true when the field holds a usable (or empty) value. */
function validateField(name, value, field, showRange = false) {
  const meta = state.bundle.features[name];
  const hint = $('.field-hint', field);
  const raw = String(value).trim();

  field.dataset.invalid = 'false';
  field.dataset.warn = 'false';

  const reset = () => {
    hint.removeAttribute('data-level');
    hint.textContent = meta.type === 'select'
      ? `${meta.options.length} options · ${meta.missing_pct}% missing in dataset`
      : `Typical ${fmt(meta.p1)} – ${fmt(meta.p99)}`;
  };

  if (raw === '') { reset(); return true; }
  if (meta.type === 'select') { reset(); return true; }

  const n = Number(raw);
  if (!Number.isFinite(n)) {
    field.dataset.invalid = 'true';
    hint.dataset.level = 'error';
    hint.textContent = 'Enter a valid number.';
    return false;
  }
  if (n < 0) {
    field.dataset.invalid = 'true';
    hint.dataset.level = 'error';
    hint.textContent = 'Must be zero or positive.';
    return false;
  }
  if (n === 0 && /power|speed|beam|hatch|thickness/i.test(name)) {
    field.dataset.invalid = 'true';
    hint.dataset.level = 'error';
    hint.textContent = 'Must be greater than zero.';
    return false;
  }
  // Outside the training envelope: warn, don't block. Tree ensembles cannot
  // extrapolate — they clamp to the edge leaf — so the user must know.
  if (showRange && (n < meta.min || n > meta.max)) {
    field.dataset.warn = 'true';
    hint.dataset.level = 'warn';
    hint.textContent = `Outside training range (${fmt(meta.min)}–${fmt(meta.max)}); ` +
      `the model cannot extrapolate.`;
    return true;
  }
  reset();
  return true;
}

function validateAll(showRange = true) {
  let ok = true;
  for (const field of $$('.field')) {
    const name = field.dataset.feature;
    const input = $('input, select', field);
    if (!validateField(name, input.value, field, showRange)) ok = false;
  }
  return ok;
}

function onFieldChange(name, value, field) {
  state.values[name] = value;
  validateField(name, value, field, false);
  updateDerived();
  // Once the user has predicted once, keep results live — but never on invalid input.
  if (state.autoPredict) {
    clearTimeout(onFieldChange._t);
    onFieldChange._t = setTimeout(() => { if (validateAll(false)) runPrediction(false); }, 260);
  }
}

/* ============================ derived features ============================ */

function currentRow() {
  const row = {};
  for (const name of [...state.bundle.raw_numeric_inputs, ...state.bundle.categorical_inputs]) {
    const v = state.values[name];
    row[name] = v === undefined || String(v).trim() === '' ? null : v;
  }
  return row;
}

function updateDerived() {
  const row = withEngineered(currentRow(), state.bundle);
  const grid = $('#derived-grid');
  const defs = {
    hatch_over_beam: ['hatch / beam', 'Track overlap — below 1 means overlapping tracks'],
    layer_over_beam: ['layer / beam', 'Penetration vs spot size'],
    aspect_h_t: ['hatch / layer', 'Melt-pool geometry proxy'],
  };
  grid.innerHTML = Object.entries(defs).map(([k, [label, tip]]) => {
    const v = row[k];
    return `<div><dt title="${escapeHtml(tip)}">${escapeHtml(label)}</dt>` +
      `<dd>${v === null || v === undefined ? '—' : fmt(v, 3)}</dd></div>`;
  }).join('');
}

/* ============================== rendering ============================== */

function showSkeletons() {
  const grid = $('#results');
  grid.setAttribute('aria-busy', 'true');
  const n = Object.keys(state.bundle.targets).length;
  grid.innerHTML = Array.from({ length: n }, () => `
    <div class="skeleton-card" aria-hidden="true">
      <div class="skeleton-line sk-sm"></div>
      <div class="skeleton-line sk-lg"></div>
      <div class="skeleton-line sk-xs"></div>
    </div>`).join('');
}

function renderResults(results) {
  const grid = $('#results');
  const row = withEngineered(currentRow(), state.bundle);
  const cards = [];

  // Regression first, then classification — matches how the thesis reports them.
  const entries = Object.entries(results).sort(([, a], [, b]) => {
    if (a.spec.task !== b.spec.task) return a.spec.task === 'reg' ? -1 : 1;
    return b.spec.cv_score - a.spec.cv_score;
  });

  for (const [name, res] of entries) {
    const spec = res.spec;
    const rel = reliability(spec);
    const scoreTxt = `${spec.metric} ${spec.cv_score.toFixed(2)} ± ${spec.cv_std.toFixed(2)}`;

    let body;
    if (spec.task === 'reg') {
      const spread = treeSpread(spec, row);
      // Prefer the cross-validated MAE — an out-of-sample error estimate — over the
      // ensemble's internal spread, which understates true uncertainty.
      const band = spec.mae != null
        ? `± ${fmtTarget(spec.mae, name)} ${escapeHtml(spec.unit)} typical error (CV MAE)`
        : spread ? `± ${fmtTarget(spread.sd, name)} across ensemble trees` : '';
      body = `
        <div class="rc-value">
          <span class="rc-number">${fmtTarget(res.value, name)}</span>
          <span class="rc-unit">${escapeHtml(spec.unit)}</span>
        </div>
        ${band ? `<p class="rc-interval">${band}</p>` : ''}`;
    } else {
      const probs = res.probabilities.slice(0, 4);
      body = `
        <div class="rc-value"><span class="rc-class">${escapeHtml(classLabel(res.label, spec))}</span></div>
        <div class="rc-probs">
          ${probs.map((p) => {
            const lbl = classLabel(p.label, spec);
            return `
            <div class="prob-row">
              <span class="prob-label" title="${escapeHtml(classTitle(p.label, spec))}">${escapeHtml(lbl)}</span>
              <span class="prob-track"><span class="prob-fill" style="width:${(p.p * 100).toFixed(1)}%"></span></span>
              <span class="prob-pct">${(p.p * 100).toFixed(0)}%</span>
            </div>`;
          }).join('')}
        </div>`;
    }

    cards.push(`
      <article class="result-card" data-reliability="${rel}">
        <div class="rc-head">
          <h3 class="rc-name">${escapeHtml(stripUnit(name))}</h3>
          <span class="rc-tag" data-reliability="${rel}">${escapeHtml(reliabilityLabel(rel))}</span>
        </div>
        ${body}
        <div class="rc-foot">
          <span>${escapeHtml(scoreTxt)}</span>
          <span><code>${escapeHtml(spec.model_name)}</code> · n=${spec.n_train}</span>
        </div>
      </article>`);
  }

  grid.innerHTML = cards.join('');
  grid.setAttribute('aria-busy', 'false');

  const badge = $('#results-badge');
  badge.hidden = false;
  badge.textContent = `${entries.length} properties predicted`;

  // Announce a concise summary rather than the whole card grid.
  $('#sr-results').textContent = entries.map(([n, r]) =>
    `${stripUnit(n)}: ${r.spec.task === 'reg'
      ? `${fmtTarget(r.value, n)} ${r.spec.unit}`
      : classLabel(r.label, r.spec)}`).join('. ');
}

function renderSampleActual() {
  const box = $('#sample-actual');
  if (!state.sampleActual || !Object.keys(state.sampleActual).length) { box.hidden = true; return; }
  const grid = $('#sample-actual-grid');
  grid.innerHTML = Object.entries(state.sampleActual).map(([name, actual]) => {
    const pred = state.lastResults?.[name];
    let delta = '';
    if (pred && typeof actual === 'number' && pred.spec.task === 'reg') {
      const err = Math.abs(pred.value - actual);
      const good = pred.spec.mae != null ? err <= pred.spec.mae : false;
      delta = `<span class="sample-delta" data-good="${good}">Δ ${fmtTarget(err, name)}</span>`;
    } else if (pred && typeof actual === 'string' && pred.spec.task === 'clf') {
      const good = pred.label === actual;
      delta = `<span class="sample-delta" data-good="${good}">${good ? '✓ match' : '✗ differs'}</span>`;
    }
    const shown = typeof actual === 'number' ? fmtTarget(actual, name) : escapeHtml(actual);
    return `<div><dt>${escapeHtml(stripUnit(name))}</dt><dd>${shown}${delta}</dd></div>`;
  }).join('');
  box.hidden = false;
}

/* ============================== actions ============================== */

async function runPrediction(withSkeleton = true) {
  if (!state.bundle) return;
  if (!validateAll(true)) {
    showFormError('Please fix the highlighted fields before predicting.');
    return;
  }
  clearFormError();

  const btn = $('#predict-btn');
  if (withSkeleton) {
    showSkeletons();
    btn.disabled = true;
    // One frame so the skeleton actually paints before the (fast) synchronous inference.
    await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 140)));
  }

  try {
    const results = predictAll(state.bundle, currentRow());
    state.lastResults = results;
    renderResults(results);
    renderSampleActual();
    state.autoPredict = true;
    $('#auto-note').textContent = 'Predictions update automatically as you type.';
  } catch (err) {
    console.error(err);
    showFormError('Prediction failed unexpectedly. Please reload the page.');
    $('#results').setAttribute('aria-busy', 'false');
  } finally {
    btn.disabled = false;
  }
}

function fillSample() {
  const samples = state.bundle.samples || [];
  if (!samples.length) { toast('No sample rows available.'); return; }
  // Walk sequentially so repeated clicks always show something new.
  state.sampleIndex = (state.sampleIndex + 1) % samples.length;
  const s = samples[state.sampleIndex];

  for (const field of $$('.field')) {
    const name = field.dataset.feature;
    const input = $('input, select', field);
    const v = s.inputs[name];
    input.value = v === null || v === undefined ? '' : v;
    state.values[name] = input.value;
    validateField(name, input.value, field, false);
  }
  state.sampleActual = s.actual || null;
  updateDerived();
  clearFormError();
  toast(`Loaded sample ${state.sampleIndex + 1} of ${samples.length}`);
  runPrediction(true);
}

function resetForm() {
  for (const field of $$('.field')) {
    const input = $('input, select', field);
    input.value = '';
    state.values[field.dataset.feature] = '';
    validateField(field.dataset.feature, '', field, false);
  }
  state.sampleActual = null;
  state.lastResults = null;
  state.autoPredict = false;
  updateDerived();
  clearFormError();
  $('#sample-actual').hidden = true;
  $('#results-badge').hidden = true;
  $('#auto-note').textContent = 'Predictions update automatically as you type.';
  $('#results').innerHTML = `
    <div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
        <path d="M4 19h16M6 19V9m5 10V5m5 14v-7" stroke-linecap="round" />
      </svg>
      <p>Enter parameters and press <strong>Predict properties</strong> to see all
         seven predictions.</p>
    </div>`;
  $('#sr-results').textContent = 'Form cleared.';
}

function showFormError(msg) {
  const el = $('#form-error');
  el.textContent = msg;
  el.hidden = false;
}
function clearFormError() { $('#form-error').hidden = true; }

let toastTimer;
function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2600);
}

/* ============================== model card ============================== */

function renderPerfTable() {
  const body = $('#perf-body');
  const rows = Object.entries(state.bundle.targets)
    .sort(([, a], [, b]) => {
      if (a.task !== b.task) return a.task === 'reg' ? -1 : 1;
      return b.cv_score - a.cv_score;
    })
    .map(([name, spec]) => {
      const rel = reliability(spec);
      const pct = Math.max(0, Math.min(100, spec.cv_score * 100));
      const err = spec.task === 'reg'
        ? `± ${fmtTarget(spec.mae, name)} ${escapeHtml(spec.unit)} <span class="muted">MAE</span>`
        : `macro-F1 ${spec.macro_f1 ?? '—'}`;
      return `
        <tr>
          <td>${escapeHtml(stripUnit(name))}</td>
          <td><code>${escapeHtml(spec.model_name)}</code></td>
          <td class="num">${spec.n_train}</td>
          <td>
            <div class="score-cell">
              <span class="num">${spec.cv_score.toFixed(3)} ± ${spec.cv_std.toFixed(3)}</span>
              <span class="score-bar"><span class="score-fill" data-reliability="${rel}" style="width:${pct}%"></span></span>
            </div>
            <span class="muted" style="font-size:.74rem">${escapeHtml(spec.metric)}</span>
          </td>
          <td class="num">${err}</td>
        </tr>`;
    });
  body.innerHTML = rows.join('');
}

/* ============================== theme ============================== */

function initTheme() {
  const stored = localStorage.getItem('lpbf-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  setTheme(stored || (prefersDark ? 'dark' : 'light'));

  $('#theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    localStorage.setItem('lpbf-theme', next);
  });
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = $('#theme-toggle');
  if (btn) btn.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`);
}

/* ============================== boot ============================== */

function setStatus(state_, text) {
  const el = $('#load-status');
  el.dataset.state = state_;
  el.innerHTML = `<span class="spinner" aria-hidden="true"></span><span>${escapeHtml(text)}</span>`;
}

function fatalError(msg, detail) {
  setStatus('error', msg);
  $('#fields').innerHTML =
    `<p class="muted" style="grid-column:1/-1">Form unavailable — models failed to load.</p>`;
  $('#results').innerHTML = `
    <div class="empty-state">
      <p><strong>Could not load the prediction models.</strong><br />${escapeHtml(detail)}</p>
      <p style="margin-top:12px">
        If you opened <code>index.html</code> directly from disk, browsers block
        <code>fetch()</code> on <code>file://</code>. Serve the folder instead:<br />
        <code>python -m http.server 8000</code> then visit
        <code>localhost:8000</code>.
      </p>
    </div>`;
  $('#perf-body').innerHTML = `<tr><td colspan="5" class="muted">Unavailable.</td></tr>`;
}

async function boot() {
  initTheme();
  setStatus('loading', 'Loading models…');

  let bundle;
  try {
    const res = await fetch('model.json', { cache: 'force-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    bundle = await res.json();
  } catch (err) {
    console.error(err);
    fatalError('Model load failed', err.message || String(err));
    return;
  }

  if (!bundle?.targets || !Object.keys(bundle.targets).length) {
    fatalError('Model file invalid', 'model.json contained no trained targets.');
    return;
  }

  state.bundle = bundle;

  try {
    buildForm();
    renderPerfTable();
  } catch (err) {
    console.error(err);
    fatalError('Interface build failed', err.message || String(err));
    return;
  }

  const nTargets = Object.keys(bundle.targets).length;
  const nTrees = Object.values(bundle.targets)
    .reduce((a, t) => a + t.model.trees.length, 0);
  setStatus('ready', `${nTargets} models ready · ${nTrees.toLocaleString()} decision trees`);

  $('#hero-rows').textContent = bundle.dataset?.rows ?? '674';
  $('#footer-meta').textContent =
    `${nTargets} targets · ${nTrees.toLocaleString()} trees · replicate-grouped nested CV`;

  $('#predict-form').addEventListener('submit', (e) => { e.preventDefault(); runPrediction(true); });
  $('#sample-btn').addEventListener('click', fillSample);
  $('#reset-btn').addEventListener('click', resetForm);

  // Optional self-check: if the Python-generated vectors are published alongside the
  // model, confirm this engine reproduces them. Absent file = silently skipped.
  fetch('test_vectors.json')
    .then((r) => (r.ok ? r.json() : null))
    .then((v) => {
      if (!v) return;
      const chk = verifyAgainstPython(bundle, v);
      if (chk.ok) {
        console.info(`[lpbf] inference verified against Python on ${chk.checked} vectors ` +
                     `(max error ${chk.maxErr.toExponential(2)})`);
      } else {
        console.error(`[lpbf] VERIFICATION FAILED — max error ${chk.maxErr}`);
        toast('Warning: model self-check failed. Predictions may be unreliable.');
      }
    })
    .catch(() => {});
}

boot();
