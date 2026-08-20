/**
 * Drives the real app in Chromium to confirm it works end-to-end: models load, the form
 * builds, prediction renders, sample data fills, validation fires, and the app is
 * keyboard/screen-reader navigable. Also captures screenshots for review.
 */
import { chromium, devices } from 'playwright';

const URL = 'http://localhost:8765/index.html';
const errors = [];
const consoleMsgs = [];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });

page.on('console', (m) => {
  consoleMsgs.push(`[${m.type()}] ${m.text()}`);
  if (m.type() === 'error') errors.push(m.text());
});
page.on('pageerror', (e) => errors.push(`PAGEERROR: ${e.message}`));

const step = (s) => console.log(`\n=== ${s} ===`);

step('1. load');
await page.goto(URL, { waitUntil: 'networkidle' });
await page.waitForFunction(
  () => document.querySelector('#load-status')?.dataset.state === 'ready',
  { timeout: 60000 },
);
console.log('status:', await page.textContent('#load-status'));

step('2. form built from metadata');
const fields = await page.$$eval('.field', (els) =>
  els.map((e) => ({
    feature: e.dataset.feature,
    tag: e.querySelector('input,select')?.tagName,
    hint: e.querySelector('.field-hint')?.textContent?.trim().slice(0, 46),
    labelled: !!e.querySelector('label')?.getAttribute('for'),
    described: !!e.querySelector('input,select')?.getAttribute('aria-describedby'),
  })));
console.log(`${fields.length} fields`);
for (const f of fields) {
  console.log(`  ${f.tag.padEnd(6)} ${String(f.feature).slice(0, 30).padEnd(32)} ` +
              `label=${f.labelled} desc=${f.described} | ${f.hint}`);
}
if (fields.some((f) => !f.labelled || !f.described)) {
  errors.push('a11y: some fields missing label[for] or aria-describedby');
}

step('3. model card table');
const perfRows = await page.$$eval('#perf-body tr', (rs) =>
  rs.map((r) => [...r.querySelectorAll('td')].map((d) => d.textContent.trim().replace(/\s+/g, ' '))));
for (const r of perfRows) console.log('  ' + r.join(' | '));
if (perfRows.length !== 7) errors.push(`expected 7 perf rows, got ${perfRows.length}`);

step('4. manual prediction');
await page.fill('#f-laser-power-w', '250');
await page.fill('#f-laser-speed-mm-s', '900');
await page.fill('#f-layer-thickness-um', '40');
await page.fill('#f-hatch-spacing-um', '110');
await page.fill('#f-beam-size-um', '80');
await page.selectOption('#f-density-measurement-method', 'Archimedes');
console.log('derived:', (await page.textContent('#derived-grid')).replace(/\s+/g, ' ').trim());
await page.click('#predict-btn');
await page.waitForSelector('.result-card', { timeout: 20000 });
const cards = await page.$$eval('.result-card', (cs) =>
  cs.map((c) => ({
    name: c.querySelector('.rc-name')?.textContent.trim(),
    value: (c.querySelector('.rc-number') || c.querySelector('.rc-class'))?.textContent.trim(),
    unit: c.querySelector('.rc-unit')?.textContent.trim() ?? '',
    tag: c.querySelector('.rc-tag')?.textContent.trim(),
    rel: c.dataset.reliability,
  })));
console.log(`${cards.length} result cards:`);
for (const c of cards) {
  console.log(`  ${c.name.padEnd(34)} ${String(c.value).padStart(9)} ${c.unit.padEnd(4)} [${c.tag}]`);
}
if (cards.length !== 7) errors.push(`expected 7 cards, got ${cards.length}`);
if (cards.some((c) => !c.value || c.value === 'NaN' || c.value === '—')) {
  errors.push('a card rendered an empty/NaN value');
}
console.log('sr announcement:', (await page.textContent('#sr-results')).slice(0, 130));

step('5. screenshot: desktop light');
await page.screenshot({ path: 'outputs/app_desktop_light.png', fullPage: true });

step('6. dark theme');
await page.click('#theme-toggle');
await page.waitForTimeout(420);
console.log('theme:', await page.getAttribute('html', 'data-theme'));
await page.screenshot({ path: 'outputs/app_desktop_dark.png', fullPage: true });
await page.click('#theme-toggle');
await page.waitForTimeout(300);

step('7. sample data button');
await page.click('#sample-btn');
await page.waitForTimeout(900);
const sampleVals = await page.$$eval('.field input, .field select', (es) => es.map((e) => e.value));
console.log('filled values:', sampleVals);
const actualShown = !(await page.getAttribute('#sample-actual', 'hidden'));
console.log('measured-values panel visible:', actualShown);
if (actualShown) {
  console.log('  ' + (await page.textContent('#sample-actual-grid')).replace(/\s+/g, ' ').trim().slice(0, 260));
}
if (sampleVals.every((v) => v === '')) errors.push('sample button filled nothing');

step('8. validation — negative and non-numeric');
await page.fill('#f-laser-power-w', '-5');
await page.locator('#f-laser-power-w').blur();
await page.waitForTimeout(260);
const invalid = await page.getAttribute('.field[data-feature="Laser power (W)"]', 'data-invalid');
const vhint = await page.textContent('#f-laser-power-w-hint');
console.log(`negative -> invalid=${invalid} hint="${vhint.trim()}"`);
if (invalid !== 'true') errors.push('negative value was not flagged invalid');

await page.click('#predict-btn');
await page.waitForTimeout(300);
const formErrVisible = !(await page.getAttribute('#form-error', 'hidden'));
console.log('predict blocked with error banner:', formErrVisible);
if (!formErrVisible) errors.push('invalid input did not block prediction');

step('9. extrapolation warning');
await page.fill('#f-laser-power-w', '99999');
await page.locator('#f-laser-power-w').blur();
await page.waitForTimeout(260);
const warn = await page.getAttribute('.field[data-feature="Laser power (W)"]', 'data-warn');
console.log(`out-of-range -> warn=${warn} hint="${(await page.textContent('#f-laser-power-w-hint')).trim()}"`);
if (warn !== 'true') errors.push('out-of-range value produced no warning');

step('10. all-blank prediction (median imputation)');
await page.click('#reset-btn');
await page.waitForTimeout(300);
await page.click('#predict-btn');
await page.waitForSelector('.result-card', { timeout: 20000 });
const blankCards = await page.$$eval('.result-card', (cs) => cs.map((c) =>
  (c.querySelector('.rc-number') || c.querySelector('.rc-class'))?.textContent.trim()));
console.log('blank-input predictions:', blankCards);
if (blankCards.some((v) => !v || v === 'NaN')) errors.push('blank input produced NaN');

step('11. keyboard navigation');
await page.click('#reset-btn');
await page.keyboard.press('Tab');
const seq = [];
for (let i = 0; i < 7; i++) {
  seq.push(await page.evaluate(() => {
    const a = document.activeElement;
    return a.id || a.tagName + (a.className ? '.' + String(a.className).split(' ')[0] : '');
  }));
  await page.keyboard.press('Tab');
}
console.log('tab order:', seq.join(' -> '));

step('12. mobile viewport');
const mob = await browser.newPage({ ...devices['iPhone 13'] });
await mob.goto(URL, { waitUntil: 'networkidle' });
await mob.waitForFunction(
  () => document.querySelector('#load-status')?.dataset.state === 'ready',
  { timeout: 60000 });
await mob.click('#sample-btn');
await mob.waitForSelector('.result-card', { timeout: 20000 });
const hScroll = await mob.evaluate(() =>
  document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
console.log('horizontal overflow on mobile:', hScroll);
if (hScroll) errors.push('mobile layout scrolls horizontally');
await mob.screenshot({ path: 'outputs/app_mobile.png', fullPage: true });
await mob.close();

step('console output');
for (const m of consoleMsgs.slice(0, 14)) console.log('  ' + m.slice(0, 150));

await browser.close();

console.log('\n' + '='.repeat(64));
if (errors.length) {
  console.log(`FAIL — ${errors.length} problem(s):`);
  for (const e of errors) console.log('  - ' + e);
  process.exit(1);
}
console.log('PASS — all browser checks succeeded');
