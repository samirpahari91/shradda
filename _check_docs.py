"""Assert the docs quote the scores the app actually ships.

Every input is read from docs/model.json — target names, display labels, scores and the
reliability thresholds — so this gate has no copy of anything it checks. If a threshold
moves in train_export.py it lands in model.json, app.js and this gate together.

Checks:
  1. model.json cv_score == nested_grouped_CV in outputs/nested_grouped_results.csv
  2. every nested score appears (2dp or 3dp) in each doc
  3. the README headline row for each target carries the tier app.js will compute
  4. no doc quotes a flat grouped_model_comparison.csv score as a headline figure

Exit 0 = consistent. Run after editing README.md, METHODOLOGY_README.md,
result_explaination.txt, or after re-running train_export.py.
"""
import json
import sys

import pandas as pd

DOCS = ["README.md", "METHODOLOGY_README.md", "result_explaination.txt"]

bundle = json.load(open("docs/model.json", encoding="utf-8"))
targets = bundle["targets"]
nested = pd.read_csv("outputs/nested_grouped_results.csv").set_index("target")
flat = pd.read_csv("outputs/grouped_model_comparison.csv")
text = {d: open(d, encoding="utf-8").read() for d in DOCS}

# Schema 2 bundles predate the exported reliability block; fall back so the gate still
# runs against an older model.json instead of dying on a KeyError.
FALLBACK = {
    "order": ["good", "moderate", "low"],
    "labels": {"good": "Reliable", "moderate": "Indicative", "low": "Low confidence"},
    "reg": {"metric": "score", "good": 0.6, "moderate": 0.45},
    "clf": {"metric": "lift", "good": 0.55, "moderate": 0.3},
}
cfg = bundle.get("reliability", FALLBACK)
if "reliability" not in bundle:
    print(f"note: model.json (schema {bundle.get('schema_version')}) has no reliability "
          f"block — using fallback; re-run train_export.py to export it")

# Mirrors reliability() in docs/js/app.js, but every number comes from cfg.
def tier(spec):
    rule = cfg[spec["task"]]
    v = spec["cv_score"]
    if rule.get("metric") == "lift":
        chance = 1 / max(len(spec.get("classes") or []), 2)
        v = (v - chance) / (1 - chance)
    order = cfg["order"]
    for name in order:
        cut = rule.get(name)
        if cut is None:
            return name
        if v >= cut:
            return name
    return order[-1]


def readme_row(label):
    """The headline table row for a target, matched on its leading cell."""
    for ln in text["README.md"].splitlines():
        if ln.startswith("| " + label + " |"):
            return ln
    return None


fails = []
for name, spec in targets.items():
    label = spec.get("display_name", name)

    if name not in nested.index:
        fails.append(f"{name}: absent from nested_grouped_results.csv")
        continue
    want = float(nested.loc[name, "nested_grouped_CV"])
    if abs(spec["cv_score"] - want) > 1e-9:
        fails.append(f"{label}: model.json cv_score {spec['cv_score']} != nested {want}")

    v3, v2 = f"{want:.3f}", f"{want:.2f}"
    for d in DOCS:
        if v3 not in text[d] and v2 not in text[d]:
            fails.append(f"{d}: nested score {v3} for {label} is never quoted")

    expect = cfg["labels"][tier(spec)]
    row = readme_row(label)
    if row is None:
        fails.append(f"README.md: no headline row starting '| {label} |'")
    elif expect.split()[0] not in row:
        fails.append(f"README.md: {label} tier should be {expect!r} "
                     f"(cv={spec['cv_score']:.4f}) — row reads: {row.strip()}")

    # A flat score must not appear as this target's headline reliability/score claim.
    sel = flat[(flat.target == name)]
    if len(sel):
        best = float(sel["grouped_CV"].max())
        if row and f"{best:.2f}" in row and abs(best - want) > 5e-3:
            fails.append(f"README.md: {label} headline row quotes flat {best:.2f} "
                         f"instead of nested {want:.3f}")

print(f"checked {len(targets)} targets across {len(DOCS)} docs "
      f"(thresholds from model.json schema {bundle.get('schema_version')})")
if fails:
    print(f"\n{len(fails)} PROBLEM(S):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("OK — docs, CSVs and shipped model agree")
