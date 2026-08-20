"""Report whether LPBF_ML_Analysis.ipynb actually executed cleanly.

This is the authoritative verdict for SOP Stage 4 — the console log is not, because
joblib/loky prints benign 'Traceback ... joblib_memmapping_folder' blocks at worker
shutdown that look like failures but occur outside any notebook cell.

Exit 0 = every code cell ran and none errored.
"""
import sys

import nbformat

NB = "LPBF_ML_Analysis.ipynb"

nb = nbformat.read(NB, as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]
with_out = [c for c in code if c.get("outputs")]
errored = [c for c in code if any(o.get("output_type") == "error"
                                 for o in c.get("outputs", []))]

print(f"{len(with_out)}/{len(code)} code cells with output, {len(errored)} error cells")

for c in errored[:5]:
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            print(f"  {o.get('ename')}: {str(o.get('evalue'))[:160]}")

if not code:
    print("PROBLEM: notebook has no code cells — did build_notebook.py run?")
    sys.exit(1)
if errored:
    print("PROBLEM: notebook contains error output — fix before publishing")
    sys.exit(1)
if len(with_out) < len(code):
    print("PROBLEM: some cells never executed — re-run python _run_nb.py")
    sys.exit(1)
print("OK — notebook executed cleanly")
