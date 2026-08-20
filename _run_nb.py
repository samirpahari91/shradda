"""Execute LPBF_ML_Analysis.ipynb in place.

Bypasses `jupyter nbconvert`, whose global config in ~/.jupyter registers a
jupyter_contrib_nbextensions preprocessor that is not installed in this env and
which aborts the run before any cell executes. nbclient is the same execution
engine nbconvert --execute uses, minus that config layer.
"""
import sys

import nbformat
from nbclient import NotebookClient

NB = "LPBF_ML_Analysis.ipynb"

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(
    nb,
    timeout=1800,
    kernel_name="python3",
    resources={"metadata": {"path": "."}},  # so relative Data.xlsx / outputs/ resolve
    allow_errors=False,
)

try:
    client.execute()
finally:
    nbformat.write(nb, NB)

executed = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
total = sum(1 for c in nb.cells if c.cell_type == "code")
print(f"executed {executed}/{total} code cells with output", file=sys.stderr)
