"""Execute a notebook in place and save its outputs.

WHY THIS EXISTS: the `jupyter` CLI inside this project's .venv is broken --
`jupyter nbconvert --version` exits 1 with no output, and `jupyter kernelspec list`
prints nothing. The Python API works fine, so we drive nbclient directly.

Usage:
    python tools/run_notebook.py 1_Tier1_Claim_Verification.ipynb
    python tools/run_notebook.py in.ipynb out.ipynb

Always run with PYTHONIOENCODING=utf-8 on Windows, or printing a progress bar
will crash the runner with a cp1252 UnicodeEncodeError.
"""
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def clean(text):
    """Drop tqdm progress-bar noise so the log stays readable."""
    keep = [ln for ln in text.splitlines()
            if all(tok not in ln for tok in
                   ("examples/s", "%|", "\r", "it/s", "B/s"))]
    return "\n".join(keep).strip()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp

    nb = nbformat.read(inp, as_version=4)
    client = NotebookClient(
        nb, timeout=6000, kernel_name="python3",
        resources={"metadata": {"path": str(Path(inp).resolve().parent)}},
    )

    start = time.time()
    with client.setup_kernel():
        total = len(nb.cells)
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            t0 = time.time()
            head = cell.source.strip().splitlines()
            preview = head[0][:80] if head else "(empty)"
            print(f"[{idx+1}/{total}] RUN  {preview}", flush=True)
            try:
                client.execute_cell(cell, idx)
            except CellExecutionError as exc:
                nbformat.write(nb, out)
                print(f"[{idx+1}/{total}] FAIL after {time.time()-t0:.1f}s", flush=True)
                print("=" * 70, flush=True)
                print(str(exc)[:6000], flush=True)
                sys.exit(1)

            for o in cell.get("outputs", []):
                kind = o.get("output_type")
                if kind == "stream":
                    txt = clean("".join(o.get("text", "")))
                    if txt:
                        print("    " + txt.replace("\n", "\n    ")[:2500], flush=True)
                elif kind == "execute_result":
                    txt = clean("".join(o.get("data", {}).get("text/plain", "")))
                    if txt:
                        print("    => " + txt.replace("\n", "\n    ")[:2500], flush=True)
                elif kind == "error":
                    print("    !! " + "\n    ".join(o.get("traceback", []))[:3000], flush=True)
            print(f"[{idx+1}/{total}] OK   {time.time()-t0:.1f}s", flush=True)

    nbformat.write(nb, out)
    print(f"DONE all cells in {time.time()-start:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
