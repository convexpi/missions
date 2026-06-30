#!/usr/bin/env python3
"""Execute every mission + starter notebook end-to-end against the published packages.

The heavier companion to check_notebook_imports.py: it actually *runs* the notebooks (mirroring
Colab) to catch logic/runtime breakage, not just import drift. Anything that would phone home is
stubbed first — `submit()` becomes a no-op (no live submission / no API key needed) and the Arena
`RemoteAgent.run()` becomes a no-op (no websocket connect). Network data pulls (e.g. yfinance) do
run, so this belongs on a nightly schedule, not every push.

Run after installing the published packages + a Jupyter kernel + the scientific stack.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
GLOBS = ["missions/**/notebook.ipynb", "starters/*.ipynb"]
TIMEOUT = 900  # seconds per notebook

# Injected as the first cell of every notebook: neutralize anything that hits the network/platform.
SETUP = """
import os
os.environ.setdefault("CONVEXPI_API_KEY", "cpk_test_ci")
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass
try:
    import convexpi.lab as _lab
    _lab.submit = lambda *a, **k: {"id": "ci-dry-run", "status": "completed", "oos_sharpe": 0.0}
except Exception:
    pass
try:
    import convexpi.arena as _arena
    if hasattr(_arena, "RemoteAgent"):
        _arena.RemoteAgent.run = lambda self, *a, **k: None
except Exception:
    pass
"""


def run_one(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(SETUP))
    NotebookClient(nb, timeout=TIMEOUT, kernel_name="python3", allow_errors=False).execute()


def main() -> int:
    notebooks = sorted({p for g in GLOBS for p in ROOT.glob(g)})
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1

    failures = []
    for nb in notebooks:
        rel = nb.relative_to(ROOT)
        t0 = time.time()
        try:
            run_one(nb)
            print(f"✓ {rel}  ({time.time() - t0:.0f}s)", flush=True)
        except CellExecutionError as e:
            last = (str(e).strip().splitlines() or ["execution error"])[-1]
            failures.append((rel, last[:300]))
            print(f"✗ {rel}  ({time.time() - t0:.0f}s) — {last[:120]}", flush=True)
        except Exception as e:  # noqa: BLE001 — kernel/timeout/etc.
            failures.append((rel, f"{type(e).__name__}: {e}"))
            print(f"✗ {rel}  (runner error: {type(e).__name__})", flush=True)

    print(f"\n{len(notebooks) - len(failures)}/{len(notebooks)} notebooks executed cleanly.")
    if failures:
        print("\nFailures:")
        for rel, err in failures:
            print(f"  ✗ {rel}\n      {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
