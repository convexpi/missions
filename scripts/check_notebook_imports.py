#!/usr/bin/env python3
"""Smoke-test every mission/starter notebook's convexpi imports against the PUBLISHED packages.

Colab installs convexpi-lab / convexpi-arena from PyPI, so a symbol that exists in the repo source
but hasn't been published yet (e.g. `submit` once did) breaks the notebooks even though the lab's
own source tests pass. This extracts every `import convexpi…` / `from convexpi… import …` from the
notebooks and runs it against the installed (published) packages — failing if any import is missing.

Run in CI after `pip install convexpi-lab convexpi-arena` (no editable/source install).
"""
from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)   # notebook cells can have stray escapes

ROOT = Path(__file__).resolve().parent.parent
NB_GLOBS = ["missions/**/*.ipynb", "starters/*.ipynb"]


def cell_convexpi_imports(src: str) -> list[str]:
    """Reconstruct convexpi import statements from a code cell (handles multiline imports/aliases;
    skips Jupyter ! and % magic lines that aren't valid Python)."""
    def parse(text: str):
        try:
            return ast.parse(text)
        except SyntaxError:
            return None
    tree = parse(src)
    if tree is None:
        clean = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        tree = parse(clean)
    if tree is None:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("convexpi"):
            names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in node.names)
            out.append(f"from {node.module} import {names}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("convexpi"):
                    out.append(f"import {a.name}" + (f" as {a.asname}" if a.asname else ""))
    return out


def notebook_imports(nb_path: Path) -> set[str]:
    nb = json.loads(nb_path.read_text())
    lines: set[str] = set()
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            lines.update(cell_convexpi_imports("".join(cell.get("source", []))))
    return lines


def main() -> int:
    notebooks = sorted({p for g in NB_GLOBS for p in ROOT.glob(g)})
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1

    where: dict[str, list[str]] = {}
    for nb in notebooks:
        for line in notebook_imports(nb):
            where.setdefault(line, []).append(nb.name)

    failures = []
    for line in sorted(where):
        try:
            exec(line, {})  # noqa: S102 — running the notebook's own import line
        except Exception as e:  # noqa: BLE001
            failures.append((line, f"{type(e).__name__}: {e}", sorted(set(where[line]))))

    print(f"Checked {len(where)} unique convexpi imports across {len(notebooks)} notebooks.")
    if failures:
        print("\nFAILED — these notebook imports don't resolve against the published packages:")
        for line, err, nbs in failures:
            print(f"  ✗ {line}\n      {err}\n      in: {', '.join(nbs)}")
        print("\nLikely a source-vs-PyPI drift — publish the package version that exports these symbols.")
        return 1
    print("All notebook imports resolve against the published packages. ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
