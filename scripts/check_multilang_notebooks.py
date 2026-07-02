#!/usr/bin/env python3
"""Validate the R and Julia mission/starter notebooks.

For every notebook: confirm it is valid nbformat JSON. For R notebooks (notebook_r.ipynb,
r_starter.ipynb), parse each code cell with `Rscript`. For Julia notebooks (notebook_julia.ipynb,
julia_starter.ipynb), parse each code cell with `julia` (Meta.parseall — syntax only, no execution).

Interpreters that aren't installed are skipped (with a note), so this runs locally with whatever you
have and fully in CI where both are installed. Exit code is nonzero if any cell fails to parse.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def code_cells(path):
    nb = json.load(open(path))
    assert nb.get("nbformat") == 4, f"{path}: not nbformat 4"
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def check_r(path, rscript):
    fails = 0
    for i, src in enumerate(code_cells(path)):
        with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as f:
            f.write(src); tmp = f.name
        p = subprocess.run([rscript, "-e", f'invisible(parse("{tmp}"))'],
                           capture_output=True, text=True)
        os.unlink(tmp)
        if p.returncode != 0:
            fails += 1
            print(f"  ✗ {os.path.relpath(path, ROOT)} cell {i}:\n{p.stderr.strip()[:300]}")
    return fails


def check_julia(path, julia):
    fails = 0
    for i, src in enumerate(code_cells(path)):
        with tempfile.NamedTemporaryFile("w", suffix=".jl", delete=False) as f:
            f.write(src); tmp = f.name
        # Meta.parseall raises on a syntax error; parsing does not execute the code.
        p = subprocess.run(
            [julia, "-e", f'Meta.parseall(read("{tmp}", String))'],
            capture_output=True, text=True)
        os.unlink(tmp)
        if p.returncode != 0:
            fails += 1
            print(f"  ✗ {os.path.relpath(path, ROOT)} cell {i}:\n{p.stderr.strip()[:300]}")
    return fails


def main():
    rscript = shutil.which("Rscript")
    julia = shutil.which("julia")
    r_paths = sorted(glob.glob(f"{ROOT}/missions/*/notebook_r.ipynb") + glob.glob(f"{ROOT}/starters/r_starter.ipynb"))
    jl_paths = sorted(glob.glob(f"{ROOT}/missions/*/notebook_julia.ipynb") + glob.glob(f"{ROOT}/starters/julia_starter.ipynb"))

    # Every notebook must be valid JSON regardless of interpreters.
    all_nb = glob.glob(f"{ROOT}/missions/*/notebook*.ipynb") + glob.glob(f"{ROOT}/starters/*.ipynb")
    for p in all_nb:
        json.load(open(p))
    print(f"JSON OK for {len(all_nb)} notebooks")

    fails = 0
    if rscript:
        for p in r_paths:
            fails += check_r(p, rscript)
        print(f"R: parse-checked {len(r_paths)} notebooks")
    else:
        print("R: Rscript not found — skipping R parse checks")
    if julia:
        for p in jl_paths:
            fails += check_julia(p, julia)
        print(f"Julia: parse-checked {len(jl_paths)} notebooks")
    else:
        print("Julia: julia not found — skipping Julia parse checks")

    if fails:
        print(f"\nFAILED: {fails} cell(s) did not parse")
        sys.exit(1)
    print("\nAll checked cells parse OK")


if __name__ == "__main__":
    main()
