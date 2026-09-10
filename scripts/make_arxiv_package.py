# SPDX-FileCopyrightText: Copyright (c) 2026 Conrard Giresse Tetsassi Feugmo
# SPDX-License-Identifier: Apache-2.0

"""Build the arXiv upload package from paper/arxiv/.

arXiv is not a LaTeX workstation: it compiles ONE document, it will not run
BibTeX for you, and anything you upload that it does not need is a way for the
build to differ from yours.  So this stages a directory containing exactly the
files needed and nothing else, then proves it by compiling that directory in a
temporary location with pdflatex alone, no bibtex, no latexmk.

What goes in:
  main.tex        the manuscript, generated from paper/main-snjnl.tex
  main.bbl        the resolved bibliography, because arXiv will not run BibTeX
  arxiv.sty       the preprint class
  figures/        only the figures main.tex actually includes
  anc/            ancillary files; the Supplementary PDF lives here, since
                  arXiv builds a single document and the SI is a separate one

What stays out: every .aux, .log, .out, .fls, .fdb_latexmk, .synctex, the
unused figures, references.bib (the .bbl supersedes it) and the SI sources.

Writes paper/arxiv-submission/ and paper/arxiv-submission.tar.gz.

Run:  python scripts/make_arxiv_package.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from htw_pdm.paths import PAPER as P

SRC = P / "arxiv"
OUT = P / "arxiv-submission"
TARBALL = P / "arxiv-submission.tar.gz"


def run(cmd, cwd, what):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stdout or "").splitlines()[-25:])
        raise SystemExit(f"{what} failed:\n{tail}")
    return r


# ---- 1. build in place so main.bbl is current -----------------------------
run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    SRC, "latexmk main.tex")
run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "supplementary.tex"],
    SRC, "latexmk supplementary.tex")

bbl = SRC / "main.bbl"
if not bbl.exists() or not bbl.read_text().strip():
    raise SystemExit("main.bbl is missing or empty; arXiv would build with no bibliography.")

# ---- 2. stage exactly what is needed --------------------------------------
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "figures").mkdir(parents=True)
(OUT / "anc").mkdir()

tex = (SRC / "main.tex").read_text(encoding="utf-8")
wanted = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)

copied = []
for ref in sorted(set(wanted)):
    name = Path(ref).name
    cand = SRC / "figures" / name
    if cand.exists():                       # explicit extension
        picks = [cand]
    else:                                   # extensionless: graphicx resolves it
        picks = [p for ext in (".pdf", ".png", ".jpg")
                 if (p := SRC / "figures" / (name + ext)).exists()][:1]
        if not picks:
            raise SystemExit(f"figure referenced but not found: {ref}")
    for p in picks:
        shutil.copy2(p, OUT / "figures" / p.name)
        copied.append(p.name)

for f in ("main.tex", "main.bbl", "arxiv.sty"):
    shutil.copy2(SRC / f, OUT / f)
shutil.copy2(SRC / "supplementary.pdf", OUT / "anc" / "supplementary.pdf")

# ---- 3. prove it: compile the staged tree alone, with pdflatex only --------
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp) / "arxiv"
    shutil.copytree(OUT, work)
    for i in (1, 2, 3):                     # arXiv runs latex repeatedly, never bibtex
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            work, f"isolated pdflatex pass {i}")
    log = (work / "main.log").read_text(errors="ignore")
    bad = {
        "undefined references": len(re.findall(r"Reference `[^']*' .*undefined", log)),
        "undefined citations": len(re.findall(r"Citation `[^']*' .*undefined", log)),
        "missing files": len(re.findall(r"File `[^']*' not found", log)),
    }
    pages = re.findall(r"main\.pdf \((\d+) pages", log)
    if any(bad.values()):
        raise SystemExit(f"isolated build is not clean: {bad}")

# ---- 4. tarball ------------------------------------------------------------
if TARBALL.exists():
    TARBALL.unlink()
with tarfile.open(TARBALL, "w:gz") as t:
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            t.add(f, arcname=str(f.relative_to(OUT)))

size = TARBALL.stat().st_size
print(f"Staged: {OUT}")
print(f"  main.tex, main.bbl, arxiv.sty, {len(copied)} figures, anc/supplementary.pdf")
print(f"Isolated pdflatex build (no bibtex): clean, {pages[-1] if pages else '?'} pages")
print(f"Tarball: {TARBALL}  ({size / 1e6:.1f} MB)")
if size > 50e6:
    print("  WARNING: arXiv's default upload limit is 50 MB.")
