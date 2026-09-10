# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the npj Materials Degradation submission zip.

The journal compiles the LaTeX source for peer review, so the package has to
build on their machine, not only on ours.  This stages exactly the files needed
and then proves it: the staged tree is copied to a temporary location and built
there from scratch, pdflatex/bibtex/pdflatex/pdflatex, with nothing else on the
path.  A tree that builds locally is not evidence.

In, per the journal's content-type guidance:
  main-snjnl.tex      the manuscript, editable, as they ask
  references.bib      so their BibTeX run resolves
  main-snjnl.bbl      and a resolved copy in case it does not
  sn-jnl.cls
  sn-nature.bst       the Springer Nature class and bibliography style
  figures/            only the figures the manuscript includes
  supplementary.pdf   the Supplementary as one separate PDF, as they ask
  cover_letter.pdf
  cover_letter.docx   the cover letter, in both formats

Comment lines are stripped from the submitted .tex.  They carry internal build
notes, venue reminders and provenance for our own use; none of it is for a
referee, and a submitted source should say only what the paper says.  Trailing
comments are left alone, because in LaTeX a line-ending % is syntax.

Writes paper/npj-submission/ and paper/npj-submission.zip.

Run:  python scripts/make_npj_package.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from htw_pdm.paths import PAPER as P

OUT = P / "npj-submission"
ZIP = P / "npj-submission.zip"


def run(cmd, cwd, what, check=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        tail = "\n".join((r.stdout or "").splitlines()[-25:])
        raise SystemExit(f"{what} failed:\n{tail}")
    return r


def strip_comments(text: str) -> str:
    """Drop whole-line comments; keep trailing %, which is LaTeX syntax."""
    out = [ln for ln in text.split("\n") if not re.match(r"\s*%", ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


# ---- build in place so the .bbl and the SI PDF are current -----------------
run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main-snjnl.tex"],
    P, "latexmk main-snjnl.tex")
run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "supplementary.tex"],
    P, "latexmk supplementary.tex")
run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "cover_letter.tex"],
    P, "latexmk cover_letter.tex")

# ---- stage ----------------------------------------------------------------
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "figures").mkdir(parents=True)

tex = (P / "main-snjnl.tex").read_text(encoding="utf-8")
clean = strip_comments(tex)
(OUT / "main-snjnl.tex").write_text(clean, encoding="utf-8")
removed = tex.count("\n") - clean.count("\n")

# references.bib carries our own verification provenance in its comments:
# which records were checked against Crossref, which page range the publisher
# never registered, which entries are held for a future revision.  All of it is
# ours to keep and none of it is a referee's business, so it is stripped too.
(OUT / "references.bib").write_text(
    strip_comments((P / "references.bib").read_text(encoding="utf-8")), encoding="utf-8")

for f in ("main-snjnl.bbl", "sn-jnl.cls", "sn-nature.bst",
          "supplementary.pdf", "cover_letter.pdf", "cover_letter.docx"):
    shutil.copy2(P / f, OUT / f)

copied = []
for ref in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))):
    name = Path(ref).name
    cand = P / "figures" / name
    picks = [cand] if cand.exists() else [
        p for ext in (".pdf", ".png", ".jpg")
        if (p := P / "figures" / (name + ext)).exists()][:1]
    if not picks:
        raise SystemExit(f"figure referenced but not found: {ref}")
    shutil.copy2(picks[0], OUT / "figures" / picks[0].name)
    copied.append(picks[0].name)

# ---- prove it: full clean build in isolation -------------------------------
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp) / "npj"
    shutil.copytree(OUT, work)
    (work / "main-snjnl.bbl").unlink()          # force their BibTeX path
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main-snjnl.tex"],
        work, "isolated pdflatex 1")
    run(["bibtex", "main-snjnl"], work, "isolated bibtex", check=False)
    for i in (2, 3):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main-snjnl.tex"],
            work, f"isolated pdflatex {i}")
    log = (work / "main-snjnl.log").read_text(errors="ignore")
    bad = {
        "undefined references": len(re.findall(r"Reference `[^']*' .*undefined", log)),
        "undefined citations": len(re.findall(r"Citation `[^']*' .*undefined", log)),
        "missing files": len(re.findall(r"File `[^']*' not found", log)),
        "overfull boxes": log.count("Overfull"),
    }
    pages = re.findall(r"main-snjnl\.pdf \((\d+) pages", log)
    blg = (work / "main-snjnl.blg")
    cited = re.findall(r"used (\d+) entries", blg.read_text(errors="ignore")) if blg.exists() else []
    if bad["undefined references"] or bad["undefined citations"] or bad["missing files"]:
        raise SystemExit(f"isolated build is not clean: {bad}")

# ---- zip -------------------------------------------------------------------
if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            z.write(f, arcname=str(f.relative_to(OUT)))

print(f"Staged: {OUT}")
print(f"  main-snjnl.tex ({removed} comment lines stripped), references.bib, main-snjnl.bbl")
print(f"  sn-jnl.cls, sn-nature.bst, {len(copied)} figures")
print(f"  supplementary.pdf, cover_letter.pdf, cover_letter.docx")
print(f"Isolated build from scratch (pdflatex + bibtex): clean, "
      f"{pages[-1] if pages else '?'} pages, {cited[-1] if cited else '?'} references, "
      f"{bad['overfull boxes']} overfull")
print(f"Zip: {ZIP}  ({ZIP.stat().st_size / 1e6:.1f} MB)")
