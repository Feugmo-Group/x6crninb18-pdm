"""Paper Figure 2: the inversion workflow.

The figure is drawn in TikZ, not matplotlib. ``paper/figures/fig_nsem_workflow.tex``
is a standalone document and is the master source; this script compiles it to PDF
(the form the manuscript includes) and rasterizes a PNG copy into outputs/paper so
The figure appears in the same place as every other generated display item.

Run:  python scripts/make_paper_fig_nsem.py
Needs: pdflatex with tikz, and pdftoppm (poppler-utils) for the raster copy.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from htw_pdm.paths import PAPER_OUT as OUT

FIGDIR = Path(__file__).resolve().parents[1] / "paper" / "figures"
STEM = "fig_nsem_workflow"
DPI = 600


def run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-2000:])
        raise SystemExit(f"{cmd[0]} failed with exit code {proc.returncode}")


src = FIGDIR / f"{STEM}.tex"
if not src.exists():
    raise SystemExit(f"missing TikZ source: {src}")

run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", src.name], FIGDIR)
run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", f"{STEM}.pdf", STEM], FIGDIR)
for junk in (".aux", ".log"):
    (FIGDIR / f"{STEM}{junk}").unlink(missing_ok=True)

OUT.mkdir(parents=True, exist_ok=True)
for name in ("fig_nsem_workflow.pdf", "fig_nsem_workflow.png"):
    shutil.copy(FIGDIR / name, OUT / name)

print(f"wrote {FIGDIR / (STEM + '.pdf')}")
print(f"wrote {FIGDIR / (STEM + '.png')} ({DPI} dpi)")
