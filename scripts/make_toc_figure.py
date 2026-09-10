# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""fig_toc -- table-of-contents / graphical-abstract figure, in PDF, PNG and SVG.

npj Materials Degradation does not require a graphical abstract, but the
Collection landing page and the Corrosion Science submission both want one, and
a thumbnail has to carry the whole argument on its own.

The drawing itself is TikZ/pgfplots (paper/figures/fig_toc.tex), matching the
other vector figures in this paper rather than the matplotlib ones, so it stays
sharp at any size.  What this script does is export the curves and points that
figure plots, straight from the committed bootstrap artifact, so the picture
cannot drift from the numbers the manuscript reports; then it compiles the
figure and rasterizes it.

Writes:
  paper/figures/fig_toc_curves.csv   mechanistic 95% band + power-law family
  paper/figures/fig_toc_points.csv   the three measured Cr thicknesses
  paper/figures/fig_toc.{pdf,png,svg}  and copies of each into outputs/paper/

Needs pdflatex with pgfplots, plus pdftoppm and pdftocairo (poppler).

Run:  python scripts/make_toc_figure.py
"""

from __future__ import annotations

import json
import subprocess

import numpy as np

from htw_pdm.baseline_fit import load_data  # noqa: E402
from htw_pdm.paths import PAPER as P  # noqa: E402
from htw_pdm.paths import PAPER_OUT as OUT  # noqa: E402

H_PER_Y = 8766.0
T_10Y = 10 * H_PER_Y
FIG = P / "figures"

band = np.load(OUT / "fig6_band.npz")
stats = json.loads((OUT / "referee_stats.json").read_text())
pl_refit = stats["power_law_weighted_refit"]["Cr"]
pl_pub = stats["power_law_scan_level"]["Cr"]

# The figure spans the calibration window through a few decades past ten years;
# earlier times are dropped because the thumbnail cannot resolve them.
t = band["t_h"]
keep = (t >= 40.0) & (t <= 60 * H_PER_Y)
t = t[keep]
mlo, mmid, mhi = band["p2p5"][keep], band["p50"][keep], band["p97p5"][keep]

rows = ["ty,mlo,mmid,mhi,pllo,plhi"]
for i, ti in enumerate(t):
    rows.append(
        f"{ti / H_PER_Y:.6g},{mlo[i]:.5g},{mmid[i]:.5g},{mhi[i]:.5g},"
        f"{pl_pub['k'] * ti ** pl_pub['n']:.5g},{pl_refit['k'] * ti ** pl_refit['n']:.5g}"
    )
(FIG / "fig_toc_curves.csv").write_text("\n".join(rows) + "\n")

means, _ = load_data()
cr = sorted(means["Cr"])
pts = ["ty,L,err"] + [
    f"{r[0] / H_PER_Y:.6g},{r[1]:.5g},{r[2] / np.sqrt(r[3]):.5g}" for r in cr
]
(FIG / "fig_toc_points.csv").write_text("\n".join(pts) + "\n")

m10 = float(np.interp(T_10Y, t, mmid))
p10 = pl_refit["k"] * T_10Y ** pl_refit["n"]

# The two annotated coordinates in fig_toc.tex are the endpoints of the
# separation arrow.  They are checked rather than injected, so a silent drift
# between the drawing and the artifact fails here instead of in print.
tex = (FIG / "fig_toc.tex").read_text()
arrow = f"(axis cs:10,{m10:.0f}) -- (axis cs:10,{p10:.0f})"
if arrow not in tex:
    raise SystemExit(
        f"fig_toc.tex separation arrow is stale: expected '{arrow}'.\n"
        "Update the \\draw[<->] coordinates in the figure source to match."
    )


def run(cmd):
    subprocess.run(cmd, cwd=FIG, check=True, capture_output=True)


run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig_toc.tex"])
run(["pdftoppm", "-r", "300", "-png", "-singlefile", "fig_toc.pdf", "fig_toc"])
run(["pdftocairo", "-svg", "fig_toc.pdf", "fig_toc.svg"])

for ext in ("pdf", "png", "svg"):
    (OUT / f"fig_toc.{ext}").write_bytes((FIG / f"fig_toc.{ext}").read_bytes())

print(f"Saved: fig_toc.pdf, .png and .svg in {FIG} and {OUT}")
print(f"  mechanistic at 10 y : {m10:.0f} nm")
print(f"  power law   at 10 y : {p10:.0f} nm  (ratio {p10 / m10:.2f})")
