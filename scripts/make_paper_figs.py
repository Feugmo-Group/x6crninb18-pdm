# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper figures F1 and F2 (PAPER_PLAN.md Section 5) under outputs/paper/.

  fig1_model_schematic.png — duplex-oxide HTW_PDM schematic: layer structure
    (metal | Ni enrichment | Cr-rich barrier layer | Fe-rich outer crystals | HTW),
    the retained interfacial reactions, and the potential distribution.
  fig2_baseline_verification.png — (a) Radau vs closed-form parity error on the
    demo set; (b) Li 2020 Table 5 regression curves (HCM12A / 316L at 500 C SCW)
    with Radau overlay — the numerical-baseline trust exhibit.

Figures 3-7 already exist (plot_results.py, plot_uncertainty_ensemble.png,
plot_sensitivity_maps.png); this script covers only the two missing ones.

Run:  python scripts/make_paper_figs.py
"""


import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htw_pdm.baseline_ode import (  # noqa: E402
    LI_TABLE5,
    HTWPDMParams,
    L_bl_closed,
    L_ol_closed,
    params_from_li_cgs,
    solve_forward,
)
from htw_pdm.paths import OUTPUTS  # noqa: E402
from htw_pdm.paths import PAPER_OUT as OUT

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

# ── Fig 1 — model schematic ─────────────────────────────────────────────
# Drawn in TikZ, not matplotlib: paper/figures/fig1_model_schematic.tex is the
# master source. Compile it and rasterize a copy so the figure lands in
# outputs/paper/ alongside every other generated display item.
FIGDIR = Path(__file__).resolve().parents[1] / "paper" / "figures"


def build_tikz(stem: str, dpi: int = 600) -> None:
    src = FIGDIR / f"{stem}.tex"
    if not src.exists():
        raise SystemExit(f"missing TikZ source: {src}")
    for cmd in (
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", src.name],
        ["pdftoppm", "-r", str(dpi), "-png", "-singlefile", f"{stem}.pdf", stem],
    ):
        proc = subprocess.run(cmd, cwd=FIGDIR, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-2000:])
            raise SystemExit(f"{cmd[0]} failed with exit code {proc.returncode}")
    for junk in (".aux", ".log"):
        (FIGDIR / f"{stem}{junk}").unlink(missing_ok=True)


build_tikz("fig1_model_schematic")
for name in ("fig1_model_schematic.pdf", "fig1_model_schematic.png"):
    shutil.copy(FIGDIR / name, OUT / name)
print(f"Saved: {OUT / 'fig1_model_schematic.png'}")

# ── Fig 2 — baseline verification ─────────────────────────────────────────────
fig2, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) parity error on the demo set.
demo = HTWPDMParams(A_bl=12.0, b3=-0.03, C_bl=0.05, PBR_eff=1.1, C_x=-0.01, L0=2.0)
t = np.linspace(0.0, 480.0, 200)
num = solve_forward(demo, t)
err_bl = np.abs(num[:, 0] - L_bl_closed(demo, t))
err_ol = np.abs(num[:, 1] - L_ol_closed(demo, t))
a.semilogy(t, np.maximum(err_bl, 1e-16), "b-", label="$|L_{bl}^{Radau} - L_{bl}^{closed}|$")
a.semilogy(t, np.maximum(err_ol, 1e-16), "g--", label="$|L_{ol}^{Radau} - L_{ol}^{closed}|$")
a.set_xlabel("t (h)")
a.set_ylabel("absolute parity error (nm)")
a.set_title("(a) Radau vs closed form, demo parameter set")
a.legend(fontsize=8)

# (b) Li Table 5 regression: HCM12A / 316L at 500 C SCW, um scale.
t5 = np.linspace(1.0, 20000.0, 300)
markers_t = np.array([100.0, 300.0, 1000.0, 3000.0, 10000.0, 20000.0])
for name, cgs, color in (("HCM12A (500 °C SCW)", LI_TABLE5["HCM12A_500C"], "tab:blue"),
                         ("316L (500 °C SCW)", LI_TABLE5["316L_500C"], "tab:orange")):
    p = params_from_li_cgs(**cgs)
    total = (L_bl_closed(p, t5) + L_ol_closed(p, t5)) / 1e3
    b.plot(t5, total, "-", color=color, lw=2, label=f"{name} — closed form (total)")
    num5 = solve_forward(p, markers_t)
    b.plot(markers_t, (num5[:, 0] + num5[:, 1]) / 1e3, "o", color=color, ms=6,
           mfc="none", label=f"{name} — Radau")
b.set_xlabel("t (h)")
b.set_ylabel("total scale thickness (µm)")
b.set_title("(b) regression vs Li 2020 Table 5 apparent parameters")
b.legend(fontsize=8)

plt.tight_layout()
out2 = OUT / "fig2_baseline_verification.png"
plt.savefig(out2, dpi=300)
print(f"Saved: {out2}")
print(f"  demo parity max: bl {err_bl.max():.2e} nm, ol {err_ol.max():.2e} nm")

# ── Figs 3-7 — copy the existing result figures into the paper archive ────────

COPIES = {
    "fig3_fits_vs_data.png": "plot_fits_vs_data.png",
    "fig4_profile_likelihood.png": "plot_profile_likelihood.png",
    # fig5 is generated directly by make_paper_fig5.py (1000-member bootstrap)
    "fig6_long_time.png": "plot_long_time.png",
    # The line panels are the main-text envelope figure; the contour maps they
    # replaced are kept for the SI, where there is room to render them legibly.
    "fig7_envelope_lines.png": "plot_envelope_lines.png",
    "figS3_sensitivity_maps.png": "plot_sensitivity_maps.png",
}
# These sources are resolved against OUTPUTS, not against this file's directory.
# The previous form, Path(__file__).parent / "outputs/...", pointed at
# scripts/outputs/... -- a path that never exists -- and because the copy was
# guarded by `if s.exists()` it silently did nothing, leaving figs 3, 4, 6 and 7
# in outputs/paper/ frozen at whatever run last wrote them.  A missing source is
# now a hard error: the figure is stale either way, and saying so is the point.
missing = [src for src in COPIES.values() if not (OUTPUTS / src).exists()]
if missing:
    raise SystemExit(
        "missing plot outputs: " + ", ".join(missing)
        + "\nrun scripts/plot_results.py and scripts/plot_sensitivity_maps.py first")
for dst, src in COPIES.items():
    shutil.copy(OUTPUTS / src, OUT / dst)
    print(f"Copied: outputs/{src} -> outputs/paper/{dst}")
