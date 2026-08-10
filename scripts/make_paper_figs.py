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

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

from htw_pdm.baseline_ode import (  # noqa: E402
    LI_TABLE5,
    HTWPDMParams,
    L_bl_closed,
    L_ol_closed,
    params_from_li_cgs,
    solve_forward,
)

from htw_pdm.paths import PAPER_OUT as OUT  # noqa: E402
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

# ── Fig 1 — model schematic ───────────────────────────────────────────────────
fig1, (ax, axp) = plt.subplots(
    2, 1, figsize=(9, 7), height_ratios=[2.2, 1.0], sharex=True
)

# Layer geometry (schematic x units).
x_metal, x_ni, x_bl, x_olroot = 0.0, 2.6, 3.0, 6.0
x_right = 9.6
zones = [
    (x_metal, x_ni, "#b0b7c0", "X6CrNiNb18-10 matrix\n17.6 Cr / 10.6 Ni /\n0.62 Nb / bal. Fe"),
    (x_ni, x_bl, "#7fbf7f", "Ni\nzone"),
    (x_bl, x_olroot, "#5a8f5a", "barrier layer (bl)\nCr-rich nanocrystalline spinel\n$FeCr_2O_4$ / $NiCr_2O_4$, $\\chi = 8/3$"),
    (x_olroot, x_right, "#cfe3f5", "HTW: 240 °C, 7 MPa\n[O$_2$] = 0.4 ppm, 0.055 µS/cm"),
]
for x0, x1, c, label in zones:
    ax.add_patch(Rectangle((x0, 0), x1 - x0, 3.0, facecolor=c, edgecolor="k", lw=0.8))
    ax.text((x0 + x1) / 2, 2.55, label, ha="center", va="top", fontsize=9)

# Outer-layer discrete crystals (Fe-rich) sitting on the bl surface, kept clear
# of the arrow/label band (y in [0.4, 2.1] near the bl/ol interface).
rng = np.random.default_rng(3)
for cx, cy, s in [(8.5, 0.45, 0.85), (9.25, 1.15, 0.65), (8.95, 1.72, 0.5)]:
    ax.add_patch(Ellipse((cx, cy), 0.85 * s, 0.65 * s, angle=float(rng.uniform(0, 60)),
                         facecolor="#d98c5f", edgecolor="k", lw=0.8, zorder=4))
ax.text(7.0, 3.08, "outer layer (ol): discrete Fe-rich crystals ($Fe_3O_4 \\to Fe_2O_3$), $\\delta = 8/3$",
        ha="center", fontsize=9)

# Interfaces.
for x, lbl in ((x_bl, "m/bl interface\n(recedes into metal)"),
               (x_olroot, "bl/ol interface")):
    ax.plot([x, x], [0, 3.0], "k-", lw=2)
    ax.annotate(lbl, (x, -0.06), ha="center", va="top", fontsize=9)

# Reactions (arrows + labels).
arrows = [
    # (x0, y0, x1, y1, label_x, label_y, label)
    (x_bl - 0.45, 1.75, x_bl + 0.45, 1.75, x_bl + 0.55, 1.86,
     "R3: m → M$_M$ + (χ/2)V$_O^{••}$ + χe′  (bl growth)"),
    (x_bl + 0.6, 1.15, x_olroot - 0.6, 1.15, (x_bl + x_olroot) / 2, 1.27,
     "V$_O^{••}$ migration →"),
    (x_olroot - 0.45, 0.55, x_olroot + 0.45, 0.55, x_olroot - 2.35, 0.20,
     "R10′: proton-assisted bl dissolution ($C_{bl}\\!\\approx\\!0$ in ultrapure HTW)"),
    (x_olroot + 0.4, 1.9, x_right - 1.9, 1.9, x_olroot + 0.4, 2.02,
     "R4/5: cation ejection → ol deposition (PBR)"),
]
for x0, y0, x1, y1, lx, ly, lbl in arrows:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, color="k", lw=1.1, zorder=5))
    ax.text(lx, ly, lbl, fontsize=8.5, ha="left", zorder=6)
ax.text(1.3, 0.55, "Ni: slow-oxidizing, accumulates\nby exclusion at m/bl\n($dL_{Ni}/dt \\approx 0$)",
        ha="center", fontsize=8.5, style="italic")
ax.set_xlim(0, x_right)
ax.set_ylim(-0.7, 3.35)
ax.axis("off")
ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontweight="bold",
        va="top", fontsize=12)

# Potential distribution panel.
xs = np.linspace(0, x_right, 400)
phi = np.piecewise(
    xs,
    [xs < x_bl, (xs >= x_bl) & (xs < x_olroot), xs >= x_olroot],
    [1.0, lambda x: 1.0 - 0.75 * (x - x_bl) / (x_olroot - x_bl), 0.25],
)
axp.plot(xs, phi, "b-", lw=2)
axp.axvline(x_bl, color="k", lw=1, ls=":")
axp.axvline(x_olroot, color="k", lw=1, ls=":")
axp.annotate("$\\varphi_{m/bl} = (1-\\alpha)V - \\hat{\\varepsilon}L_{bl} - \\beta\\,pH - \\varphi^0$",
             (x_bl - 0.1, 1.07), ha="right", fontsize=9)
axp.annotate("constant field $\\hat{\\varepsilon}$\n($\\varepsilon_f \\approx 1.7\\times10^4$ V/cm)",
             ((x_bl + x_olroot) / 2, 0.72), ha="center", fontsize=9)
axp.annotate("$\\varphi_{bl/e} = \\alpha V + \\beta\\,pH + \\varphi^0$\n(porous ol: no potential drop)",
             (x_olroot + 0.15, 0.34), fontsize=9)
axp.set_ylim(0.1, 1.35)  # headroom so the panel letter clears the curve
axp.set_ylabel("$\\varphi(x)$ (schematic)")
axp.set_yticks([])
axp.set_xticks([])
axp.set_xlabel("depth →")
axp.text(0.02, 0.98, "(b)", transform=axp.transAxes, fontweight="bold",
         va="top", fontsize=12)

plt.tight_layout()
out1 = OUT / "fig1_model_schematic.png"
plt.savefig(out1, dpi=300)
print(f"Saved: {out1}")

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
import shutil  # noqa: E402

COPIES = {
    "fig3_fits_vs_data.png": "outputs/plot_fits_vs_data.png",
    "fig4_profile_likelihood.png": "outputs/plot_profile_likelihood.png",
    # fig5 is generated directly by make_paper_fig5.py (1000-member bootstrap)
    "fig6_long_time.png": "outputs/plot_long_time.png",
    "fig7_sensitivity_maps.png": "outputs/plot_sensitivity_maps.png",
}
for dst, src in COPIES.items():
    s = Path(__file__).parent / src
    if s.exists():
        shutil.copy(s, OUT / dst)
        print(f"Copied: {src} -> outputs/paper/{dst}")
    else:
        print(f"MISSING: {src} (run its plot script first)")
