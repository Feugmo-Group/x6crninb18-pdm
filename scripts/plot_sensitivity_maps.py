# SPDX-FileCopyrightText: Copyright (c) 2026 Conrard Giresse Tetsassi Feugmo
# SPDX-License-Identifier: Apache-2.0

"""Phase E2 figures — (T, [O2]) sensitivity maps of barrier-layer growth.

Contour maps of L_bl over the BWR operating box (T in [200, 285] C at 7 MPa,
[O2] in [0.01, 8] ppm spanning NWC/HWC chemistry), computed from the closed-form
HTW_PDM solution at the condition-mapped apparent parameters
(src/physics_parametric.py, anchored at the M4 fit):

  1. plot_sensitivity_maps.png — L_bl(480 h) and L_bl(10 y) maps for the three
     dG0_R scan values {25, 50, 100} kJ/mol (the un-identifiable Arrhenius
     assumption is shown as a scan, never as a single hidden default).
  2. If outputs/parametric/predictions.npz exists (from src/parametric_trainer.py),
     the trained NSEM conditions are overlaid on the 480 h / 50 kJ map with their
     closed-form validation errors — the "parametric net" verification exhibit.

Run:  python scripts/plot_sensitivity_maps.py
"""


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed  # noqa: E402
from htw_pdm.paths import OUTPUTS as OUT_DIR  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402
from htw_pdm.physics_parametric import ConditionScan, apparent_parameters  # noqa: E402

OUT_DIR.mkdir(exist_ok=True)

REF = Parameters()  # M4
# dG0_R scan (kJ/mol). The effective Arrhenius activation energy is alpha3*dG0_R,
# so at alpha3 = 0.12 these correspond to Ea_eff = 3, 12, 30 and 50 kJ/mol. The
# upper two extend the scan into the range reported for weight-gain kinetics on
# austenitic steels in high-temperature water -- the robustness of the long-time
# prediction must be tested against that range, not only against the low values
# the transfer coefficient of a supercritical-water fit implies.
DG_SCAN_KJ = [25.0, 100.0, 250.0, 417.0]
ALPHA3 = 0.12
T_GRID = np.linspace(200.0, 285.0, 60)
O2_GRID = np.logspace(np.log10(0.01), np.log10(8.0), 60)
HORIZONS = [(480.0, "480 h (Veile window)"), (10 * 8766.0, "10 y")]


def L_bl_map(dG0_R_J: float, t_h: float) -> np.ndarray:
    scan = ConditionScan(dG0_R=dG0_R_J)
    out = np.zeros((len(O2_GRID), len(T_GRID)))
    for i, O2 in enumerate(O2_GRID):
        for j, T_C in enumerate(T_GRID):
            p = apparent_parameters(T_C, O2, REF, scan)
            bp = HTWPDMParams(A_bl=p.A_bl, b3=p.b3, C_bl=p.C_bl, PBR_eff=p.PBR_eff,
                              C_x=p.C_x, L0=p.L0, L_ol0=p.L_ol0)
            out[i, j] = L_bl_closed(bp, np.array([t_h]))[0]
    return out


plt.rcParams.update({
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

fig, axes = plt.subplots(len(HORIZONS), len(DG_SCAN_KJ),
                         figsize=(5 * len(DG_SCAN_KJ), 4.2 * len(HORIZONS)),
                         constrained_layout=True)

npz_path = OUT_DIR / "parametric" / "predictions.npz"
nsem = np.load(npz_path) if npz_path.exists() else None

# Precompute all maps so each row shares one vmin/vmax and ONE colorbar.
Z_all = {(row, col): L_bl_map(dg * 1e3, t_h)
         for row, (t_h, _) in enumerate(HORIZONS)
         for col, dg in enumerate(DG_SCAN_KJ)}

for row, (t_h, t_label) in enumerate(HORIZONS):
    z_row = [Z_all[(row, col)] for col in range(len(DG_SCAN_KJ))]
    vmin = min(z.min() for z in z_row)
    vmax = max(z.max() for z in z_row)
    levels = np.linspace(vmin, vmax, 15)
    cs = None
    for col, dg in enumerate(DG_SCAN_KJ):
        ax = axes[row, col]
        Z = Z_all[(row, col)]
        cs = ax.contourf(T_GRID, O2_GRID, Z, levels=levels, cmap="viridis")
        cl = ax.contour(T_GRID, O2_GRID, Z, levels=levels[::2],
                        colors="w", linewidths=0.6)
        ax.clabel(cl, fontsize=7, fmt="%.0f")
        ax.set_yscale("log")
        ax.set_xlabel("T (°C)")
        if col == 0:
            ax.set_ylabel("[O$_2$] (ppm)")
        ax.set_title(f"{t_label},  $\\Delta G^0_R$ = {dg:.0f} kJ/mol "
                     f"($E_a^\\mathrm{{eff}}$ = {ALPHA3 * dg:.0f} kJ/mol)",
                     fontsize=9.5)
        ax.plot(240.0, 0.4, "r*", ms=14,
                label="Veile 2024 (anchor)" if (row, col) == (0, 0) else None)
        if nsem is not None and row == 0 and dg == DG_SCAN_KJ[1]:
            conds, errs = nsem["conditions"], nsem["rel_errs"]
            ax.scatter(conds[:, 0], conds[:, 1], s=40, c="red", marker="o",
                       edgecolors="w", zorder=5,
                       label=f"NSEM conditions (worst {errs.max():.1e})")
        letter = "abcdefgh"[row * len(DG_SCAN_KJ) + col]
        ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes,
                fontweight="bold", va="top", fontsize=12, color="w")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="lower left")
    fig.colorbar(cs, ax=list(axes[row, :]), label="$L_\\mathrm{bl}$ (nm)",
                 shrink=0.9)

out = OUT_DIR / "plot_sensitivity_maps.png"
plt.savefig(out, dpi=300)
print(f"Saved: {out}")

# Scalar summary: total spread across the box per scan value.
for row, (t_h, t_label) in enumerate(HORIZONS):
    for col, dg in enumerate(DG_SCAN_KJ):
        Z = Z_all[(row, col)]
        print(f"  {t_label:22s} dG0_R = {dg:5.0f} kJ/mol "
              f"(Ea_eff = {ALPHA3 * dg:4.0f} kJ/mol):  "
              f"L_bl in [{Z.min():6.1f}, {Z.max():6.1f}] nm  "
              f"(ratio {Z.max() / Z.min():.2f})")


# ── Line figure: the same envelope, readable at \textwidth ───────────────────
# The eight contour maps above reduce to about 4 cm of page height in the
# manuscript, where their titles and contour labels are illegible, and the
# quantity the text actually quotes -- the factor by which the prediction
# varies across the box -- cannot be read off them at all. The maps are also
# nearly featureless: the contours run close to vertical, which is the result
# (temperature dominates, three decades of dissolved oxygen barely register)
# stated in a form that takes eight panels to say. These four line panels say it
# directly and carry the factors. The maps move to the SI.
O2_REF, T_REF = 0.4, 240.0
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]


def _sweep(dG0_R_J, t_h, *, temps=None, o2s=None):
    scan = ConditionScan(dG0_R=dG0_R_J)
    out = []
    for T_C, O2 in zip(temps if temps is not None else np.full(len(o2s), T_REF),
                       o2s if o2s is not None else np.full(len(temps), O2_REF)):
        p = apparent_parameters(T_C, O2, REF, scan)
        bp = HTWPDMParams(A_bl=p.A_bl, b3=p.b3, C_bl=p.C_bl, PBR_eff=p.PBR_eff,
                          C_x=p.C_x, L0=p.L0, L_ol0=p.L_ol0)
        out.append(L_bl_closed(bp, np.array([t_h]))[0])
    return np.array(out)


plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10})
fig2, ax2 = plt.subplots(2, 2, figsize=(11.0, 7.4))
for row, (t_h, t_label) in enumerate(HORIZONS):
    for col, (xlab, xs, kw) in enumerate((
            (f"T (°C)   at [O$_2$] = {O2_REF} ppm", T_GRID, {"temps": T_GRID}),
            (f"[O$_2$] (ppm)   at T = {T_REF:.0f}°C", O2_GRID, {"o2s": O2_GRID}))):
        ax = ax2[row, col]
        for dg, c in zip(DG_SCAN_KJ, COLORS):  # noqa: B905
            ax.plot(xs, _sweep(dg * 1e3, t_h, **kw), color=c, lw=2,
                    label=f"$E_a^\\mathrm{{eff}}$ = {ALPHA3 * dg:.0f} kJ/mol")
        if col == 1:
            ax.set_xscale("log")
            # Share the temperature panel's y-range. Left to autoscale, three
            # decades of oxygen fill the panel and read as a strong dependence;
            # on a common scale the line is visibly flat, which is the result.
            ax.set_ylim(ax2[row, 0].get_ylim())
            # Under the panel letter, not at the foot: the spread box holds
            # the bottom right and the two collide there.
            ax.text(0.03, 0.87,
                    "all four $E_a^\\mathrm{eff}$ curves coincide:\n"
                    "the O$_2$ channel is a prefactor shift",
                    transform=ax.transAxes, va="top", fontsize=9.5)
        # Anchor: the one condition the model was actually calibrated at.
        ax.axvline(T_REF if col == 0 else O2_REF, color="k", ls=":", lw=1)
        ax.set_xlabel(xlab)
        ax.set_ylabel("$L_\\mathrm{bl}$ (nm)")
        ax.set_title(f"{t_label}", fontsize=12)
        ax.text(0.02, 0.97, f"({'abcd'[row * 2 + col]})", transform=ax.transAxes,
                fontweight="bold", va="top", fontsize=13)
        if (row, col) == (0, 0):
            ax.legend(loc="upper left", bbox_to_anchor=(0.10, 0.98), framealpha=0.9)
    # The headline number, on the figure: the full-box spread at each scanned
    # activation energy, which is what Table 8 tabulates.
    facs = [Z_all[(row, col)].max() / Z_all[(row, col)].min()
            for col in range(len(DG_SCAN_KJ))]
    ax2[row, 1].text(
        0.97, 0.05,
        "full-box spread:\n" + ",  ".join(
            f"$\\times${f:.2f}" for f in facs),
        transform=ax2[row, 1].transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.7", lw=0.6))

plt.tight_layout()
out_lines = OUT_DIR / "plot_envelope_lines.png"
plt.savefig(out_lines, dpi=300)
print(f"Saved: {out_lines}")
