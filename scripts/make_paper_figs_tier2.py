# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper figures F8-F10 (PAPER_PLAN.md Section 5, Tier-2 items) under
outputs/paper/.

  fig8_steady_spatial.png  — (a) D-cancellation: nondim steady profile shape
    invariant under 2-3 orders of magnitude of D (only c0, the absolute
    scale, changes); (b) Newton-vs-analytic and NSEM-vs-Newton parity, both
    species, from the T2-B checkpoint (outputs/tier2/steady_{OV,CV}.npz).
  fig9_transient_qs.png — quasi-steady lag: transient NSEM profile vs the
    instantaneous quasi-steady shape at tau=1, both species, from the T2-C
    checkpoint (outputs/tier2/transient_{OV,CV}.npz), with the R1 lag
    annotated.
  fig10_composition_edx.png — copied from outputs/plot_tier2_composition.png
    (src/tier2_composition.py --fitted), already publication-quality.

Run:  python scripts/make_paper_figs_tier2.py   (AFTER
      src/tier2_steady_trainer.py, src/tier2_transient_trainer.py, and
      src/tier2_composition.py --fitted have been run)
"""

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import htw_pdm.tier2_physics as tier2_physics  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402
from htw_pdm.tier2_physics import analytic_steady, species_groups  # noqa: E402

from htw_pdm.paths import ROOT  # noqa: E402
T2_DIR = ROOT / "outputs" / "tier2"
from htw_pdm.paths import PAPER_OUT as OUT  # noqa: E402
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

# ── Fig 8 — steady: D-cancellation + Newton/NSEM parity ───────────────────────
fig8, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.3))

p1 = Parameters()
xi_demo = np.linspace(0.0, 1.0, 200)

# Empirical D-cancellation demonstration: actually CALL species_groups() (the
# real tier2_physics code path, not a re-derivation) at three D values spanning
# 4 orders of magnitude, by temporarily overriding the module-level D_OV_CM2_S/
# D_CV_CM2_S constants it reads. This is not asserted -- the overlaid curves
# below are computed independently at each D and must coincide numerically for
# the claim to be true; report the actual max profile-shape deviation and the
# actual c0 scale factor achieved, both measured from the real function calls.
D_OV_0, D_CV_0 = tier2_physics.D_OV_CM2_S, tier2_physics.D_CV_CM2_S
D_scales = [1e-2, 1.0, 1e2]
profiles = {"OV": [], "CV": []}
c0_by_scale = {"OV": [], "CV": []}
Pe_by_scale = {"OV": [], "CV": []}
for scale in D_scales:
    tier2_physics.D_OV_CM2_S = D_OV_0 * scale
    tier2_physics.D_CV_CM2_S = D_CV_0 * scale
    g_scan = species_groups(p1, 134.0)
    for name in ("OV", "CV"):
        profiles[name].append(analytic_steady(g_scan[name], xi_demo))
        c0_by_scale[name].append(g_scan[name].c0_cm3)
        Pe_by_scale[name].append(g_scan[name].Pe)
tier2_physics.D_OV_CM2_S, tier2_physics.D_CV_CM2_S = D_OV_0, D_CV_0  # restore

g0 = species_groups(p1, 134.0)  # ~480 h Tier-1 barrier thickness, default D
max_shape_dev = {}
# Color encodes species, marker encodes the D scale (markers every 40th point,
# offset per scale) — overlapping curves are then visibly overlapping instead
# of looking like missing curves.
D_markers = ("o", "s", "^")
for name, c in (("OV", "tab:blue"), ("CV", "tab:red")):
    for k, (scale, prof, mk) in enumerate(zip(D_scales, profiles[name], D_markers)):
        ax_a.plot(xi_demo, prof, c=c, ls="-", lw=1.3, marker=mk, ms=5,
                  mfc="none", markevery=(13 * k, 40))
    max_shape_dev[name] = float(np.max([
        np.abs(profiles[name][i] - profiles[name][1]).max() for i in (0, 2)
    ]))
    print(f"  [fig8 D-scan, empirical] {name}: max |shape(D) - shape(D_default)| "
          f"over D x {D_scales} = {max_shape_dev[name]:.3e} (Pe held at "
          f"{Pe_by_scale[name][0]:.6g}=={Pe_by_scale[name][1]:.6g}=="
          f"{Pe_by_scale[name][2]:.6g} across the D scan -- Pe truly does not "
          f"depend on D in this formulation); c0 scales "
          f"{c0_by_scale[name][0]:.3e} -> {c0_by_scale[name][2]:.3e} "
          f"cm^-3 (factor {c0_by_scale[name][2]/c0_by_scale[name][0]:.3g}, "
          f"vs the {D_scales[2]/D_scales[0]:.3g}x D range scanned).")
ax_a.set_xlabel(r"$\xi$ (nondim depth across barrier layer)")
ax_a.set_ylabel(r"$\hat{c}(\xi)$ (nondim concentration)")
ax_a.set_title("steady profile shape under a $10^{-2}\\times$–$10^{2}\\times$ "
               "$D$ scan", fontsize=10)
# Actual max deviation, reported honestly: exactly 0.0 (or below femto-level)
# means machine-precision coincidence.
_dev = max(max_shape_dev.values())
_dev_txt = ("$< 10^{-15}$ (machine precision)" if _dev < 1e-15
            else f"$= {_dev:.1e}$")
ax_a.text(0.97, 0.75, "max shape deviation\n" + _dev_txt,
          transform=ax_a.transAxes, ha="right", va="top", fontsize=8)
from matplotlib.lines import Line2D  # noqa: E402

_handles = [
    Line2D([], [], color="tab:blue", lw=1.6, label="OV"),
    Line2D([], [], color="tab:red", lw=1.6, label="CV"),
] + [
    Line2D([], [], color="grey", lw=0, marker=mk, mfc="none", ms=5,
           label=f"$D\\times{s:g}$" + (f" (Pe = {g0['OV'].Pe:.2g}/"
                                       f"{g0['CV'].Pe:.2g})" if s == 1.0 else ""))
    for s, mk in zip(D_scales, D_markers)
]
ax_a.legend(handles=_handles, fontsize=8, ncol=1, loc="center right")
ax_a.text(0.02, 0.98, "(a)", transform=ax_a.transAxes, fontweight="bold",
          va="top", fontsize=12)

rows = []
for name, c in (("OV", "tab:blue"), ("CV", "tab:red")):
    f = T2_DIR / f"steady_{name}.npz"
    if not f.exists():
        ax_b.text(0.5, 0.5, f"MISSING: {f.name}\n(run src/tier2_steady_trainer.py)",
                  ha="center", va="center", transform=ax_b.transAxes, fontsize=9)
        continue
    d = np.load(f)
    err_newton_analytic = np.abs(d["chat_newton"] - d["chat_exact"])
    err_nsem_newton = np.abs(d["chat_nsem"] - d["chat_newton"]) / np.abs(d["chat_newton"]).max()
    ax_b.semilogy(d["xi"], np.maximum(err_newton_analytic, 1e-16), c=c, ls="--",
                  label=f"{name}: |Newton - analytic|")
    ax_b.semilogy(d["xi"], np.maximum(err_nsem_newton, 1e-16), c=c, ls="-",
                  label=f"{name}: |NSEM - Newton| / max|Newton|")
    rows.append((name, float(err_newton_analytic.max()), float(err_nsem_newton.max())))
ax_b.axhline(1e-10, color="grey", ls=":", lw=1)
ax_b.axhline(5e-3, color="grey", ls=":", lw=1)
ax_b.text(0.02, 1.4e-10, "Newton vs analytic bound", fontsize=8, color="grey",
          va="bottom")
ax_b.text(0.02, 7e-3, "NSEM acceptance 0.5%", fontsize=8, color="grey",
          va="bottom")
ax_b.set_xlabel(r"$\xi$")
ax_b.set_ylabel("absolute / relative error")
ax_b.set_title("verification ladder, both species", fontsize=10)
ax_b.legend(fontsize=8)
ax_b.text(0.02, 0.98, "(b)", transform=ax_b.transAxes, fontweight="bold",
          va="top", fontsize=12)
plt.tight_layout()
out8 = OUT / "fig8_steady_spatial.png"
plt.savefig(out8, dpi=300)
print(f"Saved: {out8}")
for name, e1, e2 in rows:
    print(f"  {name}: Newton-vs-analytic {e1:.2e} (<=1e-10), "
          f"NSEM-vs-Newton {e2:.2e} (<=0.5%)")

# ── Fig 9 — transient: quasi-steady lag ────────────────────────────────────────
fig9, axes9 = plt.subplots(1, 2, figsize=(11, 4.3))
for k, (ax, name) in enumerate(zip(axes9, ("OV", "CV"))):
    f = T2_DIR / f"transient_{name}.npz"
    if not f.exists():
        ax.text(0.5, 0.5, f"MISSING: {f.name}\n(run src/tier2_transient_trainer.py)",
               ha="center", va="center", transform=ax.transAxes, fontsize=9)
        continue
    d = np.load(f)
    xi, tau, chat = d["xi"], d["tau"], d["chat"]
    Pe_end, jhat_end = float(d["Pe"][-1]), float(d["jhat"][-1])
    g_end = species_groups(p1, 134.0)[name]
    # instantaneous quasi-steady shape at tau=1 (jhat(tau=1) x analytic steady)
    ss_end = jhat_end * analytic_steady(g_end, xi)
    lag = float(np.max(np.abs(chat[-1] - ss_end)) / np.max(np.abs(ss_end)))
    print(f"  [fig9] {name}: quasi-steady lag = {lag:.6f} ({lag:.4%})")
    ax.plot(xi, chat[-1], "b-", lw=2, label="transient NSEM, $\\tau=1$")
    ax.plot(xi, ss_end, "k--", lw=1.5, label="instantaneous quasi-steady")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\hat{c}(\xi, \tau=1)$")
    ax.set_title(f"{name}: quasi-steady lag = {lag:.2%}", fontsize=10)
    if k == 0:
        ax.legend(fontsize=8)
    ax.text(0.02, 0.98, f"({'ab'[k]})", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=12)
plt.tight_layout()
out9 = OUT / "fig9_transient_qs.png"
plt.savefig(out9, dpi=300)
print(f"Saved: {out9}")

# ── Fig 10 — composition map (already publication-quality) ────────────────────
src10 = ROOT / "outputs" / "plot_tier2_composition.png"
if src10.exists():
    shutil.copy(src10, OUT / "fig10_composition_edx.png")
    print(f"Copied: {src10} -> outputs/paper/fig10_composition_edx.png")
else:
    print(f"MISSING: {src10} (run: python -m htw_pdm.tier2_composition --fitted)")
