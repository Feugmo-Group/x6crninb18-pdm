# SPDX-FileCopyrightText: Copyright (c) 2026 Conrard Giresse Tetsassi Feugmo
# SPDX-License-Identifier: Apache-2.0

"""fig11_f1_failure.png — the F-1 failure mode, shown rather than tabulated.

F-1 is the paper's methodological contribution, and until now it was carried
entirely by two numbers in a table: a trajectory chi2 of 0.21 against a
closed-form chi2 of 16.6 at the same recovered parameters. Those numbers state
the result; they do not show why it is dangerous. What makes the failure worth
documenting is that the trained trajectory looks like an excellent fit -- better
than the accepted model's -- while the parameters it reports are wrong. Two
panels put those side by side:

  (a) what the training loss sees: the soft-mode trajectory through the six
      measurements, chi2 = 0.21, visually a near-perfect fit.
  (b) what the physics says: the exact closed-form solution at exactly those
      recovered parameters, chi2 = 16.6, plotted against the same data with the
      hard-mode/M4 solution for reference.

Reads outputs/paper/f1_trajectory.npz (written by scripts/make_f1_trajectory.py,
which needs torch + PhysicsNeMo). This script needs only NumPy and Matplotlib.

Run:  uv run python scripts/make_paper_fig_f1.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from htw_pdm.baseline_fit import build_fit_data, fit_pdm, load_data  # noqa: E402
from htw_pdm.baseline_ode import L_bl_closed, L_ol_closed  # noqa: E402
from htw_pdm.paths import OUTPUTS  # noqa: E402

OUT = OUTPUTS / "paper"
SRC = OUT / "f1_trajectory.npz"
if not SRC.exists():
    raise SystemExit(
        f"{SRC} missing — run `uv run --extra nsem python scripts/make_f1_trajectory.py`"
    )
f1 = np.load(SRC)

means, _ = load_data()
sol4, p4, chi2_4, _dof, _ = fit_pdm(build_fit_data(means), 4)

TEXTWIDTH_IN, FIG_W = 5.15, 9.0
PT = FIG_W / TEXTWIDTH_IN
plt.rcParams.update({
    "axes.labelsize": 8 * PT, "xtick.labelsize": 7 * PT,
    "ytick.labelsize": 7 * PT, "legend.fontsize": 6.5 * PT, "font.size": 7 * PT,
})

rows = {el: sorted(means[el]) for el in ("Cr", "Fe")}
t_obs = {el: np.array([r[0] for r in rows[el]]) for el in rows}
L_obs = {el: np.array([r[1] for r in rows[el]]) for el in rows}
# The fit weights by standard error, so the error bars a reader compares the
# curves against must be the same quantity, not the population SD.
s_obs = {el: np.array([r[2] / np.sqrt(r[3]) for r in rows[el]]) for el in rows}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIG_W, 3.9), sharey=True)
td = f1["t_dense_h"]

for ax in (axA, axB):
    for el, color in (("Cr", "tab:blue"), ("Fe", "tab:orange")):
        ax.errorbar(t_obs[el], L_obs[el], yerr=s_obs[el], fmt="o", ms=5,
                    capsize=3, color="k", mfc=color, lw=1, zorder=5,
                    label=f"{el} data" if ax is axA else None)
    ax.set_xlabel("t (h)")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 300)

# (a) What the training loss sees.
axA.plot(f1["t_h"], f1["traj_bl_nm"], "-", color="tab:blue", lw=2,
         label="trajectory $L_\\mathrm{bl}$")
axA.plot(f1["t_h"], f1["traj_ol_nm"], "-", color="tab:orange", lw=2,
         label="trajectory $L_\\mathrm{ol}$")
axA.set_ylabel("L (nm)")
axA.set_title("(a) the trained soft-mode trajectory", fontsize=8 * PT)
axA.text(0.97, 0.06,
         f"$\\chi^2_\\mathrm{{traj}} = {float(f1['chi2_traj']):.2f}$\n"
         "looks like an excellent fit",
         transform=axA.transAxes, ha="right", va="bottom", fontsize=7.5 * PT,
         bbox=dict(boxstyle="round,pad=0.3", fc="#eaf6ea", ec="0.6", lw=0.6))
axA.legend(loc="upper left", framealpha=0.92, ncol=2)

# (b) What the governing equation says at the SAME recovered parameters.
axB.plot(td, f1["exact_bl_nm"], "--", color="tab:blue", lw=2,
         label="exact $L_\\mathrm{bl}$")
axB.plot(td, f1["exact_ol_nm"], "--", color="tab:orange", lw=2,
         label="exact $L_\\mathrm{ol}$")
axB.plot(td, L_bl_closed(p4, td), ":", color="0.35", lw=1.5,
         label="M4 (hard mode)")
axB.plot(td, L_ol_closed(p4, td), ":", color="0.35", lw=1.5)
axB.set_title("(b) the exact solve at those same parameters", fontsize=8 * PT)
ratio = float(f1["chi2_exact"]) / float(f1["chi2_traj"])
axB.text(0.97, 0.06,
         f"$\\chi^2_\\mathrm{{exact}} = {float(f1['chi2_exact']):.2f}$\n"
         f"factor {ratio:.0f} apart — FAIL",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=7.5 * PT,
         bbox=dict(boxstyle="round,pad=0.3", fc="#fdeaea", ec="0.6", lw=0.6))
axB.legend(loc="upper left", framealpha=0.92, ncol=2)

plt.tight_layout()
out = OUT / "fig11_f1_failure.png"
plt.savefig(out, dpi=300)
print(f"Saved: {out}")
print(f"  trajectory chi2 = {float(f1['chi2_traj']):.2f}, "
      f"exact chi2 = {float(f1['chi2_exact']):.2f} (factor {ratio:.0f}); "
      f"hard mode {chi2_4:.2f} both ways")
