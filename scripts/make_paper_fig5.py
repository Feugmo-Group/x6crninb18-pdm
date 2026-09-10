"""Regenerate paper figure 5 (bootstrap ensemble vs profile likelihood) from the
1000-member deterministic bootstrap (make_paper_stats.py), replacing the earlier
50-member PINN-bootstrap version. Profile-likelihood 1-sigma intervals are read
from the authoritative table1_parameters.csv.

Run:  python scripts/make_paper_fig5.py    (after scripts/make_paper_stats.py)
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htw_pdm.paths import OUTPUTS as OUT  # noqa: E402

PAPER = OUT / "paper"

# Placed at width=\textwidth (= 372 pt = 5.15 in) in the manuscript, so a
# figure FIG_W in wide is scaled by 5.15/FIG_W on the page.  PT inverts that:
# `fontsize=N*PT` renders at N pt, at or just below the caption size.
TEXTWIDTH_IN = 5.15
FIG_W_IN = 11.5
PT = FIG_W_IN / TEXTWIDTH_IN
plt.rcParams.update({
    "axes.labelsize": 8.0 * PT,
    "xtick.labelsize": 7.0 * PT,
    "ytick.labelsize": 7.0 * PT,
    "legend.fontsize": 6.5 * PT,
    "font.size": 7.0 * PT,
})

members = np.genfromtxt(OUT / "ensemble_members_1000.csv", delimiter=",", names=True)
names = ["A_bl", "b3", "PBR_eff", "L0", "L_ol0"]
pretty = {
    "A_bl": "$A_\\mathrm{bl}$ (nm h$^{-1}$)",
    "b3": "$b_3$ (nm$^{-1}$)",
    "PBR_eff": "$\\mathrm{PBR}_\\mathrm{eff}$ (–)",
    "L0": "$L_0$ (nm)",
    "L_ol0": "$L_\\mathrm{ol,0}$ (nm)",
}

with open(PAPER / "table1_parameters.csv") as f:
    t1 = {r["param"]: r for r in csv.DictReader(f)}

# 2 x 3 rather than 1 x 5: the single row rendered ~1 in tall at \textwidth,
# leaving every tick and axis label near 2 pt on the page.
fig, axes_grid = plt.subplots(2, 3, figsize=(FIG_W_IN, 7.6))
axes = axes_grid.ravel()
for _ax in axes[len(names):]:
    _ax.remove()
for k, (ax, name) in enumerate(zip(axes, names)):
    arr = members[name]
    ax.hist(arr, bins=30, color="steelblue", alpha=0.7, density=True,
            label="bootstrap (n=1000)" if k == 0 else None)
    lo_p = float(t1[name]["profile_1sig_lo"])
    hi_p = float(t1[name]["profile_1sig_hi"])
    ax.axvspan(lo_p, hi_p, color="red", alpha=0.15,
               label="profile 1$\\sigma$" if k == 0 else None)
    ax.axvline(float(t1[name]["m4_fit"]), color="k", ls="--", lw=1,
               label="M4 fit" if k == 0 else None)
    ax.set_xlabel(pretty[name])
    ax.set_ylabel("probability density")
    ax.text(0.04, 0.96, f"({'abcde'[k]})", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=9 * PT)
    # Two features of this figure are results rather than plotting artefacts,
    # and both were previously left for the reader to work out.
    # Guarded on sign: b_3 is negative throughout, so a bare "within 1% of the
    # maximum" test marks every one of its members as sitting on a bound.
    at_bound = (arr.min() >= 0.0) and (arr.min() < 1e-6 * arr.max())
    frac0 = float((arr < 0.01 * arr.max()).mean()) if at_bound else 0.0
    if frac0 > 0.05:
        # Members driven to the lower bound: the data do not exclude a
        # vanishing value, which is what "weakly identifiable" means here.
        ax.text(0.5, 0.55, f"{100 * frac0:.0f}% of members\nat the lower bound",
                transform=ax.transAxes, ha="center", va="top", fontsize=5.8 * PT,
                bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.7", lw=0.6))
    if name == "L0":
        # The one place bootstrap and profile disagree, and the disagreement is
        # the finding: perturbing the data cannot explore a direction the data
        # do not constrain, so the narrow histogram is not a tight constraint.
        ax.text(0.5, 0.55,
                "prior-set, not data-set:\nbootstrap $\\ll$ profile band",
                transform=ax.transAxes, ha="center", va="top", fontsize=5.8 * PT,
                bbox=dict(boxstyle="round,pad=0.25", fc="#fdeaea", ec="0.7", lw=0.6))

# One figure-level legend: inside panel (a) the box covered both the panel
# letter and the top of the histogram.
_h, _l = axes[0].get_legend_handles_labels()
fig.legend(_h, _l, fontsize=6.5 * PT, loc="upper center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(rect=(0, 0, 1, 0.965))
out = PAPER / "fig5_uncertainty_ensemble.png"
plt.savefig(out, dpi=300)
print(f"Saved: {out}")
for name in names:
    arr = members[name]
    print(f"  {name:8s}: mean {arr.mean():.5g}, std {arr.std():.3g}, "
          f"[16,84]% [{np.percentile(arr,16):.4g}, {np.percentile(arr,84):.4g}]")
