"""New paper figure: NSEM/PINN workflow schematic (referee-requested addition).

Explains, as a diagram, how the spectral-element-network (NSEM) solver is used
in this paper: (a) the shared DVR-collocation + element-network representation,
(b) the two zero-dimensional inverse modes (hard: differentiable RK4 through
the kinetics only; soft: joint field+parameter training with BRDR loss
balancing), and (c) the spatial extension where no integrator shortcut exists
and the soft-mode machinery is required. This is a schematic of the actual
code paths in src/trainer.py, src/inverse_trainer.py, src/tier2_steady_trainer.py,
src/tier2_transient_trainer.py -- not independent artwork.

Run:  python scripts/make_paper_fig_nsem.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from htw_pdm.paths import PAPER_OUT as OUT  # noqa: E402
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

fig = plt.figure(figsize=(12, 6.6))
gs = fig.add_gridspec(2, 1, height_ratios=[0.62, 1.0], hspace=0.15)


def box(ax, xy, w, h, text, fc="#eaf1fb", ec="#2a4d7a", fontsize=8.3, weight="normal"):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                       fc=fc, ec=ec, lw=1.2, zorder=2)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, zorder=3, wrap=True)
    return b


def arrow(ax, p0, p1, **kw):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                        color="k", lw=1.1, zorder=4, **kw)
    ax.add_patch(a)


# ── Panel (a): shared NSEM representation ─────────────────────────────────────
axA = fig.add_subplot(gs[0])
axA.set_xlim(0, 12)
axA.set_ylim(0.2, 3.0)
axA.axis("off")
axA.set_title("(a) Shared NSEM representation: DVR collocation + element network",
              fontsize=11, loc="left")

box(axA, (0.3, 1.1), 2.3, 1.3,
    "Node set\n(Gauss–Lobatto–Legendre\nDVR collocation,\nlog- or uniformly-clustered)",
    fc="#f5f0e6", ec="#8a6d3b")
arrow(axA, (2.6, 1.75), (3.5, 1.75))
box(axA, (3.5, 1.1), 2.6, 1.3,
    "Element network\n(MLP or KAN\nbackbone)",
    fc="#eaf1fb", ec="#2a4d7a")
arrow(axA, (6.1, 1.75), (7.0, 1.75))
box(axA, (7.0, 1.1), 2.4, 1.3,
    "Field values\non the node set\n(softplus > 0\nfor thicknesses)",
    fc="#eaf6ec", ec="#2f7a3f")
arrow(axA, (9.4, 1.75), (10.3, 1.75))
box(axA, (10.3, 0.85), 1.5, 1.8,
    "Physics\nresidual\nat every\nnode\n(first-\nintegral /\ncollocation\nform)",
    fc="#fbeaea", ec="#a33")

# ── Panel (b): three concrete uses ────────────────────────────────────────────
axB = fig.add_subplot(gs[1])
axB.set_xlim(0, 12)
axB.set_ylim(0, 11.0)
axB.axis("off")
axB.set_title("(b) Three concrete uses in this paper", fontsize=11, loc="left")

# Column headers, placed with room to grow upward (va='bottom') so they can
# never collide with the box below.
for x, title in ((2.0, "Zero-dim. inverse, HARD mode"),
                 (6.0, "Zero-dim. inverse, SOFT mode"),
                 (10.0, "Spatial defect transport")):
    axB.text(x, 10.15, title, ha="center", va="bottom", fontsize=8.6, weight="bold")

# Column 1: hard-mode 0-D inverse
box(axB, (0.2, 7.3), 3.6, 2.6,
    "Differentiable RK4 through\nkinetic parameters only.\n"
    "Physics exact by construction\n(no residual slack).",
    fc="#eaf6ec", ec="#2f7a3f", fontsize=7.6)

# Column 2: soft-mode 0-D inverse
box(axB, (4.2, 7.3), 3.6, 2.6,
    "Field networks + kinetic\nparameters trained jointly.\n"
    "Can satisfy data loss while\nODE residual stays\nunsatisfied (failure mode F-1).",
    fc="#fdeee0", ec="#b5651d", fontsize=7.6)

# Column 3: spatial (forced soft-mode) -- no shortcut exists
box(axB, (8.2, 7.3), 3.6, 2.6,
    "No differentiable integrator\nshortcut exists for the PDE:\n"
    "field network is the only\noption, verified against an\nindependent Newton solve first.",
    fc="#eaf1fb", ec="#2a4d7a", fontsize=7.6)

for x in (2.0, 6.0, 10.0):
    arrow(axB, (x, 7.3), (x, 6.5))

# Shared training machinery box
box(axB, (0.6, 4.9), 10.8, 1.5,
    "Shared optimization: loss terms combined by the balanced-residual dynamic "
    "reweighting (BRDR) aggregator →\n"
    "two-phase schedule (Adam, coarse basin-finding → L-BFGS, high-precision "
    "convergence)",
    fc="#f2f2f2", ec="#555", fontsize=8.0)

arrow(axB, (6.0, 4.9), (6.0, 4.2))

# Verification / self-consistency box
box(axB, (0.6, 2.5), 10.8, 1.6,
    "Verification (every mode, before any result is reported): forward parity against a "
    "closed-form or Newton reference\n"
    "(≤ 0.5% relative $L_\\infty$ acceptance); inverse self-consistency check "
    "(re-solve recovered parameters through an\nindependent exact forward solve; "
    "large trajectory-vs-exact gap → FAIL, as in the F-1 exhibit)",
    fc="#fbeaea", ec="#a33", fontsize=8.0)

arrow(axB, (6.0, 2.5), (6.0, 1.75))

box(axB, (2.5, 0.4), 7.0, 1.2,
    "Reported result (parameter estimate, forward prediction, or\n"
    "identifiability verdict) used in the main text",
    fc="#eaf6ec", ec="#2f7a3f", fontsize=8.2, weight="bold")

plt.tight_layout()
out = OUT / "fig_nsem_workflow.png"
plt.savefig(out, dpi=300)
print(f"Saved: {out}")
