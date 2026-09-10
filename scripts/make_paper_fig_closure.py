# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""fig12_field_closure.png — the constant-field closure is not self-consistent.

Section 3.5 refutes an assumption the PDM makes about itself, and the result is
in the abstract and the conclusions, but the section carries no figure: the
reader gets a Debye length of 0.06 nm and a six-order-of-magnitude diffusivity
gap as bare numbers. Both are distances and both have obvious yardsticks, which
is exactly the case where a figure beats a sentence.

  (a) How far the self-consistently solved field departs from uniform, against
      the defect diffusivity that would be needed to hold it there. The 1, 5 and
      10 % deviation thresholds are marked; the diffusivity actually assumed in
      the spatial model sits six orders of magnitude to the left of all three.
  (b) The Debye length against its two natural yardsticks: the barrier layer it
      is supposed to be screening across, and an interatomic spacing. At 0.06 nm
      it falls several times below the smaller of the two, which is what makes
      the closure untenable rather than merely approximate.

Reads outputs/paper/tier4_pnp.json (written by `python -m htw_pdm.tier4_pnp_solve`).

Run:  uv run python scripts/make_paper_fig_closure.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from htw_pdm.paths import OUTPUTS  # noqa: E402
from htw_pdm.tier2_physics import D_OV_CM2_S  # noqa: E402

OUT = OUTPUTS / "paper"
SRC = OUT / "tier4_pnp.json"
if not SRC.exists():
    raise SystemExit(f"{SRC} missing — run `python -m htw_pdm.tier4_pnp_solve`")
res = json.loads(SRC.read_text())

# An interatomic spacing in austenite: a/sqrt(2) for the fcc nearest neighbour
# at a = 0.36 nm. This is the yardstick that makes the result qualitative rather
# than quantitative -- a screening length below it has no physical meaning.
D_ATOMIC_NM = 0.36 / np.sqrt(2.0)
L_NM = float(res["L_nm"])
EPS_R_USED = float(res["eps_r"])

TEXTWIDTH_IN, FIG_W = 5.15, 9.0
PT = FIG_W / TEXTWIDTH_IN
plt.rcParams.update({
    "axes.labelsize": 8 * PT, "xtick.labelsize": 7 * PT,
    "ytick.labelsize": 7 * PT, "legend.fontsize": 6.5 * PT, "font.size": 7 * PT,
})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIG_W, 3.9))

# ── (a) field deviation vs the diffusivity that would be needed ──────────────
# Lam ~ 1/D at fixed flux, so the continuation parameter maps directly onto a
# required diffusivity: lam_scale = D_assumed / D.
curve = res["A3_validity"]["curve"]
ls = np.array([c["lam_scale"] for c in curve])
dev = np.array([c["max_field_dev"] for c in curve]) * 100.0
axA.loglog(D_OV_CM2_S / ls, dev, "b-o", ms=3, lw=1.6)

# Threshold labels ride the right-hand edge: the curve falls away to the right,
# so that strip is empty at every deviation level, while the left half is taken
# by the assumed-diffusivity marker.
for target, style in ((0.01, "--"), (0.05, "-."), (0.10, ":")):
    thr = res["A3_validity"]["thresholds"][f"{target:.2f}"]
    axA.axhline(target * 100.0, color="grey", ls=style, lw=1)
    axA.text(2.5e-7, target * 100.0 * 1.08, f"{target * 100:.0f}% deviation",
             fontsize=6.0 * PT, color="0.35", ha="right", va="bottom")
    axA.plot([thr["D_required_cm2_s"]["OV"]], [target * 100.0], "kv", ms=5, zorder=5)

d10 = res["A3_validity"]["thresholds"]["0.10"]["D_required_cm2_s"]["OV"]
axA.axvline(D_OV_CM2_S, color="crimson", lw=1.5)
axA.text(D_OV_CM2_S * 1.5, 24.0, "$D$ assumed here ($10^{-15}$ cm$^2$/s)",
         fontsize=6.5 * PT, color="crimson", va="top", ha="left")
axA.annotate("", xy=(d10, 0.45), xytext=(D_OV_CM2_S, 0.45),
             arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.2))
axA.text(np.sqrt(d10 * D_OV_CM2_S), 0.52,
         f"{np.log10(d10 / D_OV_CM2_S):.0f} orders of magnitude",
         ha="center", va="bottom", fontsize=6.5 * PT, color="crimson")
axA.set_xlabel("defect diffusivity $D_\\mathrm{OV}$ (cm$^2$/s)")
axA.set_ylabel("field departure from uniform (%)")
axA.set_xlim(D_OV_CM2_S / 3.0, 3e-7)
axA.set_ylim(0.35, 30)
axA.set_title("(a) uniformity needs a diffusivity the model\ndoes not have",
              fontsize=8 * PT)

# ── (b) Debye length against its two yardsticks ──────────────────────────────
per = res["A2_magnitude"]["per_eps_r"]
eps_r = np.array([float(k) for k in per])
lam_D = np.array([per[k]["debye_nm"] for k in per])
order = np.argsort(eps_r)
axB.loglog(eps_r[order], lam_D[order], "b-o", ms=5, lw=1.8,
           label="Debye length $\\lambda_D$")
axB.axhline(L_NM, color="k", ls="--", lw=1.2,
            label=f"barrier layer, $L$ = {L_NM:.0f} nm")
axB.axhline(D_ATOMIC_NM, color="crimson", ls=":", lw=1.5,
            label=f"interatomic spacing, {D_ATOMIC_NM:.2f} nm")
# The value quoted in the text, at the permittivity the solve actually used.
lam_used = float(np.interp(np.log(EPS_R_USED), np.log(eps_r[order]), lam_D[order]))
axB.plot([EPS_R_USED], [lam_used], "r*", ms=12, zorder=6,
         label=f"$\\varepsilon_r$ = {EPS_R_USED:.0f}: "
               f"$\\lambda_D \\approx {lam_used:.2f}$ nm")
axB.set_xlabel("relative permittivity $\\varepsilon_r$")
axB.set_ylabel("length (nm)")
axB.set_ylim(0.02, 400)
axB.set_title("(b) the screening length falls below\nan interatomic spacing", fontsize=8 * PT)
axB.legend(loc="center right")

plt.tight_layout()
out = OUT / "fig12_field_closure.png"
plt.savefig(out, dpi=300)
print(f"Saved: {out}")
print(f"  lambda_D at eps_r = {EPS_R_USED:.0f}: {lam_used:.3f} nm "
      f"({L_NM / lam_used:.0f} of them across the barrier layer)")
print(f"  10% threshold needs D = {d10:.2g} cm^2/s vs {D_OV_CM2_S:.0g} assumed "
      f"({np.log10(d10 / D_OV_CM2_S):.0f} orders)")
