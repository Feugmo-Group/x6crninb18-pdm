# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense-in-time Ni zone width from the fitted constant-mobility closure.

Section 2.6 rejects the constant-mobility Ni transport model on a SHAPE
argument: the model can only produce a width that grows monotonically with
exposure, while the three measured widths are flat within scatter.  Stated as
chi2 = 25.7 on one degree of freedom, that is a number the reader has to take on
trust; plotted against the data it is immediate.  The three fitted widths alone
do not show it either -- the monotonic rise is only visible on a dense grid.

Writes outputs/paper/ni_zone_curve.npz, which plot_results.py overlays on the Ni
panel of the fits figure.  The Wagner PDE solve is a few seconds per call, which
is why this is an artifact rather than an inline computation.

Run:  uv run python scripts/make_ni_curve.py
"""

from __future__ import annotations

import json

import numpy as np

from htw_pdm.baseline_fit import load_data
from htw_pdm.paths import OUTPUTS
from htw_pdm.physics import Parameters
from htw_pdm.tier2_inverse import (
    A72_MEAN,
    A72_SIG,
    NI_M,
    RISE,
    T_OBS,
    residuals,
    wagner_solve,
)

OUT = OUTPUTS / "paper"
OUT.mkdir(parents=True, exist_ok=True)

theta = np.array([
    json.loads((OUT.parent / "tier2_inverse_fit.json").read_text())[k]
    for k in ("D_Ni_eff_nm2_h", "r_Ni", "phi_ol_supply")
])
p1 = Parameters()

# Dense grid from just after the solver's start time out to the last exposure.
t_dense = np.linspace(1.0, float(T_OBS[-1]), 120)
c, xi = wagner_solve(theta, p1, t_eval=t_dense)


def _width(row):
    """Depth at which the Ni trace falls back through the width threshold.

    The fit reads this off the nearest node, which is fine for three widths but
    renders a dense curve as a staircase in units of the 2 nm node spacing.
    Interpolating the crossing costs nothing and moves each width by well under
    a nanometre, so the plotted curve and the fitted points stay consistent.
    """
    above = np.where(row > NI_M + RISE)[0]
    if above.size == 0:
        return 0.0
    j = above[-1]
    if j + 1 >= len(row):
        return float(xi[j])
    f = (row[j] - (NI_M + RISE)) / (row[j] - row[j + 1])
    return float(xi[j] + f * (xi[j + 1] - xi[j]))


W = np.array([_width(row) for row in c])

means, _ = load_data()
ni = {r[0]: (r[1], r[2], r[3]) for r in means["Ni"]}
W_obs = np.array([ni[t][0] for t in T_OBS])
W_sig = np.array([ni[t][1] for t in T_OBS])  # population SD, as in the T2-E fit
chi2 = float(np.sum(residuals(theta, p1, W_obs, W_sig) ** 2))

np.savez(OUT / "ni_zone_curve.npz", t_h=t_dense, W_nm=W,
         t_obs=T_OBS, W_obs=W_obs, W_sig=W_sig, chi2=chi2,
         a72_mean=A72_MEAN, a72_sig=A72_SIG)
print(f"Saved: {OUT / 'ni_zone_curve.npz'}")
print(f"  predicted W: {W[0]:.1f} nm at {t_dense[0]:.0f} h "
      f"-> {W[-1]:.1f} nm at {t_dense[-1]:.0f} h (monotonic rise)")
print(f"  measured  W: {np.round(W_obs, 1)} +- {np.round(W_sig, 1)} nm "
      f"(flat), chi2 = {chi2:.2f} on 1 dof")
