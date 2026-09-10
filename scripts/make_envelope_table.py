# SPDX-FileCopyrightText: Copyright (c) 2026 Conrard Giresse Tetsassi Feugmo
# SPDX-License-Identifier: Apache-2.0

"""Envelope-factor table for the manuscript's sensitivity-map claims.

Replicates the L_bl(T, [O2]) map computation of plot_sensitivity_maps.py
(same grids, same condition-mapped apparent parameters) and writes the
max/min thickness ratio ("envelope factor") over the BWR operating box per
dG0_R scan value and horizon to outputs/paper/table8_envelope_factors.csv.

Run:  python scripts/make_envelope_table.py
"""

import csv
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed  # noqa: E402
from htw_pdm.paths import PAPER_OUT as OUT  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402
from htw_pdm.physics_parametric import ConditionScan, apparent_parameters  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)

REF = Parameters()  # M4
# The manuscript table extends the original three-value scan to the activation
# energies reported for weight-gain kinetics on austenitic steels in
# high-temperature water, so the artifact has to carry all five rows.
DG_SCAN_KJ = [25.0, 50.0, 100.0, 250.0, 417.0]
ALPHA3 = 0.12  # reference transfer coefficient, as in the manuscript table
T_GRID = np.linspace(200.0, 285.0, 60)
O2_GRID = np.logspace(np.log10(0.01), np.log10(8.0), 60)
HORIZONS = {"480h": 480.0, "10y": 10 * 8766.0}
T_CAL = 240.0  # Veile et al. calibration temperature, for the O2-only sweep


def r0(x: float) -> str:
    """Round half away from zero, as the manuscript table does."""
    return str(Decimal(repr(float(x))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def L_bl_map(dG0_R_J: float, t_h: float) -> np.ndarray:
    scan = ConditionScan(dG0_R=dG0_R_J)
    out = np.zeros((len(O2_GRID), len(T_GRID)))
    for i, O2 in enumerate(O2_GRID):
        for j, T_C in enumerate(T_GRID):
            p = apparent_parameters(T_C, O2, REF, scan)
            bp = HTWPDMParams(A_bl=p.A_bl, b3=p.b3, C_bl=p.C_bl,
                              PBR_eff=p.PBR_eff, C_x=p.C_x, L0=p.L0,
                              L_ol0=p.L_ol0)
            out[i, j] = L_bl_closed(bp, np.array([t_h]))[0]
    return out


rows = []
for dg in DG_SCAN_KJ:
    factors, spans = {}, {}
    for key, t_h in HORIZONS.items():
        Z = L_bl_map(dg * 1e3, t_h)
        factors[key] = Z.max() / Z.min()
        spans[key] = (Z.min(), Z.max())
        print(f"  dG0_R = {dg:5.0f} kJ/mol, {key:5s}: "
              f"L_bl in [{Z.min():.1f}, {Z.max():.1f}] nm, "
              f"factor = {factors[key]:.3f}")
    rows.append([f"{dg:.0f}", f"{ALPHA3 * dg:.0f}",
                 r0(spans['480h'][0]), r0(spans['480h'][1]),
                 f"{factors['480h']:.3f}",
                 r0(spans['10y'][0]), r0(spans['10y'][1]),
                 f"{factors['10y']:.3f}"])

# The oxygen channel is an ideal-Nernst shift of the growth prefactor, so its
# contribution is independent of the activation energy.  The manuscript states
# that spread in the body text, so it is computed and reported here too.
o2_rows = []
for key, t_h in HORIZONS.items():
    L = np.array([
        L_bl_closed(
            HTWPDMParams(**{k: getattr(apparent_parameters(T_CAL, o, REF,
                                                           ConditionScan(dG0_R=25e3)), k)
                            for k in ("A_bl", "b3", "C_bl", "PBR_eff",
                                      "C_x", "L0", "L_ol0")}),
            np.array([t_h]))[0]
        for o in O2_GRID])
    pct = 100.0 * (L.max() / L.min() - 1.0)
    o2_rows.append([key, f"{L.min():.2f}", f"{L.max():.2f}", f"{pct:.2f}"])
    print(f"  O2 sweep at {T_CAL:.0f} C, {key:5s}: "
          f"L_bl in [{L.min():.2f}, {L.max():.2f}] nm, spread = {pct:.2f}%")

with open(OUT / "table8_o2_only_spread.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["horizon", "L_bl_min_nm", "L_bl_max_nm", "spread_percent"])
    w.writerows(o2_rows)
print(f"Saved: {OUT / 'table8_o2_only_spread.csv'}")

with open(OUT / "table8_envelope_factors.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dG0_kJmol", "alpha3_dG0_kJmol",
                "L_bl_480h_min_nm", "L_bl_480h_max_nm", "factor_480h",
                "L_bl_10y_min_nm", "L_bl_10y_max_nm", "factor_10y"])
    w.writerows(rows)
print(f"Saved: {OUT / 'table8_envelope_factors.csv'}")
