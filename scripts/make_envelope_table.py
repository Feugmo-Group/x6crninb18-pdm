# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Envelope-factor table for the manuscript's sensitivity-map claims.

Replicates the L_bl(T, [O2]) map computation of plot_sensitivity_maps.py
(same grids, same condition-mapped apparent parameters) and writes the
max/min thickness ratio ("envelope factor") over the BWR operating box per
dG0_R scan value and horizon to outputs/paper/table8_envelope_factors.csv.

Run:  python scripts/make_envelope_table.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402
from htw_pdm.physics_parametric import ConditionScan, apparent_parameters  # noqa: E402

from htw_pdm.paths import ROOT  # noqa: E402
from htw_pdm.paths import PAPER_OUT as OUT  # noqa: E402
OUT.mkdir(parents=True, exist_ok=True)

REF = Parameters()  # M4
DG_SCAN_KJ = [25.0, 50.0, 100.0]
T_GRID = np.linspace(200.0, 285.0, 60)
O2_GRID = np.logspace(np.log10(0.01), np.log10(8.0), 60)
HORIZONS = {"480h": 480.0, "10y": 10 * 8766.0}


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
    factors = {}
    for key, t_h in HORIZONS.items():
        Z = L_bl_map(dg * 1e3, t_h)
        factors[key] = Z.max() / Z.min()
        print(f"  dG0_R = {dg:5.0f} kJ/mol, {key:5s}: "
              f"L_bl in [{Z.min():.1f}, {Z.max():.1f}] nm, "
              f"factor = {factors[key]:.3f}")
    rows.append([f"{dg:.0f}", f"{factors['480h']:.3f}", f"{factors['10y']:.3f}"])

with open(OUT / "table8_envelope_factors.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dG0_kJmol", "factor_480h", "factor_10y"])
    w.writerows(rows)
print(f"Saved: {OUT / 'table8_envelope_factors.csv'}")
