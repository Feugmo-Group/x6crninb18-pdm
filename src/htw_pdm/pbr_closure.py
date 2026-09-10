"""Is the fitted PBR_eff actually in conflict with the stoichiometric bound?

Earlier revisions reported a factor-1.9 discrepancy: the likelihood puts
PBR_eff at 1.096 (model M4) while a chromium mass balance with an FeCr2O4
barrier gives about 2.05.  The discrepancy is not real, for two independent
reasons, and this script computes both.

(1) The bound is not a single number.  `src/mass_balance.py` derives the
    outer-layer production ratio for each chromium-rich phase the reported
    photoelectron spectroscopy admits; it spans roughly [0.5, 3.8], and the
    fitted value sits inside that range.

(2) Even at one fixed phase assignment, M4 and the mass balance describe
    different quantities.  M4 pins C_x = 0, so its PBR_eff measures net
    *accumulation* of outer-layer oxide, whereas the mass balance fixes
    *production*.  Any oxide released to the coolant separates the two.
    Freeing C_x (model M6) and profiling PBR_eff shows the stoichiometric value
    is reached at no cost in chi2 at all.

The conclusion is a negative one about the data, not a positive one about
release: three exposure times cannot separate production from loss, because
given L_bl(t) the outer-layer solution

    L_ol(t) = L_ol0 + PBR_eff * (L_bl(t) - L0) + C_x * t

carries three parameters against exactly three outer-layer observations.

Run:  python -m htw_pdm.pbr_closure
Writes: outputs/paper/pbr_closure.json, outputs/paper/table10_pbr_closure.csv,
        outputs/plot_pbr_closure.png
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import htw_pdm.baseline_fit as bf  # noqa: E402
import htw_pdm.mass_balance as mb  # noqa: E402
from htw_pdm.baseline_fit import build_fit_data, fit_pdm, load_data, unpack  # noqa: E402
from htw_pdm.paths import OUTPUTS as OUT  # noqa: E402
from htw_pdm.paths import PAPER_OUT as PAPER  # noqa: E402
from htw_pdm.paths import ensure_outputs  # noqa: E402

ensure_outputs()

M6 = 6  # L_ol0 + C_x free; see baseline_fit.VARIANTS
I_PBR = 2  # index of ln(PBR_eff) in the parameter vector

# Values singled out in the manuscript, profiled explicitly so the numbers in
# The text come from this file rather than from a reader's interpolation.
CALLOUTS = {
    1.096: "fitted value imposed, C_x free",
    2.053: "FeCr2O4 mass balance, iron-only",
    2.156: "FeCr2O4 mass balance, the spatial model Ni routing",
    2.433: "M1/M2 optimum (no initial precipitate)",
}


def fit_at_fixed_pbr(d, pbr: float):
    """Best M6 fit with PBR_eff held at `pbr`; returns (params, chi2_data)."""
    n = 4 + len(bf.VARIANTS[M6])
    free = [j for j in range(n) if j != I_PBR]
    x0 = np.array([np.log(0.73), np.log(0.0125), np.log(pbr), np.log(1.92),
                   np.log(50.0), np.log(0.5)])

    def res_fixed(xf):
        x = np.empty(n)
        x[free] = xf
        x[I_PBR] = np.log(pbr)
        return bf.residuals(x, d, M6, use_priors=True)

    sol = least_squares(res_fixed, x0[free], method="lm", max_nfev=20000)
    x = np.empty(n)
    x[free] = sol.x
    x[I_PBR] = np.log(pbr)
    chi2 = float(np.sum(bf.residuals(x, d, M6, use_priors=False) ** 2))
    return unpack(x, M6), chi2


def main():
    means, _ = load_data()
    d = build_fit_data(means)

    _, p4, chi2_m4, _, _ = fit_pdm(d, 4)
    _, p6, chi2_m6, _, _ = fit_pdm(d, M6)
    print(f"M4 (C_x = 0):   PBR_eff = {p4.PBR_eff:.4f}  chi2 = {chi2_m4:.4f}")
    print(f"M6 (C_x free):  PBR_eff = {p6.PBR_eff:.4f}  chi2 = {chi2_m6:.4f}  "
          f"C_x = {p6.C_x:.4f} nm/h  L_ol0 = {p6.L_ol0:.2f} nm")

    # --- stoichiometric range over the candidate barrier phases
    stoich = {f"r_Ni={r:.4f}": mb.summary(r_Ni=r) for r in (0.0, 0.2954)}
    lo, hi = stoich["r_Ni=0.0000"]["R_prod_range"]
    lo2, hi2 = stoich["r_Ni=0.2954"]["R_prod_range"]
    rng = (min(lo, lo2), max(hi, hi2))
    print(f"\nStoichiometric range over candidate barrier phases: "
          f"[{rng[0]:.2f}, {rng[1]:.2f}]")
    print(f"  M4's PBR_eff = {p4.PBR_eff:.3f} is "
          f"{'INSIDE' if rng[0] <= p4.PBR_eff <= rng[1] else 'outside'} it.")

    # --- profile PBR_eff in the C_x-free family
    grid = np.unique(np.concatenate([np.geomspace(0.5, 8.0, 61),
                                     np.array(sorted(CALLOUTS))]))
    rows = []
    for pbr in grid:
        p, chi2 = fit_at_fixed_pbr(d, float(pbr))
        rows.append({"PBR_eff": float(pbr), "chi2": chi2,
                     "dchi2_vs_M4": chi2 - chi2_m4, "C_x_nm_per_h": p.C_x,
                     "L_ol0_nm": p.L_ol0, "A_bl": p.A_bl, "b3": p.b3,
                     **mb.release_flux(p.C_x)})

    chi2s = np.array([r["chi2"] for r in rows])
    best = float(chi2s.min())
    # Delta-chi2 = 1 interval around the profile minimum (one parameter).
    inside = grid[chi2s <= best + 1.0]
    print(f"\nProfile over PBR_eff in [{grid[0]:.2f}, {grid[-1]:.2f}]: "
          f"chi2 spans {best:.3f} to {chi2s.max():.3f} "
          f"(total range {chi2s.max() - best:.3f})")
    print(f"  Delta-chi2 = 1 interval: [{inside.min():.2f}, {inside.max():.2f}]")

    print(f"\n{'PBR_eff':>9s} {'chi2':>8s} {'dchi2 vs M4':>12s} {'C_x':>9s} "
          f"{'L_ol0':>8s} {'Fe release':>12s}   note")
    callout_rows = []
    for r in rows:
        note = next((n for v, n in CALLOUTS.items()
                     if abs(r["PBR_eff"] - v) < 1e-9), None)
        if note is None:
            continue
        callout_rows.append({**r, "note": note})
        print(f"{r['PBR_eff']:9.3f} {r['chi2']:8.4f} {r['dchi2_vs_M4']:+12.4f} "
              f"{r['C_x_nm_per_h']:9.4f} {r['L_ol0_nm']:8.2f} "
              f"{r['Fe_ug_dm2_h']:9.2f} ug/dm2/h   {note}")

    # --- outputs
    with open(PAPER / "table10_pbr_closure.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(callout_rows[0]))
        w.writeheader()
        w.writerows(callout_rows)

    payload = {
        "chi2_M4": chi2_m4, "PBR_eff_M4": p4.PBR_eff,
        "chi2_M6": chi2_m6, "PBR_eff_M6": p6.PBR_eff, "C_x_M6": p6.C_x,
        "stoichiometric_range": list(rng),
        "fitted_inside_stoichiometric_range": bool(rng[0] <= p4.PBR_eff <= rng[1]),
        "profile_chi2_span": float(chi2s.max() - best),
        "dchi2_1_interval": [float(inside.min()), float(inside.max())],
        "mass_balance": stoich,
        "callouts": callout_rows,
        "profile": rows,
    }
    with open(PAPER / "pbr_closure.json", "w") as f:
        json.dump(payload, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax[0].plot(grid, chi2s - chi2_m4, "-", color="C0")
    ax[0].axhline(0.0, color="0.6", lw=0.8)
    for v, n in CALLOUTS.items():
        ax[0].axvline(v, ls=":", lw=0.9, color="C3")
    ax[0].axvspan(rng[0], rng[1], color="C2", alpha=0.15,
                  label="stoichiometric range")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\mathrm{PBR}_\mathrm{eff}$ (imposed)")
    ax[0].set_ylabel(r"$\Delta\chi^2$ vs M4")
    ax[0].legend(fontsize=7)

    ax[1].plot(grid, [r["C_x_nm_per_h"] for r in rows], "-", color="C1")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\mathrm{PBR}_\mathrm{eff}$ (imposed)")
    ax[1].set_ylabel(r"$C_x$ (nm/h)")
    fig.tight_layout()
    fig.savefig(OUT / "plot_pbr_closure.png", dpi=150)

    print(f"\nSaved: {PAPER / 'pbr_closure.json'}, "
          f"{PAPER / 'table10_pbr_closure.csv'}, {OUT / 'plot_pbr_closure.png'}")


if __name__ == "__main__":
    main()
