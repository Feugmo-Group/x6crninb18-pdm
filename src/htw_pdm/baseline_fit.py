"""Deterministic fits of the Veile 2024 layer-thickness data.

Stage 1 (validation of the digitized data): least-squares refit of the empirical power
law L = k * t^n, scan-level and linear-space as in the source paper; must recover
Cr (k = 6.521 nm, n = 0.4964) and Fe (k = 64.51 nm, n = 0.2209) within digitization error.

Stage 2 (the theoretical model): weighted least-squares fit of the reduced HTW_PDM
(closed-form solutions from baseline_ode) to the Cr (barrier layer) and Fe (outer layer)
means, in three nested variants:
    M1: C_bl = 0, C_x = 0   (4 free: A_bl, b3, PBR_eff, L0)
    M2: C_bl free           (5 free)
    M3: C_bl, C_x free      (6 free)
    M4: L_ol0 free          (5 free) -- the accepted model
    M5: L_ol0, C_bl free    (6 free)
    M6: L_ol0, C_x free     (6 free) -- diagnostic only, see VARIANTS

Stage 3: profile-likelihood identifiability scan per parameter.

Run:  python -m htw_pdm.baseline_fit
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, least_squares

from htw_pdm.baseline_ode import (  # noqa: E402
    HTWPDMParams,
    L_bl_closed,
    L_bl_steady_state,
    L_ol_closed,
)

from htw_pdm.paths import DATA  # noqa: E402
F_OVER_RT = 96485.332 / (8.314462 * 513.15)  # 1/V at 240 C
CHI = 8.0 / 3.0  # spinel-averaged cation valence in the barrier layer
ALPHA3_PRIOR = 0.12  # Li 2020 Table 4 (HCM12A), used only to convert b3 -> field strength


def load_data():
    means, scans = {}, {}
    with open(DATA / "veile2024_fig9_means.csv") as f:
        rows = [r for r in csv.DictReader(row for row in f if not row.startswith("#"))]
    for r in rows:
        means.setdefault(r["element"], []).append(
            (float(r["t_h"]), float(r["mean_nm"]), float(r["sd_nm"]), int(r["n_scans"]))
        )
    with open(DATA / "veile2024_fig9_scans.csv") as f:
        rows = [r for r in csv.DictReader(row for row in f if not row.startswith("#"))]
    for r in rows:
        scans.setdefault(r["element"], []).append((float(r["t_h"]), float(r["L_nm"])))
    return means, scans


# ----------------------------------------------------------------------------- stage 1
def fit_power_law(scans):
    """Unweighted linear-space LSM of L = k t^n on individual scans (Veile's procedure)."""
    out = {}
    for el in ("Cr", "Fe"):
        t = np.array([s[0] for s in scans[el]])
        L = np.array([s[1] for s in scans[el]])
        (k, n), _ = curve_fit(lambda t, k, n: k * t**n, t, L, p0=(10.0, 0.4))
        out[el] = (k, n)
    return out


# ----------------------------------------------------------------------------- stage 2
@dataclass
class FitData:
    t: np.ndarray
    L_cr: np.ndarray
    sig_cr: np.ndarray
    L_fe: np.ndarray
    sig_fe: np.ndarray


def build_fit_data(means) -> FitData:
    cr = sorted(means["Cr"])
    fe = sorted(means["Fe"])
    t = np.array([r[0] for r in cr])
    # sigma of the mean = population SD / sqrt(n_scans)
    return FitData(
        t=t,
        L_cr=np.array([r[1] for r in cr]),
        sig_cr=np.array([r[2] / np.sqrt(r[3]) for r in cr]),
        L_fe=np.array([r[1] for r in fe]),
        sig_fe=np.array([r[2] / np.sqrt(r[3]) for r in fe]),
    )


# Each variant: ordered list of extra free parameters beyond the M1 core
# x layout: [ln A_bl, ln(-b3), ln PBR_eff, ln L0, <extras in listed order>]
VARIANTS = {
    1: [],                       # M1: C_bl = 0, C_x = 0, L_ol0 = 0
    2: ["C_bl"],                 # M2
    3: ["C_bl", "C_x"],          # M3
    4: ["L_ol0"],                # M4: rapid initial ol precipitation
    5: ["L_ol0", "C_bl"],        # M5
    6: ["L_ol0", "C_x"],         # M6: M4 + outer-layer loss (see below)
}

# M6 exists to answer one question and is not a candidate for acceptance.
#
# A chromium mass balance fixes the ratio at which the barrier layer *produces*
# outer-layer oxide.  M4 pins C_x = 0, so its PBR_eff is forced to describe net
# *accumulation* instead, and the two numbers are then compared as if they meant
# the same thing.  They do not: any oxide lost to the coolant appears as a
# deficit in accumulation while leaving production untouched.  M6 frees C_x so
# the two can be separated -- or, as it turns out, so their inseparability can
# be demonstrated.  Given L_bl(t), the outer-layer solution is
#
#     L_ol(t) = L_ol0 + PBR_eff * (L_bl(t) - L0) + C_x * t
#
# which carries three parameters against exactly three outer-layer observations.
# It is therefore saturated, and PBR_eff is unconstrained in it: see
# htw_pdm.pbr_closure, which profiles PBR_eff from 0.5 to 8 at a total
# chi2 cost of 0.89.  M3 already had C_x free but also freed C_bl, which sends
# the fit to a degenerate ridge and hides this; M6 isolates the effect.


def _exp(v):
    return np.exp(np.clip(v, -300.0, 300.0))


def unpack(x, variant):
    """x -> HTWPDMParams. Positivity/sign by construction (log / negative-log)."""
    extras = dict(zip(VARIANTS[variant], x[4:]))
    return HTWPDMParams(
        A_bl=_exp(x[0]),
        b3=-_exp(x[1]),
        C_bl=_exp(extras["C_bl"]) if "C_bl" in extras else 0.0,
        PBR_eff=_exp(x[2]),
        C_x=-_exp(extras["C_x"]) if "C_x" in extras else 0.0,
        L0=_exp(x[3]),
        L_ol0=_exp(extras["L_ol0"]) if "L_ol0" in extras else 0.0,
    )


# Soft priors (Gaussian in log space) — the honesty knobs for 3-time-point data.
#
# PBR_eff carried a prior centred on 1.05 ("stoichiometric") until this was
# checked against a chromium mass balance and found to be unsupportable: the
# constant-volume closure that yields 1.05 requires the barrier layer to draw
# ~30 at% Cr from an 18.8 at% alloy.  A Cr mass balance on X6CrNiNb18-10 with
# an FeCr2O4 barrier gives PBR_eff = 2.09 when all non-Ni iron reports to the
# outer layer, so 1.05 is not the stoichiometric value and the prior was
# pulling the estimate toward a number with no derivation behind it.  The
# likelihood alone puts PBR_eff at 1.10 for M4, and the apparent gap to 2.09
# turned out not to be one: the stoichiometric ratio is phase-dependent and
# spans [0.52, 3.79], which contains 1.10, and freeing C_x reaches 2.09 at no
# cost in chi2 (see htw_pdm.mass_balance and htw_pdm.pbr_closure).  L0 keeps its
# prior: the exposure-time data give it no interior optimum at all (it runs to
# zero).
PRIORS = [
    # (index into x, mean of log, sigma of log, name)
    (3, np.log(2.0), 0.60, "L0 ~ 2 nm (air-formed film on high-Cr SS, 1-5 nm)"),
]


def residuals(x, d: FitData, variant: int, use_priors: bool = True):
    p = unpack(x, variant)
    r_cr = (L_bl_closed(p, d.t) - d.L_cr) / d.sig_cr
    r_fe = (L_ol_closed(p, d.t) - d.L_fe) / d.sig_fe
    res = [r_cr, r_fe]
    if use_priors:
        res.append(np.array([(x[i] - mu) / sig for i, mu, sig, _ in PRIORS]))
    return np.concatenate(res)


X0_EXTRA = {"C_bl": np.log(1e-3), "C_x": np.log(1e-3), "L_ol0": np.log(50.0)}


def fit_pdm(d: FitData, variant: int, x0=None):
    n_free = 4 + len(VARIANTS[variant])
    if x0 is None:
        x0 = np.array([np.log(3.0), np.log(0.02), np.log(1.05), np.log(2.0)]
                      + [X0_EXTRA[name] for name in VARIANTS[variant]])
    sol = least_squares(residuals, x0[:n_free], args=(d, variant), method="lm",
                        max_nfev=20000)
    # chi^2 over the data residuals only, plus the total objective including priors
    r_data = residuals(sol.x, d, variant, use_priors=False)
    chi2 = float(np.sum(r_data**2))
    total = float(np.sum(residuals(sol.x, d, variant) ** 2))
    dof = len(r_data) - n_free
    return sol, unpack(sol.x, variant), chi2, max(dof, 1), total


def power_law_chi2(d: FitData, k, n, layer: str):
    L = k * d.t**n
    if layer == "Cr":
        return float(np.sum(((L - d.L_cr) / d.sig_cr) ** 2))
    return float(np.sum(((L - d.L_fe) / d.sig_fe) ** 2))


# ----------------------------------------------------------------------------- stage 3
def profile_likelihood(d: FitData, variant: int, x_best, i_param: int, width=1.5, n=61):
    """Fix parameter i on a grid around its optimum, re-optimize the rest."""
    grid = x_best[i_param] + np.linspace(-width, width, n)
    chi2s = []
    n_free = len(x_best)
    for g in grid:
        free_idx = [j for j in range(n_free) if j != i_param]

        def res_fixed(xf):
            x = np.empty(n_free)
            x[free_idx] = xf
            x[i_param] = g
            return residuals(x, d, variant)

        s = least_squares(res_fixed, x_best[free_idx], method="lm", max_nfev=5000)
        x_full = np.empty(n_free)
        x_full[free_idx] = s.x
        x_full[i_param] = g
        r = residuals(x_full, d, variant, use_priors=False)
        chi2s.append(float(np.sum(r**2)))
    return grid, np.array(chi2s)


PARAM_NAMES_CORE = ["A_bl", "-b3", "PBR_eff", "L0"]


def param_names(variant):
    return PARAM_NAMES_CORE + [
        {"C_bl": "C_bl", "C_x": "-C_x", "L_ol0": "L_ol0"}[e] for e in VARIANTS[variant]
    ]


def main():
    means, scans = load_data()
    d = build_fit_data(means)

    print("=" * 72)
    print("Stage 1 — power-law refit (Veile procedure, scan-level, linear LSM)")
    pl = fit_power_law(scans)
    for el, target in (("Cr", (6.521, 0.4964)), ("Fe", (64.51, 0.2209))):
        k, n = pl[el]
        print(f"  {el}: k = {k:8.3f} nm  n = {n:.4f}   (Veile: k = {target[0]}, n = {target[1]})")

    print("\n" + "=" * 72)
    print("Stage 2 — HTW_PDM weighted fits to layer means (sigma = SD/sqrt(n))")
    results = {}
    for variant, label in (
        (1, "M1: core (A_bl, b3, PBR_eff, L0)"),
        (2, "M2: + C_bl"),
        (3, "M3: + C_bl, C_x"),
        (4, "M4: + L_ol0 (early ol precipitation)"),
        (5, "M5: + L_ol0, C_bl"),
    ):
        sol, p, chi2, dof, total = fit_pdm(d, variant)
        results[variant] = (sol, p, chi2, dof)
        eps_f_V_per_cm = -p.b3 / (ALPHA3_PRIOR * CHI * F_OVER_RT) * 1e7  # b3 in 1/nm -> V/cm
        print(f"\n  {label}:  chi2_data/dof = {chi2:.2f}/{dof}   total (with priors) = {total:.2f}")
        print(f"    A_bl = {p.A_bl:.4g} nm/h   b3 = {p.b3:.4g} 1/nm   C_bl = {p.C_bl:.4g} nm/h")
        print(f"    PBR_eff = {p.PBR_eff:.3f}   C_x = {p.C_x:.4g} nm/h   L0 = {p.L0:.3f} nm   "
              f"L_ol0 = {p.L_ol0:.1f} nm")
        print(f"    -> field strength eps_f = {eps_f_V_per_cm:.3g} V/cm "
              f"(alpha3 = {ALPHA3_PRIOR}, chi = 8/3)")
        print(f"    predicted L_bl: {np.round(L_bl_closed(p, d.t), 1)} vs data {d.L_cr}")
        print(f"    predicted L_ol: {np.round(L_ol_closed(p, d.t), 1)} vs data {d.L_fe}")

    chi2_pl = power_law_chi2(d, *pl["Cr"], "Cr") + power_law_chi2(d, *pl["Fe"], "Fe")
    print(f"\n  Power-law chi2 (same weights, 4 free params): {chi2_pl:.2f}")

    # 95% upper bound on the barrier-layer destruction rate: profile ln C_bl upward
    # from the M5 optimum until Delta-chi2 (data only) exceeds 2.71 (one-sided 95%).
    print("\n" + "=" * 72)
    print("Dissolution-rate upper bound (C_bl profiled from the physical M4 family, "
          "one-sided 95%)")
    sol4 = results[4][0]
    i_cbl = 4 + VARIANTS[5].index("C_bl")  # C_bl slot in the variant-5 layout
    n5 = 4 + len(VARIANTS[5])
    grid_cbl = np.linspace(np.log(1e-6), np.log(1.0), 80)
    chi2_prof = []
    x_start = np.concatenate([sol4.x[:4], [sol4.x[4]]])  # [core, L_ol0] from M4
    for ln_cbl in grid_cbl:
        free_idx = [j for j in range(n5) if j != i_cbl]

        def res_fixed(xf):
            x = np.empty(n5)
            x[free_idx] = xf
            x[i_cbl] = ln_cbl
            return residuals(x, d, 5)

        s = least_squares(res_fixed, x_start, method="lm", max_nfev=5000)
        x_full = np.empty(n5)
        x_full[free_idx] = s.x
        x_full[i_cbl] = ln_cbl
        r = residuals(x_full, d, 5, use_priors=False)
        chi2_prof.append(float(np.sum(r**2)))
    chi2_prof = np.array(chi2_prof)
    above = np.where(chi2_prof > chi2_prof.min() + 2.71)[0]
    ub = np.exp(grid_cbl[above[0]]) if len(above) else None
    if ub is not None:
        # k_d = C_bl / Omega_bl with Omega_bl = 15.6 cm3/mol; C_bl nm/h -> cm/s: /3.6e10
        k_d_ub = ub / 3.6e10 / 15.6
        print(f"  C_bl < {ub:.3g} nm/h  (k_d < {k_d_ub:.3g} mol cm^-2 s^-1; "
              f"vs growth prefactor A_bl ~ {results[4][1].A_bl:.2f} nm/h)")
    else:
        print("  no bound found below 1 nm/h (increase scan range)")

    best_variant = 4
    print("\n" + "=" * 72)
    print(f"Stage 3 — profile likelihoods (M{best_variant}), "
          "Delta-chi2 over +-1.5 in log-parameter")
    sol, p, chi2, dof = results[best_variant]
    names = param_names(best_variant)
    for i in range(len(sol.x)):
        grid, chi2s = profile_likelihood(d, best_variant, sol.x, i)
        dchi = chi2s - chi2s.min()
        inside = grid[dchi <= 1.0]
        lo, hi = np.exp(inside.min()), np.exp(inside.max())
        flat = "IDENTIFIABLE" if dchi.max() > 4.0 else "WEAK/UNIDENTIFIABLE"
        print(f"  {names[i]:8s}: 1-sigma [{lo:.4g}, {hi:.4g}]  "
              f"max Delta-chi2 = {dchi.max():7.1f}  -> {flat}")

    # long-time prediction (the npj figure): the two PDM branches vs power law.
    # M4 (C_bl = 0): unbounded direct-logarithmic growth. M5 (C_bl > 0): saturation
    # at L_bl,ss. Indistinguishable on 72-480 h data - a designed-experiment target.
    print("\n" + "=" * 72)
    print("Long-time extrapolation of the barrier layer: PDM branches vs power law")
    _, p4, _, _ = results[4]
    _, p5, _, _ = results[5]
    print(f"  (M5 steady state L_bl,ss = {L_bl_steady_state(p5):.1f} nm)")
    for years in (1, 5, 10, 40):
        t = np.array([years * 8766.0])
        print(f"  {years:3d} y: M4 (log growth) = {L_bl_closed(p4, t)[0]:7.1f} nm   "
              f"M5 (saturating) = {L_bl_closed(p5, t)[0]:7.1f} nm   "
              f"power law = {pl['Cr'][0] * t[0] ** pl['Cr'][1]:7.1f} nm")


if __name__ == "__main__":
    main()
