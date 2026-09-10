# SPDX-FileCopyrightText: Copyright (c) 2026 Conrard Giresse Tetsassi Feugmo
# SPDX-License-Identifier: Apache-2.0

"""Referee-response analyses R1-R4 -> outputs/paper/referee_response.json.

These four analyses answer questions that the original paper's own numbers
raise but do not address:

  R1  Residual and chi2 leverage decomposition of the accepted M4 fit.  Which
      of the six data points actually carries the fit?  (Answer: the Cr layer
      does essentially all of it, and one Cr point dominates the misfit.)

  R2  Prior-leverage test for PBR_eff and L0.  The paper claims the data, not
      The prior, place PBR_eff at 1.05.  A claim of that form is testable:
      widen or remove the prior and see whether the estimate moves.

  R3  Operating-envelope sensitivity over an extended dG0_R scan.  The paper
      scans 25-100 kJ/mol, i.e. effective activation energies of 3-12 kJ/mol
      at alpha3 = 0.12, while quoting literature apparent activation energies
      for inner-layer growth of several tens of kJ/mol.  The scan is extended
      to cover that range so the robustness statement is tested, not assumed.

  R4  Discrimination exposure with the FULL error budget.  The published
      2500 h threshold compares the mechanistic/power-law separation to future
      measurement scatter only; here the mechanistic parametric uncertainty and
      The power-law fit uncertainty are added, turning a lower bound into a
      realistic design number.

Run:  python scripts/make_referee_analyses.py
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

import htw_pdm.baseline_fit as bf  # noqa: E402
from htw_pdm.baseline_fit import (  # noqa: E402
    FitData,
    build_fit_data,
    fit_pdm,
    load_data,
    unpack,
)
from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed, L_ol_closed  # noqa: E402
from htw_pdm.paths import OUTPUTS as OUT  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402
from htw_pdm.physics_parametric import ConditionScan, apparent_parameters  # noqa: E402

PAPER = OUT / "paper"
PAPER.mkdir(parents=True, exist_ok=True)

HOURS_PER_YEAR = 8766.0
ALPHA3 = 0.12  # transfer coefficient used to convert dG0_R -> effective Ea

RULE = "=" * 74


# ─────────────────────────────────────────────────────── R1: leverage decomposition
def r1_leverage(d: FitData, p4) -> dict:
    """Per-point weighted residual and chi2 contribution at the M4 optimum."""
    pred_cr = L_bl_closed(p4, d.t)
    pred_fe = L_ol_closed(p4, d.t)

    rows = []
    for layer, t, obs, sig, pred in (
        *[("Cr", d.t[i], d.L_cr[i], d.sig_cr[i], pred_cr[i]) for i in range(len(d.t))],
        *[("Fe", d.t[i], d.L_fe[i], d.sig_fe[i], pred_fe[i]) for i in range(len(d.t))],
    ):
        r = (pred - obs) / sig
        rows.append({
            "layer": layer, "t_h": float(t), "obs_nm": float(obs),
            "sigma_nm": float(sig), "pred_nm": float(pred),
            "residual_sigma": float(r), "chi2_contrib": float(r * r),
        })

    chi2_total = sum(r["chi2_contrib"] for r in rows)
    per_layer = {
        lay: sum(r["chi2_contrib"] for r in rows if r["layer"] == lay)
        for lay in ("Cr", "Fe")
    }
    worst = max(rows, key=lambda r: r["chi2_contrib"])

    print(RULE)
    print("R1) Residual and chi2 leverage decomposition of the M4 fit")
    print(f"{'layer':>5} {'t (h)':>7} {'obs':>8} {'sigma':>8} {'pred':>8} "
          f"{'resid':>8} {'chi2':>8} {'% of tot':>9}")
    for r in rows:
        print(f"{r['layer']:>5} {r['t_h']:7.0f} {r['obs_nm']:8.1f} "
              f"{r['sigma_nm']:8.2f} {r['pred_nm']:8.1f} "
              f"{r['residual_sigma']:+8.2f} {r['chi2_contrib']:8.2f} "
              f"{100 * r['chi2_contrib'] / chi2_total:8.1f}%")
    print(f"  total chi2 = {chi2_total:.2f}")
    for lay in ("Cr", "Fe"):
        print(f"    {lay} layer: {per_layer[lay]:6.2f} "
              f"({100 * per_layer[lay] / chi2_total:.1f}% of total)")
    print(f"  dominant point: {worst['layer']} at {worst['t_h']:.0f} h, "
          f"{worst['residual_sigma']:+.2f} sigma, "
          f"{100 * worst['chi2_contrib'] / chi2_total:.0f}% of chi2")

    # Sign pattern of the Cr residuals: does the fitted curvature track the data?
    cr_signs = [np.sign(r["residual_sigma"]) for r in rows if r["layer"] == "Cr"]
    alternating = bool(cr_signs[0] > 0 and cr_signs[1] < 0)
    print(f"  Cr residual sign pattern: "
          f"{['+' if s > 0 else '-' for s in cr_signs]}"
          f"{'  (over/under/on -- systematic curvature miss)' if alternating else ''}")

    # Effective information content: the Fe standard errors relative to the means.
    fe_rel = [d.sig_fe[i] / d.L_fe[i] for i in range(len(d.t))]
    cr_rel = [d.sig_cr[i] / d.L_cr[i] for i in range(len(d.t))]
    print(f"  relative standard error of the mean: "
          f"Cr {['%.0f%%' % (100 * v) for v in cr_rel]}, "
          f"Fe {['%.0f%%' % (100 * v) for v in fe_rel]}")

    return {
        "points": rows,
        "chi2_total": float(chi2_total),
        "chi2_by_layer": {k: float(v) for k, v in per_layer.items()},
        "chi2_fraction_by_layer": {k: float(v / chi2_total) for k, v in per_layer.items()},
        "dominant_point": worst,
        "dominant_fraction": float(worst["chi2_contrib"] / chi2_total),
        "cr_residual_signs": [float(s) for s in cr_signs],
        "rel_sigma_mean": {"Cr": [float(v) for v in cr_rel],
                           "Fe": [float(v) for v in fe_rel]},
    }


# ────────────────────────────────────────────────────── R2: prior-leverage test
# The baseline carries only the L0 prior (see baseline_fit.PRIORS).  The PBR
# rows now run the test in the opposite direction: imposing the prior the fit
# used to carry, to show how far it displaced the likelihood optimum.
PRIOR_SCENARIOS = [
    ("baseline",             None, 0.60),
    ("impose PBR 1.05@0.20", 0.20, 0.60),
    ("impose PBR 1.05@1.00", 1.00, 0.60),
    ("L0 prior x5",          None, 3.00),
    ("L0 prior off",         None, None),
    ("both priors off",      0.20, None),
]


def _set_priors(sig_pbr, sig_l0):
    """Rebuild baseline_fit.PRIORS with the given log-space widths (None = drop)."""
    priors = []
    if sig_pbr is not None:
        priors.append((2, np.log(1.05), sig_pbr, "PBR_eff"))
    if sig_l0 is not None:
        priors.append((3, np.log(2.0), sig_l0, "L0"))
    bf.PRIORS = priors


def r2_prior_leverage(d: FitData, x0) -> dict:
    print("\n" + RULE)
    print("R2) Prior-leverage test: which parameters does the DATA determine?")
    print("    (PBR_eff now carries no prior; the test is how far imposing the")
    print("     former 1.05 prior displaces it. L0 is tested by relaxation.)")
    print(f"{'scenario':>18} {'sig_PBR':>8} {'sig_L0':>7} {'PBR_eff':>9} "
          f"{'L0 (nm)':>9} {'A_bl':>8} {'-b3':>9} {'L_ol,0':>8} {'chi2':>7}")

    original = list(bf.PRIORS)
    out = {}
    try:
        for name, s_pbr, s_l0 in PRIOR_SCENARIOS:
            _set_priors(s_pbr, s_l0)
            try:
                sol, p, chi2, dof, _ = fit_pdm(d, 4, x0=np.array(x0, dtype=float))
                rec = {"PBR_eff": p.PBR_eff, "L0": p.L0, "A_bl": p.A_bl,
                       "b3": p.b3, "L_ol0": p.L_ol0, "chi2_data": chi2,
                       "sigma_PBR": s_pbr, "sigma_L0": s_l0, "converged": True}
                print(f"{name:>18} {str(s_pbr):>8} {str(s_l0):>7} "
                      f"{p.PBR_eff:9.4f} {p.L0:9.4f} {p.A_bl:8.4f} "
                      f"{-p.b3:9.5f} {p.L_ol0:8.1f} {chi2:7.2f}")
            except Exception as exc:  # pragma: no cover - reported, not raised
                rec = {"converged": False, "error": str(exc),
                       "sigma_PBR": s_pbr, "sigma_L0": s_l0}
                print(f"{name:>18} {str(s_pbr):>8} {str(s_l0):>7}   "
                      f"FAILED TO CONVERGE ({type(exc).__name__})")
            out[name] = rec
    finally:
        bf.PRIORS = original

    base = out["baseline"]
    verdicts = {}
    for key, target, label in (("impose PBR 1.05@0.20", "PBR_eff", "PBR_eff"),
                               ("L0 prior off", "L0", "L0")):
        rec = out.get(key, {})
        if rec.get("converged"):
            shift = abs(rec[target] - base[target]) / base[target]
            # PBR_eff: the likelihood optimum is the baseline; a large shift on
            # imposing the old prior shows how much that prior was doing.
            # L0: removing its prior leaves the likelihood with no interior
            # optimum, so the estimate collapses -- the signature of a prior-set
            # parameter.
            verdicts[label] = {
                "relative_shift": float(shift),
                "data_determined": bool(shift < 0.05),
            }
            print(f"  {label}: the scenario moves the estimate by "
                  f"{100 * shift:.1f}%  -> "
                  f"{'data-determined' if shift < 0.05 else 'PRIOR-SENSITIVE'}")
        else:
            verdicts[label] = {"data_determined": False,
                               "note": "refit does not converge in this scenario"}
            print(f"  {label}: refit does not converge -> PRIOR-SET")

    return {"scenarios": out, "verdicts": verdicts}


# ─────────────────────────────────────────── R3: extended activation-energy envelope
# dG0_R values in kJ/mol.  The first three are the paper's original scan; the last
# two extend the effective activation energy alpha3*dG0_R into the range reported
# for weight-gain kinetics on austenitic steels in high-temperature water.
DG_SCAN_KJ_EXTENDED = [25.0, 50.0, 100.0, 250.0, 417.0]
T_GRID_R3 = np.linspace(200.0, 285.0, 25)
O2_GRID_R3 = np.logspace(np.log10(0.01), np.log10(8.0), 25)


def _L_bl_extremes(dG0_R_J: float, t_h: float, ref: Parameters):
    scan = ConditionScan(dG0_R=dG0_R_J)
    vals = []
    for O2 in O2_GRID_R3:
        for T_C in T_GRID_R3:
            p = apparent_parameters(float(T_C), float(O2), ref, scan)
            bp = HTWPDMParams(A_bl=p.A_bl, b3=p.b3, C_bl=p.C_bl,
                              PBR_eff=p.PBR_eff, C_x=p.C_x, L0=p.L0,
                              L_ol0=p.L_ol0)
            vals.append(L_bl_closed(bp, np.array([t_h]))[0])
    return float(np.min(vals)), float(np.max(vals))


def r3_envelope(ref: Parameters) -> dict:
    print("\n" + RULE)
    print("R3) Operating-envelope sensitivity over an EXTENDED dG0_R scan")
    print(f"    envelope: T in [200, 285] C at 7 MPa, [O2] in [0.01, 8] ppm; "
          f"alpha3 = {ALPHA3}")
    print(f"{'dG0_R':>8} {'Ea_eff':>8} | {'L(480h) range':>22} {'ratio':>7} | "
          f"{'L(10 y) range':>22} {'ratio':>7}")
    rows = []
    for dg in DG_SCAN_KJ_EXTENDED:
        ea = ALPHA3 * dg
        lo48, hi48 = _L_bl_extremes(dg * 1e3, 480.0, ref)
        lo10, hi10 = _L_bl_extremes(dg * 1e3, 10 * HOURS_PER_YEAR, ref)
        r48, r10 = hi48 / lo48, hi10 / lo10
        rows.append({"dG0_R_kJ_per_mol": dg, "Ea_eff_kJ_per_mol": ea,
                     "L480_min_nm": lo48, "L480_max_nm": hi48, "ratio_480h": r48,
                     "L10y_min_nm": lo10, "L10y_max_nm": hi10, "ratio_10y": r10})
        print(f"{dg:8.0f} {ea:8.1f} | {lo48:9.1f} - {hi48:9.1f} nm {r48:7.2f} | "
              f"{lo10:9.1f} - {hi10:9.1f} nm {r10:7.2f}")

    orig = [r for r in rows if r["dG0_R_kJ_per_mol"] <= 100.0]
    full = rows
    summary = {
        "original_scan": {
            "dG_range_kJ": [25.0, 100.0],
            "Ea_range_kJ": [ALPHA3 * 25.0, ALPHA3 * 100.0],
            "ratio_480h": [min(r["ratio_480h"] for r in orig),
                           max(r["ratio_480h"] for r in orig)],
            "ratio_10y": [min(r["ratio_10y"] for r in orig),
                          max(r["ratio_10y"] for r in orig)],
        },
        "extended_scan": {
            "dG_range_kJ": [25.0, max(DG_SCAN_KJ_EXTENDED)],
            "Ea_range_kJ": [ALPHA3 * 25.0, ALPHA3 * max(DG_SCAN_KJ_EXTENDED)],
            "ratio_480h": [min(r["ratio_480h"] for r in full),
                           max(r["ratio_480h"] for r in full)],
            "ratio_10y": [min(r["ratio_10y"] for r in full),
                          max(r["ratio_10y"] for r in full)],
        },
    }
    o, e = summary["original_scan"], summary["extended_scan"]
    print(f"  original scan (Ea {o['Ea_range_kJ'][0]:.0f}-{o['Ea_range_kJ'][1]:.0f} "
          f"kJ/mol): 10-y envelope factor {o['ratio_10y'][0]:.2f}-{o['ratio_10y'][1]:.2f}")
    print(f"  extended scan (Ea {e['Ea_range_kJ'][0]:.0f}-{e['Ea_range_kJ'][1]:.0f} "
          f"kJ/mol): 10-y envelope factor {e['ratio_10y'][0]:.2f}-{e['ratio_10y'][1]:.2f}")
    return {"rows": rows, "summary": summary}


# ──────────────────────────────── R4: discrimination threshold, full error budget
def r4_discrimination(d: FitData, p4, means) -> dict:
    print("\n" + RULE)
    print("R4) Discrimination exposure: scatter-only vs full error budget")

    # Mechanistic parametric uncertainty from the 1000-member bootstrap ensemble.
    ens_path = OUT / "ensemble_members_1000.csv"
    members = np.loadtxt(ens_path, delimiter=",", skiprows=1)
    print(f"    mechanistic uncertainty: {len(members)} bootstrap members "
          f"({ens_path.name})")

    t_scan = np.logspace(np.log10(480.0), np.log10(200000.0), 1200)
    curves = np.empty((len(members), len(t_scan)))
    for i, row in enumerate(members):
        x = np.array([np.log(row[0]), np.log(-row[1]), np.log(row[2]),
                      np.log(row[3]), np.log(row[4])])
        curves[i] = L_bl_closed(unpack(x, 4), t_scan)
    L_m4 = L_bl_closed(p4, t_scan)
    sd_m4 = curves.std(axis=0)

    # Future-campaign measurement scatter (the paper's own model).
    rel_sd_scans = float(np.max([r[2] / r[1] for r in sorted(means["Cr"])]))
    sig_future = (rel_sd_scans / np.sqrt(3)) * L_m4

    def first_crossing(sep, sigma_model, k_sigma):
        ratio = sep / sigma_model
        idx = np.argmax(ratio > k_sigma)
        return float(t_scan[idx]) if ratio[idx] > k_sigma else None

    # Both power-law branches, so the revised numbers stay parallel to the
    # paper's own two-variant reporting.
    rng = np.random.default_rng(0)
    n_pl = 1000
    res = {}

    # (i) weighted refit under the same objective as M4, with its own bootstrap
    (k_w, n_w), _ = curve_fit(lambda t, k, n: k * t**n, d.t, d.L_cr, p0=(10.0, 0.4),
                              sigma=d.sig_cr, absolute_sigma=True, maxfev=20000)
    pl_curves = np.empty((n_pl, len(t_scan)))
    for i in range(n_pl):
        L_i = d.L_cr + rng.normal(0.0, d.sig_cr)
        try:
            (k, n), _ = curve_fit(lambda t, k, n: k * t**n, d.t, L_i,
                                  p0=(10.0, 0.4), sigma=d.sig_cr,
                                  absolute_sigma=True, maxfev=20000)
        except Exception:
            k, n = k_w, n_w
        pl_curves[i] = k * t_scan**n
    sd_pl_w = pl_curves.std(axis=0)

    # (ii) Veile's published scan-level coefficients: fixed, so no fit uncertainty
    # of their own is propagated -- only the mechanistic band and future scatter.
    from htw_pdm.baseline_fit import fit_power_law  # noqa: E402
    k_p, n_p = fit_power_law(scans_global)["Cr"]

    print(f"    power-law variants: weighted refit (k = {k_w:.3f}, n = {n_w:.4f}) "
          f"with a {n_pl}-member bootstrap; published coeffs "
          f"(k = {k_p:.3f}, n = {n_p:.4f})")

    for variant, L_pl, sd_pl in (("weighted_refit", k_w * t_scan**n_w, sd_pl_w),
                                 ("published_coeffs", k_p * t_scan**n_p,
                                  np.zeros_like(t_scan))):
        sep = np.abs(L_pl - L_m4)
        sig_full = np.sqrt(sig_future**2 + sd_m4**2 + sd_pl**2)
        entry = {}
        for label, sig in (("scatter_only", sig_future), ("full_budget", sig_full)):
            entry[label] = {f"t_{k}sigma_h": first_crossing(sep, sig, k)
                            for k in (2, 3)}
        entry["inflation_factor"] = {
            f"{k}sigma": (entry["full_budget"][f"t_{k}sigma_h"]
                          / entry["scatter_only"][f"t_{k}sigma_h"])
            if (entry["full_budget"][f"t_{k}sigma_h"]
                and entry["scatter_only"][f"t_{k}sigma_h"]) else None
            for k in (2, 3)
        }
        res[variant] = entry
        print(f"    [{variant}]")
        for k in (2, 3):
            a = entry["scatter_only"][f"t_{k}sigma_h"]
            b = entry["full_budget"][f"t_{k}sigma_h"]
            fac = entry["inflation_factor"][f"{k}sigma"]
            print(f"      {k}-sigma: scatter-only {a:,.0f} h"
                  + (f"  ->  full budget {b:,.0f} h  ({fac:.1f}x later)"
                     if b else "  ->  full budget: not reached below 200,000 h"))

    res["rel_sd_scans"] = rel_sd_scans
    res["powerlaw_weighted_Cr"] = {"k": float(k_w), "n": float(n_w)}
    res["powerlaw_published_Cr"] = {"k": float(k_p), "n": float(n_p)}
    return res


# ────────────────────────────────────────────────────────────────────────── main
def main():
    global scans_global
    means, scans = load_data()
    scans_global = scans
    d = build_fit_data(means)
    sol4, p4, chi2_4, dof_4, _ = fit_pdm(d, 4)
    print(f"M4 reference fit: chi2_data = {chi2_4:.4f} on dof = {dof_4}, "
          f"p = {stats.chi2.sf(chi2_4, dof_4):.4f}\n")

    results = {
        "R1_leverage": r1_leverage(d, p4),
        "R2_prior_leverage": r2_prior_leverage(d, sol4.x),
        "R3_extended_envelope": r3_envelope(Parameters()),
        "R4_discrimination": r4_discrimination(d, p4, means),
    }

    path = PAPER / "referee_response.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
