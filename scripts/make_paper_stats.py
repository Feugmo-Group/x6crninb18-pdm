"""Referee-fix statistics for the paper (all numbers computed, none asserted).

1. Weighted power-law refit: per-layer L = k t^n refit under the SAME weighted
   least-squares objective as the PDM fits (sigma of mean = SD/sqrt(n)), so the
   chi2 comparison in Table `tab:ladder` is apples-to-apples.
2. AIC / BIC for the M4 mechanistic model and the (refit) power law; note that
   AICc is undefined at n = 6, k = 5 (n - k - 1 = 0).
3. Absolute goodness of fit: p-value of chi2 = chi2_M4 on dof = 1, and the
   error-scale factor s = sqrt(chi2/dof) whose effect on profile-likelihood
   intervals the Discussion must acknowledge.
4. 1000-member parametric bootstrap with the deterministic fitter (fast),
   propagated through the long-time extrapolation: percentile bands for
   L_bl(t) out to 1e5 years, the 10-year prediction interval, and the
   PDM-vs-power-law divergence ratio interval at 10 years.
5. Discriminability thresholds: the earliest exposure time at which the
   mechanistic and power-law predictions separate by more than two and three
   times the expected measurement uncertainty, for both power-law variants and
   for both the scatter-only and full error budgets -- the eight numbers the
   Predictions section quotes.  Where the power law's own fit uncertainty is
   carried, the separation in sigma is not monotone in t, so each threshold is
   reported with whether it still holds at the end of the scanned window.

Run:  python scripts/make_paper_stats.py
Outputs: outputs/paper/referee_stats.json, outputs/ensemble_members_1000.csv,
         outputs/paper/fig6_band.npz (percentile band for fig 6)
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

from htw_pdm.baseline_fit import (  # noqa: E402
    FitData,
    build_fit_data,
    fit_pdm,
    fit_power_law,
    load_data,
)
from htw_pdm.baseline_ode import L_bl_closed  # noqa: E402
from htw_pdm.paths import OUTPUTS as OUT  # noqa: E402

PAPER = OUT / "paper"
PAPER.mkdir(parents=True, exist_ok=True)

HOURS_PER_YEAR = 8766.0


def weighted_power_law_refit(d: FitData):
    """Per-layer weighted LSQ refit of L = k t^n under the same objective as M4."""
    out = {}
    chi2_total = 0.0
    for layer, L, sig in (("Cr", d.L_cr, d.sig_cr), ("Fe", d.L_fe, d.sig_fe)):
        (k, n), _ = curve_fit(
            lambda t, k, n: k * t**n, d.t, L, p0=(10.0, 0.4), sigma=sig,
            absolute_sigma=True, maxfev=20000,
        )
        chi2 = float(np.sum(((k * d.t**n - L) / sig) ** 2))
        out[layer] = {"k": float(k), "n": float(n), "chi2": chi2}
        chi2_total += chi2
    out["chi2_total"] = chi2_total
    return out


def information_criteria(chi2: float, k: int, n: int):
    aic = chi2 + 2 * k
    bic = chi2 + k * np.log(n)
    aicc = chi2 + 2 * k + (2 * k * (k + 1) / (n - k - 1)) if n - k - 1 > 0 else None
    return {"AIC": aic, "BIC": bic, "AICc": aicc}


def bootstrap_1000(d: FitData, sol4_x, n_members=1000, seed=0):
    """Parametric bootstrap with the deterministic M4 fitter, warm-started at the
    real-data optimum (same resampling model as src/uncertainty_ensemble.py)."""
    rng = np.random.default_rng(seed)
    members = []
    n_fail = 0
    for i in range(n_members):
        d_i = FitData(
            t=d.t,
            L_cr=d.L_cr + rng.normal(0.0, d.sig_cr),
            sig_cr=d.sig_cr,
            L_fe=d.L_fe + rng.normal(0.0, d.sig_fe),
            sig_fe=d.sig_fe,
        )
        try:
            sol, p, chi2, dof, _ = fit_pdm(d_i, 4, x0=sol4_x.copy())
            members.append((p.A_bl, p.b3, p.PBR_eff, p.L0, p.L_ol0))
        except Exception:
            n_fail += 1
    return np.array(members), n_fail


def powerlaw_bootstrap(d: FitData, n_members=1000, seed=1):
    """Power-law counterpart of the PDM parametric bootstrap.

    Resamples the same six weighted layer means under the same noise model and
    refits k, n per layer, so the two fit uncertainties entering the
    discriminability budget are computed the same way.
    """
    rng = np.random.default_rng(seed)
    members = []
    n_fail = 0
    for _ in range(n_members):
        L = d.L_cr + rng.normal(0.0, d.sig_cr)
        try:
            (k, n), _ = curve_fit(
                lambda t, k, n: k * t**n, d.t, L, p0=(10.0, 0.4), sigma=d.sig_cr,
                absolute_sigma=True, maxfev=20000,
            )
            members.append((float(k), float(n)))
        except Exception:
            n_fail += 1
    return np.array(members), n_fail


def first_crossing(t_scan, sep, level):
    """Earliest t at which the separation first exceeds `level` sigma.

    Also reports whether the criterion, once met, still holds at the end of the
    scanned window.  It need not: when the power law's own fit uncertainty is
    carried, its extrapolation variance grows with its prediction, so the
    separation measured in sigma can peak and then fall back.  Where that
    happens, "first crossing" names the start of a window rather than a
    threshold beyond which discrimination is assured, and the two must not be
    reported as the same thing.
    """
    above = sep > level
    if not above.any():
        return None, False
    idx = int(np.argmax(above))
    return float(t_scan[idx]), bool(above[idx:].all())


def main():
    means, scans = load_data()
    d = build_fit_data(means)

    # --- M4 reference fit -----------------------------------------------------
    sol4, p4, chi2_m4, dof_m4, _ = fit_pdm(d, 4)
    n_data = len(d.t) * 2  # 3 Cr + 3 Fe means

    print("=" * 72)
    print("1) Weighted power-law refit (same objective as M4)")
    pl_scan = fit_power_law(scans)  # Veile's own scan-level procedure (stage 1)
    pl_w = weighted_power_law_refit(d)
    for layer in ("Cr", "Fe"):
        print(f"   {layer}: k = {pl_w[layer]['k']:.3f}, n = {pl_w[layer]['n']:.4f}, "
              f"chi2 = {pl_w[layer]['chi2']:.2f} "
              f"(scan-level refit: k = {pl_scan[layer][0]:.3f}, n = {pl_scan[layer][1]:.4f})")
    print(f"   total weighted-refit chi2 = {pl_w['chi2_total']:.2f}  "
          f"(4 free params, dof = {n_data - 4})")

    print("\n2) Information criteria (n = 6 joint means)")
    ic_m4 = information_criteria(chi2_m4, 5, n_data)
    ic_pl = information_criteria(pl_w["chi2_total"], 4, n_data)
    print(f"   M4        : chi2 = {chi2_m4:.2f}, AIC = {ic_m4['AIC']:.1f}, "
          f"BIC = {ic_m4['BIC']:.1f}, AICc = {ic_m4['AICc']}")
    print(f"   power law : chi2 = {pl_w['chi2_total']:.2f}, AIC = {ic_pl['AIC']:.1f}, "
          f"BIC = {ic_pl['BIC']:.1f}, AICc = {ic_pl['AICc']:.1f}" if ic_pl["AICc"]
          else f"   power law : chi2 = {pl_w['chi2_total']:.2f}, AIC = {ic_pl['AIC']:.1f}, "
          f"BIC = {ic_pl['BIC']:.1f}, AICc undefined")

    print("\n3) Absolute goodness of fit (M4)")
    p_gof = float(stats.chi2.sf(chi2_m4, dof_m4))
    s_scale = float(np.sqrt(chi2_m4 / dof_m4))
    print(f"   chi2 = {chi2_m4:.2f} on dof = {dof_m4}: p = {p_gof:.4f}; "
          f"error-scale factor s = sqrt(chi2/dof) = {s_scale:.2f}")

    # --- 1000-member bootstrap --------------------------------------------------
    print("\n4) 1000-member deterministic parametric bootstrap")
    members, n_fail = bootstrap_1000(d, sol4.x)
    print(f"   {len(members)} converged members ({n_fail} failures)")
    hdr = ["A_bl", "b3", "PBR_eff", "L0", "L_ol0"]
    np.savetxt(OUT / "ensemble_members_1000.csv", members, delimiter=",",
               header=",".join(hdr), comments="")
    for j, name in enumerate(hdr):
        arr = members[:, j]
        lo, hi = np.percentile(arr, [16, 84])
        print(f"   {name:8s}: mean = {arr.mean():.5g}, std = {arr.std():.3g}, "
              f"[16,84]% = [{lo:.4g}, {hi:.4g}]")

    # Propagate to long time: percentile band of L_bl(t)
    t_grid = np.logspace(np.log10(1.0), np.log10(1e5 * HOURS_PER_YEAR), 200)
    curves = np.empty((len(members), len(t_grid)))
    from htw_pdm.baseline_fit import unpack  # local import to rebuild params per member

    for i, row in enumerate(members):
        # rebuild HTWPDMParams from raw member values via log-space x vector
        x = np.array([np.log(row[0]), np.log(-row[1]), np.log(row[2]),
                      np.log(row[3]), np.log(row[4])])
        curves[i] = L_bl_closed(unpack(x, 4), t_grid)
    band = {
        "t_h": t_grid,
        "p2p5": np.percentile(curves, 2.5, axis=0),
        "p16": np.percentile(curves, 16, axis=0),
        "p50": np.percentile(curves, 50, axis=0),
        "p84": np.percentile(curves, 84, axis=0),
        "p97p5": np.percentile(curves, 97.5, axis=0),
    }
    np.savez(PAPER / "fig6_band.npz", **band)

    t10 = np.array([10 * HOURS_PER_YEAR])
    L10 = np.array([c for c in curves[:, np.argmin(np.abs(t_grid - t10[0]))]])
    # exact evaluation at 10 y rather than nearest grid point:
    L10 = np.empty(len(members))
    for i, row in enumerate(members):
        x = np.array([np.log(row[0]), np.log(-row[1]), np.log(row[2]),
                      np.log(row[3]), np.log(row[4])])
        L10[i] = L_bl_closed(unpack(x, 4), t10)[0]
    L10_pt = float(L_bl_closed(p4, t10)[0])
    L10_pl = pl_w["Cr"]["k"] * t10[0] ** pl_w["Cr"]["n"]
    L10_pl_scan = pl_scan["Cr"][0] * t10[0] ** pl_scan["Cr"][1]
    ratio = L10_pl / L10
    lo10, med10, hi10 = np.percentile(L10, [16, 50, 84])
    lo10_95, hi10_95 = np.percentile(L10, [2.5, 97.5])
    rlo, rmed, rhi = np.percentile(ratio, [16, 50, 84])
    print(f"\n   L_bl(10 y): point = {L10_pt:.0f} nm; bootstrap median = {med10:.0f} nm, "
          f"[16,84]% = [{lo10:.0f}, {hi10:.0f}] nm, 95% = [{lo10_95:.0f}, {hi10_95:.0f}] nm")
    print(f"   power law at 10 y: weighted refit = {L10_pl:.0f} nm "
          f"(scan-level coeffs: {L10_pl_scan:.0f} nm)")
    print(f"   divergence ratio (PL/PDM) at 10 y: median = {rmed:.2f}, "
          f"[16,84]% = [{rlo:.2f}, {rhi:.2f}]")

    # --- 5) discriminability thresholds ------------------------------------------
    # Eight numbers: {published coefficients, weighted refit} x {scatter only,
    # full error budget} x {2 sigma, 3 sigma}.  The paper quotes all eight, so
    # all eight are computed here rather than only the refit/scatter pair.
    print("\n5) Discriminability thresholds (mechanistic vs power law)")
    # Expected scatter model for a future campaign: the worst observed Cr
    # scan-level relative population SD (22% at 168 h), n = 3 scans ->
    # sigma of the mean = 0.22/sqrt(3) = 12.7% of the measured thickness.
    rel_sd_scans = np.max([r[2] / r[1] for r in sorted(means["Cr"])])
    n_scans_assumed = 3
    rel_sig_mean = rel_sd_scans / np.sqrt(n_scans_assumed)
    t_scan = np.logspace(np.log10(480.0), np.log10(50000.0), 4000)
    L_m4_t = L_bl_closed(p4, t_scan)

    # Mechanistic parametric uncertainty at each t, from the 1000-member bootstrap.
    boot_curves = np.empty((len(members), len(t_scan)))
    for i_m, row in enumerate(members):
        x = np.array([np.log(row[0]), np.log(-row[1]), np.log(row[2]),
                      np.log(row[3]), np.log(row[4])])
        boot_curves[i_m] = L_bl_closed(unpack(x, 4), t_scan)
    var_pdm = boot_curves.std(axis=0) ** 2

    # Power-law fit uncertainty, by the same resampling model.  Available only
    # for the weighted refit: the published coefficients were determined
    # elsewhere, on individual scans in unweighted linear space, and no
    # covariance for them is published.  The published-coefficient budget
    # therefore carries the scatter and mechanistic terms only, which makes it
    # the more conservative (later-crossing) of the two by construction.
    pl_members, pl_fail = powerlaw_bootstrap(d)
    pl_boot = np.array([k * t_scan ** n for k, n in pl_members])
    var_pl_refit = pl_boot.std(axis=0) ** 2
    print(f"   power-law bootstrap: {len(pl_members)} members ({pl_fail} failures)")
    print(f"   scatter model: worst observed Cr scan-level rel. SD = {rel_sd_scans:.1%}, "
          f"n = {n_scans_assumed} scans -> sigma_mean = {rel_sig_mean:.1%} of L")

    variants = {
        "published_coeffs": (pl_scan["Cr"][0] * t_scan ** pl_scan["Cr"][1], None),
        "weighted_refit": (pl_w["Cr"]["k"] * t_scan ** pl_w["Cr"]["n"], var_pl_refit),
    }
    thresholds = {}
    for name, (L_pl_t, var_pl) in variants.items():
        sep_var = (rel_sig_mean * L_m4_t) ** 2
        full_var = sep_var + var_pdm + (0.0 if var_pl is None else var_pl)
        for budget, var in (("scatter_only", sep_var), ("full_budget", full_var)):
            sep = np.abs(L_pl_t - L_m4_t) / np.sqrt(var)
            entry = {}
            for level in (2.0, 3.0):
                t_x, sustained = first_crossing(t_scan, sep, level)
                entry[f"t_{int(level)}sigma_h"] = t_x
                entry[f"sustained_to_window_end_{int(level)}sigma"] = sustained
            # Where the criterion is not sustained, the exposure window over
            # which it does hold is the reportable quantity, not the crossing.
            for level in (2.0, 3.0):
                above = sep > level
                end = None
                if above.any() and not above[int(np.argmax(above)):].all():
                    i0 = int(np.argmax(above))
                    end = float(t_scan[i0 + int(np.argmin(above[i0:])) - 1])
                entry[f"t_{int(level)}sigma_window_end_h"] = end
            entry["separation_max_sigma"] = float(sep.max())
            entry["t_at_separation_max_h"] = float(t_scan[int(np.argmax(sep))])
            entry["separation_at_window_end_sigma"] = float(sep[-1])
            entry["powerlaw_fit_uncertainty_included"] = (
                var_pl is not None and budget == "full_budget")
            thresholds[f"{name}__{budget}"] = entry
            t2, t3 = entry["t_2sigma_h"], entry["t_3sigma_h"]
            note = "" if entry["sustained_to_window_end_3sigma"] else (
                f"  [3 sigma peaks at {entry['separation_max_sigma']:.2f} near "
                f"{entry['t_at_separation_max_h']:.0f} h, then falls back to "
                f"{entry['separation_at_window_end_sigma']:.2f} by 50000 h]")
            print(f"   {name:18s} {budget:13s}: "
                  f"2 sigma at {t2:8.0f} h, 3 sigma at {t3:8.0f} h{note}")

    # --- save everything ---------------------------------------------------------
    results = {
        "power_law_weighted_refit": pl_w,
        "power_law_scan_level": {el: {"k": pl_scan[el][0], "n": pl_scan[el][1]}
                                 for el in ("Cr", "Fe")},
        "m4": {"chi2": chi2_m4, "dof": dof_m4,
               "params": {"A_bl": p4.A_bl, "b3": p4.b3, "PBR_eff": p4.PBR_eff,
                          "L0": p4.L0, "L_ol0": p4.L_ol0}},
        "information_criteria": {"M4": ic_m4, "power_law": ic_pl,
                                 "n_data": n_data},
        "goodness_of_fit": {"p_value": p_gof, "error_scale_s": s_scale},
        "bootstrap": {"n_members": int(len(members)), "n_fail": int(n_fail),
                      "L10y_nm": {"point": L10_pt, "median": float(med10),
                                  "p16": float(lo10), "p84": float(hi10),
                                  "p2p5": float(lo10_95), "p97p5": float(hi10_95)},
                      "powerlaw_L10y_nm": {"weighted_refit": float(L10_pl),
                                           "scan_level": float(L10_pl_scan)},
                      "ratio_PL_over_PDM_10y": {"median": float(rmed),
                                                "p16": float(rlo), "p84": float(rhi)}},
        "discriminability": {
            "rel_sd_scans": float(rel_sd_scans),
            "n_scans_assumed": n_scans_assumed,
            "rel_sigma_mean": float(rel_sig_mean),
            # kept for backward compatibility: the weighted-refit, scatter-only pair
            "t_2sigma_h": thresholds["weighted_refit__scatter_only"]["t_2sigma_h"],
            "t_3sigma_h": thresholds["weighted_refit__scatter_only"]["t_3sigma_h"],
            "n_powerlaw_bootstrap": int(len(pl_members)),
            "thresholds": thresholds,
        },
    }
    with open(PAPER / "referee_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {PAPER / 'referee_stats.json'}, "
          f"{OUT / 'ensemble_members_1000.csv'}, {PAPER / 'fig6_band.npz'}")


if __name__ == "__main__":
    main()
