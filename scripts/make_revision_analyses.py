"""Analyses added in the npj revision, one function per reviewer point.

    R2.4  dissolution branch: M5 refit at fixed C_bl over a wide grid, multi-start
    R2.5  profiles over a wide grid, data-only and with the L0 prior
    R2.5/R3  error model: SD/sqrt(n) vs SD, and a scan-resampling bootstrap
    R2 minor  leave-one-out influence of each layer mean (replaces "leverage")
    R2.1  blind test: fit t <= 480 h, predict the later exposure(s)
    R2.7  conditional design: power of one later exposure to separate the branches

Run on the digitized data (first submission) or the raw scans:
    python scripts/make_revision_analyses.py                    # HTW_PDM_DATASET or default
    python scripts/make_revision_analyses.py --dataset veile_raw

Writes outputs/revision/<dataset>.json and outputs/revision/<dataset>_*.png.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit, least_squares  # noqa: E402
from scipy.stats import chi2 as chi2_dist, norm  # noqa: E402

from htw_pdm.baseline_fit import (  # noqa: E402
    DEFAULT_DATASET,
    VARIANTS,
    FitData,
    build_fit_data,
    fit_pdm,
    load_data,
    param_names,
    residuals,
    unpack,
)
from htw_pdm.baseline_ode import L_bl_closed, L_ol_closed  # noqa: E402
from htw_pdm.paths import OUTPUTS  # noqa: E402

OUT = OUTPUTS / "revision"
HOURS_PER_YEAR = 8766.0
T10 = 10 * HOURS_PER_YEAR
I_CBL5 = 4 + VARIANTS[5].index("C_bl")  # ln C_bl slot in the M5 layout
CBL_GRID = np.r_[0.0, np.geomspace(1e-3, 30.0, 22)]  # nm/h


# ----------------------------------------------------------------------------- fitting
def fit_fixed(d: FitData, variant: int, i_fix: int, value: float, x_start,
              use_priors=True, n_starts=12, seed=0):
    """Fix x[i_fix] = value, refit the rest from several starts, keep the best."""
    n = len(x_start) + 1
    free = [j for j in range(n) if j != i_fix]
    rng = np.random.default_rng(seed)

    def res(xf):
        x = np.empty(n)
        x[free] = xf
        x[i_fix] = value
        return residuals(x, d, variant, use_priors=use_priors)

    best = None
    for k in range(n_starts):
        x0 = np.asarray(x_start, float) + (0 if k == 0 else rng.normal(0, 1.0, n - 1))
        try:
            s = least_squares(res, x0, method="lm", max_nfev=20000)
        except Exception:
            continue
        if best is None or s.cost < best.cost:
            best = s
    x = np.empty(n)
    x[free] = best.x
    x[i_fix] = value
    return x


def chi2_data(x, d, variant):
    return float(np.sum(residuals(x, d, variant, use_priors=False) ** 2))


def chi2_total(x, d, variant):
    return float(np.sum(residuals(x, d, variant) ** 2))


def fit_best(d, variant, n_starts=12, seed=0, use_priors=True):
    """fit_pdm from the default start plus random restarts; keep the lowest objective."""
    rng = np.random.default_rng(seed)
    best = fit_pdm(d, variant)[0]
    for _ in range(n_starts - 1):
        x0 = best.x + rng.normal(0, 1.0, len(best.x))
        try:
            s = least_squares(residuals, x0, args=(d, variant, use_priors), method="lm",
                              max_nfev=20000)
        except Exception:
            continue
        if s.cost < best.cost:
            best = s
    return best


def m4_start(d):
    return fit_best(d, 4).x


def pooled_rel_sd(means, el="Cr"):
    """Per-scan SD / mean pooled over exposure times (weights n - 1)."""
    num = sum((n - 1) * (sd / L) ** 2 for _, L, sd, n in means[el] if n > 1)
    den = sum(n - 1 for *_, n in means[el] if n > 1)
    return float(np.sqrt(num / den))


def m4_as_m5(x4, ln_cbl):
    """Embed an M4 vector in the M5 layout [core, L_ol0, C_bl]."""
    return np.r_[x4[:5], ln_cbl]


# ----------------------------------------------------------------------------- R2.4
def dissolution_scan(d: FitData, t_extra=()):
    """M5 refit at each fixed C_bl.  C_bl = 0 is M4 itself."""
    x4 = m4_start(d)
    rows = []
    x_prev = x4[:5]
    for cbl in CBL_GRID:
        if cbl == 0.0:
            x = m4_as_m5(x4, np.log(1e-300))
        else:
            x = fit_fixed(d, 5, I_CBL5, np.log(cbl), x_prev)
            x_prev = np.delete(x, I_CBL5)
        p = unpack(x, 5)
        row = {
            "C_bl": float(cbl),
            "chi2_data": chi2_data(x, d, 5),
            "chi2_total": chi2_total(x, d, 5),
            "L_bl_10y": float(L_bl_closed(p, np.array([T10]))[0]),
        }
        for t in t_extra:
            row[f"L_bl_{int(t)}h"] = float(L_bl_closed(p, np.array([t]))[0])
            row[f"L_ol_{int(t)}h"] = float(L_ol_closed(p, np.array([t]))[0])
        row["x"] = x.tolist()
        rows.append(row)
    c = np.array([r["chi2_total"] for r in rows])
    best = int(np.argmin(c))
    admissible = [r["C_bl"] for r in rows if r["chi2_total"] <= c.min() + 1.0]
    return {"rows": rows, "best_C_bl": rows[best]["C_bl"],
            "best_chi2_total": float(c.min()),
            "C_bl_1sigma_range": [min(admissible), max(admissible)],
            "L_bl_10y_range_1sigma": [
                min(r["L_bl_10y"] for r in rows if r["C_bl"] in admissible),
                max(r["L_bl_10y"] for r in rows if r["C_bl"] in admissible)],
            "bounded_above": bool(rows[-1]["chi2_total"] > c.min() + 1.0)}


# ----------------------------------------------------------------------------- R2.5
def wide_profiles(d: FitData, half_width=6.0):
    """M4 profiles on +-half_width in ln-parameter, with and without the L0 prior.

    'prior' fits and scores with the prior; 'data' fits and scores without it.
    An interval end that touches the grid edge is reported as unbounded (None).
    """
    x4 = m4_start(d)
    names = param_names(4)
    out = {}
    for mode in ("prior", "data"):
        use = mode == "prior"
        if use:
            xb = x4
        else:
            xb = least_squares(lambda x: residuals(x, d, 4, use_priors=False), x4,
                               method="lm", max_nfev=20000).x
        score = (lambda x: chi2_total(x, d, 4)) if use else (lambda x: chi2_data(x, d, 4))
        out[mode] = {}
        for i, name in enumerate(names):
            # Dense near the optimum (tight parameters), coarse out to the edges.
            offs = np.unique(np.r_[np.linspace(-half_width, -0.5, 23),
                                   np.linspace(-0.5, 0.5, 41),
                                   np.linspace(0.5, half_width, 23)])
            grid = xb[i] + offs
            n = len(grid)
            i0 = int(np.argmin(np.abs(offs)))
            prof = np.empty(n)
            # Walk outward from the optimum; each point starts from its neighbour.
            for idx in (range(i0, n), range(i0 - 1, -1, -1)):
                start = np.delete(xb, i)
                for j in idx:
                    x = fit_fixed(d, 4, i, grid[j], start, use_priors=use, n_starts=12)
                    prof[j] = score(x)
                    start = np.delete(x, i)
            dchi = prof - prof.min()
            inside = np.where(dchi <= 1.0)[0]
            a, b = inside[0], inside[-1]

            def cross(j_out, j_in):  # linear interpolation of the dchi2 = 1 crossing
                f = (1.0 - dchi[j_in]) / (dchi[j_out] - dchi[j_in])
                return float(np.exp(grid[j_in] + f * (grid[j_out] - grid[j_in])))

            lo = None if a == 0 else cross(a - 1, a)
            hi = None if b == n - 1 else cross(b + 1, b)
            out[mode][name] = {"best": float(np.exp(xb[i])), "lo_1sigma": lo,
                               "hi_1sigma": hi, "max_dchi2": float(dchi.max()),
                               "grid": np.exp(grid).tolist(), "dchi2": dchi.tolist()}
    return out


# ----------------------------------------------------------------------------- R2.5 / R3
def error_models(means):
    """Refit M4 and the best M5 under two error models for the layer means."""
    out = {}
    rel = {el: pooled_rel_sd(means, el) for el in ("Cr", "Fe")}
    for label in ("SD/sqrt(n)", "SD", "pooled relative SD/sqrt(n)"):
        m = means
        if label == "SD":  # sigma = SD: pretend n = 1 so build_fit_data divides by 1
            m = {el: [(t, L, sd, 1) for (t, L, sd, _) in rows] for el, rows in means.items()}
        elif label.startswith("pooled"):
            m = {el: [(t, L, rel[el] * L, n) for (t, L, _, n) in rows] if el in rel else rows
                 for el, rows in means.items()}
        d = build_fit_data(m)
        sol = fit_best(d, 4)
        p = unpack(sol.x, 4)
        chi2 = float(np.sum(residuals(sol.x, d, 4, use_priors=False) ** 2))
        scan = dissolution_scan(d)
        out[label] = {"M4_chi2": chi2, "M4_L_bl_10y": float(L_bl_closed(p, np.array([T10]))[0]),
                      "M5_best_C_bl": scan["best_C_bl"],
                      "M5_C_bl_1sigma_range": scan["C_bl_1sigma_range"],
                      "L_bl_10y_range_1sigma": scan["L_bl_10y_range_1sigma"]}
    out["pooled_rel_sd"] = rel
    return out


def scan_bootstrap(scans, means, n_members=400, seed=3):
    """Resample whole scans within each exposure time, rebuild means and SD/sqrt(n), refit.

    A scan keeps its Cr and Fe widths together (rows of one time are in scan order
    in the scans file).  Scans from one lamella are spatial subsamples, so this
    describes the spread across the cross-section, not between exposures.  With
    3-4 scans per time there are only 10-35 distinct resamples per time.
    """
    rng = np.random.default_rng(seed)
    d0 = build_fit_data(means)
    x4 = m4_start(d0)
    by = {}
    for el in ("Cr", "Fe"):
        for t, L in scans[el]:
            by.setdefault(t, {}).setdefault(el, []).append(L)
    times = sorted(by)
    L10, L_last, n_fail = [], [], 0
    for _ in range(n_members):
        m = {"Cr": [], "Fe": []}
        for t in times:
            k = len(by[t]["Cr"])
            idx = rng.integers(0, k, size=k)
            for el in m:
                s_ = np.asarray(by[t][el])[idx]
                sd = max(float(np.std(s_, ddof=1)), 0.05 * float(np.mean(s_)), 1.0)  # floor: no zero SD
                m[el].append((t, float(np.mean(s_)), sd, k))
        try:
            d = build_fit_data(m)
            sol, p, *_ = fit_pdm(d, 4, x0=x4.copy())
            if not sol.success:
                raise RuntimeError
            L10.append(float(L_bl_closed(p, np.array([T10]))[0]))
            L_last.append(float(L_bl_closed(p, np.array([max(times)]))[0]))
        except Exception:
            n_fail += 1
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 50)),
                   float(np.percentile(a, 97.5))]
    return {"n": len(L10), "n_fail": n_fail, "M4_L_bl_10y_q": q(L10),
            "M4_L_bl_tmax_q": q(L_last)}


# ----------------------------------------------------------------------------- R2 minor
def influence(d: FitData):
    """Leave-one-out: drop each layer mean, refit M4, record the change in the forecast.

    In M4 the barrier-layer curve does not depend on the Fe data, so dropping an Fe
    point changes the forecast only through optimizer tolerance.
    """
    x4 = m4_start(d)
    p4 = unpack(x4, 4)
    base = float(L_bl_closed(p4, np.array([T10]))[0])
    out = []
    for layer in ("Cr", "Fe"):
        for k, t in enumerate(d.t):
            sig_cr, sig_fe = d.sig_cr.copy(), d.sig_fe.copy()
            (sig_cr if layer == "Cr" else sig_fe)[k] = 1e12  # weight -> 0
            dk = FitData(t=d.t, L_cr=d.L_cr, sig_cr=sig_cr, L_fe=d.L_fe, sig_fe=sig_fe)
            sol, p, *_ = fit_pdm(dk, 4, x0=x4.copy())
            out.append({"layer": layer, "t_h": float(t),
                        "L_bl_10y_without": float(L_bl_closed(p, np.array([T10]))[0]),
                        "change_pct": 100 * (float(L_bl_closed(p, np.array([T10]))[0]) / base - 1)})
    return {"L_bl_10y_all": base, "leave_one_out": out}


# ----------------------------------------------------------------------------- R2.1
def power_law_fit(d: FitData):
    (k, n), cov = curve_fit(lambda t, k, n: k * t**n, d.t, d.L_cr, p0=(10.0, 0.4),
                            sigma=d.sig_cr, absolute_sigma=True, maxfev=20000)
    return float(k), float(n), cov


def m4_prediction_sd(d: FitData, t):
    """Delta-method SD of the M4 barrier-layer prediction at times t (prior included)."""
    sol = fit_best(d, 4)
    J = sol.jac
    cov = np.linalg.pinv(J.T @ J)
    h = 1e-5
    g = np.empty((len(t), len(sol.x)))
    for i in range(len(sol.x)):
        xp = sol.x.copy()
        xp[i] += h
        g[:, i] = (L_bl_closed(unpack(xp, 4), t) - L_bl_closed(unpack(sol.x, 4), t)) / h
    return np.sqrt(np.einsum("ij,jk,ik->i", g, cov, g))


def blind_test(dataset, t_fit_max=480.0):
    means_all, _ = load_data(dataset)
    d_all = build_fit_data(means_all)
    held = [i for i, t in enumerate(d_all.t) if t > t_fit_max]
    if not held:
        return None
    means_fit, _ = load_data(dataset, t_max=t_fit_max)
    d = build_fit_data(means_fit)
    t_h = d_all.t[held]
    scan = dissolution_scan(d, t_extra=t_h)
    rows = scan["rows"]
    c = np.array([r["chi2_total"] for r in rows])
    adm = [r for r in rows if r["chi2_total"] <= c.min() + 1.0]
    k, n, cov = power_law_fit(d)
    m4_sd = m4_prediction_sd(d, t_h)
    rel = pooled_rel_sd(means_fit, "Cr")
    out = {"t_fit_max": t_fit_max, "held_out": [], "pooled_rel_sd_Cr": rel}
    for j, t, sd4 in zip(held, t_h, m4_sd):
        g = np.array([t**n, k * t**n * np.log(t)])
        pl_sd = float(np.sqrt(g @ cov @ g))
        n_t = next(r[3] for r in means_all["Cr"] if r[0] == t)
        out["held_out"].append({
            "t_h": float(t), "n_scans": int(n_t),
            "measured_Cr": float(d_all.L_cr[j]), "sigma_Cr": float(d_all.sig_cr[j]),
            "sigma_Cr_pooled": float(rel * d_all.L_cr[j] / np.sqrt(n_t)),
            "measured_Fe": float(d_all.L_fe[j]), "sigma_Fe": float(d_all.sig_fe[j]),
            "M4_Cr": rows[0][f"L_bl_{int(t)}h"], "M4_Cr_sd": float(sd4),
            "M4_Fe": rows[0][f"L_ol_{int(t)}h"],
            "M5_Cr_range_1sigma": [min(r[f"L_bl_{int(t)}h"] for r in adm),
                                   max(r[f"L_bl_{int(t)}h"] for r in adm)],
            "power_law_Cr": float(k * t**n), "power_law_Cr_sd": pl_sd,
            "power_law_n": float(n),
        })
    # Refit on everything.  A rejected fit gets no Delta-chi2 = 1 intervals.
    sc = dissolution_scan(d_all)
    best = min(sc["rows"], key=lambda r: r["chi2_total"])
    dof = 2 * len(d_all.t) - 6
    p_val = float(chi2_dist.sf(best["chi2_data"], max(dof, 1)))
    out["refit_all"] = {"M4_chi2_data": sc["rows"][0]["chi2_data"],
                        "best_C_bl": sc["best_C_bl"], "best_chi2_data": best["chi2_data"],
                        "dof": dof, "p_value": p_val, "rejected": bool(p_val < 0.05),
                        "L_bl_10y_best": best["L_bl_10y"]}
    return out


# ----------------------------------------------------------------------------- R2.7
def design(d: FitData, scan, means, n_scans=3, alpha=0.05):
    """Power of one exposure at time t to tell M4 from M5 at each C_bl in the 95% set.

    Two-sided z test of H0 'M4 is right' when M5 (at fixed C_bl) is true.  The
    per-scan SD is the pooled relative scatter of the fitted Cr means times the M5
    thickness.  The prediction spread of M4 is not included, so power is an upper
    bound.  The 80% crossing is interpolated on a log-time grid.
    """
    rows = scan["rows"]
    x4 = np.array(rows[0]["x"])
    c = np.array([r["chi2_total"] for r in rows])
    rel_sd = pooled_rel_sd(means, "Cr")
    t = np.geomspace(500, 20000, 400)
    m4 = L_bl_closed(unpack(x4, 5), t)
    z = norm.ppf(1 - alpha / 2)
    alts = []
    for r in rows[1:]:
        if r["chi2_total"] > c.min() + 3.84:
            continue
        m5 = L_bl_closed(unpack(np.array(r["x"]), 5), t)
        delta = np.abs(m4 - m5) / (rel_sd * m5 / np.sqrt(n_scans))
        power = norm.cdf(delta - z) + norm.cdf(-delta - z)
        t80 = None
        k = np.argmax(power >= 0.8)
        if power[k] >= 0.8 and k > 0:
            f = (0.8 - power[k - 1]) / (power[k] - power[k - 1])
            t80 = float(np.exp(np.log(t[k - 1]) + f * np.log(t[k] / t[k - 1])))
        alts.append({"C_bl": r["C_bl"], "dchi2": float(r["chi2_total"] - c.min()),
                     "t_power80": t80, "power_max": float(power.max())})
    return {"rel_sd": rel_sd, "n_scans": n_scans, "alpha": alpha,
            "M4_in_95pct_set": bool(c[0] <= c.min() + 3.84), "alternatives": alts}


# ----------------------------------------------------------------------------- main
def plot_scan(scan, path):
    r = scan["rows"][1:]
    c = np.array([x["C_bl"] for x in r])
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].semilogx(c, [x["chi2_data"] for x in r], "o-", label="data only")
    ax[0].semilogx(c, [x["chi2_total"] for x in r], "s--", label="with L0 prior")
    ax[0].axhline(scan["rows"][0]["chi2_data"], color="k", lw=0.8, label="M4 (C_bl = 0)")
    ax[0].set(xlabel="fixed C_bl (nm/h)", ylabel="chi2")
    ax[0].legend(fontsize=7)
    ax[1].semilogx(c, [x["L_bl_10y"] for x in r], "o-")
    ax[1].axhline(scan["rows"][0]["L_bl_10y"], color="k", lw=0.8)
    ax[1].set(xlabel="fixed C_bl (nm/h)", ylabel="barrier layer at 10 y (nm)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("HTW_PDM_DATASET", DEFAULT_DATASET))
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap members")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    means, scans = load_data(a.dataset, t_max=480.0)
    d = build_fit_data(means)
    res = {"dataset": a.dataset, "times_fit": d.t.tolist()}

    print("R2.4 dissolution scan")
    scan = dissolution_scan(d, t_extra=(2400.0,))
    res["dissolution_scan"] = scan
    plot_scan(scan, OUT / f"{a.dataset}_dissolution_scan.png")
    print(f"  best C_bl = {scan['best_C_bl']:.3g} nm/h, 1-sigma C_bl range "
          f"{scan['C_bl_1sigma_range']}, bounded above: {scan['bounded_above']}")

    print("R2.5 wide profiles")
    res["profiles"] = wide_profiles(d)
    for mode, prof in res["profiles"].items():
        for name, v in prof.items():
            print(f"  [{mode:5s}] {name:8s} best {v['best']:.4g}  1-sigma "
                  f"[{v['lo_1sigma']}, {v['hi_1sigma']}]")

    print("R2.5/R3 error models and scan bootstrap")
    res["error_models"] = error_models(means)
    res["scan_bootstrap"] = scan_bootstrap(scans, means, n_members=100 if a.quick else 400)
    print("  ", json.dumps(res["error_models"]), "\n  ", res["scan_bootstrap"])

    print("R2 minor: leave-one-out influence")
    res["influence"] = influence(d)

    print("R2.1 blind test")
    res["blind_test"] = blind_test(a.dataset)
    print("  ", "no exposure after 480 h in this dataset" if res["blind_test"] is None
          else json.dumps(res["blind_test"]["held_out"]))

    print("R2.7 design")
    res["design"] = design(d, scan, means)
    for a_ in res["design"]["alternatives"]:
        print(f"  C_bl {a_['C_bl']:.3g} (dchi2 {a_['dchi2']:.2f}): 80% power at {a_['t_power80']} h")

    path = OUT / f"{a.dataset}.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
