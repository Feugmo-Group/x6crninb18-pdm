"""T2-A — synthetic-profile identifiability gate for the Tier-2 spatial model.

TIER2_DESIGN.md Section 5 requires this go/no-go study BEFORE any Tier-2
implementation: generate synthetic EDX composition-vs-depth profiles from an assumed
parameter set with the documented instrument model, refit, and profile the likelihood
of each would-be Tier-2 parameter. Parameters that come back flat are not fittable
from the Veile line-scan data and must be dropped from the Tier-2 inverse scope.

Profile-shape parametrization and its mapping to Tier-2 physics (the gate fits the
shape parameters; a flat likelihood for a shape parameter implies a flat likelihood
for any physics parameter that only enters through it):

    L_bl      barrier-layer thickness            (Tier-1 output, known)
    w_int     m/bl transition width              <- solid-state interdiffusion D_s
    w_out     bl/ol transition width             <- outer-interface kinetics
    A_Ni      Ni enrichment peak amplitude       <- k_Ni exclusion ratio
    w_Ni      Ni enrichment zone width           <- k_Ni exclusion ratio + D_Ni
    f_Cr_bl   Cr fraction inside the bl          <- interfacial flux ratios
    g_bl      in-layer composition gradient      <- defect (cation-vacancy) transport
              (absolute % across the layer)         profile: THE parameter the spatial
                                                     PDE machinery would add

The forward model is the analytic steady constant-field profile shape (smooth-step
interfaces + linear in-layer gradient), convolved with the 2 nm EDX Gaussian — for
this gate the analytic solution replaces the Newton BVP (they coincide for the
constant-field limit, which is the designed Tier-2 starting point).

Instrument model (extraction Section 3.3/3.7): sampling every 2 nm, composition
noise 2 % relative per point (scan-to-scan agreement ~0.2 % absolute; 2 % is
conservative), position jitter sigma = 0.945 nm, 3 scans. The +-10 % relative
*systematic* accuracy is not simulated as noise: it floors the absolute-composition
parameters (f_Cr_bl, A_Ni) at +-10 % regardless of statistics — reported as an
analytic note in the verdict, not a fit result.

Heterogeneity model — the decisive ingredient: Veile's own scans show the dominant
uncertainty is PHYSICAL scan-to-scan variability, not instrument noise (Cr layer
30-48 nm across one 72 h lamella; Fig. 9 SDs are 4-70 % of the means). Each synthetic
scan therefore draws its own parameter realization (lognormal, CVs from the observed
Fig. 9 scatter), while the fit — like the future Tier-2 inverse — recovers one shared
parameter set. The resulting model error dominates chi2; intervals are calibrated by
the standard s^2 = chi2/dof scale factor (Delta-chi2 thresholds multiplied by s^2).
Without this ingredient the gate returns absurd 1e5-scale Delta-chi2 for every
parameter — pure-instrument-noise identifiability is a fiction for this data.

Run:  python -m htw_pdm.tier2_identifiability   (~1 min)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.special import erf

from htw_pdm.paths import OUTPUTS as OUT  # noqa: E402
OUT.mkdir(exist_ok=True)

# Matrix baselines (wt.%, Veile Table 1) and geometry (480 h exposure).
FE_M, CR_M, NI_M = 70.0, 18.0, 10.0
EDX_SIGMA_NM = 2.0 / 2.355  # 2 nm FWHM resolution -> Gaussian sigma
POS_JITTER_NM = 0.945
DX_NM = 2.0
N_SCANS = 3

PARAM_NAMES = ["L_bl", "w_int", "w_out", "A_Ni", "w_Ni", "f_Cr_bl", "g_bl"]
TRUTH = dict(L_bl=134.0, w_int=5.0, w_out=8.0, A_Ni=25.0, w_Ni=35.0,
             f_Cr_bl=45.0, g_bl=1.0)
# Scan-to-scan physical heterogeneity (lognormal CV per parameter), anchored to the
# observed Fig. 9 scatter: Cr thickness CV 4 % at 480 h (used for L_bl), Ni zone
# 14 % (A_Ni, w_Ni); interface widths and the defect gradient have no direct
# measurement — assigned the conservative 30 %.
HETERO_CV = dict(L_bl=0.04, w_int=0.30, w_out=0.30, A_Ni=0.15, w_Ni=0.14,
                 f_Cr_bl=0.05, g_bl=0.30)
# Scan bounds for the profile scan (x = 0 at the m/bl interface, matrix at x < 0).
X_LO, X_HI = -120.0, 220.0


def smooth_step(x, x0, w):
    """0 -> 1 transition centred at x0 with width w (erf profile)."""
    return 0.5 * (1.0 + erf((x - x0) / (np.sqrt(2.0) * np.maximum(w, 0.3))))


def profiles(x, p: dict) -> dict:
    """Element fraction traces (%, EDX-normalized metals + O) vs depth."""
    in_bl = smooth_step(x, 0.0, p["w_int"]) - smooth_step(x, p["L_bl"], p["w_out"])
    in_ol = smooth_step(x, p["L_bl"], p["w_out"])
    in_matrix = 1.0 - smooth_step(x, 0.0, p["w_int"])

    # Ni enrichment zone: one-sided peak decaying into the matrix from the interface.
    ni_zone = np.exp(-np.clip(-x, 0.0, None) / np.maximum(p["w_Ni"], 0.5)) * (x <= 0)
    ni_zone = ni_zone + np.exp(-np.clip(x, 0.0, None) / 3.0) * (x > 0)  # sharp bl side

    # In-layer gradient from the defect (cation-vacancy) profile: Cr fraction tilts
    # by g_bl absolute % across the layer (Fe compensates).
    tilt = p["g_bl"] * (x / np.maximum(p["L_bl"], 1.0) - 0.5) * in_bl

    f_cr_ol, f_fe_ol, f_o_bl, f_o_ol = 3.0, 55.0, 42.0, 40.0
    f_ni_bl = 3.0
    f_fe_bl = 100.0 - p["f_Cr_bl"] - f_ni_bl - f_o_bl

    cr = CR_M * in_matrix + (p["f_Cr_bl"] + 0.0) * in_bl + f_cr_ol * in_ol + tilt
    ni = NI_M * in_matrix + f_ni_bl * in_bl + 2.0 * in_ol \
        + (p["A_Ni"] - NI_M) * ni_zone * (1.0 - in_ol)
    fe = FE_M * in_matrix + f_fe_bl * in_bl + f_fe_ol * in_ol - tilt \
        - (p["A_Ni"] - NI_M) * ni_zone * (1.0 - in_ol)  # Ni enriches at Fe's expense
    ox = 0.0 * in_matrix + f_o_bl * in_bl + f_o_ol * in_ol

    out = {}
    for k, v in (("O", ox), ("Cr", cr), ("Fe", fe), ("Ni", ni)):
        out[k] = gaussian_filter1d(v, EDX_SIGMA_NM / (x[1] - x[0]))
    return out


def synth_scans(p: dict, rng: np.random.Generator):
    """N_SCANS noisy scans, each from its OWN physical parameter realization
    (lognormal heterogeneity at HETERO_CV) + 2 % composition noise + position jitter.
    """
    scans = []
    for _ in range(N_SCANS):
        p_i = {k: v * float(np.exp(rng.normal(0.0, HETERO_CV[k])))
               for k, v in p.items()}
        x = np.arange(X_LO, X_HI, DX_NM) + rng.normal(0.0, POS_JITTER_NM)
        f = profiles(x, p_i)
        noisy = {k: v + rng.normal(0.0, np.maximum(0.02 * np.abs(v), 0.2))
                 for k, v in f.items()}
        scans.append((x, noisy))
    return scans


def residuals(theta: np.ndarray, scans) -> np.ndarray:
    p = dict(zip(PARAM_NAMES, theta))
    res = []
    for x, data in scans:
        f = profiles(x, p)
        for k in ("O", "Cr", "Fe", "Ni"):
            sigma = np.maximum(0.02 * np.abs(data[k]), 0.2)
            res.append((f[k] - data[k]) / sigma)
    return np.concatenate(res)


def fit(scans, theta0):
    return least_squares(residuals, theta0, args=(scans,), method="lm",
                         max_nfev=20000)


def profile_scan(scans, theta_best, i_param, lo, hi, n=25):
    """Delta-chi2 profile of parameter i over [lo, hi], others re-optimized."""
    grid = np.linspace(lo, hi, n)
    chi2s = []
    free = [j for j in range(len(theta_best)) if j != i_param]
    for g in grid:
        def res_fixed(tf):
            t = np.empty(len(theta_best))
            t[free] = tf
            t[i_param] = g
            return residuals(t, scans)
        s = least_squares(res_fixed, theta_best[free], method="lm", max_nfev=5000)
        chi2s.append(float(np.sum(s.fun**2)))
    return grid, np.array(chi2s)


# Scan ranges for the profiles (physically motivated prior windows).
SCAN_RANGES = dict(L_bl=(120.0, 150.0), w_int=(1.0, 20.0), w_out=(1.0, 25.0),
                   A_Ni=(12.0, 40.0), w_Ni=(10.0, 80.0), f_Cr_bl=(35.0, 55.0),
                   g_bl=(0.0, 5.0))


N_REPLICATES = 20


def main():
    # The verdict criterion is a REPLICATE study, not a single-draw profile
    # likelihood: with scan-level heterogeneity the effective sample size is
    # N_SCANS (= 3), so a single draw carries CV/sqrt(3)-scale sampling error that
    # a within-draw profile cannot see (the recovered value can sit many profile
    # sigmas from the truth). Repeating the whole synthetic experiment and using
    # the replicate scatter of the recovered parameters is the honest uncertainty.
    rng = np.random.default_rng(0)
    theta_truth = np.array([TRUTH[k] for k in PARAM_NAMES])
    recovered = np.zeros((N_REPLICATES, len(PARAM_NAMES)))
    first_scans = None
    for r in range(N_REPLICATES):
        scans = synth_scans(TRUTH, rng)
        if first_scans is None:
            first_scans = scans
        theta0 = theta_truth * rng.uniform(0.8, 1.25, len(PARAM_NAMES))
        recovered[r] = fit(scans, theta0).x
    n_pts = sum(len(x) * 4 for x, _ in first_scans)
    print(f"{N_REPLICATES} replicates of ({N_SCANS} scans x 4 traces, {n_pts} pts), "
          f"scan-level heterogeneity ON")

    rows, verdicts = [], {}
    for i, name in enumerate(PARAM_NAMES):
        arr = recovered[:, i]
        mean, sd = float(arr.mean()), float(arr.std())
        rel_sd = sd / abs(TRUTH[name])
        bias = mean - TRUTH[name]
        if rel_sd < 0.25:
            verdict = "IDENTIFIABLE"
        elif rel_sd < 0.5:
            verdict = "WEAK"
        else:
            verdict = "UNIDENTIFIABLE"
        verdicts[name] = verdict
        rows.append([name, TRUTH[name], f"{mean:.3g}", f"{sd:.3g}",
                     f"{rel_sd:.0%}", f"{bias:+.3g}", verdict])
        print(f"  {name:8s}: truth {TRUTH[name]:7.3g}  repl mean {mean:7.3g} "
              f"+- {sd:7.3g} (rel {rel_sd:5.0%})  bias {bias:+8.3g}  -> {verdict}")
    print(f"\n  minimum detectable in-layer gradient (2x repl SD of g_bl): "
          f"{2 * recovered[:, PARAM_NAMES.index('g_bl')].std():.2g} abs% "
          f"across the layer")

    # Single-draw profile curves for the figure (visual, calibrated by s^2).
    sol = fit(first_scans, theta_truth)
    chi2_min = float(np.sum(sol.fun**2))
    s2 = chi2_min / (n_pts - len(PARAM_NAMES))
    fig_data = []
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = SCAN_RANGES[name]
        grid, chi2s = profile_scan(first_scans, sol.x, i, lo, hi, n=25)
        fig_data.append((name, grid, (chi2s - chi2s.min()) / s2))

    print("\nSystematic-accuracy note: EDX composition accuracy is +-10 % RELATIVE "
          "(systematic); f_Cr_bl and A_Ni are floored at +-10 % regardless of the "
          "statistical intervals above.")

    with open(OUT / "tier2_identifiability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "truth", "replicate_mean", "replicate_sd",
                    "rel_sd", "bias", "verdict"])
        w.writerows(rows)
    print(f"Saved: {OUT / 'tier2_identifiability.csv'}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    x_demo = np.arange(X_LO, X_HI, 0.5)
    f_demo = profiles(x_demo, TRUTH)
    ax0 = axes[0, 0]
    for k, c in (("O", "tab:red"), ("Cr", "tab:green"), ("Fe", "tab:blue"),
                 ("Ni", "tab:purple")):
        ax0.plot(x_demo, f_demo[k], c=c, label=k)
        xs, data = first_scans[0]
        ax0.plot(xs, data[k], ".", c=c, ms=2, alpha=0.4)
    ax0.set_xlabel("depth from m/bl interface (nm)")
    ax0.set_ylabel("element fraction (%)")
    ax0.set_title("synthetic EDX profiles (truth + 1 noisy scan)")
    ax0.legend(fontsize=7)
    for ax, (name, grid, dchi) in zip(axes.flat[1:], fig_data):
        ax.plot(grid, dchi, "b-")
        ax.axhline(1.0, color="grey", ls="--", lw=1)
        ax.axhline(9.0, color="grey", ls=":", lw=1)
        ax.axvline(TRUTH[name], color="k", ls="-", lw=0.8)
        ax.set_title(name)
        ax.set_ylim(0, 30)
        ax.set_xlabel(name)
        ax.set_ylabel("$\\Delta\\chi^2$")
    fig.suptitle("T2-A identifiability gate: profile likelihoods on synthetic EDX data",
                 fontsize=13)
    plt.tight_layout()
    out = OUT / "plot_tier2_identifiability.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
