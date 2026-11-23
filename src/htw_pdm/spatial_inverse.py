"""The spatial inverse — the spatial model inverse restricted to the the spatial identifiability gate-identifiable parameter set.

Fits the Ni-exclusion closure to the REAL Veile Ni observables. Per the the spatial identifiability gate gate,
defect-transport parameters are out of scope (forward-only); the identifiable
targets here are the Ni-zone parameters, which map to interfacial kinetics:

    r_Ni        Ni incorporation ratio k_Ni/k_FeCr (0 = fully excluded)
    D_Ni_eff    effective Ni back-diffusion in the metal ahead of the interface
                (grain-boundary/defect-enhanced; bulk-lattice D at 240 C is
                orders smaller — hence "effective", nm^2/h)

Forward engine — the exact moving-frame Ni transport PDE (Wagner enrichment ahead
of a receding interface), NOT the quasi-steady closure. The quasi-steady width
w = D/v grows monotonically as growth slows and misfits the flat/non-monotonic
measured widths at chi2 ~ 100; the transient PDE handles zone build-up, saturation
and re-consumption inherently. In the interface-attached frame (xi = depth into
metal ahead of the interface):

    dc/dt = D_Ni_eff c'' + v_m(t) c'                (metal streams toward xi = 0)
    D c'(0) = -v_m(t) (1 - r_Ni) c(0)               (rejection flux; steady limit
                                                     gives the classic c(0) = c_m/r)
    c(inf) = c_m,  c(xi, 0) = c_m
    v_m(t) = (dL_bl/dt) / 0.609                     (chromium balance, Methods Eq. 3)

The outer-layer iron comes from the same dissolved metal, so it adds no second
recession term (an earlier version had one, phi_ol, which double-counted).

Observed features from the solved profile with Veile's own conventions: width W(t)
where c > c_m + 2 wt%, amplitude A(t) = c(0, t).

Weighting: scan-level SDs of the Ni widths (7 wt% for the amplitude), not SD/sqrt(n) —
the model describes a typical scan and the dominant noise is lamella-to-lamella
physical heterogeneity (same reasoning as the spatial identifiability gate).

Observables: Ni zone widths at 72/168/480 h (dataset of load_data, t <= 480 h)
and the 72 h peak amplitude 23.7 +- 7 wt% (Fig. 4 scans: 22.4/31.1/17.59).
4 observables, 2 parameters.

Run:  python -m htw_pdm.spatial_inverse   (~3-5 min)
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from htw_pdm.baseline_fit import build_fit_data, fit_pdm, load_data  # noqa: E402
from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed  # noqa: E402
from htw_pdm.paths import OUTPUTS as OUT  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402

OUT.mkdir(exist_ok=True)

NI_M = 10.0  # wt% matrix baseline (the spatial identifiability gate/the composition map convention)
PBR_BL, PBR_OL = 2.05, 2.10  # FeCr2O4 / Fe3O4 (Li Table 3)
# Recession uses the chromium-balance ratio, as in tier3_wagner (one definition).
from htw_pdm.mass_balance import BARRIER_CANDIDATES, metal_recession_ratio  # noqa: E402
L_BL_PER_METAL = metal_recession_ratio(BARRIER_CANDIDATES[0])
RISE = 1.0  # wt% above c_m; the cation-fraction rule of extract_layer_thickness (NI_RISE)
T_OBS = np.array([72.0, 168.0, 480.0])
# 72 h Ni peak amplitudes from the Fig. 4 scans (extraction Section 3.3);
# population SD (heterogeneity weighting, see module docstring).
A72_MEAN, A72_SIG = 23.7, 7.0

PARAMS = ["D_Ni_eff_nm2_h", "r_Ni"]
BOUNDS = ([1e-2, 0.0], [1e2, 0.999])
X0 = np.array([2.0, 0.3])

# Moving-frame grid: metal ahead of the interface, xi in [0, XI_MAX] nm.
XI_MAX, N_XI = 300.0, 150


def v_m_of_t(theta, p1: Parameters):
    """Interface recession speed into the metal, nm/h (chromium balance)."""
    bp = HTWPDMParams(A_bl=p1.A_bl, b3=p1.b3, C_bl=p1.C_bl, PBR_eff=p1.PBR_eff,
                      C_x=p1.C_x, L0=p1.L0, L_ol0=p1.L_ol0)

    def v(t_h):
        L = float(L_bl_closed(bp, np.atleast_1d(t_h))[0])
        dL_bl = p1.A_bl * np.exp(p1.b3 * L) - p1.C_bl
        return dL_bl / L_BL_PER_METAL

    return v


def wagner_solve(theta: np.ndarray, p1: Parameters, t_eval=T_OBS):
    """Exact moving-frame Ni enrichment: method-of-lines + Radau.

    Returns (c(xi, t) at t_eval, xi grid). c in wt%, xi in nm.
    """
    D, r = theta[0], theta[1]  # a third (legacy phi) entry is ignored
    v = v_m_of_t(theta, p1)
    xi = np.linspace(0.0, XI_MAX, N_XI)
    h = xi[1] - xi[0]

    def rhs(t, c):
        vt = v(t)
        dc = np.empty_like(c)
        # ghost node at xi=-h from the rejection BC: D c'(0) = -vt (1-r) c(0)
        c_ghost = c[1] + 2.0 * h * vt * (1.0 - r) * c[0] / D
        dc[0] = D * (c[1] - 2 * c[0] + c_ghost) / h**2 \
            + vt * (c[1] - c_ghost) / (2 * h)
        dc[1:-1] = D * (c[2:] - 2 * c[1:-1] + c[:-2]) / h**2 \
            + vt * (c[2:] - c[:-2]) / (2 * h)
        dc[-1] = 0.0  # far field pinned at c_m
        return dc

    sol = solve_ivp(rhs, (1e-3, float(t_eval[-1])), np.full(N_XI, NI_M),
                    method="Radau", t_eval=t_eval, rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T, xi  # (len(t_eval), N_XI)


def model(theta: np.ndarray, p1: Parameters):
    """(widths W(t) at T_OBS, amplitude A at 72 h) from the Wagner PDE solve."""
    c, xi = wagner_solve(theta, p1)
    W = np.empty(len(T_OBS))
    for j in range(len(T_OBS)):
        above = c[j] > NI_M + RISE
        W[j] = xi[np.where(above)[0][-1]] if np.any(above) else 0.0
    return W, float(c[0, 0])


def residuals(theta, p1, W_obs, W_sig):
    W, A72 = model(theta, p1)
    return np.concatenate([(W - W_obs) / W_sig, [(A72 - A72_MEAN) / A72_SIG]])


def fit_from(x0, p1, W_obs, W_sig, i_fix=None, value=None):
    """Least squares from x0; optionally with parameter i_fix held at value."""
    if i_fix is None:
        s = least_squares(residuals, x0, bounds=BOUNDS, args=(p1, W_obs, W_sig))
        return s.x, float(np.sum(s.fun**2))
    free = [j for j in range(len(PARAMS)) if j != i_fix]

    def res_fixed(tf):
        t = np.empty(len(PARAMS))
        t[free] = tf
        t[i_fix] = value
        return residuals(t, p1, W_obs, W_sig)

    lo = [BOUNDS[0][j] for j in free]
    hi = [BOUNDS[1][j] for j in free]
    s = least_squares(res_fixed, np.clip(np.asarray(x0)[free], lo, hi), bounds=(lo, hi))
    x = np.empty(len(PARAMS))
    x[free] = s.x
    x[i_fix] = value
    return x, float(np.sum(s.fun**2))


def main():
    means, _ = load_data(t_max=float(T_OBS[-1]))
    # Frozen growth kinetics: the M4 fit to the same dataset.
    q = fit_pdm(build_fit_data(means), 4)[1]
    p1 = Parameters(A_bl=q.A_bl, b3=q.b3, C_bl=q.C_bl, PBR_eff=q.PBR_eff, C_x=q.C_x,
                    L0=q.L0, L_ol0=q.L_ol0)
    ni = {r[0]: (r[1], r[2], r[3]) for r in means["Ni"]}
    W_obs = np.array([ni[t][0] for t in T_OBS])
    W_sig = np.array([ni[t][1] for t in T_OBS])  # scan-level SD (heterogeneity)
    print(f"Observables: widths {W_obs} +- {np.round(W_sig, 2)} nm (scan SD), "
          f"A(72h) = {A72_MEAN} +- {A72_SIG} wt%")

    # Multi-start over a coarse grid of initial values.
    best = None
    for D0 in (0.5, 2.0, 8.0, 30.0):
        for r0 in (0.05, 0.3, 0.6, 0.9):
            x, c = fit_from(np.array([D0, r0]), p1, W_obs, W_sig)
            if best is None or c < best[1]:
                best = (x, c)

    # Profiles, each walked from its grid point nearest the optimum with warm
    # starts.  If a profile point beats the central fit, the central fit is
    # restarted from there and the profiles are redone, so the reported
    # optimum is the minimum over everything evaluated.
    grids = {"D_Ni_eff_nm2_h": np.geomspace(0.3, 100.0, 31),
             "r_Ni": np.linspace(0.0, 0.95, 31)}
    for _ in range(3):
        x_best, chi2 = best
        profile, improved = {}, None
        for i, name in enumerate(PARAMS):
            g = grids[name]
            i0 = int(np.argmin(np.abs(g - x_best[i])))
            vals = np.empty(len(g))
            for idx in (range(i0, len(g)), range(i0 - 1, -1, -1)):
                x0 = x_best.copy()
                for j in idx:
                    x0, vals[j] = fit_from(x0, p1, W_obs, W_sig, i_fix=i, value=g[j])
                    if vals[j] < chi2 - 1e-6 and (improved is None or vals[j] < improved[1]):
                        improved = (x0.copy(), vals[j])
            profile[name] = vals
        if improved is None:
            break
        best = fit_from(improved[0], p1, W_obs, W_sig)
        best = best if best[1] <= improved[1] else improved
    x_best, chi2 = best
    dof = len(T_OBS) + 1 - len(PARAMS)
    W_fit, A72_fit = model(x_best, p1)
    print(f"\nFit: chi2/dof = {chi2:.2f}/{dof}")
    for n, v in zip(PARAMS, x_best):
        print(f"  {n:16s} = {v:.4g}")
    print(f"  widths: {np.round(W_fit, 1)} vs {W_obs}")
    print(f"  A(72h): {A72_fit:.1f} vs {A72_MEAN}")

    fig, axes = plt.subplots(1, len(PARAMS), figsize=(9, 4))
    rows = []
    for ax, (i, name) in zip(axes, enumerate(PARAMS)):
        g = grids[name]
        dchi = profile[name] - chi2
        inside = g[dchi <= 1.0]
        lo = f"{inside.min():.3g}" if len(inside) and inside.min() > g[0] else "edge"
        hi = f"{inside.max():.3g}" if len(inside) and inside.max() < g[-1] else "edge"
        verdict = "identifiable" if dchi.max() > 4.0 else "weak"
        rows.append([name, f"{x_best[i]:.4g}", lo, hi, f"{dchi.max():.1f}", verdict])
        print(f"  {name:16s}: 1-sigma [{lo}, {hi}] max dchi2 {dchi.max():.1f} -> {verdict}")
        ax.plot(g, dchi, "b-")
        ax.axhline(1.0, color="grey", ls="--", lw=1)
        ax.axvline(x_best[i], color="k", lw=0.8)
        ax.set_title(name)
        ax.set_ylim(0, 12)
        ax.set_ylabel("$\\Delta\\chi^2$")
    fig.suptitle("Ni-exclusion inverse (chromium-balance recession)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "plot_spatial_inverse.png", dpi=150)
    print(f"Saved: {OUT / 'plot_spatial_inverse.png'}")

    with open(OUT / "spatial_inverse_fit.json", "w") as f:
        json.dump({**{n: float(v) for n, v in zip(PARAMS, x_best)},
                   "chi2": chi2, "dof": dof, "W_fit": W_fit.tolist(),
                   "W_obs": W_obs.tolist(), "A72_fit": A72_fit}, f, indent=1)
    with open(OUT / "spatial_inverse_fit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "fit", "sig1_lo", "sig1_hi", "max_dchi2", "verdict"])
        w.writerows(rows)
        w.writerow(["chi2/dof", f"{chi2:.2f}/{dof}", "", "", "", ""])
    print(f"Saved: {OUT / 'spatial_inverse_fit.json'}, spatial_inverse_fit.csv")


if __name__ == "__main__":
    main()
