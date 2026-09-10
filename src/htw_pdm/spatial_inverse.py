"""The spatial inverse — the spatial model inverse restricted to the the spatial identifiability gate-identifiable parameter set.

Fits the Ni-exclusion closure to the REAL Veile Ni observables. Per the the spatial identifiability gate gate,
defect-transport parameters are out of scope (forward-only); the identifiable
targets here are the Ni-zone parameters, which map to interfacial kinetics:

    r_Ni        Ni incorporation ratio k_Ni/k_FeCr (0 = fully excluded)
    D_Ni_eff    effective Ni back-diffusion in the metal ahead of the interface
                (grain-boundary/defect-enhanced; bulk-lattice D at 240 C is
                orders smaller — hence "effective", nm^2/h)
    phi_ol      effective outer-layer coverage: the fraction of the ol cation
                content drawn from the matrix through the m/bl interface —
                The term that repairs the Ni mass budget (see below)

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
    v_m(t) = (dL_bl/dt)/PBR_bl + phi_ol*(dL_ol/dt)/PBR_ol   (the zero-dimensional fit kinetics)

Observed features from the solved profile with Veile's own conventions: width W(t)
where c > c_m + 2 wt%, amplitude A(t) = c(0, t).

Weighting: scan-level POPULATION SDs (3.4/11.1/5.0 nm; 7 wt%), not SD/sqrt(n) —
The model describes a typical scan and the dominant noise is lamella-to-lamella
physical heterogeneity (same reasoning as the the spatial identifiability gate gate).

Observables: Ni zone widths at 72/168/480 h (Fig. 9) and the 72 h peak amplitude
23.7 +- 7 wt% (Fig. 4 scans: 22.4/31.1/17.59). 4 observables, 3 parameters.

**The mass-budget finding this fit quantifies:** barrier-layer growth alone
consumes only ~18 nm of metal by 72 h, supplying at most 1.9 nm of excluded-Ni
equivalent — but the observed zone holds ~3.7 nm. The budget only closes if the
outer-layer Fe is also drawn from the matrix (phi_ol > 0), i.e. the Ni zone is
quantitative evidence that the Fe-rich outer crystals grow from base-metal cations
ejected through the barrier layer, not from re-precipitation alone.

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

from htw_pdm.baseline_fit import load_data  # noqa: E402
from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed  # noqa: E402
from htw_pdm.paths import OUTPUTS as OUT  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402

OUT.mkdir(exist_ok=True)

NI_M = 10.0  # wt% matrix baseline (the spatial identifiability gate/the composition map convention)
PBR_BL, PBR_OL = 2.05, 2.10  # FeCr2O4 / Fe3O4 (Li Table 3)
RISE = 2.0  # wt% rise-off-baseline threshold (Veile width convention)
T_OBS = np.array([72.0, 168.0, 480.0])
# 72 h Ni peak amplitudes from the Fig. 4 scans (extraction Section 3.3);
# population SD (heterogeneity weighting, see module docstring).
A72_MEAN, A72_SIG = 23.7, 7.0

PARAMS = ["D_Ni_eff_nm2_h", "r_Ni", "phi_ol_supply"]
BOUNDS = ([1e-2, 0.0, 0.0], [1e2, 0.999, 1.0])
X0 = np.array([2.0, 0.3, 0.3])

# Moving-frame grid: metal ahead of the interface, xi in [0, XI_MAX] nm.
XI_MAX, N_XI = 300.0, 150


def v_m_of_t(theta, p1: Parameters):
    """Interface recession speed into the metal, nm/h (the zero-dimensional fit kinetics + ol supply)."""
    _, r, phi = theta
    bp = HTWPDMParams(A_bl=p1.A_bl, b3=p1.b3, C_bl=p1.C_bl, PBR_eff=p1.PBR_eff,
                      C_x=p1.C_x, L0=p1.L0, L_ol0=p1.L_ol0)

    def v(t_h):
        L = float(L_bl_closed(bp, np.atleast_1d(t_h))[0])
        dL_bl = p1.A_bl * np.exp(p1.b3 * L) - p1.C_bl
        dL_ol = p1.PBR_eff * dL_bl + p1.C_x
        return dL_bl / PBR_BL + phi * dL_ol / PBR_OL

    return v


def wagner_solve(theta: np.ndarray, p1: Parameters, t_eval=T_OBS):
    """Exact moving-frame Ni enrichment: method-of-lines + Radau.

    Returns (c(xi, t) at t_eval, xi grid). c in wt%, xi in nm.
    """
    D, r, _ = theta
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


def main():
    p1 = Parameters()
    means, _ = load_data()
    ni = {r[0]: (r[1], r[2], r[3]) for r in means["Ni"]}
    W_obs = np.array([ni[t][0] for t in T_OBS])
    W_sig = np.array([ni[t][1] for t in T_OBS])  # POPULATION SD (heterogeneity)
    print(f"Observables: widths {W_obs} +- {np.round(W_sig, 2)} nm (population SD), "
          f"A(72h) = {A72_MEAN} +- {A72_SIG} wt%")

    # Multi-start: the Wagner objective has local minima (the single-start fit
    # can land above its own phi=0 sub-fit); take the best of a coarse init grid.
    best = None
    for D0 in (1.0, 3.0, 8.0):
        for r0 in (0.05, 0.3, 0.6):
            for phi0 in (0.1, 0.5, 0.9):
                s = least_squares(residuals, np.array([D0, r0, phi0]),
                                  bounds=BOUNDS, args=(p1, W_obs, W_sig))
                c = float(np.sum(s.fun**2))
                if best is None or c < best[1]:
                    best = (s, c)
    sol, chi2 = best
    W_fit, A72_fit = model(sol.x, p1)
    print(f"\nFit: chi2/dof = {chi2:.2f}/1")
    for n, v in zip(PARAMS, sol.x):
        print(f"  {n:16s} = {v:.4g}")
    print(f"  widths: {np.round(W_fit, 1)} vs {W_obs}")
    print(f"  A(72h): {A72_fit:.1f} vs {A72_MEAN}")

    # Budget check WITHOUT the ol supply term (the finding).
    theta_no_ol = sol.x.copy()
    theta_no_ol[2] = 0.0
    sol0 = least_squares(residuals, np.array([2.0, 0.0, 0.0]),
                         bounds=([1e-3, 0.0, 0.0], [1e3, 0.999, 1e-9]),
                         args=(p1, W_obs, W_sig))
    chi2_no_ol = float(np.sum(sol0.fun**2))
    print(f"\nBudget exhibit: best fit WITHOUT ol supply (phi = 0): "
          f"chi2 = {chi2_no_ol:.1f} vs {chi2:.2f} with phi free "
          f"(Delta-chi2 = {chi2_no_ol - chi2:.1f})")

    # Profile likelihoods.
    prof = {}
    grids = {
        "D_Ni_eff_nm2_h": np.linspace(0.3, 15.0, 20),
        "r_Ni": np.linspace(0.0, 0.95, 20),
        "phi_ol_supply": np.linspace(0.0, 1.0, 20),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    rows = []
    profile_chi2 = {}
    for i, name in enumerate(PARAMS):
        chi2s = []
        for g in grids[name]:
            free = [j for j in range(3) if j != i]

            def res_fixed(tf, g=g, free=free):
                t = np.empty(3)
                t[free] = tf
                t[i] = g
                return residuals(t, p1, W_obs, W_sig)

            lo = [BOUNDS[0][j] for j in free]
            hi = [BOUNDS[1][j] for j in free]
            s = least_squares(res_fixed, np.clip(sol.x[free], lo, hi),
                              bounds=(lo, hi))
            chi2s.append(float(np.sum(s.fun**2)))
        profile_chi2[name] = np.array(chi2s)
    # Reference every profile to the GLOBAL minimum across the central fit and
    # all profile evaluations (a profile point beating the central fit would
    # otherwise produce fictitious zero-width intervals).
    global_min = min(chi2, *(float(v.min()) for v in profile_chi2.values()))
    for ax, (i, name) in zip(axes, enumerate(PARAMS)):
        dchi = profile_chi2[name] - global_min
        inside = grids[name][dchi <= 1.0]
        if len(inside) == 0:
            inside = grids[name][[int(np.argmin(dchi))]]
        verdict = "identifiable" if dchi.max() > 4.0 else "weak"
        prof[name] = (float(inside.min()), float(inside.max()), verdict)
        rows.append([name, f"{sol.x[i]:.4g}", f"{inside.min():.3g}",
                     f"{inside.max():.3g}", f"{dchi.max():.1f}", verdict])
        print(f"  {name:16s}: 1-sigma [{inside.min():.3g}, {inside.max():.3g}] "
              f"max dchi2 {dchi.max():.1f} -> {verdict}")
        ax.plot(grids[name], dchi, "b-")
        ax.axhline(1.0, color="grey", ls="--", lw=1)
        ax.axvline(sol.x[i], color="k", lw=0.8)
        ax.set_title(name)
        ax.set_ylim(0, 12)
        ax.set_ylabel("$\\Delta\\chi^2$")
    fig.suptitle("the spatial inverse: Ni-exclusion inverse on the real Veile observables",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "plot_spatial_inverse.png", dpi=150)
    print(f"Saved: {OUT / 'plot_spatial_inverse.png'}")

    with open(OUT / "spatial_inverse_fit.json", "w") as f:
        json.dump({n: float(v) for n, v in zip(PARAMS, sol.x)}, f, indent=1)
    with open(OUT / "spatial_inverse_fit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "fit", "sig1_lo", "sig1_hi", "max_dchi2", "verdict"])
        w.writerows(rows)
        w.writerow(["chi2/dof", f"{chi2:.2f}/1", "", "",
                    f"no-ol-supply chi2: {chi2_no_ol:.1f}", ""])
    print(f"Saved: {OUT / 'spatial_inverse_fit.json'}, spatial_inverse_fit.csv")


if __name__ == "__main__":
    main()
