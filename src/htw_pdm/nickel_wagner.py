"""T3 — shared Wagner moving-frame engine for the Ni-exclusion closure family.

the nickel-closure notes.md the nickel-closure gate: the the spatial inverse fit (`spatial_inverse.py`) uses a constant effective
diffusivity D_Ni_eff, which cannot reproduce the flat/saturating measured zone
widths (chi2/dof = 25.7; predicted widths grow monotonically while the data is
flat). This module generalizes the SAME moving-frame Wagner PDE

    dc/dt = D(t) c'' + v_m(t) c'
    D(t) c'(0) = -v_m(t) (1 - r_Ni) c(0) * frac(t)      (rejection BC)
    c(inf) = c_m,  c(xi, 0) = c_m

to three interchangeable closures for D(t) and frac(t), selected by `closure`:

    "const"  the spatial inverse baseline: D(t) = D0, frac(t) = 1.            (3 parameters)
    "flux"   M-A, defect-flux-coupled mobility: D(t) decays with the the zero-dimensional fit
             growth flux J(t) = dL_bl/dt, D(t) = D0 * (J(t)/J(t_ref))^n.
             n = 0 recovers "const" exactly (regression limit).  (4 parameters:
             + n_flux)
    "trap"   M-B, finite-capacity interfacial trapping: D(t) = D0, but the
             rejection flux is throttled by trap fill fraction theta(t),
             frac(t) = 1 - theta(t), d(theta)/dt = rej_flux(t) / S_cap.
             S_cap -> infinity recovers "const" exactly (regression limit).
             (4 parameters: + S_cap)

Both new closures are physically anchored: M-A reuses the the zero-dimensional fit M4 growth
kinetics already frozen from Phase B3 (no new physics input besides the
coupling exponent); M-B adds a finite trap capacity with no time-dependence
assumed. the zero-dimensional fit/the spatial model kinetic parameters (A_bl, b3, PBR_eff, C_x, L_ol0) stay
frozen at their M4 values throughout — only the Ni-zone parameters are fit.

This module intentionally does NOT modify `spatial_inverse.py` (the committed,
reported the spatial inverse result) — `closure="const"` here is a from-scratch
reimplementation verified to reproduce the spatial inverse bit-for-bit
(`tests/test_nickel.py::test_const_matches_spatial_inverse`), so the the spatial inverse
artifact stays untouched while all three closures share one integrator.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402

NI_M = 10.0  # wt% matrix baseline (the spatial identifiability gate/the composition map/the spatial inverse convention)
PBR_BL, PBR_OL = 2.05, 2.10  # FeCr2O4 / Fe3O4 (Li Table 3)
RISE = 2.0  # wt% rise-off-baseline threshold (Veile width convention)
T_OBS = np.array([72.0, 168.0, 480.0])
T_REF = 72.0  # reference exposure for the flux-coupling closure

# 72 h Ni peak amplitude from the Fig. 4 scans (the spatial inverse Section 3.3); population SD.
A72_MEAN, A72_SIG = 23.7, 7.0

# Moving-frame grid: metal ahead of the interface, xi in [0, XI_MAX] nm.
XI_MAX, N_XI = 300.0, 150

CLOSURES = ("const", "flux", "trap")

PARAMS_BY_CLOSURE = {
    "const": ["D_Ni_eff_nm2_h", "r_Ni", "phi_ol_supply"],
    "flux": ["D_Ni_eff_nm2_h", "r_Ni", "phi_ol_supply", "n_flux"],
    "trap": ["D_Ni_eff_nm2_h", "r_Ni", "phi_ol_supply", "S_cap"],
}
# S_cap upper bound (1e4) and n_flux range (+-3) are wide enough that the
# regression-limit values (S_cap -> large, n_flux = 0) sit comfortably inside.
BOUNDS_BY_CLOSURE = {
    "const": ([1e-2, 0.0, 0.0], [1e2, 0.999, 1.0]),
    "flux": ([1e-2, 0.0, 0.0, -3.0], [1e2, 0.999, 1.0, 3.0]),
    "trap": ([1e-2, 0.0, 0.0, 1e-1], [1e2, 0.999, 1.0, 1e4]),
}
X0_BY_CLOSURE = {
    "const": np.array([2.0, 0.3, 0.3]),
    "flux": np.array([2.0, 0.3, 0.3, 0.0]),
    "trap": np.array([2.0, 0.3, 0.3, 1e4]),
}
# Value of the new parameter that exactly (flux) or asymptotically (trap)
# recovers "const" (used by the regression-limit tests and as a sentinel for
# "closure adds nothing"). flux: n_flux=0 makes D(t)=D0 identically, exact
# for any trajectory. trap: theta(t) = integral(rej)/S_cap only vanishes as
# S_cap -> infinity, and the deviation from "const" is O(1/S_cap); at 1e13 that
# term is ~1e-9 wt%, an order of magnitude below the Radau floor.  The residual
# difference that remains (~8e-8 wt%) is not the trap term at all: the trap
# branch integrates an augmented system (one extra state), so the adaptive
# solver picks different steps and lands a tolerance-sized distance away.  It
# does not shrink with S_cap -- it is bounded by rtol * max|c|, and the
# regression test asserts against that bound rather than against a fixed number.
REGRESSION_LIMIT = {"flux": 0.0, "trap": 1e13}

# Radau tolerances for every wagner_solve in this module.  Exposed so the
# regression tests can size their bar off the integrator instead of hardcoding
# a number that silently goes stale when the base kinetics move.
SOLVE_RTOL = 1e-8
SOLVE_ATOL = 1e-10


def dL_bl_dt(t_h: float, p1: Parameters) -> float:
    """the zero-dimensional fit barrier-layer growth flux dL_bl/dt (frozen M4 kinetics), nm/h."""
    bp = HTWPDMParams(A_bl=p1.A_bl, b3=p1.b3, C_bl=p1.C_bl, PBR_eff=p1.PBR_eff,
                      C_x=p1.C_x, L0=p1.L0, L_ol0=p1.L_ol0)
    L = float(L_bl_closed(bp, np.atleast_1d(t_h))[0])
    return p1.A_bl * np.exp(p1.b3 * L) - p1.C_bl


def v_m_of_t(theta: np.ndarray, p1: Parameters):
    """Interface recession speed into the metal, nm/h (the zero-dimensional fit kinetics + ol supply)."""
    phi = theta[2]

    def v(t_h):
        dbl = dL_bl_dt(t_h, p1)
        dL_ol = p1.PBR_eff * dbl + p1.C_x
        return dbl / PBR_BL + phi * dL_ol / PBR_OL

    return v


def D_of_t_fn(theta: np.ndarray, p1: Parameters, closure: str):
    """Effective Ni diffusivity D(t), nm^2/h, per closure."""
    D0 = theta[0]
    if closure == "const":
        return lambda t: D0
    if closure == "flux":
        n = theta[3]
        j_ref = dL_bl_dt(T_REF, p1)
        return lambda t: D0 * (max(dL_bl_dt(t, p1), 1e-12) / j_ref) ** n
    if closure == "trap":
        return lambda t: D0
    raise ValueError(f"unknown closure {closure!r}")


def wagner_solve(theta: np.ndarray, p1: Parameters, closure: str = "const",
                  t_eval=T_OBS):
    """Exact moving-frame Ni enrichment: method-of-lines + Radau.

    Returns (c(xi, t) at t_eval, xi grid). c in wt%, xi in nm.
    """
    if closure not in CLOSURES:
        raise ValueError(f"unknown closure {closure!r}")
    r = theta[1]
    D_of_t = D_of_t_fn(theta, p1, closure)
    v = v_m_of_t(theta, p1)
    xi = np.linspace(0.0, XI_MAX, N_XI)
    h = xi[1] - xi[0]
    trap = closure == "trap"
    S_cap = theta[3] if trap else None

    def rhs(t, y):
        c = y[:N_XI]
        Dt = D_of_t(t)
        vt = v(t)
        frac = (1.0 - y[N_XI]) if trap else 1.0
        frac = max(frac, 0.0)
        rej = vt * (1.0 - r) * c[0] * frac
        c_ghost = c[1] + 2.0 * h * rej / Dt
        dc = np.empty(N_XI)
        dc[0] = Dt * (c[1] - 2 * c[0] + c_ghost) / h**2 \
            + vt * (c[1] - c_ghost) / (2 * h)
        dc[1:-1] = Dt * (c[2:] - 2 * c[1:-1] + c[:-2]) / h**2 \
            + vt * (c[2:] - c[:-2]) / (2 * h)
        dc[-1] = 0.0  # far field pinned at c_m
        if trap:
            dth = rej / S_cap
            return np.concatenate([dc, [dth]])
        return dc

    y0 = np.full(N_XI, NI_M)
    if trap:
        y0 = np.concatenate([y0, [0.0]])
    sol = solve_ivp(rhs, (1e-3, float(t_eval[-1])), y0, method="Radau",
                    t_eval=t_eval, rtol=SOLVE_RTOL, atol=SOLVE_ATOL)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:N_XI].T, xi  # (len(t_eval), N_XI)


def model(theta: np.ndarray, p1: Parameters, closure: str = "const"):
    """(widths W(t) at T_OBS, amplitude A at 72 h) from the Wagner PDE solve."""
    c, xi = wagner_solve(theta, p1, closure)
    W = np.empty(len(T_OBS))
    for j in range(len(T_OBS)):
        above = c[j] > NI_M + RISE
        W[j] = xi[np.where(above)[0][-1]] if np.any(above) else 0.0
    return W, float(c[0, 0])


def residuals(theta, p1, closure, W_obs, W_sig):
    W, A72 = model(theta, p1, closure)
    return np.concatenate([(W - W_obs) / W_sig, [(A72 - A72_MEAN) / A72_SIG]])
