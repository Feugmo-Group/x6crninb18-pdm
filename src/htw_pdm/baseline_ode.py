"""Classical numerical baseline for the HTW_PDM thickness ODEs.

Reduced (apparent-parameter) system derived in ../pdm_eqns.md Section 6:

    dL_bl/dt = A_bl * exp(b3 * L_bl) - C_bl              L_bl(0) = L0
    dL_ol/dt = PBR_eff * dL_bl/dt + C_x                  L_ol(0) = L_ol0

Units: nm for thickness, h for time (conversion helpers from Li's cgs at the bottom).
Closed-form solutions are implemented alongside the stiff Radau integration; they are
exact for constant parameters and serve as the parity target for both this module and
the NSEM solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

CM_PER_S_TO_NM_PER_H = 1e7 * 3600.0  # 1 cm/s = 3.6e10 nm/h
PER_CM_TO_PER_NM = 1e-7
CM_TO_NM = 1e7


@dataclass
class HTWPDMParams:
    """Apparent parameters of the reduced HTW_PDM (nm / h units)."""

    A_bl: float  # nm/h, growth prefactor  Omega*k3^0*exp(a3*V + c3*pH)
    b3: float  # 1/nm, < 0, = -alpha3*chi*gamma*eps_f
    C_bl: float  # nm/h, >= 0, barrier-layer destruction  Omega*k_d
    PBR_eff: float  # -, (PBR-1)*Omega_ol/Omega_bl
    C_x: float  # nm/h, <= 0, outer-layer dissolution  -Omega_ol*k11*C_O^r
    L0: float  # nm, initial barrier-layer thickness
    L_ol0: float = 0.0  # nm, initial outer-layer thickness

    def validate(self) -> None:
        if self.b3 >= 0:
            raise ValueError(f"b3 must be negative (got {self.b3})")
        if self.C_bl < 0:
            raise ValueError(f"C_bl must be >= 0 (got {self.C_bl})")
        if self.L0 <= 0:
            raise ValueError(f"L0 must be > 0 (got {self.L0})")


def rhs(t: float, y: np.ndarray, p: HTWPDMParams) -> np.ndarray:
    growth = p.A_bl * np.exp(p.b3 * y[0]) - p.C_bl
    return np.array([growth, p.PBR_eff * growth + p.C_x])


def jac(t: float, y: np.ndarray, p: HTWPDMParams) -> np.ndarray:
    d = p.A_bl * p.b3 * np.exp(p.b3 * y[0])
    return np.array([[d, 0.0], [p.PBR_eff * d, 0.0]])


def solve_forward(
    p: HTWPDMParams,
    t_eval: np.ndarray,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> np.ndarray:
    """Radau integration; returns array (len(t_eval), 2) of [L_bl, L_ol] in nm."""
    p.validate()
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(
        rhs,
        (0.0, float(t_eval[-1])),
        [p.L0, p.L_ol0],
        method="Radau",
        t_eval=t_eval,
        jac=jac,
        args=(p,),
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Radau failed: {sol.message}")
    return sol.y.T


def L_bl_closed(p: HTWPDMParams, t: np.ndarray) -> np.ndarray:
    """Closed-form barrier-layer thickness (Li Eq. 24/39; C_bl -> 0 limit handled)."""
    t = np.asarray(t, dtype=float)
    if p.C_bl == 0.0:
        # dL/dt = A e^{bL}  =>  L = L0 - (1/b) ln(1 - b A e^{b L0} t)
        return p.L0 - np.log1p(-p.b3 * p.A_bl * np.exp(p.b3 * p.L0) * t) / p.b3
    # ln u computed in log space: u = 1 + (A/C) e^{b3 L0} (e^z - 1), z = -b3 C t >= 0.
    # (e^z - 1) = e^z (1 - e^{-z}), so ln(u - 1) = ln(A/C) + b3 L0 + z + log1p(-e^{-z});
    # stable for arbitrarily large z (long-time limit L -> L_ss).
    z = -p.b3 * p.C_bl * t
    with np.errstate(divide="ignore"):  # log(0) = -inf at t = 0 is correct
        # ln(1 - e^{-z}) via expm1: exact for small z (log1p(-exp(-z)) cancels).
        ln_u_minus_1 = (
            np.log(p.A_bl / p.C_bl) + p.b3 * p.L0 + z + np.log(-np.expm1(-z))
        )
    ln_u = np.logaddexp(0.0, ln_u_minus_1)
    return p.L0 - ln_u / p.b3 - p.C_bl * t


def L_ol_closed(p: HTWPDMParams, t: np.ndarray) -> np.ndarray:
    """Closed-form outer-layer thickness: exact affine integral of the bl solution."""
    t = np.asarray(t, dtype=float)
    return p.PBR_eff * (L_bl_closed(p, t) - p.L0) + p.C_x * t + p.L_ol0


def L_ol_closed_li42(p: HTWPDMParams, t: np.ndarray) -> np.ndarray:
    """Li Eq. (42)-(45) form of the outer-layer solution, for cross-verification.

    Li parameterize dL_ol/dt = A_ol exp(b3 L_bl) - C_ol with
    A_ol = PBR_eff * A_bl and C_ol = PBR_eff * C_bl - C_x.
    """
    A_ol = p.PBR_eff * p.A_bl
    C_ol = p.PBR_eff * p.C_bl - p.C_x
    lbl = L_bl_closed(p, t)
    pref = A_ol / (p.A_bl * p.b3)
    D_ol = -pref * np.log(p.A_bl * np.exp(p.b3 * p.L0) - p.C_bl) + p.L_ol0
    return pref * np.log(p.A_bl * np.exp(p.b3 * lbl) - p.C_bl) - C_ol * t + D_ol


def L_bl_steady_state(p: HTWPDMParams) -> float:
    """L_bl,ss from dL_bl/dt = 0 (Li Eq. 20 in apparent parameters); inf if C_bl = 0."""
    if p.C_bl <= 0.0:
        return np.inf
    return np.log(p.C_bl / p.A_bl) / p.b3


def params_from_li_cgs(
    L0_cm: float,
    b3_per_cm: float,
    A_bl_cm_s: float,
    C_bl_cm_s: float,
    A_ol_cm_s: float,
    C_ol_cm_s: float,
    L_ol0_cm: float = 0.0,
) -> HTWPDMParams:
    """Convert Li 2020 Table 5 apparent parameters (cgs) to nm/h HTWPDMParams."""
    PBR_eff = A_ol_cm_s / A_bl_cm_s
    C_x = (PBR_eff * C_bl_cm_s - C_ol_cm_s) * CM_PER_S_TO_NM_PER_H
    return HTWPDMParams(
        A_bl=A_bl_cm_s * CM_PER_S_TO_NM_PER_H,
        b3=b3_per_cm * PER_CM_TO_PER_NM,
        C_bl=C_bl_cm_s * CM_PER_S_TO_NM_PER_H,
        PBR_eff=PBR_eff,
        C_x=C_x,
        L0=L0_cm * CM_TO_NM,
        L_ol0=L_ol0_cm * CM_TO_NM,
    )


# Li 2020 Table 5 apparent kinetic parameters (cgs units as printed).
LI_TABLE5 = {
    "HCM12A_500C": dict(
        L0_cm=4.98e-5,
        b3_per_cm=-6.08e2,
        A_bl_cm_s=3.15e-10,
        C_bl_cm_s=3.66e-12,
        A_ol_cm_s=3.66e-10,
        C_ol_cm_s=4.25e-12,
    ),
    "316L_500C": dict(
        L0_cm=1.01e-7,
        b3_per_cm=-7.81e2,
        A_bl_cm_s=4.03e-10,
        C_bl_cm_s=2.56e-12,
        A_ol_cm_s=3.28e-10,
        C_ol_cm_s=2.04e-12,
    ),
}


if __name__ == "__main__":
    # Parity demonstration on a generic parameter set plus the Li Table 5 sets.
    demo = HTWPDMParams(A_bl=12.0, b3=-0.03, C_bl=0.05, PBR_eff=1.1, C_x=-0.01, L0=2.0)
    t = np.linspace(0.0, 480.0, 25)
    num = solve_forward(demo, t)
    err_bl = np.max(np.abs(num[:, 0] - L_bl_closed(demo, t)))
    err_ol = np.max(np.abs(num[:, 1] - L_ol_closed(demo, t)))
    err_li42 = np.max(np.abs(L_ol_closed(demo, t) - L_ol_closed_li42(demo, t)))
    print(f"demo: max|Radau-closed| L_bl {err_bl:.3e} nm, L_ol {err_ol:.3e} nm, "
          f"affine-vs-Li42 {err_li42:.3e} nm")
    print(f"demo: L_bl_ss = {L_bl_steady_state(demo):.4f} nm")

    for name, cgs in LI_TABLE5.items():
        p = params_from_li_cgs(**cgs)
        t_h = np.array([100.0, 300.0, 1000.0, 3000.0, 10000.0, 20000.0])
        y = solve_forward(p, t_h)
        yc = np.column_stack([L_bl_closed(p, t_h), L_ol_closed(p, t_h)])
        print(f"\n{name}: max parity error {np.max(np.abs(y - yc)):.3e} nm, "
              f"L_bl_ss = {L_bl_steady_state(p) / 1e3:.2f} um")
        for ti, (lb, lo) in zip(t_h, y):
            print(f"  t = {ti:8.0f} h  L_bl = {lb / 1e3:7.3f} um  L_ol = {lo / 1e3:7.3f} um  "
                  f"total = {(lb + lo) / 1e3:7.3f} um")
