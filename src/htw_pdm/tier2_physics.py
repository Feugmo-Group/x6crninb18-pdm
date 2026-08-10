"""Tier-2 spatial HTW_PDM physics — defect transport across the barrier layer.

Milestones T2-B/T2-C of TIER2_DESIGN.md. Two defect fields on the barrier layer,
constant-field closure (the designed starting point; rpdm_nsem lesson):

    oxygen vacancies   c_OV(x,t):  z = +2,   generated at m/bl (R3), annihilated
                                   at bl/ol (R6/7/8) — migrate OUTWARD (+x)
    cation vacancies   c_CV(x,t):  z = -chi, generated at bl/ol (R4), annihilated
                                   at m/bl (R1) — migrate INWARD (-x)

Nernst-Planck flux with Einstein mobility u = D*gamma (gamma = F/RT):

    J = -D dc/dx + z*gamma*E*D * c,     E = eps_f (constant, Tier-1 value)

Nondimensionalization: xi = x/L, chat = c/c0 with c0 = |J|/|v| (migration scale,
v = z*gamma*E*D), Jhat = J/(|v| c0) = sign(v). The steady equation becomes

    d/dxi [ -(1/Pe) dchat/dxi + s*chat ] = 0,   Pe = |z|*gamma*E*L,  s = sign(z)

**Structural identifiability note (corroborates the T2-A verdict):** D cancels from
Pe — the steady nondimensional profile shape depends ONLY on z*gamma*E*L, all of
which Tier 1 already fixes. Steady profile shapes carry NO diffusivity information;
D only sets the absolute concentration scale c0 (invisible to EDX composition at
the % level unless the defect site fraction is large).

Boundary conditions (interfacial-kinetics Robin closure, kappa = k_interface/|v|):

    OV: J fixed by Tier-1 growth (J_OV = (chi/2) k3, k3 = (dL/dt)/Omega_bl);
        annihilation at xi=1: J = k_a c(1)  =>  chat(1) = 1/kappa_a
    CV: |J| set by the R4 generation rate (anchored to J_OV by the assumed ratio
        r_CV); annihilation at xi=0: |J| = k_1 c(0)  =>  chat(0) = 1/kappa_1

Assumed (NOT identified — per the T2-A gate these are forward-only): kappa_a,
kappa_1, r_CV, D_OV, D_CV. Tier-1-anchored (identified): eps_f, chi, L(t), k3(t).

Analytic steady solution (verification target for Newton and NSEM):

    chat(xi) = Jhat/s + (chat_bc - Jhat/s) * exp(s*Pe*(xi - xi_bc))

Transient (T2-C), Landau-transformed to fixed xi in [0,1] with L = L(t) from the
Tier-1 closed form (tau = t/t_c):

    dchat/dtau - xi*(Ldot/L)*dchat/dxi + (t_c/tau_mig(L)) * dJhat/dxi = 0

with tau_mig(L) = L/|v| the migration transit time. Quasi-steadiness holds when
tau_mig << t_c (with the assumed D_OV = 1e-15 cm^2/s: tau_mig ~ 5 h << 480 h).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed
from htw_pdm.physics import CHI, F_OVER_RT, Parameters

# Tier-1 anchors and assumed transport constants (documented above).
EPS_F_V_CM = 1.73e4  # V/cm — Tier-1 recovered field strength
OMEGA_BL_CM3_MOL = 15.6  # cm^3/mol (Li Table 3, spinel)
D_OV_CM2_S = 1e-15  # ASSUMED
D_CV_CM2_S = 1e-16  # ASSUMED
KAPPA_A = 1.25  # ASSUMED: k_annihilation/|v| at bl/ol (OV)
KAPPA_1 = 1.25  # ASSUMED: k_annihilation/|v| at m/bl (CV)
R_CV = 0.5  # ASSUMED: |J_CV|/J_OV generation ratio
NM_TO_CM = 1e-7


@dataclass
class SpeciesGroups:
    """Nondimensional groups of one defect species' steady/transient problem."""

    name: str
    Pe: float  # |z| * gamma * E * L
    s: float  # sign(z) = migration direction along +x
    chat_bc: float  # Robin value at the annihilation boundary
    xi_bc: float  # 0.0 (m/bl) or 1.0 (bl/ol)
    c0_cm3: float  # dimensional migration concentration scale (1/cm^3)
    tau_mig_h: float  # migration transit time L/|v| in hours


def growth_k3(p1: Parameters, L_nm: float) -> float:
    """Tier-1-anchored growth rate constant k3 (mol cm^-2 s^-1) at thickness L."""
    dLdt_nm_h = p1.A_bl * np.exp(p1.b3 * L_nm) - p1.C_bl
    return dLdt_nm_h * NM_TO_CM / 3600.0 / OMEGA_BL_CM3_MOL


def species_groups(p1: Parameters, L_nm: float) -> dict[str, SpeciesGroups]:
    """Both species' nondim groups at barrier-layer thickness L (Tier-1 anchored)."""
    L_cm = L_nm * NM_TO_CM
    N_A = 6.02214076e23
    J_OV = (CHI / 2.0) * growth_k3(p1, L_nm)  # mol cm^-2 s^-1
    out = {}
    for name, z, D, kappa, xi_bc, J in (
        ("OV", +2.0, D_OV_CM2_S, KAPPA_A, 1.0, J_OV),
        ("CV", -CHI, D_CV_CM2_S, KAPPA_1, 0.0, R_CV * J_OV),
    ):
        v = abs(z) * F_OVER_RT * EPS_F_V_CM * D  # |migration velocity|, cm/s
        out[name] = SpeciesGroups(
            name=name,
            Pe=abs(z) * F_OVER_RT * EPS_F_V_CM * L_cm,
            s=np.sign(z),
            chat_bc=1.0 / kappa,
            xi_bc=xi_bc,
            c0_cm3=J / v * N_A,
            tau_mig_h=L_cm / v / 3600.0,
        )
    return out


def analytic_steady(g: SpeciesGroups, xi: np.ndarray) -> np.ndarray:
    """Exact steady chat(xi): migration plateau + interfacial boundary layer.

    Jhat = s everywhere (flux along the migration direction), so the particular
    solution is chat = Jhat/s = 1 for BOTH species; the homogeneous mode
    exp(s*Pe*xi) decays away from the annihilation boundary at xi_bc.
    """
    xi = np.asarray(xi, dtype=float)
    plateau = 1.0
    return plateau + (g.chat_bc - plateau) * np.exp(g.s * g.Pe * (xi - g.xi_bc))


def flux_hat(chat: torch.Tensor, D1: torch.Tensor, g: SpeciesGroups) -> torch.Tensor:
    """Nondim flux Jhat(xi) = -(1/Pe) dchat/dxi + s*chat on the DVR nodes."""
    return -(D1 @ chat) / g.Pe + g.s * chat


def steady_residuals(
    chat: torch.Tensor, D1: torch.Tensor, g: SpeciesGroups
) -> dict[str, torch.Tensor]:
    """Loss terms of the steady BVP: interior flux-divergence + the two BCs.

    The flux is constant (= s) in steady state, so instead of the ill-conditioned
    second-derivative form we penalize (Jhat - s) directly at every node — this is
    the first-integral form of d(Jhat)/dxi = 0 with the flux BC built in.
    """
    J = flux_hat(chat, D1, g)
    i_bc = 0 if g.xi_bc == 0.0 else -1
    return {
        "flux": ((J - g.s) ** 2).mean(),
        "bc": (chat[i_bc] - g.chat_bc) ** 2,
    }


def newton_steady(g: SpeciesGroups, D1: torch.Tensor, tol: float = 1e-13,
                  max_iter: int = 10) -> torch.Tensor:
    """Newton solve of the steady BVP on the DVR nodes (machine-precision ref).

    System: rows 0..N-2 from (Jhat - s) at all nodes except the Robin node, whose
    row enforces chat(xi_bc) = chat_bc. Linear problem -> converges in 1 step; the
    loop verifies the residual anyway (general-case discipline).
    """
    N = D1.shape[0]
    dtype = D1.dtype
    i_bc = 0 if g.xi_bc == 0.0 else N - 1
    chat = torch.full((N,), float(g.chat_bc), dtype=dtype)

    def F(c):
        J = flux_hat(c, D1, g)
        r = J - g.s
        r = torch.cat([r[:i_bc], r[i_bc + 1:], (c[i_bc] - g.chat_bc).reshape(1)])
        return r

    for _ in range(max_iter):
        r = F(chat)
        if float(r.abs().max()) < tol:
            break
        Jac = torch.autograd.functional.jacobian(F, chat)
        chat = chat - torch.linalg.solve(Jac, r)
    return chat


@dataclass
class TransientGroups:
    """Fixed-reference nondim setup for the Landau-transformed transient problem.

    Reference scales frozen at L_ref = L(t_ref): c0_ref, v (both L-independent for
    fixed D). Time-varying pieces enter as tensors over the tau grid:
        Pe(tau) = |z|*gamma*E*L(tau)         (grows with the layer)
        jhat(tau) = k3(L(tau))/k3(L_ref)     (Tier-1 growth-flux ratio, -> 1)
        adv(tau) = d(ln L)/dtau              (Landau advection strength)
        mig(tau) = t_c / tau_mig(L(tau))     (flux-divergence prefactor)
    """

    s: float
    kappa: float  # Robin ratio at the annihilation boundary
    xi_gen: float  # generation boundary (flux imposed): 0.0 or 1.0
    Pe: torch.Tensor  # (Nt,)
    jhat: torch.Tensor  # (Nt,)
    adv: torch.Tensor  # (Nt,)
    mig: torch.Tensor  # (Nt,)


def transient_groups(p1: Parameters, name: str, tau: np.ndarray, t_c_h: float,
                     dtype=torch.float64) -> TransientGroups:
    L, dlnL = tier1_L_of_tau(p1, tau, t_c_h)
    L_ref = float(L[-1])
    g_ref = species_groups(p1, L_ref)[name]
    k3_ratio = growth_k3(p1, L) / growth_k3(p1, L_ref)  # vectorized over L
    kappa = KAPPA_A if name == "OV" else KAPPA_1
    return TransientGroups(
        s=g_ref.s,
        kappa=kappa,
        xi_gen=1.0 - g_ref.xi_bc,
        Pe=torch.tensor(g_ref.Pe * L / L_ref, dtype=dtype),
        jhat=torch.tensor(k3_ratio, dtype=dtype),
        adv=torch.tensor(dlnL, dtype=dtype),
        mig=torch.tensor(t_c_h / (g_ref.tau_mig_h * L / L_ref), dtype=dtype),
    )


def steady_at(tg: TransientGroups, j: int, xi: torch.Tensor) -> torch.Tensor:
    """Quasi-steady profile at time node j: jhat(j) x the analytic steady shape."""
    xi_bc = 1.0 - tg.xi_gen
    shape = 1.0 + (1.0 / tg.kappa - 1.0) * torch.exp(
        tg.s * tg.Pe[j] * (xi - xi_bc)
    )
    return tg.jhat[j] * shape


def transient_residuals(
    chat: torch.Tensor,  # (Nt, Nx) field on the space-time tensor grid
    D1x: torch.Tensor,  # (Nx, Nx) spatial DVR derivative
    t_grid: torch.Tensor,  # (Nt,) tau nodes
    xi: torch.Tensor,  # (Nx,) spatial nodes
    tg: TransientGroups,
) -> dict[str, torch.Tensor]:
    """CN residuals of the Landau-transformed transient problem.

    dchat/dtau = xi*adv(tau)*dchat/dxi - mig(tau)*d(Jhat)/dxi
    Jhat_j = -(1/Pe_j) dchat/dxi + s*chat
    BCs per time node: Jhat = s*jhat(tau) at the generation boundary (Tier-1
    growth-flux anchor), Jhat = s*kappa*chat at the annihilation boundary (Robin).
    IC: quasi-steady profile at tau_0.
    """
    Nt = chat.shape[0]
    rhs = torch.zeros_like(chat)
    J_all = torch.zeros_like(chat)
    for j in range(Nt):
        dc = D1x @ chat[j]
        J = -dc / tg.Pe[j] + tg.s * chat[j]
        J_all[j] = J
        rhs[j] = xi * tg.adv[j] * dc - tg.mig[j] * (D1x @ J)
    dt = (t_grid[1:] - t_grid[:-1]).unsqueeze(1)
    r_pde = chat[1:] - chat[:-1] - dt * 0.5 * (rhs[1:] + rhs[:-1])

    i_gen = 0 if tg.xi_gen == 0.0 else -1
    i_ann = -1 if tg.xi_gen == 0.0 else 0
    r_gen = J_all[:, i_gen] - tg.s * tg.jhat
    r_ann = J_all[:, i_ann] - tg.s * tg.kappa * chat[:, i_ann]
    return {
        "pde": (r_pde**2).mean(),
        "bc_gen": (r_gen**2).mean(),
        "bc_ann": (r_ann**2).mean(),
        "ic": ((chat[0] - steady_at(tg, 0, xi)) ** 2).mean(),
    }


def tier1_L_of_tau(p1: Parameters, tau: np.ndarray, t_c_h: float):
    """L_bl(tau) and d(ln L)/dtau from the Tier-1 closed form."""
    bp = HTWPDMParams(A_bl=p1.A_bl, b3=p1.b3, C_bl=p1.C_bl, PBR_eff=p1.PBR_eff,
                      C_x=p1.C_x, L0=p1.L0, L_ol0=p1.L_ol0)
    t_h = np.asarray(tau) * t_c_h
    L = L_bl_closed(bp, t_h)
    dLdt = (p1.A_bl * np.exp(p1.b3 * L) - p1.C_bl) * t_c_h  # nm per unit tau
    return L, dLdt / L
