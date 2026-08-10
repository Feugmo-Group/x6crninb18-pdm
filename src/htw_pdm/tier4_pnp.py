"""Tier-4 T4-A — self-consistent Poisson-Nernst-Planck transport in the barrier layer.

Tier 2 imposes the PDM's constant-field closure: E(x) = eps_f everywhere, with
eps_f fixed by the Tier-1 mapping b3 = -alpha3*chi*gamma*eps_f. That closure is
the single structural assumption behind the field strength quoted in the paper,
and it is untested: the migrating defects carry charge, so they generate space
charge, which bends the field they migrate in.

Here the field is solved *with* the transport rather than imposed on it. Steady
state, nondimensional on xi = x/L, with chat_i = c_i/c0_i as in tier2_physics
and the field measured against the Tier-1 value, Ehat = E/eps_f:

    Nernst-Planck (first-integral form, flux constant in steady state):
        Jhat_i = -(1/Pe_i) dchat_i/dxi + s_i * Ehat * chat_i = s_i
    Poisson:
        dEhat/dxi = sum_i z_i * Lam_i * chat_i
    Gauge (preserves the Tier-1 anchor):
        int_0^1 Ehat dxi = 1   <=>  the total potential drop across the layer is
        still eps_f * L, so eps_f retains its meaning as the *mean* field and the
        b3 <-> eps_f mapping is untouched; what changes is the field's shape.

The space-charge parameter

    Lam_i = e * L * c0_i / (eps_r * eps_0 * eps_f)

is the whole story. Lam -> 0 recovers Ehat == 1 identically and with it the
analytic constant-field solution of tier2_physics.analytic_steady, which is the
exact reference this module is graded against. Lam = O(1) means the constant-field
closure is quantitatively wrong.

Unlike the Tier-2 steady problem, the coupled system is *nonlinear* (the product
Ehat*chat) and has no closed form, so Newton iteration genuinely iterates here.
Newton remains the accuracy reference; the NSEM solve is graded against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from htw_pdm.physics import CHI, F_OVER_RT, Parameters
from htw_pdm.tier2_physics import (
    D_CV_CM2_S,
    D_OV_CM2_S,
    EPS_F_V_CM,
    KAPPA_1,
    KAPPA_A,
    NM_TO_CM,
    R_CV,
    growth_k3,
)

# Physical constants.
E_CHARGE_C = 1.602176634e-19  # C
EPS_0_F_CM = 8.8541878128e-14  # F/cm
N_A = 6.02214076e23

# Relative permittivity of the Cr-rich spinel barrier layer. Reported values for
# Fe/Cr spinels and passive films on stainless steel span roughly 10-100; the
# smallest value gives the largest space charge, so 12 is the conservative
# (most-space-charge) choice and EPS_R_SCAN brackets the range.
EPS_R = 12.0
EPS_R_SCAN = (10.0, 20.0, 40.0, 100.0)


@dataclass
class PNPSpecies:
    """One defect species in the coupled steady PNP problem."""

    name: str
    z: float  # signed charge number
    Pe: float  # |z| * gamma * eps_f * L  (field reference = Tier-1 eps_f)
    s: float  # sign(z): migration direction along +x
    chat_bc: float  # Robin value at the annihilation boundary
    xi_bc: float  # 0.0 (m/bl) or 1.0 (bl/ol)
    Lam: float  # space-charge parameter e*L*c0/(eps_r*eps_0*eps_f)
    c0_cm3: float


def pnp_species(p1: Parameters, L_nm: float, eps_r: float = EPS_R) -> list[PNPSpecies]:
    """Both defect species' nondimensional groups, including space charge.

    Mirrors tier2_physics.species_groups (same anchors, same assumed constants)
    and adds Lam. Keeping the two constructors parallel is deliberate: T4-A must
    reduce to T2-B exactly when Lam -> 0.
    """
    L_cm = L_nm * NM_TO_CM
    J_OV = (CHI / 2.0) * growth_k3(p1, L_nm)  # mol cm^-2 s^-1
    out: list[PNPSpecies] = []
    for name, z, D, kappa, xi_bc, J in (
        ("OV", +2.0, D_OV_CM2_S, KAPPA_A, 1.0, J_OV),
        ("CV", -CHI, D_CV_CM2_S, KAPPA_1, 0.0, R_CV * J_OV),
    ):
        v = abs(z) * F_OVER_RT * EPS_F_V_CM * D  # |migration velocity|, cm/s
        c0 = J / v * N_A  # cm^-3
        out.append(
            PNPSpecies(
                name=name,
                z=z,
                Pe=abs(z) * F_OVER_RT * EPS_F_V_CM * L_cm,
                s=float(np.sign(z)),
                chat_bc=1.0 / kappa,
                xi_bc=xi_bc,
                Lam=E_CHARGE_C * L_cm * c0 / (eps_r * EPS_0_F_CM * EPS_F_V_CM),
                c0_cm3=c0,
            )
        )
    return out


def pnp_residuals(
    chats: list[torch.Tensor],
    Ehat: torch.Tensor,
    D1: torch.Tensor,
    w: torch.Tensor,
    species: list[PNPSpecies],
    lam_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Residual terms of the coupled steady PNP system on the DVR nodes.

    `lam_scale` multiplies every Lam, so lam_scale=0 collapses the system to the
    Tier-2 constant-field problem whose analytic solution is known exactly. This
    is the continuation knob used both for verification and for Newton homotopy.
    """
    terms: dict[str, torch.Tensor] = {}
    rho = torch.zeros_like(Ehat)
    for sp, chat in zip(species, chats):
        J = -(D1 @ chat) / sp.Pe + sp.s * Ehat * chat
        i_bc = 0 if sp.xi_bc == 0.0 else -1
        terms[f"flux_{sp.name}"] = ((J - sp.s) ** 2).mean()
        terms[f"bc_{sp.name}"] = (chat[i_bc] - sp.chat_bc) ** 2
        rho = rho + sp.z * (lam_scale * sp.Lam) * chat
    terms["poisson"] = ((D1 @ Ehat - rho) ** 2).mean()
    terms["gauge"] = ((w * Ehat).sum() - 1.0) ** 2
    return terms


def _pack(chats: list[torch.Tensor], Ehat: torch.Tensor) -> torch.Tensor:
    return torch.cat([*chats, Ehat])


def _unpack(u: torch.Tensor, n_species: int, N: int):
    parts = [u[i * N:(i + 1) * N] for i in range(n_species)]
    return parts, u[n_species * N:]


def pnp_system(
    u: torch.Tensor,
    D1: torch.Tensor,
    w: torch.Tensor,
    species: list[PNPSpecies],
    lam_scale: float = 1.0,
) -> torch.Tensor:
    """Square residual vector (3N,) of the coupled system, for Newton.

    Per species: (Jhat - s) at every node except its Robin node, which instead
    carries chat(xi_bc) - chat_bc. For the field: Poisson at every node except
    node 0, whose row carries the gauge constraint (the field equation is first
    order, so exactly one condition is required and the gauge supplies it).
    """
    N = D1.shape[0]
    chats, Ehat = _unpack(u, len(species), N)
    rows = []
    rho = torch.zeros_like(Ehat)
    for sp, chat in zip(species, chats):
        J = -(D1 @ chat) / sp.Pe + sp.s * Ehat * chat
        i_bc = 0 if sp.xi_bc == 0.0 else N - 1
        r = J - sp.s
        rows.append(
            torch.cat([r[:i_bc], r[i_bc + 1:], (chat[i_bc] - sp.chat_bc).reshape(1)])
        )
        rho = rho + sp.z * (lam_scale * sp.Lam) * chat
    r_p = D1 @ Ehat - rho
    rows.append(torch.cat([r_p[1:], ((w * Ehat).sum() - 1.0).reshape(1)]))
    return torch.cat(rows)


def newton_pnp(
    species: list[PNPSpecies],
    D1: torch.Tensor,
    w: torch.Tensor,
    lam_scale: float = 1.0,
    n_homotopy: int = 8,
    tol: float = 1e-12,
    max_iter: int = 60,
):
    """Newton solve of the coupled PNP system, with continuation in Lam.

    The nonlinearity is the Ehat*chat product, so a cold Newton start at full
    space charge can diverge. We ramp lam_scale from 0 (where the exact
    constant-field solution is the initial guess) to its target in n_homotopy
    steps, re-solving at each. Returns (chats, Ehat, max |residual|).
    """
    from htw_pdm.tier2_physics import SpeciesGroups, analytic_steady

    N = D1.shape[0]
    xi_np = np.linspace(0.0, 1.0, N)  # only used for the cold start shape
    chats0 = []
    for sp in species:
        g = SpeciesGroups(sp.name, sp.Pe, sp.s, sp.chat_bc, sp.xi_bc, sp.c0_cm3, 0.0)
        chats0.append(torch.tensor(analytic_steady(g, xi_np), dtype=D1.dtype))
    u = _pack(chats0, torch.ones(N, dtype=D1.dtype))

    ramp = np.linspace(0.0, lam_scale, n_homotopy + 1)[1:] if lam_scale else [0.0]
    res = float("nan")
    for ls in ramp:
        for _ in range(max_iter):
            r = pnp_system(u, D1, w, species, float(ls))
            res = float(r.abs().max())
            if res < tol:
                break
            Jac = torch.autograd.functional.jacobian(
                lambda uu: pnp_system(uu, D1, w, species, float(ls)), u
            )
            u = u - torch.linalg.solve(Jac, r)
    chats, Ehat = _unpack(u, len(species), N)
    return chats, Ehat, res


def field_diagnostics(Ehat: torch.Tensor, w: torch.Tensor) -> dict[str, float]:
    """How far the self-consistent field departs from the constant-field closure.

    `max_dev` is the quantity that decides whether the Tier-1 eps_f is a
    description of the field or only of its average; `interface_ratio` is the
    ratio of the field at the two interfaces, which is what the interfacial rate
    constants actually see (the PDM's rate laws are exponential in the local
    field at the reacting interface, not in its spatial mean).
    """
    E = Ehat.detach()
    return {
        "mean": float((w * E).sum()),
        "max_dev": float((E - 1.0).abs().max()),
        "E_at_mbl": float(E[0]),
        "E_at_blol": float(E[-1]),
        "interface_ratio": float(E[0] / E[-1]),
    }
