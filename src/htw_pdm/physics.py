"""HTW_PDM physics residuals for the NSEM (SCEN/DVR) solver.

Reduced Point Defect Model for oxide growth on X6CrNiNb18-10 in high-temperature
water (derivation in ../pdm_eqns.md). Nondimensional system on tau = t/t_c,
lam = L/L_c:

    dlam_bl/dtau = Ahat * exp(bhat * lam_bl) - Chat          lam_bl(0) = lam0
    dlam_ol/dtau = PBReff * dlam_bl/dtau + Cxhat             lam_ol(0) = lamol0

The film-growth ODEs are enforced in Crank-Nicolson integral form on the DVR
time grid (the spectral first-derivative matrix is too ill-conditioned to serve
as the residual operator - same lesson as rpdm_nsem). All parameters are O(1)
by construction of the characteristic scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from htw_pdm._optional import require_torch

if TYPE_CHECKING:  # annotations are strings under `from __future__ import annotations`
    import torch

# Characteristic scales: exposure window and observed Cr-layer thickness scale.
T_C_H = 480.0  # h
L_C_NM = 100.0  # nm

# Physical constants for parameter recovery (240 C).
F_OVER_RT = 96485.332 / (8.314462 * 513.15)  # 1/V
CHI = 8.0 / 3.0  # spinel-averaged cation valence in the barrier layer


@dataclass
class Parameters:
    """Dimensional apparent parameters (nm / h units).

    Defaults are the deterministic fit (model M4) to the Veile 2024
    layer-thickness data - see IMPL_REPORT.md.  Refreshed after the PBR_eff
    prior was removed from baseline_fit.PRIORS: PBR_eff 1.051 -> 1.096 and
    L_ol0 118.7 -> 115.6 nm.  Tier 2/3/4 read these defaults, so they must be
    regenerated whenever the M4 fit changes.
    """

    A_bl: float = 0.7269  # nm/h
    b3: float = -0.01249  # 1/nm (< 0)
    C_bl: float = 0.0  # nm/h (>= 0)
    PBR_eff: float = 1.096  # -
    C_x: float = 0.0  # nm/h (<= 0)
    L0: float = 1.924  # nm
    L_ol0: float = 115.6  # nm


@dataclass
class NondimGroups:
    Ahat: float
    bhat: float
    Chat: float
    PBReff: float
    Cxhat: float
    lam0: float
    lamol0: float

    @classmethod
    def from_parameters(cls, p: Parameters) -> NondimGroups:
        return cls(
            Ahat=p.A_bl * T_C_H / L_C_NM,
            bhat=p.b3 * L_C_NM,
            Chat=p.C_bl * T_C_H / L_C_NM,
            PBReff=p.PBR_eff,
            Cxhat=p.C_x * T_C_H / L_C_NM,
            lam0=p.L0 / L_C_NM,
            lamol0=p.L_ol0 / L_C_NM,
        )

    def to_parameters(self) -> Parameters:
        return Parameters(
            A_bl=self.Ahat * L_C_NM / T_C_H,
            b3=self.bhat / L_C_NM,
            C_bl=self.Chat * L_C_NM / T_C_H,
            PBR_eff=self.PBReff,
            C_x=self.Cxhat * L_C_NM / T_C_H,
            L0=self.lam0 * L_C_NM,
            L_ol0=self.lamol0 * L_C_NM,
        )


# Clamp for the growth exponential: bhat < 0 and lam > 0 keep the argument
# negative in the physical region, but during optimisation transients the
# product can go positive; cap it well below overflow.
EXP_CLAMP = 60.0


def growth_rate(lam_bl: torch.Tensor, g: NondimGroups, kinetics: dict | None = None):
    """f(lam_bl) = Ahat exp(bhat lam_bl) - Chat, with optional learnable overrides.

    kinetics: dict with any of {"Ahat", "bhat", "Chat", "PBReff", "Cxhat",
    "lam0", "lamol0"} as scalar tensors (inverse problem). Missing keys fall
    back to the fixed NondimGroups values.
    """
    torch = require_torch()
    k = kinetics or {}
    Ahat = k.get("Ahat", g.Ahat)
    bhat = k.get("bhat", g.bhat)
    Chat = k.get("Chat", g.Chat)
    return Ahat * torch.exp(torch.clamp(bhat * lam_bl, max=EXP_CLAMP)) - Chat


def pdm_residuals(
    lam_bl: torch.Tensor,
    lam_ol: torch.Tensor,
    t_grid: torch.Tensor,
    g: NondimGroups,
    kinetics: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Crank-Nicolson residuals of the two film-growth ODEs plus IC terms.

    lam_bl, lam_ol: (Nt,) nondimensional thicknesses on the DVR time nodes.
    Returns scalar loss terms {ode_bl, ode_ol, ic}.
    """
    k = kinetics or {}
    PBReff = k.get("PBReff", g.PBReff)
    Cxhat = k.get("Cxhat", g.Cxhat)
    lam0 = k.get("lam0", g.lam0)
    lamol0 = k.get("lamol0", g.lamol0)

    dt = t_grid[1:] - t_grid[:-1]  # (Nt-1,)
    f = growth_rate(lam_bl, g, kinetics)  # (Nt,)
    f_mid = 0.5 * (f[1:] + f[:-1])

    r_bl = lam_bl[1:] - lam_bl[:-1] - dt * f_mid
    r_ol = lam_ol[1:] - lam_ol[:-1] - dt * (PBReff * f_mid + Cxhat)

    ic = (lam_bl[0] - lam0) ** 2 + (lam_ol[0] - lamol0) ** 2
    return {
        "ode_bl": (r_bl**2).mean(),
        "ode_ol": (r_ol**2).mean(),
        "ic": ic,
    }


def data_loss(
    lam: torch.Tensor,
    t_grid: torch.Tensor,
    t_data: torch.Tensor,
    L_data_nm: torch.Tensor,
    sigma_nm: torch.Tensor,
) -> torch.Tensor:
    """1/sigma^2-weighted MSE of the network thickness against measured means.

    Linear interpolation of lam onto the (dimensional, hours) data times.
    """
    torch = require_torch()
    tau_data = t_data / T_C_H
    idx = torch.searchsorted(t_grid, tau_data).clamp(1, t_grid.numel() - 1)
    t0, t1 = t_grid[idx - 1], t_grid[idx]
    w = (tau_data - t0) / (t1 - t0)
    lam_at = (1 - w) * lam[idx - 1] + w * lam[idx]
    r = (lam_at * L_C_NM - L_data_nm) / sigma_nm
    return (r**2).mean()


def recovered_field_strength(b3_per_nm: float, alpha3: float = 0.12) -> float:
    """eps_f [V/cm] from the fitted b3 = -alpha3*chi*gamma*eps_f."""
    return -b3_per_nm / (alpha3 * CHI * F_OVER_RT) * 1e7
