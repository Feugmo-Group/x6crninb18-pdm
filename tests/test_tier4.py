"""Tier-4 regression tests: PNP self-consistency, the differentiable steppers.

Every Tier-4 module claims to reduce to an already-verified Tier-2/Tier-3 result
in a limit. These tests pin those limits, because they are what make the new,
unverifiable-by-construction results trustworthy.

Run from the example root:  python -m pytest tests/test_tier4.py -q
"""

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# Tier 4 is the neural half throughout -- there is no classical subset here to
# preserve, so the whole module stands down when the dependencies are absent.
pytest.importorskip("torch", reason="PyTorch not installed (Tier-4 is neural)")
pytest.importorskip(
    "physicsnemo", reason="PhysicsNeMo not installed; see docs/NSEM_DEPENDENCY.md"
)

import torch  # noqa: E402
from physicsnemo.experimental.models.scen import DVRMapper  # noqa: E402

from htw_pdm.physics import Parameters  # noqa: E402
from htw_pdm.tier2_physics import (  # noqa: E402
    SpeciesGroups,
    analytic_steady,
    transient_groups,
    transient_residuals,
)
from htw_pdm.tier3_wagner import NI_M, RISE, T_OBS  # noqa: E402
from htw_pdm.tier3_wagner import model as wagner_scipy  # noqa: E402
from htw_pdm.tier4_ni_closure import _tgrid, soft_width, wagner_torch  # noqa: E402
from htw_pdm.tier4_pnp import field_diagnostics, newton_pnp, pnp_species  # noqa: E402
from htw_pdm.tier4_transient_inverse import (  # noqa: E402
    NT,
    NX,
    T_C_H,
    TAU0,
    cn_forward,
    scaled_groups,
)

P1 = Parameters()
L_TEST_NM = 134.785  # L_bl(480 h) at the M4 kinetics


def _mapper(n=48):
    torch.set_default_dtype(torch.float64)
    return DVRMapper(n, 0.0, 1.0, 0.0, dtype=torch.float64)


# --------------------------------------------------------------------------
# T4-A: the coupled PNP solver must collapse onto the Tier-2 constant field.
# --------------------------------------------------------------------------

def test_pnp_zero_space_charge_matches_analytic():
    """lam_scale = 0 reproduces the Tier-2 analytic solution to machine precision."""
    m = _mapper()
    species = pnp_species(P1, L_TEST_NM)
    chats, Ehat, res = newton_pnp(species, m.D1, m.weights, lam_scale=0.0)
    assert res < 1e-11
    for sp, chat in zip(species, chats):
        g = SpeciesGroups(sp.name, sp.Pe, sp.s, sp.chat_bc, sp.xi_bc, sp.c0_cm3, 0.0)
        ref = torch.tensor(analytic_steady(g, m.nodes.numpy()), dtype=torch.float64)
        assert float((chat - ref).abs().max() / ref.abs().max()) < 1e-12
    # With no space charge the field must be exactly the imposed constant one.
    assert float((Ehat - 1.0).abs().max()) < 1e-12


def test_pnp_gauge_is_enforced():
    """The mean field is pinned to the Tier-1 value, so eps_f keeps its meaning."""
    m = _mapper()
    species = pnp_species(P1, L_TEST_NM)
    _, Ehat, res = newton_pnp(species, m.D1, m.weights, lam_scale=1e-7)
    assert res < 1e-9
    assert abs(field_diagnostics(Ehat, m.weights)["mean"] - 1.0) < 1e-9


def test_pnp_space_charge_is_monotone_in_lambda():
    """More space charge bends the field further from the constant-field closure."""
    m = _mapper()
    species = pnp_species(P1, L_TEST_NM)
    devs = []
    for ls in (1e-9, 1e-8, 1e-7):
        _, Ehat, res = newton_pnp(species, m.D1, m.weights, lam_scale=ls)
        assert res < 1e-9
        devs.append(float((Ehat - 1.0).abs().max()))
    assert devs[0] < devs[1] < devs[2]


def test_pnp_lambda_scales_inversely_with_diffusivity():
    """Lam ~ c0 ~ 1/D at fixed growth flux -- the basis of the D threshold."""
    import htw_pdm.tier4_pnp as T

    base = pnp_species(P1, L_TEST_NM)[0].Lam
    old = T.D_OV_CM2_S
    try:
        T.D_OV_CM2_S = old * 10.0
        scaled = pnp_species(P1, L_TEST_NM)[0].Lam
    finally:
        T.D_OV_CM2_S = old
    assert scaled == pytest.approx(base / 10.0, rel=1e-10)


# --------------------------------------------------------------------------
# T4-B: the classical Crank-Nicolson stepper must satisfy the Tier-2 residuals.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["OV", "CV"])
def test_cn_stepper_satisfies_transient_residuals(name):
    """The independent forward solve used by the F-1 check is itself exact."""
    m = DVRMapper(NX, 0.0, 1.0, 0.0, dtype=torch.float64)
    t_grid = TAU0 + (1.0 - TAU0) * torch.linspace(0.0, 1.0, NT, dtype=torch.float64)
    tg = transient_groups(P1, name, t_grid.numpy(), T_C_H, dtype=torch.float64)
    c = cn_forward(tg, m.D1, t_grid, m.nodes)
    r = transient_residuals(c, m.D1, t_grid, m.nodes, tg)
    assert max(float(v) for v in r.values()) < 1e-16


@pytest.mark.parametrize("name", ["OV", "CV"])
def test_scaled_groups_identity(name):
    """Unit scaling factors must leave the physics groups untouched."""
    t_grid = TAU0 + (1.0 - TAU0) * torch.linspace(0.0, 1.0, NT, dtype=torch.float64)
    tg = transient_groups(P1, name, t_grid.numpy(), T_C_H, dtype=torch.float64)
    tg2 = scaled_groups(tg, 1.0, 1.0)
    assert tg2.kappa == tg.kappa
    assert torch.allclose(tg2.mig, tg.mig)


def test_cn_stepper_responds_to_diffusivity():
    """A different migration prefactor must give a different field.

    Guards the inverse: if `mig` did not move the solution, recovering it would
    be meaningless regardless of what the optimizer reported.
    """
    m = DVRMapper(NX, 0.0, 1.0, 0.0, dtype=torch.float64)
    t_grid = TAU0 + (1.0 - TAU0) * torch.linspace(0.0, 1.0, NT, dtype=torch.float64)
    tg = transient_groups(P1, "OV", t_grid.numpy(), T_C_H, dtype=torch.float64)
    a = cn_forward(scaled_groups(tg, 1.0, 0.5), m.D1, t_grid, m.nodes)
    b = cn_forward(scaled_groups(tg, 1.0, 2.0), m.D1, t_grid, m.nodes)
    assert float((a - b).abs().max()) > 1e-6


# --------------------------------------------------------------------------
# T4-C: the differentiable Wagner solver must reproduce the Tier-3 integrator.
# --------------------------------------------------------------------------

def test_wagner_torch_matches_scipy_const_closure():
    """Torch Crank-Nicolson vs Tier-3 scipy/Radau at matched constant D."""
    theta = np.array([2.0, 0.30, 0.30])
    W_ref, A_ref = wagner_scipy(theta, P1, "const")
    c, xi = wagner_torch(lambda t: torch.tensor(theta[0], dtype=torch.float64),
                         torch.tensor(theta[1], dtype=torch.float64),
                         float(theta[2]), P1, _tgrid())
    W = [float(xi[(c[j] > NI_M + RISE).nonzero()[-1]]) for j in range(len(T_OBS))]
    assert np.allclose(W, W_ref, rtol=0, atol=1e-9)
    assert float(c[0, 0]) == pytest.approx(A_ref, rel=2e-3)


def test_wagner_torch_is_differentiable_in_D():
    """A gradient must flow to the mobility -- the whole point of the module."""
    D0 = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    c, xi = wagner_torch(lambda t: D0, torch.tensor(0.30, dtype=torch.float64),
                         0.30, P1, _tgrid(60))
    soft_width(c[-1], xi).backward()
    assert D0.grad is not None and torch.isfinite(D0.grad)
    assert float(D0.grad.abs()) > 0.0


def test_soft_width_converges_to_hard_width():
    """The smoothed width is a faithful surrogate as its smoothing goes to zero."""
    c, xi = wagner_torch(lambda t: torch.tensor(2.0, dtype=torch.float64),
                         torch.tensor(0.30, dtype=torch.float64), 0.30, P1,
                         _tgrid())
    j = len(T_OBS) - 1
    hard = float(xi[(c[j] > NI_M + RISE).nonzero()[-1]])
    widths = [float(soft_width(c[j], xi, eps=e)) for e in (1.0, 0.25, 0.05)]
    errs = [abs(w - hard) for w in widths]
    assert errs[0] > errs[1] > errs[2]
    # The residual gap is set by the spatial grid, not by the smoothing: the
    # hard width snaps to a node while the smoothed integral resolves the
    # crossing to within one cell. Requiring better than h would be requiring
    # sub-grid accuracy from a quadrature on that same grid.
    h = float(xi[1] - xi[0])
    assert errs[-1] < h
