"""Parity and regression tests for the HTW_PDM baseline and NSEM physics.

Run from the example root:  python -m pytest tests/ -q
"""


import numpy as np
import pytest

from htw_pdm._optional import have

# The classical parity and digitization tests are the ones that pin the
# manuscript's headline numbers, and they must run on a NumPy/SciPy-only
# install.  Only the tests that genuinely exercise the neural path are skipped
# when its dependencies are absent -- an import-time `import torch` here would
# have failed collection for the whole module instead.
requires_torch = pytest.mark.skipif(
    not have("torch"), reason="PyTorch not installed (neural path only)"
)
requires_physicsnemo = pytest.mark.skipif(
    not have("physicsnemo"),
    reason="PhysicsNeMo not installed; see docs/NSEM_DEPENDENCY.md",
)

if have("torch"):
    import torch


from htw_pdm.baseline_ode import (  # noqa: E402
    LI_TABLE5,
    HTWPDMParams,
    L_bl_closed,
    L_bl_steady_state,
    L_ol_closed,
    L_ol_closed_li42,
    params_from_li_cgs,
    solve_forward,
)

DEMO = HTWPDMParams(A_bl=12.0, b3=-0.03, C_bl=0.05, PBR_eff=1.1, C_x=-0.01, L0=2.0)
T = np.linspace(0.0, 480.0, 25)


class TestClosedFormParity:
    def test_radau_vs_closed_form(self):
        num = solve_forward(DEMO, T)
        assert np.max(np.abs(num[:, 0] - L_bl_closed(DEMO, T))) < 1e-8
        assert np.max(np.abs(num[:, 1] - L_ol_closed(DEMO, T))) < 1e-8

    def test_affine_vs_li42(self):
        assert np.max(np.abs(L_ol_closed(DEMO, T) - L_ol_closed_li42(DEMO, T))) < 1e-10

    def test_t0_exact_and_long_time_limit(self):
        assert L_bl_closed(DEMO, np.array([0.0]))[0] == pytest.approx(DEMO.L0)
        lss = L_bl_steady_state(DEMO)
        assert L_bl_closed(DEMO, np.array([1e9]))[0] == pytest.approx(lss, rel=1e-10)

    def test_no_dissolution_limit(self):
        p0 = HTWPDMParams(A_bl=12.0, b3=-0.03, C_bl=0.0, PBR_eff=1.1, C_x=0.0, L0=2.0)
        peps = HTWPDMParams(A_bl=12.0, b3=-0.03, C_bl=1e-12, PBR_eff=1.1, C_x=0.0, L0=2.0)
        assert np.max(np.abs(L_bl_closed(p0, T) - L_bl_closed(peps, T))) < 1e-6

    def test_li_table5_regression(self):
        for cgs in LI_TABLE5.values():
            p = params_from_li_cgs(**cgs)
            t = np.array([100.0, 1000.0, 20000.0])
            num = solve_forward(p, t)
            ref = np.column_stack([L_bl_closed(p, t), L_ol_closed(p, t)])
            assert np.max(np.abs(num - ref)) < 1e-6  # nm, on um-scale values


class TestVeileDigitization:
    def test_power_law_refit_cr(self):
        from htw_pdm.baseline_fit import fit_power_law, load_data

        _, scans = load_data()
        k, n = fit_power_law(scans)["Cr"]
        assert k == pytest.approx(6.521, rel=0.01)
        assert n == pytest.approx(0.4964, abs=0.005)


class TestNSEMPhysics:
    @requires_torch
    def test_cn_residual_of_exact_solution(self):
        """CN residuals of the closed-form solution vanish at O(dt^3)."""
        from htw_pdm.physics import L_C_NM, T_C_H, NondimGroups, Parameters, pdm_residuals

        p = Parameters()
        g = NondimGroups.from_parameters(p)
        t_grid = torch.linspace(0.0, 1.0, 201, dtype=torch.float64)
        bp = HTWPDMParams(A_bl=p.A_bl, b3=p.b3, C_bl=p.C_bl, PBR_eff=p.PBR_eff,
                          C_x=p.C_x, L0=p.L0, L_ol0=p.L_ol0)
        t_h = t_grid.numpy() * T_C_H
        lam_bl = torch.tensor(L_bl_closed(bp, t_h) / L_C_NM)
        lam_ol = torch.tensor(L_ol_closed(bp, t_h) / L_C_NM)
        terms = pdm_residuals(lam_bl, lam_ol, t_grid, g)
        assert float(terms["ode_bl"]) < 1e-10
        assert float(terms["ode_ol"]) < 1e-10
        assert float(terms["ic"]) < 1e-20

    def test_nondim_roundtrip(self):
        from htw_pdm.physics import NondimGroups, Parameters

        p = Parameters(A_bl=1.3, b3=-0.02, C_bl=0.1, PBR_eff=1.2, C_x=-0.05,
                       L0=3.0, L_ol0=50.0)
        q = NondimGroups.from_parameters(p).to_parameters()
        for f in ("A_bl", "b3", "C_bl", "PBR_eff", "C_x", "L0", "L_ol0"):
            assert getattr(q, f) == pytest.approx(getattr(p, f), rel=1e-12)

    @requires_physicsnemo
    def test_hard_integrator_vs_closed_form(self):
        from htw_pdm.inverse_trainer import KineticParams, integrate_hard
        from htw_pdm.physics import L_C_NM, T_C_H, NondimGroups, Parameters

        p = Parameters()
        g = NondimGroups.from_parameters(p)
        kin = KineticParams(g, learn_Chat=False, init_scale=1.0)
        t_grid = torch.linspace(0.0, 1.0, 25, dtype=torch.float64)
        with torch.no_grad():
            lam_bl, lam_ol = integrate_hard(kin, g, t_grid)
        bp = HTWPDMParams(A_bl=p.A_bl, b3=p.b3, C_bl=p.C_bl, PBR_eff=p.PBR_eff,
                          C_x=p.C_x, L0=p.L0, L_ol0=p.L_ol0)
        ref_bl = L_bl_closed(bp, t_grid.numpy() * T_C_H) / L_C_NM
        assert np.max(np.abs(lam_bl.numpy() - ref_bl)) < 1e-9


class TestTier2Spatial:
    @requires_physicsnemo
    def test_newton_vs_analytic_both_species(self):
        from physicsnemo.experimental.models.scen import DVRMapper

        from htw_pdm.physics import Parameters
        from htw_pdm.tier2_physics import analytic_steady, newton_steady, species_groups

        mapper = DVRMapper(24, 0.0, 1.0, 0.0, dtype=torch.float64)
        for g in species_groups(Parameters(), 134.0).values():
            chat = newton_steady(g, mapper.D1)
            exact = torch.tensor(analytic_steady(g, mapper.nodes.numpy()))
            assert float((chat - exact).abs().max()) < 1e-10

    @requires_physicsnemo
    def test_steady_flux_is_constant(self):
        """The analytic steady profile carries Jhat = s at every node."""
        from physicsnemo.experimental.models.scen import DVRMapper

        from htw_pdm.physics import Parameters
        from htw_pdm.tier2_physics import analytic_steady, flux_hat, species_groups

        mapper = DVRMapper(32, 0.0, 1.0, 0.0, dtype=torch.float64)
        for g in species_groups(Parameters(), 134.0).values():
            chat = torch.tensor(analytic_steady(g, mapper.nodes.numpy()))
            J = flux_hat(chat, mapper.D1, g)
            assert float((J - g.s).abs().max()) < 1e-8

    @requires_torch
    def test_transient_groups_reduce_to_steady_at_reference(self):
        """jhat(tau_end) = 1 and Pe(tau_end) matches the steady groups."""
        import numpy as np

        from htw_pdm.physics import Parameters
        from htw_pdm.tier2_physics import species_groups, transient_groups

        p1 = Parameters()
        tau = np.linspace(0.05, 1.0, 8)
        from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed

        bp = HTWPDMParams(A_bl=p1.A_bl, b3=p1.b3, C_bl=p1.C_bl,
                          PBR_eff=p1.PBR_eff, C_x=p1.C_x, L0=p1.L0,
                          L_ol0=p1.L_ol0)
        L_ref = float(L_bl_closed(bp, np.array([480.0]))[0])
        for name in ("OV", "CV"):
            tg = transient_groups(p1, name, tau, 480.0)
            g = species_groups(p1, L_ref)[name]
            assert float(tg.jhat[-1]) == pytest.approx(1.0, rel=1e-12)
            assert float(tg.Pe[-1]) == pytest.approx(g.Pe, rel=1e-12)

    def test_wagner_no_enrichment_at_full_incorporation(self):
        """r_Ni -> 1: Ni enters the oxide like Fe/Cr, no enrichment zone."""
        import numpy as np

        from htw_pdm.physics import Parameters
        from htw_pdm.tier2_inverse import NI_M, wagner_solve

        c, _ = wagner_solve(np.array([2.0, 0.999, 0.0]), Parameters())
        assert float(np.abs(c - NI_M).max()) < 0.05


class TestParametricPhysics:
    def test_reference_condition_reproduces_m4(self):
        from htw_pdm.physics import Parameters
        from htw_pdm.physics_parametric import ConditionScan, apparent_parameters

        ref = Parameters()
        p = apparent_parameters(240.0, 0.4, ref, ConditionScan(dG0_R=50e3))
        for n in ("A_bl", "b3", "C_bl", "PBR_eff", "L0", "L_ol0"):
            assert getattr(p, n) == pytest.approx(getattr(ref, n), rel=1e-12)

    def test_channel_signs(self):
        """Arrhenius: A_bl increases with T; field: |b3| decreases with T;
        ECP: A_bl increases with [O2]; dissolution scales as sqrt(C_O)."""
        from htw_pdm.physics import Parameters
        from htw_pdm.physics_parametric import ConditionScan, apparent_parameters

        ref = Parameters()
        scan = ConditionScan(dG0_R=50e3)
        p_lo = apparent_parameters(210.0, 0.4, ref, scan)
        p_hi = apparent_parameters(280.0, 0.4, ref, scan)
        assert p_hi.A_bl > p_lo.A_bl
        assert abs(p_hi.b3) < abs(p_lo.b3)

        p_o2lo = apparent_parameters(240.0, 0.02, ref, scan)
        p_o2hi = apparent_parameters(240.0, 4.0, ref, scan)
        assert p_o2hi.A_bl > p_o2lo.A_bl
        # ECP off -> [O2] has no effect when C_bl_ref = 0
        scan_off = ConditionScan(dG0_R=50e3, ecp_channel=False)
        q1 = apparent_parameters(240.0, 0.02, ref, scan_off)
        q2 = apparent_parameters(240.0, 4.0, ref, scan_off)
        assert q1.A_bl == pytest.approx(q2.A_bl, rel=1e-14)

        scan_diss = ConditionScan(dG0_R=0.0, ecp_channel=False, C_bl_ref=0.1)
        d1 = apparent_parameters(240.0, 0.4, ref, scan_diss)
        d2 = apparent_parameters(240.0, 1.6, ref, scan_diss)
        assert d2.C_bl == pytest.approx(d1.C_bl * 2.0, rel=1e-12)  # sqrt(4x) = 2x

    def test_liquid_range_enforced(self):
        from htw_pdm.physics_parametric import rho_water

        with pytest.raises(ValueError):
            rho_water(np.array([300.0]))  # steam at 7 MPa
        assert 0.74 < rho_water(np.array([285.0]))[0] < 0.87
