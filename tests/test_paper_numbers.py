"""Regression tests pinning the numbers the manuscript actually claims.

Everything else in tests/ checks that the machinery is internally consistent --
closed form against Radau, Newton against analytic, the differentiable stepper
against the classical one. None of it would notice if a refactor moved
$\\chi^2$ from 5.79 to 6.4, which is the change that would matter, because it
is the number in the abstract.

This module closes that gap. Every value asserted here appears in
paper/main-snjnl.tex, and the tolerances are the precision at which the
manuscript quotes them, so a drift large enough to make the paper wrong fails
here and a drift smaller than the last reported digit does not.

Classical path only: no PyTorch, no PhysicsNeMo. The headline claims are
reachable without either, and that is itself worth pinning.
"""

from __future__ import annotations

import numpy as np
import pytest

from htw_pdm.baseline_fit import (
    build_fit_data,
    fit_pdm,
    fit_power_law,
    load_data,
    power_law_chi2,
    profile_likelihood,
)
from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed, L_ol_closed
from htw_pdm.paths import PAPER_OUT
from htw_pdm.physics import Parameters, recovered_field_strength

HOURS_PER_YEAR = 8766.0


@pytest.fixture(scope="module")
def fit():
    means, scans = load_data()
    d = build_fit_data(means)
    sol, p, chi2, dof, _total = fit_pdm(d, 4)  # M4, the accepted model
    return d, scans, sol, p, chi2, dof


class TestDigitizedData:
    """The digitization is validated by reproducing the source paper's own fits."""

    def test_chromium_refit_reproduces_veile(self):
        """The manuscript's stated digitization check, to four significant figures.

        This is the only refit the paper claims reproduces the source, and it is
        what bounds the read-off error on the layer that carries 95% of the fit.
        """
        _means, scans = load_data()
        pl = fit_power_law(scans)
        assert pl["Cr"][0] == pytest.approx(6.521, abs=0.01)
        assert pl["Cr"][1] == pytest.approx(0.4964, abs=0.001)

    def test_iron_refit_does_not_reproduce_veile(self):
        """And the iron refit does not -- pinned so it cannot be quietly assumed.

        The outer layer is discrete crystals; its scan-to-scan physical scatter
        exceeds digitization precision and the Fe scan positions are figure
        read-offs. The manuscript therefore treats the transcribed Fe *means*
        as authoritative and never uses this refit. If a future change makes
        this agree, the digitization improved and the claim can be widened; the
        failure a silent assertion here would hide is the reverse.
        """
        _means, scans = load_data()
        pl = fit_power_law(scans)
        assert pl["Fe"][0] == pytest.approx(54.6, rel=0.02)
        assert pl["Fe"][1] == pytest.approx(0.249, rel=0.02)

    def test_six_points_five_parameters(self, fit):
        """The paper's central premise: the fit is this sparse."""
        d, _scans, sol, _p, _chi2, dof = fit
        assert len(d.t) == 3
        assert d.L_cr.size + d.L_fe.size == 6
        assert len(sol.x) == 5
        assert dof == 1


class TestAcceptedModel:
    """Table 1 / Table 2 of the manuscript."""

    def test_m4_chi2(self, fit):
        *_, chi2, _dof = fit
        assert chi2 == pytest.approx(5.79, abs=0.01)

    def test_m4_parameters(self, fit):
        _d, _scans, _sol, p, *_ = fit
        assert p.A_bl == pytest.approx(0.7269, rel=1e-3)
        assert p.b3 == pytest.approx(-0.01249, rel=1e-3)
        assert p.PBR_eff == pytest.approx(1.096, rel=1e-3)
        assert p.L0 == pytest.approx(1.924, rel=1e-3)
        assert p.L_ol0 == pytest.approx(115.6, rel=1e-3)
        assert p.C_bl == 0.0
        assert p.C_x == 0.0

    def test_physics_defaults_match_the_fit(self, fit):
        """physics.Parameters is read by the spatial and closure solves; it must not go stale."""
        _d, _scans, _sol, p, *_ = fit
        ref = Parameters()
        for name in ("A_bl", "b3", "PBR_eff", "L0", "L_ol0"):
            assert getattr(ref, name) == pytest.approx(getattr(p, name), rel=2e-3), name

    def test_field_strength(self, fit):
        _d, _scans, _sol, p, *_ = fit
        assert recovered_field_strength(p.b3) == pytest.approx(1.73e4, rel=0.01)

    def test_m1_chi2_without_initial_outer_layer(self, fit):
        d, *_ = fit
        _sol, _p, chi2, _dof, _t = fit_pdm(d, 1)
        assert chi2 == pytest.approx(7.15, abs=0.02)


class TestExtrapolation:
    """The comparison the abstract leads with."""

    def test_ten_year_barrier_layer(self, fit):
        _d, _scans, _sol, p, *_ = fit
        L10 = float(L_bl_closed(p, np.array([10 * HOURS_PER_YEAR]))[0])
        assert L10 == pytest.approx(535.0, abs=1.0)

    def test_power_law_extrapolates_far_above(self, fit):
        _d, scans, _sol, p, *_ = fit
        pl = fit_power_law(scans)
        t10 = 10 * HOURS_PER_YEAR
        L_pl = pl["Cr"][0] * t10 ** pl["Cr"][1]
        L_pdm = float(L_bl_closed(p, np.array([t10]))[0])
        assert L_pl == pytest.approx(1853.0, rel=0.01)
        # "a factor of 3.5 to 6.3 above the mechanistic barrier layer"
        assert 3.4 < L_pl / L_pdm < 6.4

    def test_power_law_fits_calibration_data_comparably(self, fit):
        """Both descriptions fit; only the extrapolations part company."""
        d, scans, _sol, _p, chi2, _dof = fit
        pl = fit_power_law(scans)
        chi2_pl = power_law_chi2(d, *pl["Cr"], "Cr") + power_law_chi2(d, *pl["Fe"], "Fe")
        assert chi2_pl > chi2  # the mechanistic fit is not worse


class TestIdentifiability:
    """The paper's thesis, as a test: some of these parameters are not determined."""

    def test_L0_is_unidentifiable(self, fit):
        d, _scans, sol, *_ = fit
        _grid, chi2s = profile_likelihood(d, 4, sol.x, 3)  # index 3 = ln L0
        assert (chi2s - chi2s.min()).max() < 4.0

    @pytest.mark.parametrize("i,name", [(0, "A_bl"), (1, "-b3"), (4, "L_ol0")])
    def test_sharply_determined_parameters(self, fit, i, name):
        d, _scans, sol, *_ = fit
        _grid, chi2s = profile_likelihood(d, 4, sol.x, i)
        assert (chi2s - chi2s.min()).max() > 4.0, name


class TestResidualLeverage:
    """Abstract: 95% of the fit statistic on three chromium points, 80% on one."""

    def test_chromium_carries_the_fit(self, fit):
        d, _scans, _sol, p, chi2, _dof = fit
        r_cr = (L_bl_closed(p, d.t) - d.L_cr) / d.sig_cr
        share = float(np.sum(r_cr**2) / chi2)
        assert share == pytest.approx(0.95, abs=0.02)

    def test_one_point_carries_most_of_it(self, fit):
        d, _scans, _sol, p, chi2, _dof = fit
        r_cr = (L_bl_closed(p, d.t) - d.L_cr) / d.sig_cr
        i168 = int(np.argmin(np.abs(d.t - 168.0)))
        assert float(r_cr[i168] ** 2 / chi2) == pytest.approx(0.80, abs=0.02)


class TestOneWayCoupling:
    """README: the barrier layer never reads PBR_eff, C_x or L_ol0.

    This is the structural fact that makes every barrier-layer result immune to
    The outer-layer degeneracy, so it is asserted rather than trusted.
    """

    def test_barrier_layer_ignores_outer_layer_parameters(self, fit):
        from dataclasses import replace

        _d, _scans, _sol, p, *_ = fit
        t = np.array([72.0, 168.0, 480.0, 10 * HOURS_PER_YEAR])
        base = L_bl_closed(p, t)
        perturbed = replace(p, PBR_eff=p.PBR_eff * 3.0, C_x=-0.5, L_ol0=p.L_ol0 * 2)
        assert np.allclose(L_bl_closed(perturbed, t), base, rtol=0, atol=0)


class TestNeuralArtefacts:
    """The committed artefacts behind the two neural exhibits.

    Regenerating these needs the `nsem` extra; *checking* them must not. The
    F-1 exhibit and the field-closure exhibit are the two places where a
    manuscript number comes out of a torch run rather than a closed form, so
    without this class nothing in the classical tier would notice if either
    artefact drifted away from the value printed in the paper.
    """

    @pytest.fixture(scope="class")
    def f1(self):
        path = PAPER_OUT / "f1_trajectory.npz"
        if not path.exists():
            pytest.skip("f1_trajectory.npz not present; "
                        "run scripts/make_f1_trajectory.py --extra nsem")
        return np.load(path)

    def test_soft_mode_trajectory_chi2_matches_the_text(self, f1):
        """Section 2.4 and Table 6: the trajectory's own chi2 is 0.21."""
        assert float(f1["chi2_traj"]) == pytest.approx(0.21, abs=0.005)

    def test_exact_solve_at_those_parameters_matches_the_text(self, f1):
        """Section 2.4 and Table 6: the closed form at the same parameters, 16.6."""
        assert float(f1["chi2_exact"]) == pytest.approx(16.64, abs=0.005)

    def test_the_gap_is_the_stated_two_orders(self, f1):
        """'a factor of roughly 80 apart' -- the diagnostic, not the values."""
        ratio = float(f1["chi2_exact"]) / float(f1["chi2_traj"])
        assert 60.0 < ratio < 100.0

    def test_trajectory_fits_the_data_better_than_the_accepted_model(self, f1, fit):
        """The point of the exhibit: a wrong answer that looks like a better fit."""
        _d, _scans, _sol, _p, chi2_m4, _dof = fit
        assert float(f1["chi2_traj"]) < chi2_m4

    def test_the_stored_exact_curve_is_the_closed_form_at_the_stored_parameters(self, f1):
        """The artefact must be internally consistent, not just carry two scalars.

        Recomputing the closed form here from the recovered parameters catches a
        stale or mismatched dump, which the two chi2 values alone would not.
        """
        p = HTWPDMParams(A_bl=float(f1["fit_A_bl"]), b3=float(f1["fit_b3"]),
                         C_bl=0.0, PBR_eff=float(f1["fit_PBR_eff"]), C_x=0.0,
                         L0=float(f1["fit_L0"]), L_ol0=float(f1["fit_L_ol0"]))
        t = f1["t_dense_h"]
        assert np.allclose(L_bl_closed(p, t), f1["exact_bl_nm"], rtol=1e-10)
        assert np.allclose(L_ol_closed(p, t), f1["exact_ol_nm"], rtol=1e-10)

    def test_recovered_parameters_are_badly_wrong(self, f1, fit):
        """Section 2.4: the soft mode misses A_bl by 74% and L_ol0 by 55%."""
        _d, _scans, _sol, p4, *_ = fit
        assert abs(float(f1["fit_A_bl"]) - p4.A_bl) / p4.A_bl == pytest.approx(0.744, abs=0.01)
        assert abs(float(f1["fit_L_ol0"]) - p4.L_ol0) / p4.L_ol0 == pytest.approx(0.553, abs=0.01)

    def test_the_artefact_says_where_it_came_from(self, f1):
        """A committed binary that nothing can trace is not a reproducible one.

        The run that produced this file is recorded inside it -- overrides,
        seed, dtype and solver versions -- because the artefact outlives the
        environment that made it.
        """
        import json
        prov = json.loads(str(f1["provenance"]))
        assert prov["seed"] == 0
        assert prov["dtype"] == "float64"
        assert "inverse.lambda_phys=1000.0" in prov["overrides"]
        assert prov["torch"] and prov["nvidia-physicsnemo"]

    @pytest.fixture(scope="class")
    def pnp(self):
        import json
        path = PAPER_OUT / "field_closure.json"
        if not path.exists():
            pytest.skip("field_closure.json not present; run htw_pdm.field_closure_solve")
        return json.loads(path.read_text())

    def test_debye_length_matches_the_abstract(self, pnp):
        """Conclusions: 'the implied Debye length is approximately 0.06 nm'."""
        per = pnp["A2_magnitude"]["per_eps_r"]
        eps = np.array(sorted(float(k) for k in per))
        lam = np.array([per[f"{e}"]["debye_nm"] for e in eps])
        used = float(pnp["eps_r"])
        assert float(np.interp(np.log(used), np.log(eps), lam)) == pytest.approx(0.06, abs=0.005)

    def test_the_diffusivity_gap_is_six_orders(self, pnp):
        """Section 3.5: uniformity would need D six orders above the assumed value."""
        from htw_pdm.spatial_physics import D_OV_CM2_S
        d10 = pnp["A3_validity"]["thresholds"]["0.10"]["D_required_cm2_s"]["OV"]
        assert np.log10(d10 / D_OV_CM2_S) == pytest.approx(6.0, abs=0.3)

    def test_the_reduction_check_that_licenses_the_result(self, pnp):
        """Section 3.5: with the space charge off it reproduces the analytic
        constant-field solution to 1e-15. Without that, the refutation is just
        a solver disagreeing with a formula."""
        rel = pnp["A1_verification"]["rel_err_vs_analytic"]
        assert max(rel.values()) < 1e-14

    def test_nickel_closure_is_rejected_on_shape(self):
        """Section 2.6: chi2 = 25.7 on one dof, and the rejection is that the
        model can only rise while the data are flat."""
        path = PAPER_OUT / "ni_zone_curve.npz"
        if not path.exists():
            pytest.skip("ni_zone_curve.npz not present; run scripts/make_ni_curve.py")
        nc = np.load(path)
        assert float(nc["chi2"]) == pytest.approx(25.71, abs=0.05)
        assert np.all(np.diff(nc["W_nm"]) >= -1e-9), "closure prediction must be monotone"
        w = nc["W_obs"]
        assert w.max() / w.min() < 1.5, "measured widths are flat within scatter"


class TestDiscriminabilityThresholds:
    """Section 3.6 quotes eight exposure thresholds. Every one of them has to
    come out of referee_stats.json, because a threshold nobody can re-derive is
    a recommendation nobody can check."""

    @pytest.fixture(scope="class")
    def thresholds(self):
        import json
        path = PAPER_OUT / "referee_stats.json"
        if not path.exists():
            pytest.skip("referee_stats.json not present; run scripts/make_paper_stats.py")
        stats = json.loads(path.read_text())["discriminability"]
        if "thresholds" not in stats:
            pytest.skip("referee_stats.json predates the eight-threshold computation")
        return stats["thresholds"]

    @pytest.mark.parametrize(
        ("key", "t_2sigma", "t_3sigma"),
        [
            ("published_coeffs__scatter_only", 2546.0, 4014.0),
            ("published_coeffs__full_budget", 2988.0, 5129.0),
            ("weighted_refit__scatter_only", 1437.0, 2020.0),
            ("weighted_refit__full_budget", 1928.0, 5493.0),
        ],
    )
    def test_every_quoted_threshold_matches_the_artifact(
        self, thresholds, key, t_2sigma, t_3sigma
    ):
        entry = thresholds[key]
        assert entry["t_2sigma_h"] == pytest.approx(t_2sigma, rel=0.005)
        assert entry["t_3sigma_h"] == pytest.approx(t_3sigma, rel=0.005)

    def test_two_sigma_once_reached_is_never_lost(self, thresholds):
        """The 2 sigma recommendation is a threshold, and the paper states it
        as one."""
        for key, entry in thresholds.items():
            assert entry["sustained_to_window_end_2sigma"], key

    def test_three_sigma_against_the_refit_is_a_window_not_a_threshold(self, thresholds):
        """Carrying the power law's own fit uncertainty makes the separation
        non-monotone in exposure time: it peaks near 12000 h and falls back
        through 3 sigma. Section 3.6 says so, and would be wrong if this
        stopped holding."""
        entry = thresholds["weighted_refit__full_budget"]
        assert entry["powerlaw_fit_uncertainty_included"]
        assert not entry["sustained_to_window_end_3sigma"]
        assert entry["separation_max_sigma"] == pytest.approx(3.18, abs=0.05)
        assert entry["t_at_separation_max_h"] == pytest.approx(12047.0, rel=0.02)
        assert entry["t_3sigma_window_end_h"] == pytest.approx(34400.0, rel=0.02)
        assert entry["separation_at_window_end_sigma"] < 3.0

    def test_the_published_branch_carries_no_power_law_fit_uncertainty(self, thresholds):
        """Those coefficients were fitted elsewhere and no covariance is
        published, which is why that branch is the optimistic one at 3 sigma."""
        for key in ("published_coeffs__scatter_only", "published_coeffs__full_budget"):
            assert not thresholds[key]["powerlaw_fit_uncertainty_included"]
