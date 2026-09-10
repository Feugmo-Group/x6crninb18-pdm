"""Tests for the mass balance and the M6 (C_x-free) closure diagnostic.

Run from the example root:  python -m pytest tests/ -q
"""


import numpy as np
import pytest

import htw_pdm.baseline_fit as bf  # noqa: E402
import htw_pdm.mass_balance as mb  # noqa: E402
from htw_pdm.baseline_fit import build_fit_data, fit_pdm, load_data, unpack  # noqa: E402


class TestMassBalance:
    def test_molar_volumes_against_known_densities(self):
        """Lattice-derived molar volumes must reproduce tabulated densities."""
        # rho = M / V; magnetite 5.15-5.20, chromite 4.95-5.10 g/cm^3.
        m_mag = 3 * mb.M["Fe"] + 4 * mb.M["O"]
        assert 5.15 < m_mag / mb.MAGNETITE.V < 5.25
        chromite = next(p for p in mb.BARRIER_CANDIDATES if p.name == "FeCr2O4")
        m_chr = mb.M["Fe"] + 2 * mb.M["Cr"] + 4 * mb.M["O"]
        assert 4.95 < m_chr / chromite.V < 5.15

    def test_alloy_composition_matches_manuscript(self):
        """The nominal AISI 347 composition must give the ~18.8 at% Cr quoted."""
        x = mb.atom_fractions()
        assert 100 * x["Cr"] == pytest.approx(18.8, abs=0.5)
        assert 100 * x["Ni"] == pytest.approx(10.0, abs=1.0)
        assert sum(x.values()) == pytest.approx(1.0, rel=1e-12)

    def test_fecr2o4_production_ratio_is_the_quoted_bound(self):
        """The manuscript's 2.09 must fall out of the derivation, not be typed in."""
        chromite = next(p for p in mb.BARRIER_CANDIDATES if p.name == "FeCr2O4")
        assert mb.production_ratio(chromite) == pytest.approx(2.09, rel=0.05)

    def test_barrier_pbr_matches_spatial_constant(self):
        """barrier_pbr must reproduce the 2.05 hardcoded in composition_map.

        These are different quantities from production_ratio and the whole point
        of computing both is that they must not be conflated; this pins the one
        that already had a value elsewhere in the codebase.
        """
        from htw_pdm.composition_map import PBR_BL

        chromite = next(p for p in mb.BARRIER_CANDIDATES if p.name == "FeCr2O4")
        assert mb.barrier_pbr(chromite) == pytest.approx(PBR_BL, rel=0.05)

    def test_iron_poor_barrier_is_infeasible(self):
        """A barrier richer in Fe than the alloy can supply cannot grow at all."""
        greedy = mb.Phase("FeCr0.1O", n_Fe=10.0, n_Cr=0.1, n_Ni=0.0, V=44.0)
        assert mb.production_ratio(greedy) == 0.0

    def test_ni_routing_raises_the_ratio(self):
        """Routing nickel to the oxide adds outer-layer volume, never removes it."""
        chromite = next(p for p in mb.BARRIER_CANDIDATES if p.name == "FeCr2O4")
        assert (mb.production_ratio(chromite, r_Ni=0.2954)
                > mb.production_ratio(chromite, r_Ni=0.0))

    def test_release_flux_scales_linearly_and_has_sane_magnitude(self):
        a = mb.release_flux(-0.25)
        b = mb.release_flux(-0.50)
        assert b["Fe_ug_dm2_h"] == pytest.approx(2 * a["Fe_ug_dm2_h"], rel=1e-12)
        # 0.25 nm/h of magnetite is order 10 ug Fe per dm^2 per hour.
        assert 5.0 < a["Fe_ug_dm2_h"] < 15.0
        assert mb.release_flux(0.0)["Fe_ug_dm2_h"] == 0.0


class TestM6Closure:
    def test_m6_reduces_to_m4_as_cx_vanishes(self):
        """M6 with C_x driven to zero must reproduce the M4 fit."""
        d = build_fit_data(load_data()[0])
        _, p4, chi4, _, _ = fit_pdm(d, 4)

        n = 4 + len(bf.VARIANTS[6])
        x = np.empty(n)
        x[0], x[1] = np.log(p4.A_bl), np.log(-p4.b3)
        x[2], x[3] = np.log(p4.PBR_eff), np.log(p4.L0)
        x[4] = np.log(p4.L_ol0)
        x[5] = np.log(1e-12)  # C_x -> 0
        chi2 = float(np.sum(bf.residuals(x, d, 6, use_priors=False) ** 2))
        assert chi2 == pytest.approx(chi4, rel=1e-9)
        assert unpack(x, 6).C_x == pytest.approx(0.0, abs=1e-11)

    def test_m6_is_at_least_as_good_as_m4(self):
        """M6 nests M4, so its optimum cannot be worse."""
        d = build_fit_data(load_data()[0])
        _, _, chi4, _, _ = fit_pdm(d, 4)
        _, _, chi6, _, _ = fit_pdm(d, 6)
        assert chi6 <= chi4 + 1e-9

    def test_pbr_eff_is_unconstrained_once_cx_is_free(self):
        """The paper's central claim: 3 outer-layer points cannot pin PBR_eff.

        The outer-layer solution L_ol = L_ol0 + PBR_eff*(L_bl - L0) + C_x*t has
        three parameters against three observations, so imposing any PBR_eff
        over a wide range costs essentially nothing.
        """
        from htw_pdm.pbr_closure import fit_at_fixed_pbr

        d = build_fit_data(load_data()[0])
        _, chi_lo = fit_at_fixed_pbr(d, 1.0)
        _, chi_hi = fit_at_fixed_pbr(d, 4.0)
        _, _, chi4, _, _ = fit_pdm(d, 4)
        # A factor of 4 in PBR_eff must cost less than 1 unit of chi2.
        assert abs(chi_hi - chi_lo) < 1.0
        assert chi_lo < chi4 + 1.0 and chi_hi < chi4 + 1.0

    def test_mass_balance_value_is_admissible(self):
        """Imposing the stoichiometric ratio must not degrade the fit."""
        from htw_pdm.pbr_closure import fit_at_fixed_pbr

        d = build_fit_data(load_data()[0])
        _, _, chi4, _, _ = fit_pdm(d, 4)
        chromite = next(p for p in mb.BARRIER_CANDIDATES if p.name == "FeCr2O4")
        p, chi2 = fit_at_fixed_pbr(d, mb.production_ratio(chromite))
        assert chi2 <= chi4  # it actually improves
        assert p.C_x < 0.0  # paid for by outer-layer loss

    def test_fitted_value_lies_inside_the_stoichiometric_range(self):
        """The other half of the resolution: the bound is not a single number."""
        d = build_fit_data(load_data()[0])
        _, p4, _, _, _ = fit_pdm(d, 4)
        lo, hi = mb.summary()["R_prod_range"]
        assert lo < p4.PBR_eff < hi
