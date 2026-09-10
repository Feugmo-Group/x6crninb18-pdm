"""The nickel-closure gate regression tests for the shared Wagner closure engine (nickel_wagner.py).

Run from the example root:  python -m pytest tests/ -q
"""


import numpy as np
import pytest

from htw_pdm.nickel_wagner import (  # noqa: E402
    BOUNDS_BY_CLOSURE,
    PARAMS_BY_CLOSURE,
    REGRESSION_LIMIT,
    SOLVE_ATOL,
    SOLVE_RTOL,
    T_OBS,
    X0_BY_CLOSURE,
    model,
    residuals,
    wagner_solve,
)
from htw_pdm.physics import Parameters  # noqa: E402

P1 = Parameters()
THETA_CONST = np.array([3.8, 0.29, 0.57])  # the spatial inverse fitted optimum (IMPL_REPORT the spatial inverse)


def test_const_matches_spatial_inverse():
    """closure='const' must reproduce spatial_inverse.py's committed engine exactly."""
    from htw_pdm.spatial_inverse import wagner_solve as t2_wagner_solve

    c3, xi3 = wagner_solve(THETA_CONST, P1, closure="const")
    c2, xi2 = t2_wagner_solve(THETA_CONST, P1)
    np.testing.assert_allclose(xi3, xi2)
    np.testing.assert_allclose(c3, c2, rtol=0, atol=1e-10)


@pytest.mark.parametrize("closure", ["flux", "trap"])
def test_regression_limit_matches_const(closure):
    """New-parameter regression limit must reproduce 'const'.

    flux (n_flux = 0) makes D(t) = D0 identically and integrates the *same*
    system as 'const', so it agrees to roundoff.  trap integrates an augmented
    system (the extra trap-fill state), so Radau selects different steps and
    The two solutions land a tolerance-sized distance apart -- a gap that does
    not shrink as S_cap grows.  Its bar is therefore the integrator's own
    bound, rtol * max|c|, not a hardcoded number: the previous fixed 5e-8 was
    just the value this happened to take at the old M4 kinetics and went stale
    The moment PBR_eff moved.  The genuine trap physics -- deviation O(1/S_cap)
    -- is asserted separately below.
    """
    theta_new = np.concatenate([THETA_CONST, [REGRESSION_LIMIT[closure]]])
    c_new, xi_new = wagner_solve(theta_new, P1, closure=closure)
    c_const, xi_const = wagner_solve(THETA_CONST, P1, closure="const")
    np.testing.assert_allclose(xi_new, xi_const)

    dev = np.max(np.abs(c_new - c_const))
    if closure == "flux":
        assert dev < 1e-8
    else:
        assert dev < SOLVE_RTOL * np.max(np.abs(c_const)) + SOLVE_ATOL


def test_trap_deviation_is_first_order_in_capacity():
    """The trap closure's departure from 'const' must decay as 1/S_cap.

    This is the claim that makes S_cap -> infinity a true regression limit, and
    it is the part test_regression_limit_matches_const cannot see (there the
    S_cap-dependent term is already buried under the integrator floor).  Taking
    The deviation at a large sentinel as the floor, the *excess* over that floor
    must drop by ~10x per decade of S_cap.
    """
    c_const, _ = wagner_solve(THETA_CONST, P1, closure="const")

    def dev(S_cap):
        c, _ = wagner_solve(np.concatenate([THETA_CONST, [S_cap]]), P1,
                            closure="trap")
        return np.max(np.abs(c - c_const))

    floor = dev(1e17)  # solver-step difference only; S_cap term is ~1e-13 here
    excess = np.array([dev(s) - floor for s in (1e12, 1e13, 1e14)])
    assert np.all(excess > 0)
    ratios = excess[:-1] / excess[1:]
    assert np.all(ratios > 5.0), f"not first order in 1/S_cap: {ratios}"


@pytest.mark.parametrize("closure", ["const", "flux", "trap"])
def test_model_shapes(closure):
    theta = X0_BY_CLOSURE[closure].copy()
    W, A72 = model(theta, P1, closure)
    assert W.shape == T_OBS.shape
    assert np.all(np.isfinite(W))
    assert np.isfinite(A72)


@pytest.mark.parametrize("closure", ["flux", "trap"])
def test_noiseless_synthetic_recovery(closure):
    """Noiseless synthetic data must be fittable to near-zero chi2 by SOME
    point found via multi-start (catches implementation bugs -- forward model
    inconsistency, broken residuals/bounds -- before the noisy replicate gate,
    The nickel-closure gate). This does NOT assert the recovered point equals the truth: with 4
    observables and 4 free parameters a noiseless multi-start can land on a
    different exact root (a genuine near-degenerate ridge, not a bug -- see
    the 'trap' case checked at src/nickel_identifiability.py time). Uniqueness
    under REALISTIC noise is what the identifiability gate itself tests.
    """
    from scipy.optimize import least_squares

    truth = {
        "flux": np.array([3.8, 0.29, 0.57, 0.8]),
        "trap": np.array([3.8, 0.29, 0.57, 40.0]),
    }[closure]
    W_obs, A72_obs = model(truth, P1, closure)
    # Realistic-scale weighting (real Veile population SDs), not an
    # artificially tight sigma -- tighter than the ~2 nm width-grid
    # resolution makes the objective non-smooth and breaks gradient descent.
    W_sig = np.array([3.4, 11.1, 5.0])

    def res(theta):
        r = residuals(theta, P1, closure, W_obs, W_sig)
        W_fit, A72_fit = model(theta, P1, closure)
        r[-1] = (A72_fit - A72_obs) / 7.0
        return r

    lo, hi = BOUNDS_BY_CLOSURE[closure]
    best = None
    for scale in (0.7, 1.0, 1.3):
        x0 = np.clip(truth * scale, lo, hi)
        s = least_squares(res, x0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14)
        c = float(np.sum(s.fun**2))
        if best is None or c < best[1]:
            best = (s, c)
    sol, chi2 = best
    assert chi2 < 1e-4, f"multi-start failed to drive chi2 to ~0 (got {chi2:.3g})"
    W_fit, A72_fit = model(sol.x, P1, closure)
    np.testing.assert_allclose(W_fit, W_obs, atol=1e-3)
    np.testing.assert_allclose(A72_fit, A72_obs, atol=1e-3)


def test_param_names_and_bounds_consistent():
    for closure in ("const", "flux", "trap"):
        n = len(PARAMS_BY_CLOSURE[closure])
        assert len(BOUNDS_BY_CLOSURE[closure][0]) == n
        assert len(BOUNDS_BY_CLOSURE[closure][1]) == n
        assert len(X0_BY_CLOSURE[closure]) == n
