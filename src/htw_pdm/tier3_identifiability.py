"""T3-A -- synthetic identifiability gate for the Ni-exclusion closure family.

TIER3_PLAN.md T3-A: before fitting either candidate closure (M-A flux-coupled
mobility, M-B finite-capacity trapping) to the real Veile Ni observables,
prove the new parameter is recoverable from data of the same shape and noise
as the real experiment (4 observables -- 3 zone widths + the 72 h amplitude,
population-SD weighted) -- the T2-A/M3-trap-avoidance discipline: never fit a
parameter that has not been shown recoverable.

Procedure per closure (D0, r_Ni, phi_ol fixed at the T2-E fitted optimum,
IMPL_REPORT.md T2-E):
  1. pick a truth for the new parameter via a grid search that minimizes the
     width trajectory's coefficient of variation -- the "data-relevant
     regime" the plan calls for (T2-E's own const closure gives monotonically
     growing widths; the whole point of Tier 3 is testing whether a closure
     CAN go flat/saturating, so the gate should exercise that regime);
  2. draw N_REPLICATES synthetic experiments: W_obs, A72_obs from the truth
     model plus the SAME population-SD noise scale as the real Veile data
     (population SD already encodes scan-to-scan physical heterogeneity per
     the T2-A finding -- these are already-aggregated summary observables,
     not raw line-scan points, so no separate heterogeneity draw is needed);
  3. refit each replicate with the same multi-start discipline as T2-E;
     report replicate mean/SD -> verdict.

Acceptance (TIER3_PLAN.md T3-A): new-parameter replicate rel. SD <= 30% AND
the base parameters (D0, r_Ni, phi_ol) do not degrade beyond 2x their T2-E
profile-likelihood 1-sigma half-widths (T2E_PROFILE_HALFWIDTH below, from the
IMPL_REPORT.md T2-E profile-likelihood scan). If BOTH closures fail -> re-scope
Tier 3 to forward predict-and-compare (a valid, documented outcome, not a bug
to route around).

Run:  python -m htw_pdm.tier3_identifiability 
"""

from __future__ import annotations

import os

# Each Radau solve is a tiny (150-node) dense linear system -- BLAS threading
# adds pure oversubscription overhead here (verified empirically: ~20x wall
# time with default threading vs single-threaded). Parallelism instead comes
# from running independent replicates as separate single-threaded processes.
# Must be set BEFORE numpy/scipy import -- BLAS backends read these at load.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from htw_pdm.baseline_fit import load_data  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402
from htw_pdm.tier3_wagner import (  # noqa: E402
    A72_SIG,
    BOUNDS_BY_CLOSURE,
    PARAMS_BY_CLOSURE,
    T_OBS,
    model,
)

from htw_pdm.paths import OUTPUTS as OUT  # noqa: E402
OUT.mkdir(exist_ok=True)

# T2-E fitted optimum (D_Ni_eff, r_Ni, phi_ol_supply) -- IMPL_REPORT.md T2-E.
BASE = np.array([3.8, 0.29, 0.57])

# T2-E profile-likelihood 1-sigma HALF-WIDTHS, from a fresh rerun of
# tier2_inverse.py's const-closure fit (chi2/dof = 25.71/1, matching
# IMPL_REPORT.md T2-E exactly) plus a refined profile scan around the
# optimum (the T2-E script's own 20-pt wide-range grid gives degenerate
# single-point intervals for D_Ni_eff/r_Ni -- too coarse to resolve how
# narrow the minimum actually is; phi_ol_supply's refined interval
# [0.521, 0.926] matches the report's quoted [0.53, 0.90] closely).
# Used to calibrate the "does not degrade beyond 2x" acceptance criterion.
T2E_PROFILE_HALFWIDTH = {
    "D_Ni_eff_nm2_h": 0.6074,
    "r_Ni": 0.08038,
    "phi_ol_supply": 0.35242,
}

NEW_PARAM_GRID = {
    "flux": np.linspace(-2.0, 2.0, 41),
    # S_cap: the rejection flux integrates to O(1e3-1e4) nm*wt%*h over 480 h
    # at the T2-E base kinetics (checked empirically) -- S_cap <~ 500 (the
    # first grid tried) saturates the trap within hours, giving W = 0 at
    # every observation time (rejected by pick_truth's viability filter,
    # hence the original NO-GO-by-crash). The flattest width trajectory
    # (CV = 0.136) sits near S_cap ~ 1e3; 1e5+ is indistinguishable from
    # the "const" limit (matches REGRESSION_LIMIT["trap"] = 1e13 recovering
    # "const" exactly).
    "trap": np.geomspace(2e2, 2e4, 41),
}
MULTISTART_NEW = {
    "flux": [-1.0, 0.0, 1.0],
    "trap": [3e2, 1.5e3, 8e3],
}
# Anchored multi-start: (D0, r0, phi0) base combos -- the T2-E fit itself
# (base params are already tightly constrained; T2-E's own multi-start only
# needed to escape the phi=0 local minimum, not explore D0/r0 broadly) plus
# 4 corner off-starts to catch analogous local minima in the extended
# 4-parameter system. Full 3x3x3x3=81-start grid was measured impractical
# (hours single-threaded); this 5-combo anchor set was validated against it
# on the noiseless-recovery check (test_tier3.py) with matching chi2~0 result.
MULTISTART_BASE = [
    (3.85, 0.29, 0.57),   # T2-E optimum
    (1.5, 0.10, 0.30),
    (8.0, 0.60, 0.85),
    (1.5, 0.60, 0.85),
    (8.0, 0.10, 0.30),
]
N_REPLICATES = 20
N_WORKERS = min(16, os.cpu_count() or 4)


def flatness(W: np.ndarray) -> float:
    """Coefficient of variation of the width trajectory (lower = flatter)."""
    return float(np.std(W) / np.mean(W))


def pick_truth(closure: str, p1: Parameters):
    """New-parameter value giving the flattest width trajectory at the T2-E
    base kinetics -- the qualitative regime the flat/saturating real data is in."""
    best = None
    for g in NEW_PARAM_GRID[closure]:
        theta = np.concatenate([BASE, [g]])
        try:
            W, A72 = model(theta, p1, closure)
        except RuntimeError:
            continue
        # Require a non-trivial zone at every observation time (W.min() > 3 nm)
        # -- otherwise the search can pick a degenerate "trap saturates
        # immediately, zone never forms" truth, which is flat only because
        # it is zero, not because it reproduces the real flat-nonzero data.
        if not np.all(np.isfinite(W)) or np.any(W <= 3.0):
            continue
        f = flatness(W)
        if best is None or f < best[0]:
            best = (f, g, W, A72)
    if best is None:
        raise RuntimeError(f"no viable truth found for closure {closure!r}")
    return best[1], best[2], best[3]


def multistart_fit(closure, p1, W_obs, W_sig, A72_obs, new_grid):
    def res(theta):
        W_fit, A72_fit = model(theta, p1, closure)
        return np.concatenate([(W_fit - W_obs) / W_sig,
                                [(A72_fit - A72_obs) / A72_SIG]])

    lo, hi = BOUNDS_BY_CLOSURE[closure]
    best = None
    for D0, r0, phi0 in MULTISTART_BASE:
        for g0 in new_grid:
            x0 = np.clip([D0, r0, phi0, g0], lo, hi)
            # max_nfev bounds worst-case cost per start (some far-from-optimum
            # starts otherwise take 50-100+ Radau solves to converge; measured
            # empirically -- 300 covers every observed multi-start in practice).
            s = least_squares(res, x0, bounds=(lo, hi), max_nfev=300)
            c = float(np.sum(s.fun**2))
            if best is None or c < best[1]:
                best = (s, c)
    return best[0]


def _fit_one_replicate(args):
    closure, p1, W_truth, A72_truth, W_sig, seed = args
    rng = np.random.default_rng(seed)
    W_obs = np.clip(W_truth + rng.normal(0.0, W_sig), 1.0, None)
    A72_obs = A72_truth + rng.normal(0.0, A72_SIG)
    sol = multistart_fit(closure, p1, W_obs, W_sig, A72_obs,
                         MULTISTART_NEW[closure])
    return sol.x


def run_gate(closure: str, p1: Parameters, W_sig: np.ndarray, seed0: int = 0):
    g_truth, W_truth, A72_truth = pick_truth(closure, p1)
    theta_truth = np.concatenate([BASE, [g_truth]])
    print(f"\n=== closure={closure}: truth new-param = {g_truth:.4g}, "
          f"widths {np.round(W_truth, 1)}, A72={A72_truth:.1f} "
          f"(CV={flatness(W_truth):.2f}) ===", flush=True)

    names = PARAMS_BY_CLOSURE[closure]
    jobs = [(closure, p1, W_truth, A72_truth, W_sig, seed0 + rep)
            for rep in range(N_REPLICATES)]
    with Pool(N_WORKERS) as pool:
        results = pool.map(_fit_one_replicate, jobs)
    recovered = np.array(results)
    return names, theta_truth, recovered


def main():
    p1 = Parameters()
    means, _ = load_data()
    ni = {r[0]: (r[1], r[2], r[3]) for r in means["Ni"]}
    W_sig = np.array([ni[t][1] for t in T_OBS])
    print(f"Population-SD noise model (real Veile Ni data): W_sig={W_sig}, "
          f"A72_sig={A72_SIG}")

    halfwidth = T2E_PROFILE_HALFWIDTH
    if halfwidth is None:
        raise RuntimeError(
            "T2E_PROFILE_HALFWIDTH is unset -- patch it from a fresh "
            "tier2_inverse.py profile-likelihood rerun before running the gate "
            "(see TIER3_PLAN.md T3-A acceptance criterion)."
        )

    all_rows = []
    verdicts = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for closure_idx, (ax, closure) in enumerate(zip(axes, ("flux", "trap"))):
        names, theta_truth, recovered = run_gate(
            closure, p1, W_sig, seed0=1000 * (closure_idx + 1))
        i_new = len(names) - 1
        new_ok = None
        base_ok = True
        for i, name in enumerate(names):
            arr = recovered[:, i]
            mean, sd = float(arr.mean()), float(arr.std())
            rel_sd = sd / max(abs(theta_truth[i]), 1e-6)
            if i == i_new:
                new_ok = rel_sd <= 0.30
                tag = f"rel_sd<=30%: {'PASS' if new_ok else 'FAIL'}"
            else:
                thresh = 2.0 * halfwidth[name]
                base_ok &= sd <= thresh
                tag = f"sd<=2x profile HW ({thresh:.3g}): " \
                      f"{'PASS' if sd <= thresh else 'FAIL'}"
            print(f"  [{closure}] {name:16s}: truth {theta_truth[i]:.4g}  "
                  f"repl mean {mean:.4g} +- {sd:.4g} (rel {rel_sd:.0%})  {tag}")
            all_rows.append([closure, name, theta_truth[i], mean, sd, rel_sd])
        verdicts[closure] = "GO" if (new_ok and base_ok) else "NO-GO"
        print(f"  [{closure}] verdict -> {verdicts[closure]}")

        ax.boxplot([recovered[:, i_new]], tick_labels=[names[i_new]])
        ax.axhline(theta_truth[i_new], color="k", ls="--", lw=1)
        ax.set_title(f"{closure}: {names[i_new]} recovery ({N_REPLICATES} reps)")

    plt.tight_layout()
    plt.savefig(OUT / "plot_tier3_identifiability.png", dpi=150)
    print(f"\nSaved: {OUT / 'plot_tier3_identifiability.png'}")

    with open(OUT / "tier3_identifiability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["closure", "param", "truth", "repl_mean", "repl_sd", "rel_sd"])
        w.writerows(all_rows)
    print(f"Saved: {OUT / 'tier3_identifiability.csv'}")

    overall = "GO" if any(v == "GO" for v in verdicts.values()) else "NO-GO"
    print(f"\nT3-A verdict per closure: {verdicts} -> overall {overall} "
          f"({'proceed to T3-B/C with the passing closure(s)' if overall == 'GO' else 'both closures fail the gate -- re-scope Tier 3 to forward predict-and-compare'})")
    with open(OUT / "tier3_identifiability_verdict.json", "w") as f:
        json.dump({"verdicts": verdicts, "overall": overall}, f, indent=1)


if __name__ == "__main__":
    main()
