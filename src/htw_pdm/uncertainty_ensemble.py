"""Phase D3 — seed ensemble for parameter uncertainty on the real Veile data.

Parametric bootstrap: resample the Veile means within their own reported SDs (the
sampling distribution implied by the data), refit the hard-mode NSEM inverse (fast:
kinetics-only differentiable RK4, no field networks) from each resampled draw, and
report the ensemble spread of the recovered M4 parameters. This is the "posterior/
ensemble spread" the plan (PLAN_PDM_X6CrNiNb18.md Phase D2/D3) asks to compare against
The B3 deterministic profile-likelihood intervals (baseline_fit.py).

Run:  python -m htw_pdm.uncertainty_ensemble [n_ensemble]
"""

from __future__ import annotations

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from htw_pdm.baseline_fit import (  # noqa: E402
    build_fit_data,
    fit_pdm,
    load_data,
    profile_likelihood,
)
from htw_pdm.inverse_trainer import KineticParams, integrate_hard  # noqa: E402
from htw_pdm.paths import OUTPUTS as OUT_DIR  # noqa: E402
from htw_pdm.physics import NondimGroups, Parameters, data_loss  # noqa: E402

OUT_DIR.mkdir(exist_ok=True)

PARAM_NAMES = ["A_bl", "b3", "PBR_eff", "L0", "L_ol0"]
DTYPE = torch.float64

# M4 reference (Phase B3 deterministic fit) — center of the resampling and the
# initialization for every ensemble member's optimizer.
P_REF = Parameters(A_bl=0.7269, b3=-0.01249, C_bl=0.0, PBR_eff=1.096, L0=1.924, L_ol0=115.6)


def build_time_grid_simple(Nt: int, alpha_t: float) -> torch.Tensor:
    """log-clustered nodes on tau in [0, 1], matching src/trainer.py's mapping_t=log."""
    xi = torch.linspace(-1.0, 1.0, Nt, dtype=DTYPE)
    tau = torch.expm1(alpha_t * (xi + 1.0) / 2.0) / math.expm1(alpha_t)
    return tau


def fit_one(t_cr, L_cr, s_cr, t_fe, L_fe, s_fe, g0: NondimGroups, t_grid: torch.Tensor,
            n_adam: int = 500):
    kin = KineticParams(g0, learn_Chat=False, init_scale=1.0, dtype=DTYPE)
    # Mirrors baseline_fit.PRIORS: L0 only. The PBR_eff prior was removed there
    # (its 1.05 center fails a chromium mass balance), so it must be absent here
    # too, or the 50-member PINN bootstrap stops being comparable with the
    # 1000-member deterministic one.
    prior_lam0 = (math.log(2.0 / 100.0), 0.60)

    def loss():
        lam_bl, lam_ol = integrate_hard(kin, g0, t_grid)
        d = (
            data_loss(lam_bl, t_grid, t_cr, L_cr, s_cr) * len(t_cr)
            + data_loss(lam_ol, t_grid, t_fe, L_fe, s_fe) * len(t_fe)
        )
        prior = ((kin.log_lam0 - prior_lam0[0]) / prior_lam0[1]) ** 2
        return d + prior

    opt = torch.optim.Adam(kin.parameters(), lr=1e-2)
    for _ in range(n_adam):
        opt.zero_grad()
        loss().backward()
        opt.step()
    lbfgs = torch.optim.LBFGS(kin.parameters(), max_iter=100, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-13, tolerance_change=1e-15)

    def closure():
        lbfgs.zero_grad()
        l = loss()
        l.backward()
        return l

    lbfgs.step(closure)
    p_fit = kin.to_groups(g0).to_parameters()
    return {n: getattr(p_fit, n) for n in PARAM_NAMES}


def main(n_ensemble: int = 50):
    means, _ = load_data()
    d = build_fit_data(means)
    g0 = NondimGroups.from_parameters(P_REF)
    t_grid = build_time_grid_simple(Nt=24, alpha_t=1.5)

    t_cr = torch.tensor(d.t, dtype=DTYPE)
    t_fe = torch.tensor(d.t, dtype=DTYPE)
    s_cr = torch.tensor(d.sig_cr, dtype=DTYPE)
    s_fe = torch.tensor(d.sig_fe, dtype=DTYPE)

    rng = np.random.default_rng(0)
    results = {n: [] for n in PARAM_NAMES}
    print(f"Running {n_ensemble}-member parametric bootstrap (hard-mode NSEM inverse)...")
    for i in range(n_ensemble):
        L_cr_i = torch.tensor(d.L_cr + rng.normal(0.0, d.sig_cr), dtype=DTYPE)
        L_fe_i = torch.tensor(d.L_fe + rng.normal(0.0, d.sig_fe), dtype=DTYPE)
        fit = fit_one(t_cr, L_cr_i, s_cr, t_fe, L_fe_i, s_fe, g0, t_grid)
        for n in PARAM_NAMES:
            results[n].append(fit[n])
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_ensemble} done")

    # Profile-likelihood 1-sigma intervals on the actual (unresampled) data fit, for
    # comparison. M4 variant = 4 in baseline_fit's VARIANTS; params in the same order
    # as PARAM_NAMES plus L_ol0 last (param_names(4) = [A_bl, -b3, PBR_eff, L0, L_ol0]).
    sol4, p4, chi2_4, dof_4, _ = fit_pdm(d, 4)
    prof_intervals = {}
    for i, name in enumerate(["A_bl", "b3", "PBR_eff", "L0", "L_ol0"]):
        grid, chi2s = profile_likelihood(d, 4, sol4.x, i)
        dchi = chi2s - chi2s.min()
        inside = grid[dchi <= 1.0]
        lo, hi = np.exp(inside.min()), np.exp(inside.max())
        if name == "b3":
            lo, hi = -hi, -lo  # profile_likelihood scans -b3 (positive log-parameter)
        prof_intervals[name] = (lo, hi)

    print("\n" + "=" * 78)
    print(f"{'param':10s} {'ensemble mean':>14s} {'ensemble std':>13s} "
          f"{'ens [16,84]%':>22s} {'profile 1-sigma':>22s}")
    summary_rows = []
    for name in PARAM_NAMES:
        arr = np.array(results[name])
        lo_e, hi_e = np.percentile(arr, [16, 84])
        lo_p, hi_p = prof_intervals[name]
        print(f"{name:10s} {arr.mean():14.5g} {arr.std():13.3g} "
              f"[{lo_e:9.4g}, {hi_e:9.4g}]  [{lo_p:9.4g}, {hi_p:9.4g}]")
        summary_rows.append((name, getattr(P_REF, name), arr.mean(), arr.std(),
                             lo_e, hi_e, lo_p, hi_p))

    # Machine-readable outputs for the paper tables (member-level + summary).
    import csv

    with open(OUT_DIR / "ensemble_members.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PARAM_NAMES)
        w.writerows(zip(*(results[n] for n in PARAM_NAMES)))
    with open(OUT_DIR / "ensemble_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "m4_fit", "ens_mean", "ens_std",
                    "ens_p16", "ens_p84", "profile_1sig_lo", "profile_1sig_hi"])
        w.writerows(summary_rows)
    print(f"Saved: {OUT_DIR / 'ensemble_members.csv'}, {OUT_DIR / 'ensemble_summary.csv'}")

    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(4 * len(PARAM_NAMES), 4))
    fig.suptitle(
        f"D3 ensemble ({n_ensemble} bootstrap resamples) vs B3 profile likelihood",
        fontsize=12,
    )
    for ax, name in zip(axes, PARAM_NAMES):
        arr = np.array(results[name])
        ax.hist(arr, bins=15, color="steelblue", alpha=0.7, density=True)
        lo_p, hi_p = prof_intervals[name]
        ax.axvspan(lo_p, hi_p, color="red", alpha=0.15, label="profile 1-sigma")
        ax.axvline(getattr(P_REF, name), color="k", ls="--", lw=1, label="M4 (real data)")
        ax.set_title(name)
        ax.legend(fontsize=7)
    plt.tight_layout()
    out = OUT_DIR / "plot_uncertainty_ensemble.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    main(n)
