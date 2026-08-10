"""Tier-4 T4-B — joint moving-boundary inverse: defect fields and parameters together.

Tier 2 solves the Landau-transformed transient problem *forward*, with every
transport constant assumed. This module inverts it: the space-time defect field
and the transport parameters are recovered in one optimization.

Why this is the case that needs a neural solver. The zero-dimensional kinetics
have a closed form, so Tier 1 can build a hard-mode inversion in which the physics
is exact by construction and the parameters are the only unknowns. No such
construction exists here: the state is a field on a domain whose boundary moves
with the solution, so a classical inverse must run a full forward PDE solve inside
every optimizer iteration. The soft-mode neural inversion instead carries the field
and the parameters as one set of unknowns and solves once.

That is also exactly the mode that failed in Tier 1 (failure mode F-1: the data
loss goes to zero while the governing equation is left unsatisfied). So the point
of this module is not only to show the inverse works, but to run the F-1
self-consistency check in the setting the check was designed for -- one where no
hard-mode alternative exists to fall back on.

The independent forward solve required by that check is provided here as a
classical Crank-Nicolson stepper (`cn_forward`). The Landau-transformed operator
is linear in the field at fixed parameters, so each step is a single linear solve;
this is a genuinely independent implementation, sharing no code path with the
neural residual assembly beyond the physics module's group definitions.

Inverted parameters (per species, in log space so they stay positive):
    kappa  -- the interfacial Robin ratio at the annihilation boundary
    mig    -- the migration prefactor t_c/tau_mig, which is proportional to the
              defect diffusivity D. The Tier-2 structural result showed steady
              profile shapes carry no diffusivity information; the transient
              problem is where that information lives, so recovering `mig` from
              transient data is the direct test of that claim.

Run:
    python -m htw_pdm.tier4_transient_inverse
    python -m htw_pdm.tier4_transient_inverse --smoke
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import replace

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from htw_pdm.physics import Parameters  # noqa: E402
from htw_pdm.tier2_physics import (  # noqa: E402
    TransientGroups,
    steady_at,
    transient_groups,
    transient_residuals,
)

from physicsnemo.experimental.models.scen import DVRMapper  # noqa: E402
from physicsnemo.optim import TwoPhaseOptimizer, build_aggregator  # noqa: E402
from htw_pdm.paths import PAPER_OUT  # noqa: E402

T_C_H = 480.0
NX, NT = 24, 16
TAU0 = 0.05
SPECIES = ("OV", "CV")

# Ground truth for the synthetic experiment, as multiplicative factors on the
# Tier-2 assumed values. Deliberately off 1.0 so a fit that simply keeps its
# initial guess cannot be mistaken for a successful recovery.
TRUTH = {"OV": {"kappa": 1.40, "mig": 0.60}, "CV": {"kappa": 0.75, "mig": 1.80}}
NOISE_REL = 0.02
N_OBS_XI, N_OBS_TAU = 6, 5


def scaled_groups(tg: TransientGroups, kappa_f, mig_f) -> TransientGroups:
    """Apply multiplicative factors to the two inverted groups.

    Accepts floats or 0-d tensors, so the same helper serves the classical
    forward solve and the differentiable residual assembly.
    """
    return replace(tg, kappa=tg.kappa * kappa_f, mig=tg.mig * mig_f)


def cn_forward(tg: TransientGroups, D1x: torch.Tensor, t_grid: torch.Tensor,
               xi: torch.Tensor) -> torch.Tensor:
    """Classical Crank-Nicolson march of the Landau-transformed transient problem.

    Independent of the neural path: assembles and solves the linear system at each
    step directly. Returns chat on the (Nt, Nx) grid.

    Step operator, from tier2_physics.transient_residuals:
        dchat/dtau = A_j chat,
        A_j = diag(xi*adv_j) D1x - mig_j D1x ( -D1x/Pe_j + s I )
    with the two boundary rows replaced by the flux BCs at the new time level.
    """
    Nt, Nx = len(t_grid), len(xi)
    dtype = D1x.dtype
    I = torch.eye(Nx, dtype=dtype)
    i_gen = 0 if tg.xi_gen == 0.0 else -1
    i_ann = -1 if tg.xi_gen == 0.0 else 0

    def A_of(j):
        Jop = -D1x / tg.Pe[j] + tg.s * I
        return torch.diag(xi * tg.adv[j]) @ D1x - tg.mig[j] * (D1x @ Jop)

    chat = torch.zeros(Nt, Nx, dtype=dtype)
    chat[0] = steady_at(tg, 0, xi)
    for j in range(Nt - 1):
        dt = t_grid[j + 1] - t_grid[j]
        M = I - 0.5 * dt * A_of(j + 1)
        b = (I + 0.5 * dt * A_of(j)) @ chat[j]
        Jop_n = -D1x / tg.Pe[j + 1] + tg.s * I
        M[i_gen] = Jop_n[i_gen]
        b[i_gen] = tg.s * tg.jhat[j + 1]
        M[i_ann] = Jop_n[i_ann] - tg.s * tg.kappa * I[i_ann]
        b[i_ann] = torch.zeros((), dtype=dtype)
        chat[j + 1] = torch.linalg.solve(M, b)
    return chat


def make_observations(groups: dict, D1x, t_grid, xi, seed: int = 0):
    """Synthetic composition observations from the truth parameters.

    Sampled sparsely in space and time, at the density a depth-profile campaign
    could realistically deliver, with 2 % relative noise.
    """
    rng = np.random.default_rng(seed)
    i_xi = np.linspace(0, len(xi) - 1, N_OBS_XI).round().astype(int)
    i_t = np.linspace(1, len(t_grid) - 1, N_OBS_TAU).round().astype(int)
    obs = {}
    for name in SPECIES:
        tg_true = scaled_groups(groups[name], TRUTH[name]["kappa"],
                                TRUTH[name]["mig"])
        truth = cn_forward(tg_true, D1x, t_grid, xi)
        sub = truth[np.ix_(i_t, i_xi)]
        sig = NOISE_REL * sub.abs().clamp_min(1e-12)
        noisy = sub + torch.tensor(rng.normal(size=tuple(sub.shape)),
                                   dtype=sub.dtype) * sig
        obs[name] = {"i_xi": i_xi, "i_t": i_t, "y": noisy, "sig": sig,
                     "truth_field": truth}
    return obs


class SpaceTimeNet(nn.Module):
    """(xi, tau) -> chat, positive by construction."""

    def __init__(self, hidden: int = 32, n_layers: int = 3, dtype=torch.float64):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.to(dtype)

    def forward(self, grid):
        return torch.nn.functional.softplus(self.net(grid).squeeze(-1)) + 1e-8


def chi2_at(field: torch.Tensor, ob: dict) -> float:
    """Weighted chi^2 of a field against the observations."""
    pred = field[np.ix_(ob["i_t"], ob["i_xi"])]
    return float((((pred - ob["y"]) / ob["sig"]) ** 2).sum())


def invert_species(name, tg, obs, D1x, t_grid, xi, grid, n_adam, seed=0) -> dict:
    """Soft-mode joint inversion of field + (kappa, mig) for one species."""
    torch.manual_seed(seed)
    net = SpaceTimeNet(dtype=D1x.dtype)
    log_k = torch.zeros((), dtype=D1x.dtype, requires_grad=True)
    log_m = torch.zeros((), dtype=D1x.dtype, requires_grad=True)
    params = list(net.parameters()) + [log_k, log_m]
    ob = obs[name]
    Nt, Nx = len(t_grid), len(xi)

    def fields():
        return net(grid).view(Nt, Nx)

    # Warm start on the quasi-steady profile at the initial parameter guess.
    tgt = torch.stack([steady_at(tg, j, xi) for j in range(Nt)])
    opt_pre = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(800):
        opt_pre.zero_grad()
        loss = ((fields() - tgt) ** 2).mean()
        loss.backward()
        opt_pre.step()

    term_names = ["pde", "bc_gen", "bc_ann", "ic", "data"]
    brdr = build_aggregator("brdr", params, num_losses=len(term_names),
                            weights=[1.0] * len(term_names))
    adam = torch.optim.Adam(params, lr=1e-3)
    lbfgs = torch.optim.LBFGS(params, line_search_fn="strong_wolfe", max_iter=400,
                              tolerance_grad=1e-13, tolerance_change=1e-15)
    opt = TwoPhaseOptimizer(adam, lbfgs, aggregator=brdr)
    step_c = [0]

    def closure():
        tg_c = scaled_groups(tg, torch.exp(log_k), torch.exp(log_m))
        c = fields()
        terms = transient_residuals(c, D1x, t_grid, xi, tg_c)
        pred = c[np.ix_(ob["i_t"], ob["i_xi"])]
        terms["data"] = (((pred - ob["y"]) / ob["sig"]) ** 2).mean()
        total = brdr({n: terms[n] for n in term_names}, step_c[0])
        if brdr.training:
            step_c[0] += 1
        return total

    opt.run(closure, n_adam_steps=n_adam, n_lbfgs_steps=3, verbose=False)

    with torch.no_grad():
        kf, mf = float(torch.exp(log_k)), float(torch.exp(log_m))
        traj = fields()
        tg_hat = scaled_groups(tg, kf, mf)
        # F-1 self-consistency: re-solve the recovered parameters through the
        # independent classical stepper and compare data chi^2 with the
        # trajectory's own.
        fwd = cn_forward(tg_hat, D1x, t_grid, xi)
        chi2_traj, chi2_fwd = chi2_at(traj, ob), chi2_at(fwd, ob)
        pde_traj = transient_residuals(traj, D1x, t_grid, xi, tg_hat)
        field_gap = float((traj - fwd).abs().max() / fwd.abs().max())

    return {
        "truth": TRUTH[name],
        "recovered": {"kappa": kf, "mig": mf},
        "rel_err": {"kappa": abs(kf - TRUTH[name]["kappa"]) / TRUTH[name]["kappa"],
                    "mig": abs(mf - TRUTH[name]["mig"]) / TRUTH[name]["mig"]},
        "chi2_trajectory": chi2_traj,
        "chi2_forward_solve": chi2_fwd,
        "chi2_ratio": chi2_fwd / max(chi2_traj, 1e-30),
        "field_gap_rel_Linf": field_gap,
        "pde_residual": {k: float(v) for k, v in pde_traj.items()},
    }


def nested_loop_inverse(name, tg, obs, D1x, t_grid, xi) -> dict:
    """Control experiment: the classical inverse the neural route claims to avoid.

    Outer Levenberg-Marquardt over (kappa, mig) in log space, with a full
    `cn_forward` PDE solve inside every residual evaluation -- exactly the
    nested-loop structure a physics-informed joint inversion is supposed to make
    unnecessary. Its purpose here is diagnostic: if this recovers the truth, then
    the information is present in the data and any soft-mode failure is an
    optimization pathology rather than an identifiability limit.
    """
    from scipy.optimize import least_squares

    ob = obs[name]

    def resid(logp):
        tg_c = scaled_groups(tg, float(np.exp(logp[0])), float(np.exp(logp[1])))
        c = cn_forward(tg_c, D1x, t_grid, xi)
        pred = c[np.ix_(ob["i_t"], ob["i_xi"])]
        return (((pred - ob["y"]) / ob["sig"]).flatten()).numpy()

    n_solves = [0]

    def counted(logp):
        n_solves[0] += 1
        return resid(logp)

    sol = least_squares(counted, np.zeros(2), method="lm", xtol=1e-12, ftol=1e-12)
    kf, mf = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))
    return {
        "recovered": {"kappa": kf, "mig": mf},
        "rel_err": {"kappa": abs(kf - TRUTH[name]["kappa"]) / TRUTH[name]["kappa"],
                    "mig": abs(mf - TRUTH[name]["mig"]) / TRUTH[name]["mig"]},
        "chi2": float((sol.fun ** 2).sum()),
        "n_forward_solves": n_solves[0],
    }


def main(smoke: bool = False) -> dict:
    torch.set_default_dtype(torch.float64)
    p1 = Parameters()
    mapper = DVRMapper(NX, 0.0, 1.0, 0.0, dtype=torch.float64)
    xi, D1x = mapper.nodes, mapper.D1
    t_unit = torch.linspace(0.0, 1.0, NT, dtype=torch.float64)
    t_grid = TAU0 + (1.0 - TAU0) * t_unit
    grid = torch.stack([xi.repeat(NT), t_grid.repeat_interleave(NX)], dim=1)

    groups = {n: transient_groups(p1, n, t_grid.numpy(), T_C_H,
                                  dtype=torch.float64) for n in SPECIES}

    # B0: the classical stepper must reproduce the Tier-2 forward physics --
    # its own CN residuals, at the parameters it solved for, must be ~machine zero.
    b0 = {}
    for n in SPECIES:
        c = cn_forward(groups[n], D1x, t_grid, xi)
        r = transient_residuals(c, D1x, t_grid, xi, groups[n])
        b0[n] = {k: float(v) for k, v in r.items()}
    print("[B0] classical stepper self-residuals:",
          {n: f"{max(v.values()):.2e}" for n, v in b0.items()})

    obs = make_observations(groups, D1x, t_grid, xi)
    n_adam = 300 if smoke else 4000
    res = {"n_adam": n_adam, "truth": TRUTH, "noise_rel": NOISE_REL,
           "n_obs": N_OBS_XI * N_OBS_TAU, "B0_stepper_residuals": b0,
           "species": {}}
    for n in SPECIES:
        out = invert_species(n, groups[n], obs, D1x, t_grid, xi, grid, n_adam)
        ctl = nested_loop_inverse(n, groups[n], obs, D1x, t_grid, xi)
        out["nested_loop_control"] = ctl
        res["species"][n] = out
        print(f"[B3] {n}: nested-loop control  kappa -> {ctl['recovered']['kappa']:.3f} "
              f"({ctl['rel_err']['kappa']*100:.1f}%)  "
              f"mig -> {ctl['recovered']['mig']:.3f} "
              f"({ctl['rel_err']['mig']*100:.1f}%)  chi2 = {ctl['chi2']:.2f}  "
              f"[{ctl['n_forward_solves']} forward solves]")
        print(f"[B1] {n}: kappa {out['truth']['kappa']:.3f} -> "
              f"{out['recovered']['kappa']:.3f} "
              f"({out['rel_err']['kappa']*100:.1f}%)   "
              f"mig {out['truth']['mig']:.3f} -> {out['recovered']['mig']:.3f} "
              f"({out['rel_err']['mig']*100:.1f}%)")
        print(f"[B2] {n}: chi2 trajectory {out['chi2_trajectory']:.3f} vs "
              f"forward-solve {out['chi2_forward_solve']:.3f} "
              f"(ratio {out['chi2_ratio']:.2f}), field gap "
              f"{out['field_gap_rel_Linf']:.2e}")

    # F-1 acceptance: the paper's heuristic is that a trustworthy soft-mode fit
    # keeps the forward-solve chi^2 within a factor of a few of the trajectory's.
    res["f1_pass"] = all(
        0.2 <= res["species"][n]["chi2_ratio"] <= 5.0 for n in SPECIES
    )
    print(f"[B2] F-1 self-consistency: {'PASS' if res['f1_pass'] else 'FAIL'}")

    out_dir = str(PAPER_OUT)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tier4_transient_inverse.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"[T4-B] wrote {path}")
    return res


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
