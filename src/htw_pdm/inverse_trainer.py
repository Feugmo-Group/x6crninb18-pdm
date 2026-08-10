"""Inverse HTW_PDM trainer (NSEM): joint field + kinetic-parameter identification.

Learns the apparent PDM parameters {Ahat, bhat, PBReff, lam0, lamol0} (optionally
Chat) jointly with the two thickness networks, from layer-thickness data:

* synthetic mode (data.synthetic=true): data generated from the closed form at
  the physics.* ground truth + Gaussian noise — the recovery test that must pass
  before the real-data inversion is trusted;
* real mode: the Veile 2024 Cr (barrier layer) and Fe (outer layer) means.

Parameterization follows the rpdm_nsem discipline: positivity by construction
(log for Ahat/Chat/lam0/lamol0/PBReff, negative-log for bhat), kinetics frozen
during the field pretrain, a soft prior on lam0 as a fixed-weight loss
terms (the identifiability analysis in IMPL_REPORT.md shows 3 time points cannot
constrain L0, and PBR_eff only weakly).

Run:
    python -m htw_pdm.inverse_trainer                       # real Veile data
    python -m htw_pdm.inverse_trainer data.synthetic=true   # recovery test
"""

from __future__ import annotations

import csv
import math
import os
import sys


import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed, L_ol_closed  # noqa: E402
from htw_pdm.physics import (  # noqa: E402
    L_C_NM,
    T_C_H,
    NondimGroups,
    Parameters,
    data_loss,
    pdm_residuals,
    recovered_field_strength,
)
from htw_pdm.trainer import build_nets, build_time_grid, make_fields, pretrain_ic  # noqa: E402

from physicsnemo.optim import TwoPhaseOptimizer, build_aggregator  # noqa: E402
from physicsnemo.utils import save_checkpoint, set_default_dtype  # noqa: E402
from physicsnemo.utils.logging import PythonLogger  # noqa: E402
from htw_pdm.paths import CONF, ROOT  # noqa: E402


def integrate_hard(kin: "KineticParams", g0: NondimGroups, t_grid: torch.Tensor,
                   n_sub: int = 8):
    """Differentiable RK4 integration of the film-growth ODE (physics exact by
    construction — no residual slack). Returns (lam_bl, lam_ol) on t_grid.

    The bl ODE is integrated with n_sub RK4 sub-steps per grid interval; the ol
    solution follows exactly from the affine relation (pdm_eqns.md Section 5.2).
    """
    k = kin.kinetics()
    Ahat, bhat = k["Ahat"], k["bhat"]
    Chat = k.get("Chat", torch.zeros(()))
    lam0, lamol0, PBReff = k["lam0"], k["lamol0"], k["PBReff"]
    Cxhat = torch.as_tensor(g0.Cxhat)

    def f(lam):
        return Ahat * torch.exp(torch.clamp(bhat * lam, max=60.0)) - Chat

    lam_list = [lam0]
    lam = lam0
    for j in range(1, t_grid.numel()):
        h = (t_grid[j] - t_grid[j - 1]) / n_sub
        for _ in range(n_sub):
            k1 = f(lam)
            k2 = f(lam + 0.5 * h * k1)
            k3 = f(lam + 0.5 * h * k2)
            k4 = f(lam + h * k3)
            lam = lam + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        lam_list.append(lam)
    lam_bl = torch.stack(lam_list)
    lam_ol = PBReff * (lam_bl - lam0) + Cxhat * t_grid + lamol0
    return lam_bl, lam_ol


class KineticParams(torch.nn.Module):
    """Learnable apparent PDM parameters, positivity/sign by construction."""

    def __init__(self, g0: NondimGroups, learn_Chat: bool, init_scale: float = 1.0,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        s = init_scale

        def logp(v):
            return torch.nn.Parameter(torch.tensor(math.log(v), dtype=dtype))

        self.log_Ahat = logp(g0.Ahat * s)
        self.log_negbhat = logp(-g0.bhat * s)
        self.log_PBReff = logp(g0.PBReff)
        self.log_lam0 = logp(g0.lam0)
        self.log_lamol0 = logp(max(g0.lamol0, 1e-3))
        self.learn_Chat = learn_Chat
        if learn_Chat:
            self.log_Chat = logp(max(g0.Chat, 1e-4))

    def kinetics(self) -> dict:
        k = {
            "Ahat": torch.exp(self.log_Ahat),
            "bhat": -torch.exp(self.log_negbhat),
            "PBReff": torch.exp(self.log_PBReff),
            "lam0": torch.exp(self.log_lam0),
            "lamol0": torch.exp(self.log_lamol0),
        }
        if self.learn_Chat:
            k["Chat"] = torch.exp(self.log_Chat)
        return k

    def to_groups(self, g0: NondimGroups) -> NondimGroups:
        k = {n: float(v) for n, v in self.kinetics().items()}
        return NondimGroups(
            Ahat=k["Ahat"], bhat=k["bhat"], Chat=k.get("Chat", 0.0),
            PBReff=k["PBReff"], Cxhat=g0.Cxhat,
            lam0=k["lam0"], lamol0=k["lamol0"],
        )


def load_means(csv_path: str, dtype: torch.dtype):
    """(t, L, sigma_mean) per element from the means CSV."""
    rows = []
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    out = {}
    for el in ("Cr", "Fe"):
        sel = sorted((float(r["t_h"]), float(r["mean_nm"]),
                      float(r["sd_nm"]) / math.sqrt(int(r["n_scans"])))
                     for r in rows if r["element"] == el)
        t, L, s = (torch.tensor(v, dtype=dtype) for v in zip(*sel))
        out[el] = (t, L, s)
    return out


def make_synthetic(p_true: Parameters, t_h: np.ndarray, noise_rel: float, seed: int,
                   dtype: torch.dtype):
    """Closed-form data at the ground truth + relative Gaussian noise."""
    bp = HTWPDMParams(A_bl=p_true.A_bl, b3=p_true.b3, C_bl=p_true.C_bl,
                      PBR_eff=p_true.PBR_eff, C_x=p_true.C_x, L0=p_true.L0,
                      L_ol0=p_true.L_ol0)
    rng = np.random.default_rng(seed)
    out = {}
    for el, fn in (("Cr", L_bl_closed), ("Fe", L_ol_closed)):
        L = fn(bp, t_h)
        sigma = np.maximum(noise_rel * L, 1.0)
        Ln = L + rng.normal(0.0, sigma)
        out[el] = tuple(torch.tensor(v, dtype=dtype) for v in (t_h, Ln, sigma))
    return out


@hydra.main(config_path=str(CONF), config_name="inverse_config", version_base="1.3")
def main(cfg: DictConfig) -> dict:
    logger = PythonLogger("x6_pdm_inv")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.train.seed)
    dtype = torch.float64 if cfg.train.dtype == "float64" else torch.float32
    set_default_dtype(dtype)

    p_ref = Parameters(**{k: float(v) for k, v in cfg.physics.items()})
    g0 = NondimGroups.from_parameters(p_ref)

    # Data
    if bool(cfg.data.synthetic):
        t_h = np.array([72.0, 168.0, 480.0])
        data = make_synthetic(p_ref, t_h, float(cfg.data.noise_rel),
                              int(cfg.train.seed), dtype)
        logger.info("Synthetic data generated from physics.* ground truth")
    else:
        data = load_means(str(ROOT / str(cfg.data.csv)), dtype)
        logger.info(f"Veile 2024 means loaded from {cfg.data.csv}")
    (t_cr, L_cr, s_cr), (t_fe, L_fe, s_fe) = data["Cr"], data["Fe"]

    t_grid = build_time_grid(cfg.domain, dtype)
    Nt = int(cfg.domain.Nt)

    kin = KineticParams(g0, bool(cfg.inverse.learn_Chat),
                        float(cfg.inverse.init_scale), dtype)

    mode = str(cfg.inverse.get("mode", "hard"))
    # Only L0 carries a prior; this must mirror baseline_fit.PRIORS exactly, or
    # the neural inversion and the classical fitter stop minimizing the same
    # objective and their agreement becomes meaningless. The PBR_eff prior that
    # used to sit here was removed with it (its 1.05 center fails a chromium
    # mass balance -- see the comment in baseline_fit.PRIORS).
    prior_lam0 = (math.log(2.0 / L_C_NM), 0.6)
    lambda_prior_h = float(cfg.inverse.lambda_prior)

    if mode == "hard":
        # Physics enforced by construction: differentiable RK4 through the
        # kinetics; only the kinetic parameters are optimized. This is the
        # reference inverse mode for the 0-D model (no ODE-slack leakage; see
        # IMPL_REPORT Phase D). The soft mode below is the joint field+parameter
        # PINN needed once no integrator shortcut exists (spatial Tier-2 model).
        logger.info("Inverse mode: HARD (differentiable RK4, kinetics-only)")

        def hard_loss():
            lam_bl, lam_ol = integrate_hard(kin, g0, t_grid)
            loss = (
                data_loss(lam_bl, t_grid, t_cr, L_cr, s_cr) * len(t_cr)
                + data_loss(lam_ol, t_grid, t_fe, L_fe, s_fe) * len(t_fe)
            )
            prior = ((kin.log_lam0 - prior_lam0[0]) / prior_lam0[1]) ** 2
            return loss + lambda_prior_h * prior

        opt_h = torch.optim.Adam(kin.parameters(), lr=1e-2)
        n_adam = int(cfg.train.n_adam)
        for step in range(n_adam):
            opt_h.zero_grad()
            loss = hard_loss()
            loss.backward()
            opt_h.step()
            if step % int(cfg.output.log_every) == 0:
                logger.info(f"  hard Adam {step:5d}: loss = {loss.item():.4e}")
        lbfgs_h = torch.optim.LBFGS(
            kin.parameters(), max_iter=int(cfg.train.lbfgs_max_iter),
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-13, tolerance_change=1e-15,
        )

        def _cl():
            lbfgs_h.zero_grad()
            loss = hard_loss()
            loss.backward()
            return loss

        final = lbfgs_h.step(_cl).item()
        logger.info(f"  hard L-BFGS: loss = {final:.4e}")
        with torch.no_grad():
            lam_bl, lam_ol = integrate_hard(kin, g0, t_grid)
        history = [{"loss": final}]
        return _report(cfg, logger, kin, g0, p_ref, lam_bl, lam_ol, t_grid,
                       (t_cr, L_cr, s_cr), (t_fe, L_fe, s_fe), history,
                       models=None, hard=True)

    net_bl, net_ol = build_nets(cfg, Nt)
    fields = make_fields(net_bl, net_ol, Nt)
    field_params = list(net_bl.parameters()) + list(net_ol.parameters())
    all_params = field_params + list(kin.parameters())

    # Stage 1 — field pretrain with kinetics frozen. Targets are linear ramps
    # IC -> last data point (a constant target gives the kinetics no
    # time-variation gradient — rpdm_nsem inverse lesson).
    n_pre = int(cfg.train.n_pretrain)
    if n_pre > 0:
        logger.info(f"Pre-training fields ({n_pre} Adam + L-BFGS, kinetics frozen) ...")
        ramp = t_grid / float(t_grid[-1])
        tgt_bl = g0.lam0 + (float(L_cr[-1]) / L_C_NM - g0.lam0) * ramp
        tgt_ol = g0.lamol0 + (float(L_fe[-1]) / L_C_NM - g0.lamol0) * ramp
        pretrain_ic(fields, field_params, (tgt_bl, tgt_ol), n_pre,
                    float(cfg.train.pretrain_lr), logger)

    # Stage 2 — joint physics + data training.
    lambda_ic = float(cfg.train.lambda_ic)
    lambda_data = float(cfg.inverse.lambda_data)
    lambda_prior = float(cfg.inverse.lambda_prior)
    # Fixed multiplier on the ODE terms. Without it the joint optimum leaves
    # ~1e-3 of slack in the physics residuals and "fits" the data by bending the
    # trajectory instead of moving the kinetics (accumulated CN slack of a few
    # nm/step is enough to absorb the whole data misfit) — the recovered
    # parameters then fail the closed-form self-consistency check.
    lambda_phys = float(cfg.inverse.get("lambda_phys", 1000.0))
    term_names = ["ode_bl", "ode_ol", "ic", "data_bl", "data_ol", "prior"]
    brdr = build_aggregator("brdr", all_params, num_losses=len(term_names),
                            weights=[1.0] * len(term_names))
    adam = torch.optim.Adam(all_params, lr=float(cfg.train.adam_lr))
    lbfgs = torch.optim.LBFGS(
        all_params, line_search_fn="strong_wolfe",
        max_iter=int(cfg.train.lbfgs_max_iter),
        tolerance_grad=float(cfg.train.lbfgs_tol_grad),
        tolerance_change=float(cfg.train.lbfgs_tol_change),
    )
    opt = TwoPhaseOptimizer(adam, lbfgs, aggregator=brdr)
    brdr_step = [0]

    def closure() -> torch.Tensor:
        lam_bl, lam_ol = fields()
        kd = kin.kinetics()
        terms = pdm_residuals(lam_bl, lam_ol, t_grid, g0, kinetics=kd)
        terms["ode_bl"] = lambda_phys * terms["ode_bl"]
        terms["ode_ol"] = lambda_phys * terms["ode_ol"]
        terms["ic"] = lambda_ic * terms["ic"]
        terms["data_bl"] = lambda_data * data_loss(lam_bl, t_grid, t_cr, L_cr, s_cr)
        terms["data_ol"] = lambda_data * data_loss(lam_ol, t_grid, t_fe, L_fe, s_fe)
        terms["prior"] = lambda_prior * (
            ((kin.log_lam0 - prior_lam0[0]) / prior_lam0[1]) ** 2
        )
        total = brdr({n: terms[n] for n in term_names}, brdr_step[0])
        if brdr.training:
            brdr_step[0] += 1
        return total

    os.makedirs(cfg.output.dir, exist_ok=True)
    history = opt.run(
        closure,
        n_adam_steps=int(cfg.train.n_adam),
        n_lbfgs_steps=int(cfg.train.n_lbfgs),
        verbose=True,
        log_every=int(cfg.output.log_every),
        grad_clip=float(cfg.train.get("grad_clip", 0)) or None,
    )
    logger.info(f"Final loss: {history[-1]['loss']:.4e}")

    with torch.no_grad():
        lam_bl, lam_ol = fields()
        terms = pdm_residuals(lam_bl, lam_ol, t_grid, g0, kinetics=kin.kinetics())
    for name, val in terms.items():
        logger.info(f"  residual[{name}] = {float(val):.3e}")
    return _report(cfg, logger, kin, g0, p_ref, lam_bl, lam_ol, t_grid,
                   (t_cr, L_cr, s_cr), (t_fe, L_fe, s_fe), history,
                   models=[net_bl, net_ol], hard=False)


def _report(cfg, logger, kin, g0, p_ref, lam_bl, lam_ol, t_grid,
            cr, fe, history, models, hard: bool):
    """Recovered-parameter report, self-consistency check, checkpointing."""
    (t_cr, L_cr, s_cr), (t_fe, L_fe, s_fe) = cr, fe
    with torch.no_grad():
        p_fit = kin.to_groups(g0).to_parameters()
        chi2 = float(
            data_loss(lam_bl, t_grid, t_cr, L_cr, s_cr) * len(t_cr)
            + data_loss(lam_ol, t_grid, t_fe, L_fe, s_fe) * len(t_fe)
        )
    logger.info("Recovered apparent parameters (vs reference/ground truth):")
    for name in ("A_bl", "b3", "C_bl", "PBR_eff", "L0", "L_ol0"):
        v, r = getattr(p_fit, name), getattr(p_ref, name)
        rel = abs(v - r) / abs(r) if r != 0 else float("nan")
        logger.info(f"  {name:8s} = {v:12.5g}   ref {r:12.5g}   rel dev {rel:8.2%}")
    logger.info(f"  field strength eps_f = {recovered_field_strength(p_fit.b3):.3g} V/cm "
                f"(alpha3 = 0.12)")
    logger.info(f"  data chi2 (trajectory) = {chi2:.2f}")

    # Self-consistency: the recovered kinetics integrated EXACTLY (closed form)
    # must also fit the data. A large gap means the trajectory absorbed the
    # misfit through ODE slack and the parameters are not trustworthy (this is
    # what the hard mode eliminates by construction).
    bp_fit = HTWPDMParams(
        A_bl=p_fit.A_bl, b3=p_fit.b3, C_bl=p_fit.C_bl, PBR_eff=p_fit.PBR_eff,
        C_x=p_fit.C_x, L0=p_fit.L0, L_ol0=p_fit.L_ol0,
    )
    chi2_exact = float(
        np.sum(((L_bl_closed(bp_fit, t_cr.numpy()) - L_cr.numpy()) / s_cr.numpy()) ** 2)
        + np.sum(((L_ol_closed(bp_fit, t_fe.numpy()) - L_fe.numpy()) / s_fe.numpy()) ** 2)
    )
    logger.info(f"  data chi2 (closed form at recovered params) = {chi2_exact:.2f}")
    consistency = "PASS" if chi2_exact < max(2.0 * chi2, chi2 + 2.0) else "FAIL"
    logger.info(f"  self-consistency: {consistency}")

    os.makedirs(cfg.output.dir, exist_ok=True)
    meta = {"final_loss": float(history[-1]["loss"]), "chi2": chi2,
            "chi2_exact": chi2_exact, "mode": "hard" if hard else "soft",
            **{f"fit_{n}": float(getattr(p_fit, n))
               for n in ("A_bl", "b3", "C_bl", "PBR_eff", "L0", "L_ol0")}}
    if models is not None:
        save_checkpoint(cfg.output.dir, models=models,
                        epoch=int(cfg.train.n_adam), metadata=meta)
    else:
        torch.save({"kinetics": {n: float(v) for n, v in kin.kinetics().items()},
                    "meta": meta},
                   os.path.join(cfg.output.dir, "kinetics_hard.pt"))
    return {"chi2": chi2, "chi2_exact": chi2_exact, "consistency": consistency,
            "p_fit": {n: float(getattr(p_fit, n))
                      for n in ("A_bl", "b3", "PBR_eff", "L0", "L_ol0")}}


if __name__ == "__main__":
    main()
