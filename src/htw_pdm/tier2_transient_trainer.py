"""T2-C — moving-boundary transient defect transport (Landau transform, NSEM).

Solves the Landau-transformed transient problem of src/tier2_physics.py for both
species over tau in [tau0, 1] (t in [24, 480] h), with L(tau) imposed from the
Tier-1 closed form and the m/bl growth flux jhat(tau) as the Tier-1 anchor.
Space-time MLP chat(xi, tau) evaluated on the (Nt x Nx) tensor grid; CN in time,
DVR in space; BRDR + TwoPhaseOptimizer.

Regression checks (the Tier-1 reduction discipline of TIER2_DESIGN Section 3):
  R1 quasi-steadiness: final-time profile vs the analytic steady shape at
     L(480 h). The DEVIATION is itself the physical deliverable — it measures the
     quasi-steady approximation error, expected O(tau_mig/t_growth): ~1 % for OV
     (tau_mig ~ 5 h), ~5-10 % for CV (tau_mig ~ 36 h).
  R2 Tier-1 flux reduction: m/bl flux at tau = 1 equals the Tier-1 growth flux
     (jhat = 1) within 1 %.

Run:
    python -m htw_pdm.tier2_transient_trainer
    python -m htw_pdm.tier2_transient_trainer train.n_adam=500   # smoke
"""

from __future__ import annotations

import os

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from physicsnemo.experimental.models.scen import DVRMapper  # noqa: E402
from physicsnemo.optim import TwoPhaseOptimizer, build_aggregator  # noqa: E402
from physicsnemo.utils import set_default_dtype  # noqa: E402
from physicsnemo.utils.logging import PythonLogger  # noqa: E402

from htw_pdm.paths import CONF  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402
from htw_pdm.tier2_physics import (  # noqa: E402
    steady_at,
    transient_groups,
    transient_residuals,
)
from htw_pdm.trainer import build_time_grid  # noqa: E402

T_C_H = 480.0


class SpaceTimeNet(nn.Module):
    """(xi, tau) -> chat MLP, evaluated on the tensor grid (pnp_2d discipline)."""

    def __init__(self, hidden_dim: int, n_layers: int, dtype: torch.dtype):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.to(dtype)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.net(grid).squeeze(-1)) + 1e-6


@hydra.main(config_path=str(CONF), config_name="tier2_config", version_base="1.3")
def main(cfg: DictConfig) -> dict:
    logger = PythonLogger("t2c")
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.train.seed)
    dtype = torch.float64 if cfg.train.dtype == "float64" else torch.float32
    set_default_dtype(dtype)

    p1 = Parameters(**{k: float(v) for k, v in cfg.physics.items()})
    Nx, Nt = int(cfg.domain.Nx), int(cfg.domain.Nt)
    tau0 = float(cfg.domain.tau0)

    mapper_x = DVRMapper(Nx, 0.0, 1.0, float(cfg.domain.alpha_x), dtype=dtype)
    xi, D1x = mapper_x.nodes, mapper_x.D1
    # log-clustered tau grid on [tau0, 1] (reuse Tier-1's builder then rescale).
    t_unit = build_time_grid(cfg.domain, dtype)  # nodes on [0, 1]
    t_grid = tau0 + (1.0 - tau0) * t_unit

    grid = torch.stack(
        [xi.repeat(Nt), t_grid.repeat_interleave(Nx)], dim=1
    )  # (Nt*Nx, 2)

    results = {}
    os.makedirs(cfg.output.dir, exist_ok=True)
    for name in ("OV", "CV"):
        tg = transient_groups(p1, name, t_grid.numpy(), T_C_H, dtype=dtype)
        logger.info(f"[{name}] Pe(tau0..1) = {float(tg.Pe[0]):.2f}..{float(tg.Pe[-1]):.2f}  "
                    f"jhat(tau0) = {float(tg.jhat[0]):.2f}  "
                    f"mig(tau=1) = {float(tg.mig[-1]):.1f}")
        net = SpaceTimeNet(int(cfg.model.hidden_dim), int(cfg.model.n_layers), dtype)
        params = list(net.parameters())

        def fields():
            return net(grid).view(Nt, Nx)

        # Pretrain to the quasi-steady profile at every time node (warm start —
        # the transient solve then only has to learn the lag corrections).
        tgt = torch.stack([steady_at(tg, j, xi) for j in range(Nt)])
        opt_pre = torch.optim.Adam(params, lr=float(cfg.train.pretrain_lr))
        for step in range(int(cfg.train.n_pretrain)):
            opt_pre.zero_grad()
            loss = ((fields() - tgt) ** 2).mean()
            loss.backward()
            opt_pre.step()
        logger.info(f"[{name}] pretrain MSE = {float(loss):.3e}")

        term_names = ["pde", "bc_gen", "bc_ann", "ic"]
        brdr = build_aggregator("brdr", params, num_losses=4, weights=[1.0] * 4)
        adam = torch.optim.Adam(params, lr=float(cfg.train.adam_lr))
        lbfgs = torch.optim.LBFGS(
            params, line_search_fn="strong_wolfe",
            max_iter=int(cfg.train.lbfgs_max_iter),
            tolerance_grad=1e-13, tolerance_change=1e-15,
        )
        opt = TwoPhaseOptimizer(adam, lbfgs, aggregator=brdr)
        step_c = [0]

        def closure():
            terms = transient_residuals(fields(), D1x, t_grid, xi, tg)
            terms["ic"] = float(cfg.train.lambda_ic) * terms["ic"]
            total = brdr({n: terms[n] for n in term_names}, step_c[0])
            if brdr.training:
                step_c[0] += 1
            return total

        history = opt.run(closure, n_adam_steps=int(cfg.train.n_adam),
                          n_lbfgs_steps=int(cfg.train.n_lbfgs), verbose=True,
                          log_every=int(cfg.output.log_every))
        with torch.no_grad():
            chat = fields()
            terms = transient_residuals(chat, D1x, t_grid, xi, tg)
            for tn, tv in terms.items():
                logger.info(f"[{name}] residual[{tn}] = {float(tv):.3e}")

            # R1: quasi-steadiness at tau = 1.
            ss_end = steady_at(tg, Nt - 1, xi)
            r1 = float((chat[-1] - ss_end).abs().max() / ss_end.abs().max())
            # R2: Tier-1 flux reduction at the generation boundary, tau = 1.
            i_gen = 0 if tg.xi_gen == 0.0 else -1
            J_end = -(D1x @ chat[-1]) / tg.Pe[-1] + tg.s * chat[-1]
            r2 = float(abs(J_end[i_gen] - tg.s * tg.jhat[-1]))
        logger.info(f"[{name}] R1 quasi-steady deviation at tau=1: {r1:.3e}")
        logger.info(f"[{name}] R2 |m/bl flux - Tier-1 growth flux| at tau=1: {r2:.3e}")
        results[name] = {"r1_quasi_steady": r1, "r2_flux": r2,
                         "final_loss": float(history[-1]["loss"])}
        np.savez(os.path.join(cfg.output.dir, f"transient_{name}.npz"),
                 xi=xi.numpy(), tau=t_grid.numpy(), chat=chat.numpy(),
                 Pe=tg.Pe.numpy(), jhat=tg.jhat.numpy())

    tol_r1 = {"OV": 0.03, "CV": 0.12}  # quasi-steady lag scales with tau_mig
    ok = all(results[n]["r1_quasi_steady"] <= tol_r1[n] and
             results[n]["r2_flux"] <= 0.01 for n in results)
    logger.info("T2-C acceptance: "
                + "  ".join(f"{n}: R1 {results[n]['r1_quasi_steady']:.2e} "
                            f"(<={tol_r1[n]}), R2 {results[n]['r2_flux']:.2e} (<=0.01)"
                            for n in results)
                + f" -> {'PASS' if ok else 'FAIL'}")
    return results


if __name__ == "__main__":
    main()
