"""T2-B — steady 1-D spatial solve: Newton BVP reference + NSEM vs Newton.

Solves the steady defect-transport BVP (src/tier2_physics.py) for both species
(OV, CV) at the 480 h Tier-1 geometry, three ways:

  1. analytic closed form (exact),
  2. Newton on the DVR nodes (machine-precision reference; must hit analytic),
  3. NSEM: one SCEN spatial network per species, BRDR + TwoPhaseOptimizer,
     first-integral flux residuals (per-term diagnostics printed).

Acceptance (TIER2_DESIGN milestone T2-B): NSEM rel Linf vs Newton <= 0.5 % per
species; Newton vs analytic <= 1e-10.

Run:
    python -m htw_pdm.tier2_steady_trainer
    python -m htw_pdm.tier2_steady_trainer domain.Nx=32 train.n_adam=4000
"""

from __future__ import annotations

import os

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from physicsnemo.experimental.models.scen import DVRMapper  # noqa: E402
from physicsnemo.optim import TwoPhaseOptimizer, build_aggregator  # noqa: E402
from physicsnemo.utils import set_default_dtype  # noqa: E402
from physicsnemo.utils.logging import PythonLogger  # noqa: E402

from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed  # noqa: E402
from htw_pdm.paths import CONF  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402
from htw_pdm.tier2_physics import (  # noqa: E402
    analytic_steady,
    newton_steady,
    species_groups,
    steady_residuals,
)
from htw_pdm.trainer import build_nets  # noqa: E402


@hydra.main(config_path=str(CONF), config_name="tier2_config", version_base="1.3")
def main(cfg: DictConfig) -> dict:
    logger = PythonLogger("t2b")
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.train.seed)
    dtype = torch.float64 if cfg.train.dtype == "float64" else torch.float32
    set_default_dtype(dtype)

    p1 = Parameters(**{k: float(v) for k, v in cfg.physics.items()})
    bp = HTWPDMParams(A_bl=p1.A_bl, b3=p1.b3, C_bl=p1.C_bl, PBR_eff=p1.PBR_eff,
                      C_x=p1.C_x, L0=p1.L0, L_ol0=p1.L_ol0)
    t_h = float(cfg.domain.t_h)
    L_nm = float(L_bl_closed(bp, np.array([t_h]))[0])
    groups = species_groups(p1, L_nm)
    for g in groups.values():
        logger.info(f"  {g.name}: Pe = {g.Pe:.3f}  s = {g.s:+.0f}  "
                    f"chat_bc = {g.chat_bc:.3f} at xi = {g.xi_bc:.0f}  "
                    f"c0 = {g.c0_cm3:.3g} cm^-3  tau_mig = {g.tau_mig_h:.2f} h")

    Nx = int(cfg.domain.Nx)
    mapper = DVRMapper(Nx, 0.0, 1.0, float(cfg.domain.alpha_x), dtype=dtype)
    xi, D1 = mapper.nodes, mapper.D1

    results = {}
    os.makedirs(cfg.output.dir, exist_ok=True)
    for name, g in groups.items():
        # Newton reference + analytic parity.
        chat_newton = newton_steady(g, D1)
        chat_exact = torch.tensor(analytic_steady(g, xi.numpy()), dtype=dtype)
        err_newton = float((chat_newton - chat_exact).abs().max())
        logger.info(f"[{name}] Newton vs analytic: Linf = {err_newton:.3e}")

        # NSEM solve.
        net, _ = build_nets(cfg, Nx)
        params = list(net.parameters())

        def fields():
            return torch.nn.functional.softplus(net().view(Nx)) + 1e-6

        # IC-style pretrain to the Robin plateau (flat start, Tier-1 discipline).
        tgt = torch.full((Nx,), float(abs(g.chat_bc)), dtype=dtype)
        opt_pre = torch.optim.Adam(params, lr=float(cfg.train.pretrain_lr))
        for _ in range(int(cfg.train.n_pretrain)):
            opt_pre.zero_grad()
            ((fields() - tgt) ** 2).mean().backward()
            opt_pre.step()

        term_names = ["flux", "bc"]
        brdr = build_aggregator("brdr", params, num_losses=2, weights=[1.0, 1.0])
        adam = torch.optim.Adam(params, lr=float(cfg.train.adam_lr))
        lbfgs = torch.optim.LBFGS(
            params, line_search_fn="strong_wolfe",
            max_iter=int(cfg.train.lbfgs_max_iter),
            tolerance_grad=1e-13, tolerance_change=1e-15,
        )
        opt = TwoPhaseOptimizer(adam, lbfgs, aggregator=brdr)
        step = [0]

        def closure():
            # CV migrates inward: the field is |chat| but flux sign is carried by
            # g.s, so the positive softplus output is the physical magnitude.
            terms = steady_residuals(fields(), D1, g)
            total = brdr({n: terms[n] for n in term_names}, step[0])
            if brdr.training:
                step[0] += 1
            return total

        history = opt.run(closure, n_adam_steps=int(cfg.train.n_adam),
                          n_lbfgs_steps=int(cfg.train.n_lbfgs), verbose=True,
                          log_every=int(cfg.output.log_every))
        with torch.no_grad():
            chat_nsem = fields()
            terms = steady_residuals(chat_nsem, D1, g)
        for tn, tv in terms.items():
            logger.info(f"[{name}] residual[{tn}] = {float(tv):.3e}")
        rel = float((chat_nsem - chat_newton).abs().max() / chat_newton.abs().max())
        logger.info(f"[{name}] NSEM vs Newton: rel Linf = {rel:.3e}")
        results[name] = {"newton_vs_exact": err_newton, "nsem_vs_newton": rel,
                         "final_loss": float(history[-1]["loss"])}
        np.savez(os.path.join(cfg.output.dir, f"steady_{name}.npz"),
                 xi=xi.numpy(), chat_newton=chat_newton.numpy(),
                 chat_nsem=chat_nsem.numpy(), chat_exact=chat_exact.numpy(),
                 Pe=g.Pe, s=g.s, c0_cm3=g.c0_cm3, L_nm=L_nm)

    worst_newton = max(r["newton_vs_exact"] for r in results.values())
    worst_nsem = max(r["nsem_vs_newton"] for r in results.values())
    ok = worst_newton <= 1e-10 and worst_nsem <= 5e-3
    logger.info(f"T2-B acceptance: Newton-vs-analytic {worst_newton:.2e} (<=1e-10), "
                f"NSEM-vs-Newton {worst_nsem:.2e} (<=0.5%) -> "
                f"{'PASS' if ok else 'FAIL'}")
    return results


if __name__ == "__main__":
    main()
