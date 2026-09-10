"""Forward transient HTW_PDM trainer (NSEM / SCEN-DVR).

Two time-only SCEN networks — lam_bl(tau) and lam_ol(tau) — trained against the
Crank-Nicolson residuals of the reduced PDM film-growth ODEs (src/physics.py),
with BRDR loss balancing and TwoPhaseOptimizer (Adam -> L-BFGS).

Stage 1 pretrains both networks to the constant initial-condition shape
(lam = lam0 / lamol0 everywhere), same discipline as rpdm_nsem. Stage 2 runs
physics training. Verification against the Radau/closed-form baseline runs at
The end and reports the relative L-inf error (acceptance: <= 0.5 %).

Run:
    python -m htw_pdm.trainer
    python -m htw_pdm.trainer model.backbone=kan train.n_adam=4000
"""

from __future__ import annotations

import os

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from physicsnemo.experimental.models.scen import (  # noqa: E402
    DVRMapper,
    SCENElementNetwork,
)
from physicsnemo.optim import TwoPhaseOptimizer, build_aggregator  # noqa: E402
from physicsnemo.utils import save_checkpoint, set_default_dtype  # noqa: E402
from physicsnemo.utils.logging import PythonLogger  # noqa: E402

from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed, L_ol_closed  # noqa: E402
from htw_pdm.paths import CONF  # noqa: E402
from htw_pdm.physics import (  # noqa: E402
    L_C_NM,
    T_C_H,
    NondimGroups,
    Parameters,
    pdm_residuals,
)


def build_time_grid(dom: DictConfig, dtype: torch.dtype):
    mapper_t = DVRMapper(
        int(dom.Nt),
        0.0,
        float(dom.yf),
        float(dom.alpha_t),
        mapping=str(dom.get("mapping_t", "kte")),
        dtype=dtype,
    )
    return mapper_t.nodes  # (Nt,)


def build_nets(cfg: DictConfig, Nt: int):
    time_element = [{"N": Nt, "a": -1.0, "b": 1.0}]
    kwargs = dict(
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone,
        poly_degree=cfg.model.poly_degree,
        dtype=cfg.train.dtype,
    )
    return SCENElementNetwork(time_element, **kwargs), SCENElementNetwork(
        time_element, **kwargs
    )


def make_fields(net_bl, net_ol, Nt: int):
    def fields():
        # softplus keeps both thicknesses strictly positive.
        lam_bl = torch.nn.functional.softplus(net_bl().view(Nt)) + 1e-6
        lam_ol = torch.nn.functional.softplus(net_ol().view(Nt)) + 1e-6
        return lam_bl, lam_ol

    return fields


def pretrain_ic(fields, all_params, targets, n_steps, lr, logger):
    """Stage 1: drive both networks to constant IC shapes (Adam + L-BFGS polish)."""
    tgt_bl, tgt_ol = targets

    def _loss():
        lam_bl, lam_ol = fields()
        return ((lam_bl - tgt_bl) ** 2).mean() + ((lam_ol - tgt_ol) ** 2).mean()

    opt = torch.optim.Adam(all_params, lr=lr)
    for step in range(n_steps):
        opt.zero_grad()
        loss = _loss()
        loss.backward()
        opt.step()
        if step % max(1, n_steps // 4) == 0:
            logger.info(f"  pretrain Adam {step:5d}: MSE = {loss.item():.3e}")
    lbfgs = torch.optim.LBFGS(
        all_params,
        max_iter=300,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-13,
        tolerance_change=1e-15,
    )

    def _cl():
        lbfgs.zero_grad()
        loss = _loss()
        loss.backward()
        return loss

    val = lbfgs.step(_cl).item()
    logger.info(f"  pretrain L-BFGS: MSE = {val:.3e}")
    return val


@hydra.main(config_path=str(CONF), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> dict:
    logger = PythonLogger("x6_pdm")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.train.seed)
    dtype = torch.float64 if cfg.train.dtype == "float64" else torch.float32
    set_default_dtype(dtype)

    p = Parameters(**{k: float(v) for k, v in cfg.physics.items()})
    g = NondimGroups.from_parameters(p)
    logger.info(
        f"Nondim groups: Ahat={g.Ahat:.4f} bhat={g.bhat:.4f} Chat={g.Chat:.4f} "
        f"PBReff={g.PBReff:.4f} Cxhat={g.Cxhat:.4f} lam0={g.lam0:.5f} "
        f"lamol0={g.lamol0:.4f}"
    )

    t_grid = build_time_grid(cfg.domain, dtype)
    Nt = int(cfg.domain.Nt)

    net_bl, net_ol = build_nets(cfg, Nt)
    all_params = list(net_bl.parameters()) + list(net_ol.parameters())
    fields = make_fields(net_bl, net_ol, Nt)

    # Stage 1 — IC pretrain (constant shapes; rpdm_nsem discipline).
    n_pre = int(cfg.train.n_pretrain)
    if n_pre > 0:
        logger.info(f"Pre-training to IC shape ({n_pre} Adam + L-BFGS) ...")
        tgt_bl = torch.full((Nt,), g.lam0, dtype=dtype)
        tgt_ol = torch.full((Nt,), g.lamol0, dtype=dtype)
        pretrain_ic(
            fields, all_params, (tgt_bl, tgt_ol), n_pre,
            float(cfg.train.pretrain_lr), logger,
        )

    # Stage 2 — physics training.
    lambda_ic = float(cfg.train.lambda_ic)
    term_names = ["ode_bl", "ode_ol", "ic"]
    brdr = build_aggregator(
        "brdr", all_params, num_losses=len(term_names), weights=[1.0] * len(term_names)
    )
    adam = torch.optim.Adam(all_params, lr=float(cfg.train.adam_lr))
    lbfgs = torch.optim.LBFGS(
        all_params,
        line_search_fn="strong_wolfe",
        max_iter=int(cfg.train.lbfgs_max_iter),
        tolerance_grad=float(cfg.train.lbfgs_tol_grad),
        tolerance_change=float(cfg.train.lbfgs_tol_change),
    )
    opt = TwoPhaseOptimizer(adam, lbfgs, aggregator=brdr)
    brdr_step = [0]

    def closure() -> torch.Tensor:
        lam_bl, lam_ol = fields()
        terms = pdm_residuals(lam_bl, lam_ol, t_grid, g)
        terms["ic"] = lambda_ic * terms["ic"]
        total = brdr({name: terms[name] for name in term_names}, brdr_step[0])
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

    # Verification against the Radau/closed-form baseline.
    with torch.no_grad():
        lam_bl, lam_ol = fields()
        terms = pdm_residuals(lam_bl, lam_ol, t_grid, g)
        for name, val in terms.items():
            logger.info(f"  residual[{name}] = {float(val):.3e}")

        bp = HTWPDMParams(
            A_bl=p.A_bl, b3=p.b3, C_bl=p.C_bl, PBR_eff=p.PBR_eff,
            C_x=p.C_x, L0=p.L0, L_ol0=p.L_ol0,
        )
        t_h = (t_grid.cpu().numpy() * T_C_H).astype(float)
        ref_bl = L_bl_closed(bp, t_h)
        ref_ol = L_ol_closed(bp, t_h)
        pred_bl = lam_bl.cpu().numpy() * L_C_NM
        pred_ol = lam_ol.cpu().numpy() * L_C_NM
        rel_bl = float(np.max(np.abs(pred_bl - ref_bl)) / np.max(np.abs(ref_bl)))
        rel_ol = float(np.max(np.abs(pred_ol - ref_ol)) / np.max(np.abs(ref_ol)))
    logger.info(f"vs closed form: rel Linf L_bl = {rel_bl:.3e}, L_ol = {rel_ol:.3e}")
    status = "PASS" if max(rel_bl, rel_ol) <= 5e-3 else "FAIL"
    logger.info(f"Acceptance (<= 0.5 %): {status}")

    for name, net in zip(["lbl", "lol"], [net_bl, net_ol]):
        net.save(
            os.path.join(
                cfg.output.dir,
                cfg.output.checkpoint.replace(".mdlus", f"_{name}.mdlus"),
            )
        )
    final_loss = float(history[-1]["loss"])
    save_checkpoint(
        cfg.output.dir,
        models=[net_bl, net_ol],
        optimizer=adam,
        epoch=int(cfg.train.n_adam),
        metadata={"final_loss": final_loss, "rel_bl": rel_bl, "rel_ol": rel_ol},
    )
    return {"final_loss": final_loss, "rel_bl": rel_bl, "rel_ol": rel_ol}


if __name__ == "__main__":
    main()
