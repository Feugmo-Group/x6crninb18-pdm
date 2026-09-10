"""Phase E2 — parametric NSEM training over the (T, [O2]) operating box.

Trains the forward HTW_PDM at Nc Sobol-sampled conditions in
T in [200, 285] C x [O2] in [0.01, 8] ppm (log-uniform) simultaneously: each
condition gets its own pair of SCEN time networks (same recipe as src/trainer.py),
all optimized jointly by one TwoPhaseOptimizer with a shared BRDR aggregator —
the rpdm_nsem parametric-trainer discipline (cross-condition loss-scale sharing).

The condition -> apparent-parameter map is src/physics_parametric.py (Arrhenius +
1/T field + Nernstian ECP channels, anchored at the M4 fit; dG0_R is the explicit
scan assumption). Every condition is verified against the closed-form solution at
its own parameters; the acceptance criterion is the same 0.5 % rel Linf as the
single-condition forward run.

Results (per-condition L(t) curves + errors) are saved to outputs/parametric/
predictions.npz for plot_sensitivity_maps.py.

Run:
    python -m htw_pdm.parametric_trainer
    python -m htw_pdm.parametric_trainer conditions.n=4 train.n_adam=500   # smoke
"""

from __future__ import annotations

import math
import os

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from physicsnemo.optim import TwoPhaseOptimizer, build_aggregator  # noqa: E402
from physicsnemo.utils import set_default_dtype  # noqa: E402
from physicsnemo.utils.logging import PythonLogger  # noqa: E402
from torch.quasirandom import SobolEngine  # noqa: E402

from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed, L_ol_closed  # noqa: E402
from htw_pdm.paths import CONF  # noqa: E402
from htw_pdm.physics import (  # noqa: E402
    L_C_NM,
    T_C_H,
    NondimGroups,
    Parameters,
    pdm_residuals,
)
from htw_pdm.physics_parametric import ConditionScan, apparent_parameters  # noqa: E402
from htw_pdm.trainer import build_nets, build_time_grid, make_fields, pretrain_ic  # noqa: E402


def sample_conditions(n: int, seed: int) -> np.ndarray:
    """Sobol points over T in [200, 285] C x log10[O2] in [log10(0.01), log10(8)].

    The reference condition (240 C, 0.4 ppm) is always prepended, so the run
    reproduces the validated single-condition forward problem as its first entry.
    """
    u = SobolEngine(2, scramble=True, seed=seed).draw(n - 1).numpy()
    T = 200.0 + u[:, 0] * (285.0 - 200.0)
    O2 = 10.0 ** (math.log10(0.01) + u[:, 1] * (math.log10(8.0) - math.log10(0.01)))
    return np.vstack([[240.0, 0.4], np.column_stack([T, O2])])


@hydra.main(config_path=str(CONF), config_name="parametric_config", version_base="1.3")
def main(cfg: DictConfig) -> dict:
    logger = PythonLogger("x6_pdm_param")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.train.seed)
    dtype = torch.float64 if cfg.train.dtype == "float64" else torch.float32
    set_default_dtype(dtype)

    ref = Parameters(**{k: float(v) for k, v in cfg.physics.items()})
    scan = ConditionScan(
        dG0_R=float(cfg.conditions.dG0_R),
        ecp_channel=bool(cfg.conditions.ecp_channel),
        C_bl_ref=float(cfg.conditions.C_bl_ref),
    )
    conds = sample_conditions(int(cfg.conditions.n), int(cfg.train.seed))
    logger.info(f"{len(conds)} conditions (first = reference 240 C / 0.4 ppm):")
    groups = []
    for T_C, O2 in conds:
        p = apparent_parameters(T_C, O2, ref, scan)
        groups.append(NondimGroups.from_parameters(p))
        logger.info(f"  T = {T_C:6.1f} C  [O2] = {O2:6.3f} ppm  "
                    f"A_bl = {p.A_bl:.4f}  b3 = {p.b3:.6f}  C_bl = {p.C_bl:.4g}")

    t_grid = build_time_grid(cfg.domain, dtype)
    Nt = int(cfg.domain.Nt)

    nets, fields_list, all_params = [], [], []
    for _ in conds:
        net_bl, net_ol = build_nets(cfg, Nt)
        nets.append((net_bl, net_ol))
        fields_list.append(make_fields(net_bl, net_ol, Nt))
        all_params += list(net_bl.parameters()) + list(net_ol.parameters())

    # Stage 1 — per-condition IC pretrain (constant shapes).
    n_pre = int(cfg.train.n_pretrain)
    if n_pre > 0:
        logger.info(f"Pre-training {len(conds)} condition nets to IC shapes ...")
        for g, fields, (net_bl, net_ol) in zip(groups, fields_list, nets):
            params = list(net_bl.parameters()) + list(net_ol.parameters())
            tgt_bl = torch.full((Nt,), g.lam0, dtype=dtype)
            tgt_ol = torch.full((Nt,), g.lamol0, dtype=dtype)
            pretrain_ic(fields, params, (tgt_bl, tgt_ol), n_pre,
                        float(cfg.train.pretrain_lr), logger)

    # Stage 2 — joint physics training, shared BRDR across all conditions.
    lambda_ic = float(cfg.train.lambda_ic)
    term_names = [f"{t}_{i}" for i in range(len(conds)) for t in ("ode_bl", "ode_ol", "ic")]
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
        terms = {}
        for i, (g, fields) in enumerate(zip(groups, fields_list)):
            lam_bl, lam_ol = fields()
            t = pdm_residuals(lam_bl, lam_ol, t_grid, g)
            terms[f"ode_bl_{i}"] = t["ode_bl"]
            terms[f"ode_ol_{i}"] = t["ode_ol"]
            terms[f"ic_{i}"] = lambda_ic * t["ic"]
        total = brdr(terms, brdr_step[0])
        if brdr.training:
            brdr_step[0] += 1
        return total

    out_dir = str(cfg.output.dir)
    os.makedirs(out_dir, exist_ok=True)
    history = opt.run(
        closure,
        n_adam_steps=int(cfg.train.n_adam),
        n_lbfgs_steps=int(cfg.train.n_lbfgs),
        verbose=True,
        log_every=int(cfg.output.log_every),
        grad_clip=float(cfg.train.get("grad_clip", 0)) or None,
    )
    logger.info(f"Final loss: {history[-1]['loss']:.4e}")

    # Verification: every condition vs its own closed form.
    t_h = (t_grid.cpu().numpy() * T_C_H).astype(float)
    pred_bl = np.zeros((len(conds), Nt))
    pred_ol = np.zeros((len(conds), Nt))
    rel_errs = []
    with torch.no_grad():
        for i, (g, fields) in enumerate(zip(groups, fields_list)):
            lam_bl, lam_ol = fields()
            pred_bl[i] = lam_bl.cpu().numpy() * L_C_NM
            pred_ol[i] = lam_ol.cpu().numpy() * L_C_NM
            p = g.to_parameters()
            bp = HTWPDMParams(A_bl=p.A_bl, b3=p.b3, C_bl=p.C_bl, PBR_eff=p.PBR_eff,
                              C_x=p.C_x, L0=p.L0, L_ol0=p.L_ol0)
            ref_bl, ref_ol = L_bl_closed(bp, t_h), L_ol_closed(bp, t_h)
            rel = max(
                float(np.max(np.abs(pred_bl[i] - ref_bl)) / np.max(np.abs(ref_bl))),
                float(np.max(np.abs(pred_ol[i] - ref_ol)) / np.max(np.abs(ref_ol))),
            )
            rel_errs.append(rel)
            logger.info(f"  cond {i:2d} (T={conds[i,0]:6.1f} C, O2={conds[i,1]:6.3f} ppm): "
                        f"rel Linf = {rel:.3e}")
    worst = max(rel_errs)
    status = "PASS" if worst <= 5e-3 else "FAIL"
    logger.info(f"Worst condition rel Linf = {worst:.3e} — acceptance (<= 0.5 %): {status}")

    np.savez(
        os.path.join(out_dir, "predictions.npz"),
        conditions=conds, t_h=t_h, L_bl=pred_bl, L_ol=pred_ol,
        rel_errs=np.array(rel_errs),
        dG0_R=scan.dG0_R, ecp_channel=scan.ecp_channel, C_bl_ref=scan.C_bl_ref,
    )
    logger.info(f"Saved: {os.path.join(out_dir, 'predictions.npz')}")
    return {"final_loss": float(history[-1]["loss"]), "worst_rel": worst}


if __name__ == "__main__":
    main()
