"""T4-A runner — is the PDM's constant-field closure self-consistent?

Four questions, in order:

  A1 VERIFICATION. With the space charge switched off (lam_scale = 0) the coupled
     solver must reproduce the Tier-2 analytic constant-field solution exactly.
     This is the exact reference that makes the rest trustworthy.

  A2 MAGNITUDE. Evaluate the space-charge parameter Lam and the Debye length at
     the identified Tier-1 kinetics and the Tier-2 assumed transport constants.

  A3 VALIDITY BOUNDARY. Continue in lam_scale and find where the self-consistent
     field departs from the constant-field closure by 1 %, 5 % and 10 %. Convert
     each threshold into the defect diffusivity (equivalently, defect
     concentration) at which the closure stops being usable, since Lam ~ 1/D.

  A4 NEURAL SOLVE. Grade an NSEM solve of the coupled nonlinear system against
     the Newton reference, inside the convergent regime. Unlike the Tier-2 steady
     problem this system has no closed form, so this is the first place in the
     project where the neural solver is asked to do something a closed form
     cannot -- and it is still checked, because Newton is available here.

Writes outputs/paper/tier4_pnp.json.

Run:
    python -m htw_pdm.tier4_pnp_solve
"""

from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from physicsnemo.experimental.models.scen import DVRMapper  # noqa: E402
from physicsnemo.optim import TwoPhaseOptimizer, build_aggregator  # noqa: E402

from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed  # noqa: E402
from htw_pdm.paths import PAPER_OUT  # noqa: E402
from htw_pdm.physics import F_OVER_RT, Parameters  # noqa: E402
from htw_pdm.tier2_physics import (  # noqa: E402
    D_CV_CM2_S,
    D_OV_CM2_S,
    EPS_F_V_CM,
    NM_TO_CM,
    SpeciesGroups,
    analytic_steady,
)
from htw_pdm.tier4_pnp import (  # noqa: E402
    E_CHARGE_C,
    EPS_0_F_CM,
    EPS_R,
    EPS_R_SCAN,
    newton_pnp,
    pnp_residuals,
    pnp_species,
)

N_NODES = 48
T_EVAL_H = 480.0
DEV_TARGETS = (0.01, 0.05, 0.10)
D_ASSUMED = {"OV": D_OV_CM2_S, "CV": D_CV_CM2_S}


def _L_at(p1: Parameters, t_h: float) -> float:
    bp = HTWPDMParams(A_bl=p1.A_bl, b3=p1.b3, C_bl=p1.C_bl, PBR_eff=p1.PBR_eff,
                      C_x=p1.C_x, L0=p1.L0, L_ol0=p1.L_ol0)
    return float(L_bl_closed(bp, np.array([t_h]))[0])


def a1_verification(species, D1, w, xi_np) -> dict:
    """lam_scale = 0 must reproduce the Tier-2 analytic solution exactly."""
    chats, Ehat, res = newton_pnp(species, D1, w, lam_scale=0.0)
    errs = {}
    for sp, chat in zip(species, chats):
        g = SpeciesGroups(sp.name, sp.Pe, sp.s, sp.chat_bc, sp.xi_bc, sp.c0_cm3, 0.0)
        ref = torch.tensor(analytic_steady(g, xi_np), dtype=D1.dtype)
        errs[sp.name] = float((chat - ref).abs().max() / ref.abs().max())
    return {
        "newton_residual": res,
        "rel_err_vs_analytic": errs,
        "max_field_dev": float((Ehat - 1.0).abs().max()),
    }


def a2_magnitude(p1: Parameters, L_nm: float) -> dict:
    """Lam and the Debye length at the identified kinetics, versus eps_r."""
    out = {"L_nm": L_nm, "per_eps_r": {}}
    kT_V = 1.0 / F_OVER_RT  # RT/F in volts
    for eps_r in EPS_R_SCAN:
        species = pnp_species(p1, L_nm, eps_r=eps_r)
        n_eff = sum(sp.z**2 * sp.c0_cm3 for sp in species)
        lam_D_nm = float(
            np.sqrt(eps_r * EPS_0_F_CM * kT_V / (E_CHARGE_C * n_eff)) / NM_TO_CM
        )
        out["per_eps_r"][str(eps_r)] = {
            "Lam": {sp.name: sp.Lam for sp in species},
            "c0_cm3": {sp.name: sp.c0_cm3 for sp in species},
            "debye_nm": lam_D_nm,
            "L_over_debye": L_nm / lam_D_nm,
        }
    return out


def a3_validity(species, D1, w) -> dict:
    """Where does the constant-field closure break?

    Bisect lam_scale for each target field deviation, then translate: since
    Lam ~ 1/D at fixed flux, a threshold lam_scale* means the closure holds only
    for D >= D_assumed / lam_scale*. Reported per species using its own Lam.
    """
    def dev(ls: float) -> float:
        try:
            _, Ehat, res = newton_pnp(species, D1, w, lam_scale=ls, n_homotopy=10)
            if not np.isfinite(res) or res > 1e-8:
                return float("inf")
            return float((Ehat - 1.0).abs().max())
        except Exception:
            return float("inf")

    # Bracket: grow from a tiny value until the deviation exceeds the largest target.
    lo, hi = 1e-12, 1e-12
    while dev(hi) < max(DEV_TARGETS) and hi < 1.0:
        lo, hi = hi, hi * 10.0

    out = {"thresholds": {}, "curve": []}
    for ls in np.logspace(np.log10(max(lo, 1e-12)) - 1, np.log10(hi), 12):
        d = dev(float(ls))
        if np.isfinite(d):
            out["curve"].append({"lam_scale": float(ls), "max_field_dev": d})

    for target in DEV_TARGETS:
        a, b = 1e-14, hi
        for _ in range(45):
            m = np.sqrt(a * b)
            if dev(float(m)) < target:
                a = m
            else:
                b = m
        ls_star = float(np.sqrt(a * b))
        out["thresholds"][f"{target:.2f}"] = {
            "lam_scale": ls_star,
            "Lam_at_threshold": {sp.name: sp.Lam * ls_star for sp in species},
            "D_required_cm2_s": {
                sp.name: D_ASSUMED[sp.name] / ls_star for sp in species
            },
            "c0_max_cm3": {sp.name: sp.c0_cm3 * ls_star for sp in species},
        }
    return out


class FieldNet(nn.Module):
    """xi -> (chat_OV, chat_CV, Ehat). Positivity on concentrations only."""

    def __init__(self, hidden: int = 48, n_layers: int = 4, dtype=torch.float64):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)
        self.to(dtype)

    def forward(self, xi: torch.Tensor):
        y = self.net(xi.unsqueeze(-1))
        c = torch.nn.functional.softplus(y[:, :2]) + 1e-8
        return [c[:, 0], c[:, 1]], y[:, 2]


def a4_nsem(species, D1, w, xi, lam_scale: float, seed: int = 0) -> dict:
    """NSEM solve of the coupled system, graded against Newton at the same lam."""
    torch.manual_seed(seed)
    chats_ref, Ehat_ref, res_ref = newton_pnp(species, D1, w, lam_scale=lam_scale)
    if not np.isfinite(res_ref) or res_ref > 1e-8:
        return {"status": "newton_reference_unavailable", "lam_scale": lam_scale}

    net = FieldNet(dtype=D1.dtype)
    params = list(net.parameters())

    # Warm start on the constant-field solution: the neural solve then only has
    # to learn the space-charge correction (same discipline as T2-C pretraining).
    tgt_c = torch.stack(
        [torch.tensor(analytic_steady(
            SpeciesGroups(sp.name, sp.Pe, sp.s, sp.chat_bc, sp.xi_bc, sp.c0_cm3, 0.0),
            xi.numpy()), dtype=D1.dtype) for sp in species]
    )
    opt_pre = torch.optim.Adam(params, lr=1e-2)
    for _ in range(1500):
        opt_pre.zero_grad()
        c, E = net(xi)
        loss = ((torch.stack(c) - tgt_c) ** 2).mean() + ((E - 1.0) ** 2).mean()
        loss.backward()
        opt_pre.step()

    names = ["flux_OV", "bc_OV", "flux_CV", "bc_CV", "poisson", "gauge"]
    brdr = build_aggregator("brdr", params, num_losses=len(names),
                            weights=[1.0] * len(names))
    adam = torch.optim.Adam(params, lr=1e-3)
    lbfgs = torch.optim.LBFGS(params, line_search_fn="strong_wolfe", max_iter=400,
                              tolerance_grad=1e-13, tolerance_change=1e-15)
    opt = TwoPhaseOptimizer(adam, lbfgs, aggregator=brdr)
    step_c = [0]

    def closure():
        c, E = net(xi)
        terms = pnp_residuals(c, E, D1, w, species, lam_scale=lam_scale)
        total = brdr({n: terms[n] for n in names}, step_c[0])
        if brdr.training:
            step_c[0] += 1
        return total

    opt.run(closure, n_adam_steps=3000, n_lbfgs_steps=3, verbose=False)

    with torch.no_grad():
        c, E = net(xi)
        terms = pnp_residuals(c, E, D1, w, species, lam_scale=lam_scale)
        rel = {sp.name: float((ci - cr).abs().max() / cr.abs().max())
               for sp, ci, cr in zip(species, c, chats_ref)}
        rel["Ehat"] = float((E - Ehat_ref).abs().max() / Ehat_ref.abs().max())
    return {
        "status": "ok",
        "lam_scale": lam_scale,
        "newton_residual": res_ref,
        "nsem_vs_newton_rel_Linf": rel,
        "nsem_residuals": {k: float(v) for k, v in terms.items()},
        "max_field_dev": float((Ehat_ref - 1.0).abs().max()),
    }


def main() -> dict:
    torch.set_default_dtype(torch.float64)
    p1 = Parameters()
    L_nm = _L_at(p1, T_EVAL_H)
    mapper = DVRMapper(N_NODES, 0.0, 1.0, 0.0, dtype=torch.float64)
    xi, D1, w = mapper.nodes, mapper.D1, mapper.weights
    species = pnp_species(p1, L_nm, eps_r=EPS_R)

    print(f"[T4-A] L({T_EVAL_H:.0f} h) = {L_nm:.3f} nm, eps_r = {EPS_R}, "
          f"{N_NODES} nodes")
    for sp in species:
        print(f"  {sp.name}: z = {sp.z:+.3f}  Pe = {sp.Pe:.3f}  "
              f"c0 = {sp.c0_cm3:.3e} cm^-3  Lam = {sp.Lam:.4e}")

    res = {"L_nm": L_nm, "eps_r": EPS_R, "n_nodes": N_NODES,
           "eps_f_V_cm": EPS_F_V_CM}

    res["A1_verification"] = a1_verification(species, D1, w, xi.numpy())
    print(f"[A1] lam=0 vs analytic: "
          f"{res['A1_verification']['rel_err_vs_analytic']}  "
          f"field dev {res['A1_verification']['max_field_dev']:.2e}")

    res["A2_magnitude"] = a2_magnitude(p1, L_nm)
    base = res["A2_magnitude"]["per_eps_r"][str(float(EPS_R))] \
        if str(float(EPS_R)) in res["A2_magnitude"]["per_eps_r"] \
        else list(res["A2_magnitude"]["per_eps_r"].values())[0]
    print(f"[A2] Debye = {base['debye_nm']:.4f} nm, L/lam_D = "
          f"{base['L_over_debye']:.0f}, Lam = {base['Lam']}")

    res["A3_validity"] = a3_validity(species, D1, w)
    for tgt, blk in res["A3_validity"]["thresholds"].items():
        print(f"[A3] {float(tgt)*100:.0f}% field deviation at lam_scale = "
              f"{blk['lam_scale']:.3e} -> D_required = "
              + ", ".join(f"{k} {v:.3e}" for k, v in blk["D_required_cm2_s"].items()))

    ls_nsem = res["A3_validity"]["thresholds"]["0.10"]["lam_scale"]
    res["A4_nsem"] = a4_nsem(species, D1, w, xi, lam_scale=float(ls_nsem))
    print(f"[A4] NSEM vs Newton: {res['A4_nsem'].get('nsem_vs_newton_rel_Linf')}")

    out_dir = str(PAPER_OUT)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tier4_pnp.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[T4-A] wrote {path}")
    return res


if __name__ == "__main__":
    main()
