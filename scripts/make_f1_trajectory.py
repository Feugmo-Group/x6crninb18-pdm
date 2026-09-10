# SPDX-FileCopyrightText: Copyright (c) 2026 Conrard Giresse Tetsassi Feugmo
# SPDX-License-Identifier: Apache-2.0

"""Dump the soft-mode (F-1) trajectory to a small artifact for the F-1 figure.

The F-1 exhibit needs the actual field-network trajectory, not just the two
chi-squared numbers of Table 6: the whole point is that the trajectory passes
through the data while the exact forward solution at the same recovered
parameters does not.  That trajectory lives in the trained field networks, so
producing it needs torch + PhysicsNeMo -- but the figure itself should not.
This script is the bridge: it runs the seed-0 soft-mode inversion, evaluates the
field networks on the DVR node set, and writes everything the figure needs to

    outputs/paper/f1_trajectory.npz

so that scripts/make_paper_fig_f1.py depends only on NumPy and Matplotlib.

The run is deterministic (torch.manual_seed(cfg.train.seed), float64, CPU), and
reproduces the manuscript's Table 6 numbers exactly: trajectory chi2 = 0.21,
closed-form chi2 at the recovered parameters = 16.64.

Run:  uv run --extra nsem python scripts/make_f1_trajectory.py
"""

from __future__ import annotations

import json
import tempfile
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from physicsnemo.utils import set_default_dtype

from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed, L_ol_closed
from htw_pdm.inverse_trainer import load_means
from htw_pdm.inverse_trainer import main as run_inverse
from htw_pdm.paths import CONF, OUTPUTS, ROOT
from htw_pdm.physics import L_C_NM, T_C_H
from htw_pdm.trainer import build_time_grid, make_fields

OUT = OUTPUTS / "paper"
OUT.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as run_dir:
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        cfg = compose(
            config_name="inverse_config",
            overrides=[
                "inverse.mode=soft",
                "inverse.lambda_phys=1000.0",
                f"output.dir={run_dir}",
            ],
        )
    result = run_inverse(cfg)

    set_default_dtype(torch.float64)
    Nt = int(cfg.domain.Nt)
    # Rebuild the two element networks and restore the trained weights.  The
    # checkpoint is a PhysicsNeMo .mdlus archive, so it is reloaded through
    # Module.from_checkpoint rather than torch.load.
    import physicsnemo

    net_bl = physicsnemo.Module.from_checkpoint(
        f"{run_dir}/SCENElementNetwork0.0.{int(cfg.train.n_adam)}.mdlus")
    net_ol = physicsnemo.Module.from_checkpoint(
        f"{run_dir}/SCENElementNetwork1.0.{int(cfg.train.n_adam)}.mdlus")
    net_bl.double()
    net_ol.double()
    with torch.no_grad():
        lam_bl, lam_ol = make_fields(net_bl, net_ol, Nt)()
    tau = build_time_grid(cfg.domain, torch.float64).numpy()

p = result["p_fit"]
bp = HTWPDMParams(A_bl=p["A_bl"], b3=p["b3"], C_bl=0.0, PBR_eff=p["PBR_eff"],
                  C_x=0.0, L0=p["L0"], L_ol0=p["L_ol0"])

# Dense grid for the exact forward solution at the recovered parameters -- the
# curve the trajectory should have followed and did not.
t_dense = np.linspace(0.0, T_C_H, 400)

data = load_means(str(ROOT / str(cfg.data.csv)), torch.float64)
(t_cr, L_cr, s_cr), (t_fe, L_fe, s_fe) = data["Cr"], data["Fe"]

# Provenance, stored in the artefact rather than in a comment somewhere: the
# artefact outlives the environment that made it, and a reader checking the F-1
# figure should be able to see exactly which run and which solver revision it
# came from without reading this script.
try:
    _pnm_version = version("nvidia-physicsnemo")
except PackageNotFoundError:  # installed from a git checkout without metadata
    _pnm_version = "unknown"

np.savez(
    OUT / "f1_trajectory.npz",
    provenance=np.array(json.dumps({
        "script": "scripts/make_f1_trajectory.py",
        "overrides": ["inverse.mode=soft", "inverse.lambda_phys=1000.0"],
        "seed": int(cfg.train.seed),
        "dtype": str(cfg.train.dtype),
        "n_adam": int(cfg.train.n_adam),
        "n_lbfgs": int(cfg.train.n_lbfgs),
        "backbone": str(cfg.model.backbone),
        "torch": torch.__version__,
        "nvidia-physicsnemo": _pnm_version,
    })),
    t_h=tau * T_C_H,
    traj_bl_nm=lam_bl.numpy() * L_C_NM,
    traj_ol_nm=lam_ol.numpy() * L_C_NM,
    t_dense_h=t_dense,
    exact_bl_nm=L_bl_closed(bp, t_dense),
    exact_ol_nm=L_ol_closed(bp, t_dense),
    chi2_traj=result["chi2"],
    chi2_exact=result["chi2_exact"],
    **{f"fit_{k}": v for k, v in p.items()},
)
print(f"Saved: {OUT / 'f1_trajectory.npz'}")
print(f"  chi2 trajectory = {result['chi2']:.2f}, "
      f"chi2 closed form = {result['chi2_exact']:.2f} ({result['consistency']})")

# ── The self-consistency exhibit itself ──────────────────────────────────────
# The table that accompanies the figure used to be assembled from whatever .pt
# checkpoints happened to be lying in outputs/, which silently went stale when
# The training schedule changed and left the artefact disagreeing with the
# manuscript.  Both remaining runs are re-run here instead, so the exhibit and
# The figure come from one execution of one script.
RUNS = (
    ("hard (real data)", "physics exact by construction",
     ["inverse.mode=hard"]),
    ("soft, lambda_phys=1e3 (real data)", "joint field+parameter PINN",
     ["inverse.mode=soft", "inverse.lambda_phys=1000.0"]),
    ("soft, lambda_data=10 (real data)", "joint field+parameter PINN",
     ["inverse.mode=soft", "inverse.lambda_data=10.0"]),
)

exhibit = []
for label, note, overrides in RUNS:
    if overrides == ["inverse.mode=soft", "inverse.lambda_phys=1000.0"]:
        r = result  # already run above for the trajectory
    else:
        with tempfile.TemporaryDirectory() as rd:
            with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
                c = compose(config_name="inverse_config",
                            overrides=[*overrides, f"output.dir={rd}"])
            r = run_inverse(c)
    exhibit.append({"mode": label, "note": note, "overrides": overrides,
                    "chi2_trajectory": float(r["chi2"]),
                    "chi2_closed_form": float(r["chi2_exact"]),
                    "self_consistent": r["consistency"]})
    print(f"  {label}: chi2_traj = {r['chi2']:.4f}, "
          f"chi2_closed_form = {r['chi2_exact']:.4f} ({r['consistency']})")

with open(OUT / "f1_runs.json", "w") as fh:
    json.dump({"provenance": {"script": "scripts/make_f1_trajectory.py",
                              "seed": int(cfg.train.seed),
                              "torch": torch.__version__,
                              "nvidia-physicsnemo": _pnm_version},
               "runs": exhibit}, fh, indent=2)
print(f"Saved: {OUT / 'f1_runs.json'}")
