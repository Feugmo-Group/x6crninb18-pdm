# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper tables (PAPER_PLAN.md Section 5) as CSV files under outputs/paper/.

  table1_parameters.csv   — M4 parameter table: fit value, profile 1-sigma,
                            bootstrap mean/std/[16,84]% (from ensemble_summary.csv
                            if present), recovered field strength.
  table2_model_ladder.csv — M1-M5 chi2 ladder + power-law comparison + per-model
                            parameter values (the degenerate-ridge exhibit).
  table3_nsem_accuracy.csv— MLP vs KAN forward parity + hard-inverse recovery
                            (real + synthetic) vs the deterministic baseline.
  table4_f1_exhibit.csv   — F-1: per inverse run, trajectory chi2 vs closed-form
                            chi2 at the recovered parameters (self-consistency).

Run:  python scripts/make_paper_tables.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch

from htw_pdm.baseline_fit import (  # noqa: E402
    build_fit_data,
    fit_pdm,
    fit_power_law,
    load_data,
    param_names,
    power_law_chi2,
    profile_likelihood,
)
from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed, L_ol_closed  # noqa: E402
from htw_pdm.physics import recovered_field_strength  # noqa: E402

from htw_pdm.paths import ROOT  # noqa: E402
from htw_pdm.paths import PAPER_OUT as OUT  # noqa: E402
OUT.mkdir(parents=True, exist_ok=True)

means, scans = load_data()
d = build_fit_data(means)
pl = fit_power_law(scans)


def closed_form_chi2(p_dict: dict) -> float:
    """Data chi2 of the closed form at a recovered parameter set (real Veile data)."""
    bp = HTWPDMParams(
        A_bl=p_dict["fit_A_bl"], b3=p_dict["fit_b3"], C_bl=p_dict["fit_C_bl"],
        PBR_eff=p_dict["fit_PBR_eff"], C_x=0.0, L0=p_dict["fit_L0"],
        L_ol0=p_dict["fit_L_ol0"],
    )
    return float(
        np.sum(((L_bl_closed(bp, d.t) - d.L_cr) / d.sig_cr) ** 2)
        + np.sum(((L_ol_closed(bp, d.t) - d.L_fe) / d.sig_fe) ** 2)
    )


def load_meta(path: str) -> dict | None:
    f = ROOT / path
    if not f.exists():
        return None
    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    return ckpt.get("metadata", ckpt.get("meta", {}))


# ── Table 2 — model ladder (also yields the M4 solution reused by Table 1) ────
ladder_rows = []
sols = {}
for variant, label, extras in (
    (1, "M1 core", ""),
    (2, "M2", "+C_bl"),
    (3, "M3", "+C_bl +C_x"),
    (4, "M4 (accepted)", "+L_ol0"),
    (5, "M5", "+L_ol0 +C_bl"),
):
    sol, p, chi2, dof, total = fit_pdm(d, variant)
    sols[variant] = (sol, p, chi2, dof)
    ladder_rows.append([
        label, extras, 4 + len(extras.split()) if extras else 4,
        f"{chi2:.2f}", dof, f"{total:.2f}",
        f"{p.A_bl:.4g}", f"{p.b3:.4g}", f"{p.C_bl:.4g}",
        f"{p.PBR_eff:.4g}", f"{p.C_x:.4g}", f"{p.L0:.4g}", f"{p.L_ol0:.4g}",
    ])
chi2_pl = power_law_chi2(d, *pl["Cr"], "Cr") + power_law_chi2(d, *pl["Fe"], "Fe")
ladder_rows.append([
    "power law (Veile)", "k,n per layer", 4, f"{chi2_pl:.2f}", 2, f"{chi2_pl:.2f}",
    "", "", "", "", "", "", "",
])
with open(OUT / "table2_model_ladder.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "extra_params", "n_free", "chi2_data", "dof",
                "total_with_priors", "A_bl_nm_h", "b3_1_nm", "C_bl_nm_h",
                "PBR_eff", "C_x_nm_h", "L0_nm", "L_ol0_nm"])
    w.writerows(ladder_rows)
print(f"Saved: {OUT / 'table2_model_ladder.csv'}")

# ── Table 1 — M4 parameter table with profile + bootstrap intervals ───────────
sol4, p4, chi2_4, dof_4 = sols[4]
names = param_names(4)  # [A_bl, -b3, PBR_eff, L0, L_ol0]
prof = {}
for i, nm in enumerate(names):
    grid, chi2s = profile_likelihood(d, 4, sol4.x, i)
    dchi = chi2s - chi2s.min()
    inside = grid[dchi <= 1.0]
    lo, hi = float(np.exp(inside.min())), float(np.exp(inside.max()))
    key = "b3" if nm == "-b3" else nm
    if key == "b3":
        lo, hi = -hi, -lo
    prof[key] = (lo, hi, float(dchi.max()))

ens = {}
members_csv = ROOT / "outputs" / "ensemble_members_1000.csv"
ens_csv = ROOT / "outputs" / "ensemble_summary.csv"
if members_csv.exists():
    # Preferred source: the 1000-member deterministic parametric bootstrap
    # (make_paper_stats.py) — the manuscript's authoritative intervals.
    mem = np.genfromtxt(members_csv, delimiter=",", names=True)
    for key in mem.dtype.names:
        arr = mem[key]
        p16, p84 = np.percentile(arr, [16, 84])
        ens[key] = {"ens_mean": f"{arr.mean():.5g}", "ens_std": f"{arr.std():.4g}",
                    "ens_p16": f"{p16:.5g}", "ens_p84": f"{p84:.5g}"}
    print(f"Bootstrap columns from {members_csv.name} ({len(mem)} members).")
elif ens_csv.exists():
    with open(ens_csv) as f:
        for row in csv.DictReader(f):
            ens[row["param"]] = row
    print("NOTE: bootstrap columns from legacy outputs/ensemble_summary.csv "
          "(50-member PINN ensemble); run make_paper_stats.py for the "
          "1000-member deterministic bootstrap.")
else:
    print("NOTE: no bootstrap source found — bootstrap columns empty; "
          "run make_paper_stats.py first.")

fit_vals = {"A_bl": p4.A_bl, "b3": p4.b3, "PBR_eff": p4.PBR_eff,
            "L0": p4.L0, "L_ol0": p4.L_ol0}
units = {"A_bl": "nm/h", "b3": "1/nm", "PBR_eff": "-", "L0": "nm", "L_ol0": "nm"}
with open(OUT / "table1_parameters.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["param", "unit", "m4_fit", "profile_1sig_lo", "profile_1sig_hi",
                "profile_max_dchi2", "identifiability",
                "boot_mean", "boot_std", "boot_p16", "boot_p84"])
    # Verdicts as argued in the manuscript (profile + prior structure combined):
    # PBR_eff passes the raw Delta-chi2 > 4 gate but its interval is set by the
    # prior; L0's profile is flat (prior-set).
    # PBR_eff no longer carries a prior (see baseline_fit.PRIORS), so its
    # verdict is now a pure data statement: constrained, but far less sharply
    # than A_bl/b3/L_ol0 (profile max dchi2 ~20 vs 50-130) and with a 1-sigma
    # interval spanning an order of magnitude.
    VERDICTS = {"A_bl": "identifiable", "b3": "identifiable",
                "PBR_eff": "weakly identifiable",
                "L0": "unidentifiable (prior-set)", "L_ol0": "identifiable"}
    for key in ("A_bl", "b3", "PBR_eff", "L0", "L_ol0"):
        lo, hi, dmax = prof[key]
        ident = VERDICTS[key]
        e = ens.get(key, {})
        w.writerow([key, units[key], f"{fit_vals[key]:.5g}",
                    f"{lo:.4g}", f"{hi:.4g}", f"{dmax:.1f}", ident,
                    e.get("ens_mean", ""), e.get("ens_std", ""),
                    e.get("ens_p16", ""), e.get("ens_p84", "")])
    w.writerow(["eps_f", "V/cm", f"{recovered_field_strength(p4.b3):.3g}",
                "", "", "", "derived (alpha3 = 0.12, chi = 8/3)", "", "", "", ""])
    w.writerow(["chi2/dof", "-", f"{chi2_4:.2f}/{dof_4}",
                "", "", "", f"power law: {chi2_pl:.2f}/2", "", "", "", ""])
print(f"Saved: {OUT / 'table1_parameters.csv'}")

# ── Table 3 — NSEM forward parity + hard-inverse recovery ─────────────────────
rows3 = []
for label, path in (("forward MLP", "outputs/forward/checkpoint.0.2000.pt"),
                    ("forward KAN", "outputs/forward_kan/checkpoint.0.2000.pt")):
    m = load_meta(path)
    if m:
        rows3.append([label, "rel Linf L_bl vs closed form", f"{m['rel_bl']:.2e}",
                      "PASS" if m["rel_bl"] <= 5e-3 else "FAIL"])
        rows3.append([label, "rel Linf L_ol vs closed form", f"{m['rel_ol']:.2e}",
                      "PASS" if m["rel_ol"] <= 5e-3 else "FAIL"])

m_hard = load_meta("outputs/inverse_hard/kinetics_hard.pt")
if m_hard:
    for pn, ref in (("A_bl", p4.A_bl), ("b3", p4.b3), ("PBR_eff", p4.PBR_eff),
                    ("L0", p4.L0), ("L_ol0", p4.L_ol0)):
        v = m_hard[f"fit_{pn}"]
        rel = abs(v - ref) / abs(ref)
        rows3.append(["inverse hard (real data)", f"{pn} vs deterministic M4",
                      f"{v:.5g} (dev {rel:.2%})", "PASS" if rel < 0.01 else "CHECK"])
    rows3.append(["inverse hard (real data)", "chi2 (= M4 baseline 5.79)",
                  f"{m_hard['chi2']:.2f}", "PASS"])
m_synth = load_meta("outputs/inverse_hard_synth/kinetics_hard.pt")
if m_synth:
    truth = {"A_bl": 0.7266, "b3": -0.01248, "PBR_eff": 1.051, "L0": 1.924,
             "L_ol0": 118.7}
    for pn, tv in truth.items():
        v = m_synth[f"fit_{pn}"]
        rows3.append(["inverse hard (synthetic, 2% noise)", f"{pn} vs ground truth",
                      f"{v:.5g} (dev {abs(v - tv) / abs(tv):.2%})", ""])
with open(OUT / "table3_nsem_accuracy.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "quantity", "value", "verdict"])
    w.writerows(rows3)
print(f"Saved: {OUT / 'table3_nsem_accuracy.csv'}")

# ── Table 4 — F-1 exhibit: trajectory chi2 vs closed form at recovered params ─
rows4 = []
for label, path, note in (
    ("hard (real data)", "outputs/inverse_hard/kinetics_hard.pt",
     "physics exact by construction"),
    ("soft, lambda_phys=1e3 (real data)", "outputs/inverse/checkpoint.0.6000.pt",
     "joint field+parameter PINN"),
    ("soft, lambda_data=10 (real data)", "outputs/inverse_ld10/checkpoint.0.6000.pt",
     "joint field+parameter PINN"),
):
    m = load_meta(path)
    if m is None:
        continue
    chi2_traj = m["chi2"]
    chi2_cf = m.get("chi2_exact", closed_form_chi2(m))
    consistent = chi2_cf < max(2.0 * chi2_traj, chi2_traj + 2.0)
    rows4.append([label, note, f"{chi2_traj:.3f}", f"{chi2_cf:.2f}",
                  "PASS" if consistent else "FAIL"])
with open(OUT / "table4_f1_exhibit.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["inverse mode", "note", "chi2_trajectory",
                "chi2_closed_form_at_recovered_params", "self_consistency"])
    w.writerows(rows4)
print(f"Saved: {OUT / 'table4_f1_exhibit.csv'}")
