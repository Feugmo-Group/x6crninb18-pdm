# SPDX-FileCopyrightText: Copyright (c) 2026 Conrard Giresse Tetsassi Feugmo
# SPDX-License-Identifier: Apache-2.0

"""Paper tables T5-T7 (PAPER_PLAN.md Section 5, Tier-2/3 items) under
outputs/paper/.

  table5_t2a_identifiability.csv — T2-A synthetic identifiability gate:
    shape parameter, truth, replicate mean/SD, rel SD, verdict.
    From outputs/tier2_identifiability.csv (src/tier2_identifiability.py).
  table6_t2e_ni_exhibit.csv — T2-E Ni-exclusion fit: r_Ni, D_Ni_eff,
    phi_ol_supply, profile intervals, chi2/dof, mass-budget Delta-chi2.
    From outputs/tier2_inverse_fit.csv (src/tier2_inverse.py).
  table7_t3_closure_nogo.csv — Tier-3 closure identifiability gate: 2
    closures x 4 parameters, replicate rel SD vs acceptance threshold,
    verdict. From outputs/tier3_identifiability.csv
    (src/tier3_identifiability.py) + outputs/tier3_identifiability_verdict.json.

Run:  python scripts/make_paper_tables_tier2.py   (AFTER
      src/tier2_identifiability.py, src/tier2_inverse.py, and
      src/tier3_identifiability.py have been run)
"""

import csv
import json

from htw_pdm.paths import PAPER_OUT as OUT  # noqa: E402
from htw_pdm.paths import ROOT  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)


def _missing(name: str, cmd: str) -> None:
    print(f"MISSING: outputs/{name} -- run `{cmd}` first")


# ── Table 5 — T2-A identifiability gate ────────────────────────────────────────
src5 = ROOT / "outputs" / "tier2_identifiability.csv"
if src5.exists():
    with open(src5) as f:
        rows = list(csv.DictReader(f))
    with open(OUT / "table5_t2a_identifiability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape_param", "physics_mapping", "truth", "replicate_mean",
                    "replicate_sd", "rel_sd", "verdict"])
        mapping = {
            "L_bl": "Tier-1 output (known)", "w_int": "solid-state interdiffusion D_s",
            "w_out": "outer-interface kinetics", "A_Ni": "k_Ni exclusion ratio",
            "w_Ni": "k_Ni ratio + D_Ni", "f_Cr_bl": "interfacial flux ratios",
            "g_bl": "defect (cation-vacancy) transport gradient",
        }
        for r in rows:
            w.writerow([r["param"], mapping.get(r["param"], ""), r["truth"],
                        r["replicate_mean"], r["replicate_sd"], r["rel_sd"],
                        r["verdict"]])
    print(f"Saved: {OUT / 'table5_t2a_identifiability.csv'}")
else:
    _missing("tier2_identifiability.csv", "python -m htw_pdm.tier2_identifiability")

# ── Table 6 — T2-E Ni-exclusion exhibit ────────────────────────────────────────
src6 = ROOT / "outputs" / "tier2_inverse_fit.csv"
if src6.exists():
    with open(src6) as f:
        rows = list(csv.DictReader(f))
    with open(OUT / "table6_t2e_ni_exhibit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "fit", "profile_1sig_lo", "profile_1sig_hi",
                    "max_dchi2", "verdict"])
        # A zero-width interval (sig1_lo == sig1_hi) is a degenerate profile:
        # only one coarse-grid point fell inside Delta-chi2 <= 1 because the
        # D_Ni_eff-r_Ni ridge lets the profile minimum sit off the central fit.
        # It is NOT a real interval — blank it and flag the degeneracy instead
        # of reporting a bogus pair that does not bracket the fit value.
        partner = {"D_Ni_eff_nm2_h": "r_Ni", "r_Ni": "D_Ni_eff"}
        for r in rows:
            lo, hi, verdict = r["sig1_lo"], r["sig1_hi"], r["verdict"]
            if lo and hi and lo == hi:
                lo, hi = "", ""
                verdict = (f"degenerate (correlated with "
                           f"{partner.get(r['param'], 'ridge partner')})")
            w.writerow([r["param"], r["fit"], lo, hi, r["max_dchi2"], verdict])
    print(f"Saved: {OUT / 'table6_t2e_ni_exhibit.csv'}")
    print("  Note: chi2/dof = 25.71/1 (constant-D closure); shape tension "
          "(monotone-predicted vs flat-measured widths) documented in "
          "IMPL_REPORT.md T2-E and TIER3_PLAN.md -- NOT resolved by this fit; "
          "report both the fit AND the tension in the Discussion (item 11).")
else:
    _missing("tier2_inverse_fit.csv", "python -m htw_pdm.tier2_inverse")

# ── Table 7 — Tier-3 closure NO-GO summary ─────────────────────────────────────
src7 = ROOT / "outputs" / "tier3_identifiability.csv"
verdict7 = ROOT / "outputs" / "tier3_identifiability_verdict.json"
if src7.exists():
    with open(src7) as f:
        rows = list(csv.DictReader(f))
    verdicts = {}
    if verdict7.exists():
        verdicts = json.load(open(verdict7))["verdicts"]
    with open(OUT / "table7_t3_closure_nogo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["closure", "param", "truth", "replicate_mean",
                    "replicate_sd", "rel_sd", "closure_verdict"])
        for r in rows:
            w.writerow([r["closure"], r["param"], r["truth"], r["repl_mean"],
                        r["repl_sd"], f"{float(r['rel_sd']):.1%}",
                        verdicts.get(r["closure"], "")])
    print(f"Saved: {OUT / 'table7_t3_closure_nogo.csv'}")
    print(f"  Closure verdicts: {verdicts}")
else:
    _missing("tier3_identifiability.csv", "python -m htw_pdm.tier3_identifiability")
