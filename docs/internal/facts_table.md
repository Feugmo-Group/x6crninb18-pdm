# Facts table — verified numbers for drafting

**Purpose**: every number that can legally appear in the manuscript, in one
place, each tagged with its source so a number is never typed from memory
while drafting. When writing a sentence with a number in it, look it up here
first; if it isn't here, it doesn't go in the paper until it's added here
with a source. Numbers regenerated fresh this session (2026-07-11) in the
`dgx_arm` conda env with real PhysicsNeMo — all matched IMPL_REPORT.md within
expected run-to-run stochastic variation (noted below where relevant).

Sources: `T1`-`T7` = `outputs/paper/table{1-7}_*.csv` (this session, exact).
`IR` = `../IMPL_REPORT.md` (prior sessions' narrative numbers not captured in
a CSV — spot-check before use, most are also reproduced above).

---

## A. Model parameters (M4, the accepted deterministic fit) — source T1, T2

| param | unit | M4 fit | profile 1σ | max Δχ² | verdict | bootstrap mean±std | bootstrap [16,84]% |
|---|---|---|---|---|---|---|---|
| A_bl | nm/h | 0.7266 | [0.625, 0.844] | 50.1 | identifiable | 0.756 ± 0.136 | [0.618, 0.866] |
| b3 | 1/nm | −0.01248 | [−0.0145, −0.0107] | 129.9 | identifiable | −0.01283 ± 0.00207 | [−0.0148, −0.0105] |
| PBR_eff | – | 1.0512 | [0.235, 2.34] | 16.8 | identifiable | 1.061 ± 0.0355 | [1.024, 1.096] |
| L0 | nm | 1.9237 | [0.429, 8.20] | 1.0 | **weak/prior-dominated** | 1.917 ± 0.0340 | [1.884, 1.946] |
| L_ol0 | nm | 118.69 | [71.99, 160.2] | 81.5 | identifiable | 107.3 ± 44.0 | [67.4, 153.4] |
| eps_f (ε_f) | V/cm | 1.73×10⁴ | — | — | derived (α3=0.12, χ=8/3) | — | — |
| χ²/dof (M4) | – | 5.79/1 | — | — | vs power law 20.10/2 | — | — |

**Caution**: bootstrap columns for PBR_eff and L0 are narrower than their
profile-likelihood intervals — this is EXPECTED and must be explained, not
just reported (bootstrap perturbs only data; when data has no leverage on a
parameter, resamples collapse onto the prior). **Always quote the profile
interval, not the bootstrap interval, for PBR_eff and L0** when stating an
uncertainty range in prose.

**C_bl (dissolution rate) bound — IMPORTANT CORRECTION**: an earlier
IMPL_REPORT Phase-B3 draft quoted "C_bl < 0.71 nm/h (95%)". This is STALE.
The corrected Phase-E1 statement: **no bound found below 1 nm/h from
thickness data alone**; the two PDM branches (unbounded-log M4 vs saturating
M5) are indistinguishable to 10⁵ y. Use the corrected statement; do not
reuse 0.71 nm/h anywhere in the manuscript.

---

## B. Model ladder (M1–M5 + power law) — source T2

| model | extra params | n free | χ²(data) | dof | total (w/ priors) |
|---|---|---|---|---|---|
| M1 core | — | 4 | 11.49 | 2 | 12.43 |
| M2 | +C_bl | 5 | 11.49 | 1 | 12.43 |
| M3 | +C_bl +C_x | 6 | 11.49 | 1 | 12.43 (degenerate ridge: A_bl~C_bl→∞, b3→0) |
| **M4 (accepted)** | +L_ol0 | 5 | **5.79** | 1 | 5.80 |
| M5 | +L_ol0 +C_bl | 6 | 5.79 | 1 | 5.80 (C_bl→0⁺, overfitting exhibit) |
| power law (Veile coeffs, evaluated) | k,n per layer | 4 | **20.10** | 2 | 20.10 |
| power law (weighted refit, same objective) | k,n per layer | 4 | **7.84** | 2 | 7.84 |

**REVISED 2026-07-13 (referee fix, source: outputs/paper/referee_stats.json):**
the 20.10 row is Veile's *published* coefficients evaluated under our weighted
objective, NOT a refit. Refit under the same objective the power law reaches
χ² = 7.84 (Cr: k=2.964, n=0.6184, χ²=7.60; Fe: k=52.93, n=0.2571, χ²=0.25).
Information criteria (n=6): M4 AIC 15.8 / BIC 14.8 / AICc undefined (n−k−1=0);
power law AIC 15.8 / BIC 15.0 / AICc 55.8. **AIC is tied — do NOT claim the
mechanistic model "outperforms" on fit quality.** The honest claims: (i)
comparable fit at same/similar cost, (ii) mechanistic interpretability +
coupling, (iii) divergent extrapolation: refit PL gives 3375 nm at 10 y
(6.3× PDM), published coeffs give 1853 nm (3.5×) — either way far outside the
PDM bootstrap band. GoF: χ²=5.79/1 dof → p = 0.0161, error-scale s = 2.41.
Bootstrap n=1000 (deterministic fitter, 0 failures): L_bl(10 y) point 535 nm,
median 537, 68% [482, 607], 95% [445, 705] nm. Per-parameter (mean±std, source
outputs/ensemble_members_1000.csv — now the tab:params bootstrap column):
A_bl 0.736±0.122, b3 −0.01245±0.00202, PBR_eff 1.053±0.0338, L0 1.921±0.0345,
L_ol0 117.4±43.1. The old 50-member PINN-bootstrap values (0.756±0.136 etc.)
are retained in the manuscript only as the hard-mode cross-check sentence. Discriminability (worst Cr
scan rel-SD 22%, n=3 scans → σ_mean 12.7%): 2σ at 1437 h (refit PL) / 2546 h
(published PL); 3σ at 2020 h / 4014 h.

---

## C. NSEM forward + inverse accuracy — source T3

| run | quantity | value | acceptance |
|---|---|---|---|
| forward MLP | rel L∞ L_bl | 1.37×10⁻³ | ≤0.5% PASS |
| forward MLP | rel L∞ L_ol | 1.20×10⁻³ | ≤0.5% PASS |
| forward KAN | rel L∞ L_bl | 1.09×10⁻³ | ≤0.5% PASS |
| forward KAN | rel L∞ L_ol | 3.95×10⁻⁴ | ≤0.5% PASS |
| inverse hard (real) | A_bl dev vs M4 | 0.55% | PASS |
| inverse hard (real) | b3 dev vs M4 | 0.51% | PASS |
| inverse hard (real) | PBR_eff dev vs M4 | 0.00% | PASS |
| inverse hard (real) | L0 dev vs M4 | 0.01% | PASS |
| inverse hard (real) | L_ol0 dev vs M4 | 0.00% | PASS |
| inverse hard (real) | χ² | 5.79 (= M4) | PASS |
| inverse hard (synthetic, 2% noise) | A_bl dev | 1.35% | — |
| inverse hard (synthetic, 2% noise) | b3 dev | 3.36% | — |
| inverse hard (synthetic, 2% noise) | PBR_eff dev | 0.48% | — |
| inverse hard (synthetic, 2% noise) | **L0 dev** | **10.96%** | — (the identifiability-flagged parameter — say so in prose) |
| inverse hard (synthetic, 2% noise) | L_ol0 dev | 0.07% | — |

Phrasing guide: "≤0.6% on every parameter" for the real-data hard-inverse row
(max of the 5 devs above = 0.55%, round generously to "≤0.6%" per
IMPL_REPORT's own convention — do not say "≤0.5%" here, that's the
*forward*-parity bar, a different number).

---

## D. F-1 finding (self-consistency exhibit) — source T4

| inverse mode | χ² trajectory | χ² closed-form at recovered params | self-consistency |
|---|---|---|---|
| hard (real data) | 5.795 | 5.79 | **PASS** |
| soft, λ_phys=1e3 (real) | 0.127 | 33.69 | **FAIL** |
| soft, λ_data=10 (real) | 0.501 | 21.38 | **FAIL** |

**Note on reproducibility**: an earlier run (cited in IMPL_REPORT.md) found
soft/λ_data=10 at chi2 0.004/47.3 — this session's rerun (stochastic PINN
training, different seed path) landed at 0.501/21.38. Both show the SAME
qualitative F-1 signature (tiny trajectory χ², large closed-form χ², FAIL) —
**cite the qualitative pattern and this session's numbers** (they're the
freshly verified ones); do not mix old and new numbers in the same table.

---

## E. Tier-2 (T2-A) identifiability gate — source T5

| shape param | physics mapping | truth | replicate mean±SD | rel SD | verdict |
|---|---|---|---|---|---|
| L_bl | Tier-1 output (known) | 134.0 | 133 ± 3.25 | 2% | IDENTIFIABLE |
| w_int | solid-state interdiffusion D_s | 5.0 | 4.8 ± 0.838 | 17% | IDENTIFIABLE |
| w_out | outer-interface kinetics | 8.0 | 8.09 ± 1.54 | 19% | IDENTIFIABLE |
| A_Ni | k_Ni exclusion ratio | 25.0 | 24.5 ± 1.71 | 7% | IDENTIFIABLE |
| w_Ni | k_Ni ratio + D_Ni | 35.0 | 35 ± 2.69 | 8% | IDENTIFIABLE |
| f_Cr_bl | interfacial flux ratios | 45.0 | 45.6 ± 1.02 | 2% | IDENTIFIABLE |
| **g_bl** | **defect (cation-vacancy) transport gradient** | 1.0 | 0.819 ± 0.466 | **47%** | **WEAK** |

**Minimum detectable in-layer gradient**: 0.93 abs% (2× replicate SD of
g_bl) — this is THE headline number for the T2-A scoping decision, use it
verbatim.

**Systematic floor**: EDX composition accuracy is ±10% RELATIVE
(systematic, not statistical) — f_Cr_bl and A_Ni are floored at ±10%
regardless of the statistical intervals above; state this as a caveat, not
as an error in the table.

---

## F. Tier-2 composition map (T2-D) — source: session run, `outputs/tier2_composition_comparison.csv` (regenerate to re-verify before final draft; numbers below transcribed from this session's run, `--fitted` mode)

| exposure | Cr pred (nm) | Cr meas (nm) | Cr meas SD | Cr dev % | Ni pred (nm) | Ni meas (nm) | Ni meas SD | Ni dev % |
|---|---|---|---|---|---|---|---|---|
| 72 h | 45.5 | 37.3 | 7.7 | +21.9% | 19.0 | 31.3 | 3.4 | −39.4% |
| 168 h | 79.0 | 98.2 | 21.6 | −19.6% | 28.0 | 43.0 | 11.1 | −34.9% |
| 480 h | 138.5 | 134.3 | 5.4 | +3.1% | 53.0 | 36.0 | 5.0 | +47.2% |

**Headline zero-parameter number**: predicted Cr plateau composition =
**46.7 wt%** (from FeCr2O4 stoichiometry: 2×52.0/222.7), vs measured
plateau **~45%** — this is the number to lead the Tier-2 section with; it
uses NO fitted parameters at all (pure stoichiometry + Tier-1-frozen
geometry).

**Predicted in-layer defect gradient** (g_bl_pred, from the T2-B cation-
vacancy profile): 1.45 abs% at 72h, 0.96 abs% at 168h, 0.46 abs% at 480h —
compare directly against the 0.93 abs% detection floor from section E above
(72h marginally detectable, 480h undetectable) — this is the direct link
between T2-D and T2-A, state it explicitly in prose.

---

## G. Tier-2 steady solve (T2-B) — source: this session's fresh NSEM run

| species | Newton vs analytic (L∞) | NSEM vs Newton (rel L∞) | acceptance |
|---|---|---|---|
| OV (O vacancies) | 8.88×10⁻¹⁶ | 2.47×10⁻⁴ | ≤1e-10 / ≤0.5% both PASS |
| CV (cation vacancies) | 2.15×10⁻¹³ | 5.38×10⁻⁵ | ≤1e-10 / ≤0.5% both PASS |

**Structural finding — EMPIRICALLY VERIFIED, not just derived**: D
(diffusivity) cancels from Pe = |z|·γ·E·L. This was directly measured (not
just argued analytically) by calling the real `species_groups()`/
`analytic_steady()` code path at D scaled by 0.01×, 1×, 100× (a 10⁴× range):
Pe was IDENTICAL to 6 significant figures across the scan (10.4849 for OV,
13.9798 for CV, all three D values), and the resulting profile shapes
differed by exactly **0.000e+00** (max abs deviation, to float64 precision)
— while c0 (the dimensional concentration scale, NOT plotted, not visible to
EDX) scaled by exactly 1e-4, inversely proportional to the D range scanned,
as the theory predicts. Cite as: "verified by direct computation over a
10⁴-fold range in D" — this is a genuine numerical demonstration, safe to
describe as empirically confirmed in the manuscript, not merely asserted.
State as independent corroboration of the T2-A statistical gate (section E),
by a completely different (structural/analytic, now also numerically
verified) argument.

Predicted defect concentration scales (from IMPL_REPORT, not rechecked this
session — verify before final use): c0_OV ≈ 2.5×10²⁰ cm⁻³, c0_CV ≈
9.3×10²⁰ cm⁻³ (~2% of spinel cation sites — physical origin of the g_bl~1%
scale used in the T2-A gate).

---

## H. Tier-2 transient (T2-C) — source: this session's fresh NSEM run

| species | R1 quasi-steady lag at τ=1 | R2 Tier-1 flux-reduction error | acceptance |
|---|---|---|---|
| OV (fast, τ_mig~5h) | 0.65% (script) / 0.69% (fig recompute) | 7.69×10⁻⁴ | ≤3% / ≤1% PASS |
| CV (slow, τ_mig~36h) | 5.75% (script) / 5.79% (fig recompute) | 2.70×10⁻⁴ | ≤12% / ≤1% PASS |

Use the trainer script's own reported numbers (0.65%, 5.75%) as the primary
citation; the ~0.04pp difference from the figure-generation script's
independent recomputation is a benign artifact of slightly different
reference-scale handling, not a discrepancy worth mentioning in text.

**Physical interpretation**: lag scales as τ_mig/t_growth, matching physical
expectation — the SLOWER species (larger τ_mig) shows the BIGGER lag,
exactly as the quasi-steady approximation's validity condition predicts.

---

## I. Tier-2 Ni-exclusion inverse (T2-E) — source T6

| param | fit | profile 1σ | max Δχ² | verdict |
|---|---|---|---|---|
| D_Ni_eff | 3.846 nm²/h | (degenerate, see IR note) | 101.6 | identifiable |
| r_Ni | 0.2937 | (degenerate, see IR note) | 123.6 | identifiable |
| phi_ol_supply | 0.5739 | [0.526, 0.895] | 3.3 | weak |
| χ²/dof | 25.71/1 | — | — | — |

**Mass-budget finding**: dropping the outer-layer supply term (phi_ol=0)
worsens χ² to 32.1 (Δχ²=6.4) — evidence the Fe-rich outer crystals draw
cations from the base metal through the barrier layer, not from
re-precipitation alone. State this Δχ²=6.4 number explicitly, it's the
quantitative support for a genuinely novel mechanistic claim.

**r_Ni interpretation**: 0.29 means Ni incorporates at ~30% of the Fe/Cr
rate — PARTIAL exclusion, explicitly NOT total exclusion (contrast with the
original `TIER2_DESIGN.md` k_Ni~0 assumption).

**Documented tension (do not omit)**: predicted Ni-zone widths grow
monotonically (16/24/44 nm at 72/168/480h) while measured widths are flat
(31/43/36 nm) — χ²/dof=25.7 reflects this shape misfit, not just scatter.
State plainly this fit is NOT fully satisfactory and say why (Tier-3
addresses it, see below).

---

## J. Tier-3 closure identifiability gate — source T7

| closure | new param | truth | replicate mean±SD | rel SD | verdict |
|---|---|---|---|---|---|
| flux (M-A) | n_flux | 1.5 | 1.28 ± 0.41 | 27% (PASS own bar) | overall **NO-GO** (D_Ni_eff degrades to 37%) |
| trap (M-B) | S_cap | 1125 | 1699 ± 1478 | **131%** | **NO-GO** (decisive failure) |

Base-parameter degradation under each closure (for the Discussion paragraph,
not a headline number): flux closure pushes D_Ni_eff to 37% replicate SD
(vs 2×profile-halfwidth bar of ~32%, narrowly over); trap closure pushes
D_Ni_eff to 60%, r_Ni to 79% — decisively worse.

**Forward envelope finding (T3-B', no fit)**: scanning n_flux over [-2,3] at
frozen T2-E kinetics — n_flux≈0 (const closure) reference reproduces T2-E
exactly (χ²=25.7); the qualitatively flat-like region (n_flux∈[1.4,3]) has
WORSE forward χ² (35–70+) than the near-const region; best forward χ² (22.9,
only modest improvement over 25.7) sits at n_flux≈0.4, itself still
monotonically growing. **No single value is both shape-correct and
quantitatively competitive** — state this as the paper's honest conclusion
on the Ni-zone tension, not a resolved result.

---

## K. Predictions (long-time, sensitivity maps) — source IR (verify against
`outputs/plot_long_time.png` and `plot_sensitivity_maps.png` numeric log
output before final draft; these matched exactly on this session's rerun)

| quantity | value |
|---|---|
| L_bl at 10 y, PDM (M4) | 535 nm |
| L_bl at 10 y, PDM (M5, saturating branch) | 535 nm (tracks M4 within ~5% to 10⁵ y) |
| L_bl at 10 y, power-law extrapolation | 1853 nm |
| **Divergence ratio (power law / PDM at 10 y)** | **3.5×** |
| Designed-experiment threshold (PDM vs power law) | >2000 h |
| PDM-branch discrimination | not feasible at any practical exposure (to 10⁵ y) |

| sensitivity map quantity | range |
|---|---|
| L_bl spread at 480h across BWR box, ΔG⁰_R=25 kJ/mol | ratio 1.20 ([122.9, 147.2] nm) |
| L_bl spread at 480h, ΔG⁰_R=50 kJ/mol | ratio 1.27 ([119.4, 151.2] nm) |
| L_bl spread at 480h, ΔG⁰_R=100 kJ/mol | ratio 1.42 ([112.5, 159.4] nm) |
| L_bl spread at 10y, ΔG⁰_R=25 kJ/mol | ratio 1.19 ([491.8, 582.9] nm) |
| L_bl spread at 10y, ΔG⁰_R=50 kJ/mol | ratio 1.21 ([487.4, 587.8] nm) |
| L_bl spread at 10y, ΔG⁰_R=100 kJ/mol | ratio 1.25 ([478.6, 597.7] nm) |
| BWR box | 200–285°C at 7 MPa, 0.01–8 ppm O₂ |
| Liquid-phase validity cap | 285.8°C at 7 MPa (300°C nominal endpoint unreachable) |

Phrasing guide: "1.2–1.4× across the operating box" is the safe rounded
summary spanning all three ΔG⁰_R scans at 480h; "1.19–1.25×" for the 10y
horizon — do not conflate the two time horizons into one range.

---

## L. Field strength ordering (physical interpretation, for Discussion)

| regime | ε_f (V/cm) | source |
|---|---|---|
| This work (240°C BWR HTW) | 1.73×10⁴ | section A above |
| Li 2020 (500°C SCW) | ~10² | IR, cite Li 2020 directly if quoting |
| Room-T passive films | ~10⁶ | IR, cite a room-T passivity reference if quoting |

Physically sensible ordering with temperature (higher T → lower field
strength, consistent with thermally-assisted ionic transport) — state as
supporting physical evidence for the fitted b3, not as a proven trend (only
3 literature anchor points).

---

## M. Digitization acceptance (Phase A, for Data section)

| element | k (fit) | k (Veile) | n (fit) | n (Veile) |
|---|---|---|---|---|
| Cr | 6.522 | 6.521 | 0.4964 | 0.4964 |
| Fe | 54.6 | 64.51 | 0.249 | 0.221 |

**Note**: Fe read-offs are digitization-limited (large physical scatter in
the discrete-crystal outer layer, SD 50-80% of mean); state that the MEANS
file (not scan-level Fe) carries the authoritative values used in all fits.

---

## N0. Baseline validation matrix (for Model/Methods sections) — source IR
(Phase B1-B2 validation matrix; re-verify against `tests/test_ode_parity.py`
before final use if not re-running this session)

| check | result |
|---|---|
| Radau vs closed form (generic + Li Table 5 params) | < 1e-8 nm |
| Affine L_ol closed form vs Li's own Eq. 42 form | < 1e-10 nm |
| Crank-Nicolson residual of the exact solution | < 1e-10 |
| Nondimensional roundtrip | 1e-12 |
| Demo-set parity, THIS session's fig2 regeneration | bl 4.63e-11 nm, ol 5.07e-11 nm (source: `make_paper_figs.py` stdout, this session) |

Use "< 1e-10 nm" for the affine-vs-Li's-Eq.-42 cross-check specifically (not
the unrelated "8.5e-14 nm" figure that appeared in early planning notes but
was never independently re-confirmed this session — do not use that number
without rerunning the specific comparison it refers to).

---

## N. Test suite (for Methods, reproducibility statement)

25/25 tests pass (`tests/test_ode_parity.py` + `tests/test_tier3.py`),
verified fresh this session with real PhysicsNeMo installed. Do not quote a
stale count from IMPL_REPORT (which predates the Tier-3 test additions —
12/12 was the Tier-1/2 count before Tier 3 existed).

---

## O. Known reproducibility notes (read before drafting numbers, not for the paper itself)

1. All Tier-1/Tier-2 numbers regenerated fresh this session in a real
   PhysicsNeMo environment match IMPL_REPORT.md to the precision reported
   (deterministic scipy fits, deterministic Newton solves) — safe to cite
   either source interchangeably for those.
2. NSEM-trained numbers (forward parity, hard-inverse recovery, F-1 soft
   exhibit, T2-B/T2-C NSEM-vs-Newton/quasi-steady-lag) are seed/training-
   stochastic — this session's numbers are given above and are the ones
   verified in THIS environment; they agree with IMPL_REPORT to within
   normal training variation except the soft/λ_data=10 F-1 row (section D),
   which differs in magnitude but not qualitative signature. **Use this
   session's numbers as primary** (freshest, verified end-to-end together)
   and do not silently blend old and new numbers for the same quantity.
3. Sections F, G (part), K numbers are transcribed from this session's
   terminal output during the conversation rather than re-read from a
   freshly reopened CSV/log file at doc-writing time — spot-check the
   underlying `outputs/tier2_composition_comparison.csv` and script stdout
   once more immediately before they go in the actual manuscript draft, as
   a final safety check.
