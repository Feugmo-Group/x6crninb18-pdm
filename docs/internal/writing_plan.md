# Writing plan — HTW_PDM for X6CrNiNb18-10 (paper 4 of the series)

Stage 1 of the scientific-writing skill's two-stage process: section outlines
with bullet-point key claims, citations, and data points. This is scaffolding
for drafting, not manuscript prose — see `references/writing_principles.md`
discipline: every bullet below becomes flowing paragraphs in Stage 2, never
bullet lists in the actual manuscript (Methods materials lists excepted).

Source documents this plan draws on (do not re-derive, cite/reference):
- `PAPER_PLAN.md` — positioning, contributions, venue, figure/table list
- `literature_review.md` — 6 threads, ~24 references, gap synthesis
- `../IMPL_REPORT.md` — every numeric result, phase-by-phase
- `../pdm_eqns.md` — full derivation
- `../TIER2_DESIGN.md`, `../TIER3_PLAN.md` — spatial model scope + gates

**Both blockers RESOLVED (2026-07-11)**: venue is *npj Materials
Degradation* (derivation to SI, nuclear/BWR framing leads the Introduction);
arXiv:2510.02872 is confirmed our own prior work (required self-citation,
Introduction states directly this paper completes its applied half, no
hedging). `PAPER_PLAN.md` §0/§2/§6 updated accordingly. Drafting can now
proceed without venue-agnostic hedging anywhere below.

---

## Title (candidates, unchanged from PAPER_PLAN.md §3)

Primary: *"Mechanistic Identification of Passive-Film Growth Kinetics on
Nb-Stabilized Stainless Steel in Boiling-Water-Reactor Hydrothermal Water: A
Point Defect Model Study with Physics-Informed Inverse Analysis"*

Alt (identifiability-first, riskier/more distinctive): *"What Three Exposure
Times Can and Cannot Tell You: Point Defect Model Identifiability for Duplex
Oxide Growth on AISI 347 in High-Temperature Water"*

---

## Abstract (draft last, ~200-250 words, single flowing paragraph, NO
labeled sub-sections per the skill's abstract rule)

Key points to weave in (not headers — one continuous paragraph):
- Motivation: AISI 347/X6CrNiNb18-10 in BWR HTW has no mechanistic
  oxide-growth model; Veile et al. 2024 supply EDX/SIMS depth data but stop
  at empirical power laws.
- What we did: derived HTW_PDM from Li 2020 SCW_PDM, inverted it against the
  real 3-exposure dataset with a physics-informed NSEM (hard-mode
  differentiable-RK4), cross-validated with a deterministic baseline,
  bootstrap ensemble, and an independent spatial (Tier-2) EDX composition
  check.
- Headline numbers: χ² 5.8 (PDM) vs 20.1 (power law); ε_f ≈ 1.7e4 V/cm;
  46.7 wt% zero-parameter Cr-plateau prediction vs ~45% measured; 3.5×
  divergence between PDM and power-law extrapolation at 10 y.
- Methodological contribution: identifiability audit (profile likelihood +
  bootstrap) showing which of 5 parameters 3 exposure times can and cannot
  constrain; documented PINN inverse failure mode (F-1) and a
  self-consistency-check fix/reporting standard.
- So-what: >2000 h designed-experiment recommendation; operating-envelope
  robustness (1.2-1.4× L_bl variation across the full BWR box).

---

## 1. Introduction

**Target**: ~800-1000 words, 4-5 paragraphs. Two-audience interleaving per
`PAPER_PLAN.md`'s venue table synthesis.

Outline:
- **Para 1 (material/application motivation)**: AISI 347 in BWR internals
  (weld-decay resistance via Nb stabilization); current predictive practice
  is either operational ECP/dissolved-O2 threshold criteria (HWC/NWC
  guidance, non-mechanistic) or pure data-driven ML of crack-growth rate
  (cite the 2025 npj MD IGSCC-UQ paper, lit review Thread 4 item 4) — neither
  predicts oxide chemistry/thickness. Cite Veile et al. 2024 as the first
  detailed characterization of this alloy in BWR HTW, and note it stops at
  per-element empirical power laws with no shared physical parameter set
  (lit review Thread 3).
- **Para 2 (PDM lineage + the model gap)**: Macdonald 1992 PDM foundation,
  Macdonald 2011 review, Li et al. 2020 SCW_PDM as the direct ancestor of our
  model (lit review Thread 1-2). No prior PDM/MCM work targets subcritical
  BWR HTW + this alloy + depth-profile inversion together (Thread 1 item 5 is
  the closest analogue, PWR/316L, EIS-fitted not depth-profile-fitted).
- **Para 3 (identifiability gap — methodological)**: 40 years of PDM
  parameter estimation has never been paired with formal identifiability
  analysis (profile likelihood/bootstrap), despite chronic 4-6-parameter fits
  to handfuls of data points; the technique is mature elsewhere (cite the
  2021 battery P2D practical-identifiability paper, systems-biology
  profile-likelihood references, lit review Thread 6).
- **Para 4 (the PINN-PDM precedent + what's still missing)**: name
  arXiv:2510.02872 explicitly as OUR OWN prior work (CONFIRMED — cite
  directly, no hedging) — validates PINN-PDM forward/inverse against
  synthetic FEM benchmarks, catalogues 4 training-stability failure modes,
  never touches real data/identifiability/extrapolation. State plainly that
  this paper completes that paper's applied half: real data replaces
  synthetic FEM benchmarks, plus identifiability analysis, the F-1
  sparse-real-data failure mode, and long-time/operating-envelope
  prediction, none of which 2510.02872 attempts. Also cite PINNverse
  (arXiv:2504.05248) as independent corroboration that the F-1 soft-mode
  failure mode is a real, recognized, active problem class beyond our
  system specifically — this is the ONE place a non-self citation
  strengthens the self-citation's claim (an outside group hit the same wall
  from a different domain).
- **Para 5 (contributions + roadmap)**: enumerate contributions using
  `PAPER_PLAN.md` §1 items 1-11 (condensed to the ~5-6 that fit an
  Introduction paragraph — lead with 1, 3, 4, 8, 5 per the venue-specific
  emphasis chosen), one sentence naming each headline number, close with a
  one-sentence section roadmap.

---

## 2. Model (HTW_PDM derivation)

**Target**: ~500-700 words + equations. DECIDED: full derivation goes to SI
(`pdm_eqns.md` is the SI source); main text carries only the summary below.

Outline:
- Reaction set in the high-density/EO limit (duplex oxide: Cr-rich spinel
  barrier layer + Fe-rich outer-layer crystals + Ni enrichment zone).
- The exact constant-volume collapse: dL_ol/dt = PBR_eff·dL_bl/dt -
  Ω·k11·C_O^r — note explicitly that this is SIMPLER than Li's Eq. 42 and
  both forms were cross-verified to 8.5e-14 nm (IMPL_REPORT Phase A).
- Reduced 6-parameter apparent system {A_bl, b3, C_bl, PBR_eff, C_x, L0}
  (+L_ol0 in Phase B) and the closed-form solutions (log/expm1-stabilized,
  exact at t=0 and t->infinity to 12 digits).
- One paragraph explicitly contrasting with Li 2020's SCW (supercritical,
  500°C) regime vs our subcritical BWR HTW (240°C/7MPa) — this is the
  physical adaptation, not a rederivation from scratch.

---

## 3. Data and deterministic inverse

**Target**: ~700-900 words + Table 2 + Fig 3/4.

Outline:
- Veile Fig. 9 digitization + acceptance check: scan-level power-law refit
  reproduces published k, n exactly (Cr 6.522 vs 6.521, n 0.4964 vs 0.4964).
- Model ladder M1-M5 (Table 2): M1 core fails to reproduce the thick early Fe
  layer; M3's unpenalized degenerate ridge (A_bl~C_bl->infinity, b3->0) as
  the classic PDM identifiability trap, demonstrated explicitly — this is a
  genuinely didactic result worth a full paragraph, since Thread 6 of the
  lit review found no prior PDM paper shows this failure mode happening.
- M4 as the accepted physical model: χ²=5.8 vs power law's 20.1 (Table 2
  last row) — mechanistic model *outperforms* the empirical fit because the
  constant-volume constraint couples Cr+Fe layers; note explicitly this is
  NOT a moral victory from extra parameters (comparable parameter count).
  PBR_eff lands on stoichiometric spinel/magnetite value from the DATA, not
  the prior (report the profile 1-sigma [0.23,2.34] to show the prior isn't
  doing the work).
- Profile-likelihood identifiability map (Fig 4, Table 1 profile columns):
  A_bl/b3/L_ol0 sharply identifiable, PBR_eff weak, L0 unidentifiable
  (needs the air-film prior), C_bl bounded only above ~1 nm/h (cite the
  corrected E1 statement, NOT the stale 0.71 nm/h figure —
  `PAPER_PLAN.md` §6 checklist item).
- Physical recovery: eps_f ~ 1.7e4 V/cm, ordered between Li's SCW value
  (~1e2) and room-T passive films (~1e6) — sensible temperature ordering,
  worth a sentence of physical interpretation.

---

## 4. Physics-informed inverse (NSEM)

**Target**: ~800-1000 words + Table 3/4 + methods cross-reference.

Outline:
- Hard mode (differentiable RK4 through learnable kinetics) as the
  reference: physics exact by construction, recovers the deterministic M4
  optimum to <=0.6% on every parameter (Table 3), chi2 5.79=5.79 exactly.
- Finding F-1: soft-mode joint field+parameter PINN inversion on sparse data
  equilibrates at data-chi2~0 with real ODE residual slack; recovered
  kinetics FAIL the closed-form self-consistency check (Table 4: chi2
  trajectory 0.13 vs closed-form 33.7, and a second run 0.50 vs 21.4 —
  same qualitative signature, both FAIL). Explain the mechanism: on the
  "data fits exactly" manifold physics can't tighten further and its
  gradient there is smaller than the data gradient off it; raising
  lambda_phys does not move the equilibrium (BRDR renormalizes) — this is
  the mechanistic explanation reviewers will want, not just the symptom.
- Propose the self-consistency check (re-solve recovered params exactly,
  compare chi2) as a reporting standard — explicitly extend paper 3's
  Newton-verification-for-forward-problems standard to inverse problems.
  Frame as the concrete, general-purpose remedy that goes beyond
  arXiv:2510.02872's diagnosis-only failure-mode catalogue (lit review
  Thread 5) and beyond PINNverse's constrained-optimization alternative
  (cite as a complementary fix operating on a different lever — ours
  changes the integrator, PINNverse changes the loss formulation).
- Synthetic recovery validation at 2% noise (Table 3 bottom rows: A_bl
  1.35%, b3 3.36%, PBR_eff 0.48%, L0 10.96% [[[the one that matches the
  identifiability-flagged parameter — say so explicitly]]], L_ol0 0.07%) and
  mention the 10% noise ridge-growth result from IMPL_REPORT without
  re-deriving numbers not in a table (or add if time permits).
- Bootstrap ensemble vs profile likelihoods (Fig 5, Table 1 boot columns):
  A_bl/b3/L_ol0 agree closely between methods (cross-validation of both);
  PBR_eff/L0 bootstrap collapses onto the prior while profile stays wide —
  explain WHY (bootstrap perturbs only data; when data has no leverage,
  every resample collapses to the informative prior) and state the
  practical conclusion explicitly: quote profile intervals for
  prior-dominated parameters, not the falsely-narrow bootstrap spread.
- MLP vs KAN parity (Table 3 top rows, 0.14%/0.12% vs 0.11%/0.04%): parity
  not superiority is the honest headline — one sentence, don't oversell.

---

## 5. Spatial validation (Tier 2) — NEW SECTION

**Target**: ~700-900 words + Fig 8/9/10 + Table 5.

Outline:
- Frame explicitly as the independent cross-check that Section 3-4's 0-D
  kinetics are not merely curve-fit but structurally consistent with spatial
  compositional data never used in the Tier-1 fit.
- T2-B steady solve: two defect species (O vacancies outward, cation
  vacancies inward) under the FROZEN Tier-1 constant field; Newton vs
  analytic to 2e-13, NSEM vs Newton to 2.5e-4 (Fig 8b). Structural finding
  (Fig 8a): D cancels from Pe = z*gamma*E*L — the steady profile SHAPE
  carries no diffusivity information, independent of and corroborating the
  T2-A statistical identifiability gate (Table 5) by a completely different
  argument. State explicitly this is (to our knowledge, per lit review
  Thread 1/2) the first explicit statement of this structural-cancellation
  identifiability limit in the PDM literature.
- T2-C transient: quasi-steady error quantification (Fig 9) — 0.65% lag for
  the fast species (tau_mig~5h) vs 5.7% for the slow species (tau_mig~36h),
  scaling as tau_mig/t_growth as physically expected. Frame as quantifying
  an approximation every classical (non-spatial) PDM application makes
  silently — again, first explicit quantification per the lit review.
- T2-D composition map (Fig 10, headline figure): ZERO free parameters —
  Cr-plateau composition from spinel stoichiometry alone (46.7 wt% predicted
  vs ~45% measured), widths at Tier-1 fit quality (+22/-20/+3% across
  exposures). Emphasize this is forward prediction against data NEVER used
  in fitting, the strongest kind of validation.
- T2-A identifiability gate (Table 5) as the discipline that SCOPED Tier 2:
  in-layer defect-transport gradient is weak (47% replicate SD, min
  detectable ~0.9 abs%) — state plainly that Tier 2's inverse scope is
  therefore restricted to interface/zone/composition parameters, defect
  transport stays forward-only, and this restraint is itself evidence of
  rigor (a paper that fit everything anyway would be less trustworthy).

---

## 6. Predictions

**Target**: ~600-800 words + Fig 6/7.

Outline:
- Long-time extrapolation (Fig 6): 10 y PDM prediction 535 nm vs power-law
  1853 nm — 3.5x divergence, directly consequential for structural-material
  life/metal-loss prediction (the nuclear-audience "so what").
- Branch indistinguishability: unbounded-log (M4) vs saturating (M5)
  PDM branches track within ~5% of each other out to 1e5 y — NOT
  distinguishable on any feasible exposure. State the designed-experiment
  logic explicitly: >2000 h discriminates PDM from power law (the
  discriminating experiment IS feasible); no feasible exposure discriminates
  the PDM branches (dissolution rate must come from water chemistry
  reasoning, not more thickness data) — this is a genuine, actionable
  research-program recommendation, worth stating as such.
- (T,[O2]) sensitivity maps (Fig 7): across the full BWR operating box
  (200-285C, 0.01-8 ppm O2), L_bl varies only 1.2-1.4x at 480h and 1.19-1.25x
  at 10y; temperature dominates (Arrhenius + field attenuation), O2
  contributes only a mild Nernstian tilt while dissolution ~ 0. State the
  liquid-phase validity bound explicitly (300C nominal endpoint unreachable
  at 7 MPa; box capped at 285.8C) as a modeling-honesty note, not buried.
- Field-strength ordering across temperature regimes: eps_f ~1.7e4 V/cm
  here, vs Li's ~1e2 (500C SCW) and ~1e6 (room-T passive) — one paragraph
  of physical interpretation tying this back to Section 3.

---

## 7. Discussion

**Target**: ~900-1200 words. This is the section carrying the most
"honesty work" — do not let it read as an afterthought.

Outline:
- Mechanistic reinterpretation of Veile's power-law exponents (parabolic Cr
  -> near-zero dissolution; cubic-like Fe -> constant-volume-constrained
  deposition from a large initial precipitate population) — connect back to
  Section 3's chi2 comparison as the quantitative version of this claim.
- Ni enrichment as an exclusion zone (T2-E, Table 6): partial exclusion
  r_Ni=0.29 (NOT total exclusion — state this contrasts with the naive
  k_Ni~0 assumption in `TIER2_DESIGN.md`'s original design), mass-budget
  evidence (Delta-chi2=6.4 when the outer-layer supply term is dropped) that
  Fe-rich outer crystals draw cations from the base metal through the
  barrier layer, not from re-precipitation alone — a genuinely novel
  mechanistic claim, worth its own paragraph.
- **COMPACT** paragraph, not a subsection: the T2-E shape tension
  (chi2/dof=25.7, predicted widths grow monotonically while data is flat)
  and the Tier-3 response — two physically motivated time-dependent
  Ni-mobility closures (flux-coupled, finite-capacity trapping) BOTH fail
  the synthetic identifiability gate before touching real data (Table 7).
  State this as a second, independent demonstration of the "3-4 exposure
  times cannot support this many parameters" theme (item 3's sibling), not
  a separate result requiring its own section — and as evidence the paper's
  identifiability discipline is applied consistently, not just where
  convenient.
- Limitations, stated plainly and specifically (not a vague hedging
  paragraph): single temperature (240C only — the (T,[O2]) maps in Section
  6 are forward SENSITIVITY, not validated against multi-temperature data);
  open-circuit potential absorbed into A_bl (the Nernst-ECP channel in the
  sensitivity maps is a LOWER BOUND on true O2 sensitivity — measured
  ECP-vs-O2 curves are steeper in the ppb transition region, cite a
  BWR water-chemistry reference per `PAPER_PLAN.md` §6 checklist); 3 time
  points (the central limitation, already the paper's main methodological
  point); EDX-thickness vs total-oxide-mass caveats.
- Close with the SINGLE fix that resolves both the T2-A and T3-A
  identifiability gates: more exposure time points (the >2000 h
  recommendation from Section 6, generalized) — tie Discussion back to
  Predictions so the paper ends on one coherent actionable recommendation,
  not two disconnected ones.

---

## 8. Methods

**Target**: ~600-800 words, can be terser/more technical than other
sections (methods papers 1/3 carry the deep architecture description).

Outline:
- NSEM/SCEN + DVR one-paragraph summary, citing papers 1 and 3 for the
  architecture (this paper does not re-derive the method).
- BRDR loss balancing, TwoPhaseOptimizer (Adam -> L-BFGS) — brief, cite
  paper 1/3 for details.
- MLP/KAN backbone parity as a methods-table entry (Table 3), not
  re-discussed here beyond a pointer.
- Radau/closed-form parity as the numerical-baseline trust exhibit (Fig 2,
  <1e-8 nm generic set, <1e-10 CN residual) — this grounds every other
  claim in the paper, worth stating explicitly as "before any PINN result,
  we establish the deterministic baseline to machine precision."
- Test suite: point to the 25-test suite (tests/test_ode_parity.py +
  tests/test_tier3.py) as the reproducibility artifact, not enumerate every
  test.

---

## 9. Data / Code availability

**Target**: ~150-200 words.
- This example directory (`examples/electrochemistry/x6crninb18_pdm/`) +
  PhysicsNeMo `experimental.models.scen`.
- Digitized Veile 2024 data CSVs with explicit provenance notes
  (`data/veile2024_fig9_means.csv`, `_scans.csv`) — state clearly this is
  digitization of PUBLISHED figures, cited not permission-requiring (flag
  per `PAPER_PLAN.md` §6 sign-off checklist item, confirm before submission).
- Full reproduction commands per script docstring (`Run:` lines already in
  every `src/*.py` file — point to them rather than restate).

---

## Citation map (which literature-review references land where)

Quick cross-reference so drafting doesn't require re-searching:

| Section | Primary citations (from `literature_review.md`) |
|---|---|
| Intro para 1 | Veile 2024 (Thread 3.1); HWC/NWC literature (Thread 4.1); npj MD IGSCC-UQ 2025 (Thread 4.4) |
| Intro para 2 | Macdonald 1992, 2011 (Thread 1.1-2); Li 2020 (Thread 2.1); 316L/321/PWR analogues (Thread 1.4-5) |
| Intro para 3 | Battery P2D identifiability 2021 (Thread 6.2); systems-biology profile likelihood (Thread 6.3); 1996 identifiability foundations (Thread 6.1) |
| Intro para 4 | arXiv:2510.02872 (Thread 5.1); PINNverse arXiv:2504.05248 (Thread 5.3); battery PINN precedents (Thread 5.2) |
| Model | Li 2020 (Thread 2.1) |
| Data/inverse | Veile 2024 (Thread 3.1) |
| Discussion (mixed-conduction analogue) | Bojinov et al. MCM (Thread 1.3, 4.2) |
| Discussion (limitations, ECP) | Dissolved-O2/surface-treatment 304 papers (Thread 3.3-4) |

---

## Next step (per the skill's Stage 2) — UNBLOCKED, drafting underway

Both blockers resolved 2026-07-11. Drafting order (per the skill's suggested
sequence, adapted to this paper's dependency structure): Methods -> Results
(3, 4, 5, 6 in that order, each drafted straight from an existing
figure/table) -> Discussion -> Introduction -> Abstract -> Title. Draft
sections accumulate in `paper/draft.md` as they're written, each pulling
numbers only from `facts_table.md`.
