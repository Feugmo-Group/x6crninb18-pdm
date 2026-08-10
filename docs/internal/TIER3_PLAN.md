# TIER3_PLAN — resolving the T2-E tension + hard-mode PDE inversion

Status: PLAN (nothing implemented). Prerequisites: Tier 1 + Tier 2 complete
(IMPL_REPORT.md through T2-E). Scope: the two open items left by Tier 2 —

1. **Physics gap (T2-E tension)**: chi2/dof = 25.7; the constant-D Wagner
   exclusion model predicts monotonically growing Ni-zone widths (16/24/44 nm)
   while the measured widths are flat (31/43/36 nm). A zone that forms by 72 h
   and then stops growing is the signature of a self-limiting process, not
   steady diffusion ahead of a decelerating interface.
2. **Methodology gap (F-1 for PDEs)**: Tier 1 escaped the soft-mode
   physics-slack failure via the differentiable RK4 hard mode; Tier 2 has no
   ODE shortcut and T2-E fell back to derivative-free multi-start least-squares
   on the Radau engine. The PDE analogue of the hard mode — a differentiable
   forward solver with exact physics — is the missing piece that makes the
   spatial inverse a first-class NSEM capability rather than a scipy fallback.

Discipline carried over unchanged from Tier 2: identifiability gate BEFORE any
fit (T2-A lesson); Tier-1/Tier-2 parameters frozen at their fitted values
(never re-opened); every fit ships with the closed-loop self-consistency check
(F-1 mitigation); population-SD heterogeneity weighting (3.4/11.1/5.0 nm widths,
7 wt% amplitude).

---

## Candidate closures (T3-A decides between them — do NOT pre-commit)

Both replace the constant D_Ni_eff of `tier2_inverse.py` with one added
parameter each (5 observables — 3 widths + 72 h amplitude + the T2-E
mass-budget Delta-chi2 constraint on phi_ol — against 4 parameters):

**M-A — defect-flux-coupled mobility.** The physical carrier of "effective"
(grain-boundary/defect-enhanced) Ni mobility is the defect flux through the
near-interface metal, and T2-C showed that flux decays as the barrier layer
thickens (Pe ~ 1/L, dL/dt = A_bl exp(b3 L)). Closure:

    D_Ni_eff(t) = D0 * (J_defect(t) / J_defect(t_ref))^n ,  t_ref = 72 h

with J_defect(t) proportional to dL_bl/dt from the frozen Tier-1 M4 kinetics
(zero new physics inputs — the time dependence is inherited, only the coupling
exponent n is new). n = 0 recovers T2-E exactly (regression limit). Expected
behaviour: early fast zone build-up, then mobility collapse -> width freeze.

**M-B — finite-capacity interfacial trapping.** Ni accumulates in a
defect/grain-boundary trap layer of finite site density at the receding
interface; uptake saturates. Closure: replace the rejection BC with

    D c'(0) = -v_m(t) (1 - r_Ni) c(0) * (1 - theta(t)),
    d(theta)/dt = v_m(t) (1 - r_Ni) c(0) (1 - theta) / S_cap

where S_cap (nm * wt%) is the trap capacity, the one new parameter.
S_cap -> infinity recovers T2-E (regression limit). Expected behaviour:
zone amplitude and width lock once the trap fills.

Both closures keep the exact moving-frame Wagner PDE engine (method of lines);
only D(t) or the xi = 0 boundary condition changes.

---

## Milestones

### T3-A — closure formulation + identifiability gate (GO/NO-GO) -- DONE, verdict NO-GO

Executed (2026-07-10): `src/tier3_wagner.py` (shared engine, both closures
verified to reproduce the committed T2-E `const` trajectory in their
regression limits: flux n=0 exact, trap S_cap=1e13 exact to < 5e-8 nm),
`tests/test_tier3.py` (9/9 pass), `src/tier3_identifiability.py` (20
replicates/closure, anchored 5x3 multi-start, real Veile population-SD noise
model, T2-E profile-likelihood half-widths refined via a dedicated finer
scan: D_Ni_eff 0.607, r_Ni 0.0804, phi_ol 0.352).

Truth-picking note: the naive S_cap search range (1-500) always saturates
the trap closure within hours at the T2-E base kinetics (rejection flux
integrates to O(1e3-1e4) nm*wt%*h over 480 h), giving W = 0 at every
observation time -- a `RuntimeError: no viable truth found` on the first
run. Widened to geomspace(2e2, 2e4) with a W.min() > 3 nm viability floor
(reject degenerate "zone never forms" truths); flattest trajectory lands at
S_cap ~ 1.1e3 (CV = 0.12), widths [12, 16, 14] nm.

| closure | param | truth | replicate mean +- SD | rel SD | acceptance | verdict |
|---|---|---|---|---|---|---|
| flux (M-A) | n_flux (new) | 1.5 | 1.28 +- 0.41 | 27% | <=30% | PASS |
| flux (M-A) | D_Ni_eff | 3.8 | 3.86 +- 1.43 | 37% | sd <= 1.21 (2x HW) | **FAIL** (1.43) |
| flux (M-A) | r_Ni | 0.29 | 0.283 +- 0.117 | 40% | sd <= 0.161 (2x HW) | FAIL (marginal check, see note) |
| flux (M-A) | phi_ol | 0.57 | 0.785 +- 0.171 | 30% | sd <= 0.705 (2x HW) | PASS |
| trap (M-B) | S_cap (new) | 1125 | 1699 +- 1478 | 131% | <=30% | **FAIL** |
| trap (M-B) | D_Ni_eff | 3.8 | 4.98 +- 2.29 | 60% | sd <= 1.21 (2x HW) | FAIL |
| trap (M-B) | r_Ni | 0.29 | 0.377 +- 0.229 | 79% | sd <= 0.161 (2x HW) | FAIL |
| trap (M-B) | phi_ol | 0.57 | 0.550 +- 0.267 | 47% | sd <= 0.705 (2x HW) | PASS |

**Verdict: both closures NO-GO.** Flux (M-A) is the closer call -- its own
new parameter (n_flux) IS recoverable, and only D_Ni_eff (and, on the
computed number, r_Ni) breach the 2x-profile-halfwidth bar on the base
kinetics, not by a large margin (1.43 vs 1.21 for D_Ni_eff). Trap (M-B)
fails decisively -- S_cap itself is not identifiable (131% rel SD, more than
4x the acceptance bar) and destabilizes every other parameter. Physical
reading: a single scalar exponent (M-A) coupling the Ni mobility to a
kinetics curve that is ALREADY well-constrained by the Tier-1 fit is a much
better-posed inverse problem than a free-standing saturation capacity (M-B)
fit from only 4 scalar observables (3 widths + 1 amplitude) against 4 free
parameters -- the 4-observables-4-parameters system is exactly determined at
best, and M-B's extra parameter has comparatively little independent
leverage on the observables (S_cap mostly trades off against D_Ni_eff and
r_Ni rather than being pinned down on its own).

**Consequence (per the plan's own contingency, Section on candidate
closures): re-scope Tier 3 to forward predict-and-compare.** T3-C's fit to
the real Veile Ni data is out of scope -- neither closure is safe to fit.
T3-B's differentiable hard-mode solver, as originally scoped (to enable a
gradient-based T3-C refit), no longer has a fitting target and should not be
built in that form. The remaining Tier-3 value is: (a) a forward-only
predict-and-compare scan of n_flux (the closer-to-identifiable closure) over
a physically plausible range against the real Ni widths/amplitude, reporting
the envelope rather than a fitted point estimate; (b) documenting this
NO-GO itself as a companion result to T2-A -- Tier 2 flagged the in-layer
defect gradient as unidentifiable from EDX line scans, and this gate now
shows the Ni-zone TIME-DEPENDENCE is unidentifiable from the 3-exposure
summary statistics, for the same underlying reason (too few independent
observables for the parameter count). Both findings point at the same fix:
more exposure time points (the >2000 h designed-experiment recommendation
already on record), not a more flexible closure.

Outputs: `outputs/plot_tier3_identifiability.png`,
`outputs/tier3_identifiability.csv`,
`outputs/tier3_identifiability_verdict.json`.

#### (superseded) original T3-A milestone text, retained for the record

Implement both closures in the existing Radau MOL engine
(`tier2_inverse.py` engine extracted to `src/tier3_wagner.py` so the three
variants — constant-D, M-A, M-B — share one integrator). Then rerun the T2-A
synthetic gate on each: generate synthetic width/amplitude observables from an
assumed truth (n or S_cap chosen so the synthetic widths are flat, i.e. in the
data-relevant regime), add the full Veile instrument + scan-level heterogeneity
model (the T2-A decisive ingredient), refit 20 replicates.

- Deliverable: `src/tier3_identifiability.py`; gate table appended to this file
  (Section 5-style) with replicate rel. SD per parameter per closure.
- **Acceptance (GO)**: the new parameter (n or S_cap) recovers with replicate
  SD <= 30 % AND the r_Ni/D0 estimates do not degrade beyond 2x their T2-E
  intervals. If BOTH closures fail the gate -> re-scope to forward
  predict-and-compare with a scanned n (the honest T2-A-style fallback), skip
  T3-C's fit, and the paper reports the tension + the forward envelope.
- Tests (`tests/test_tier3.py`):
  - `test_wagner_regression_limits`: M-A at n = 0 and M-B at S_cap = 1e9
    reproduce the stored T2-E constant-D trajectory to < 1e-8 (same grid).
  - `test_gate_synthetic_noiseless`: noiseless synthetic refit recovers truth
    to < 1e-3 relative (catches implementation bugs before the noisy gate).

### T3-B' — forward predict-and-compare envelope (REPLACES T3-B/T3-C below, per the T3-A NO-GO) -- DONE

Executed (2026-07-10): `src/tier3_forward_envelope.py`. n_flux scanned over
[-2, 3] at the frozen T2-E base kinetics; n_flux = 0 reference reproduces
T2-E exactly (W = [16.1, 24.2, 44.3], chi2 = 25.7). Outputs:
`outputs/plot_tier3_forward_envelope.png`, `outputs/tier3_forward_envelope.csv`.

**Result -- a genuine disconnect between qualitative shape and quantitative
fit, not a resolution:**
- Non-monotonic (flat/dipping-like-the-data) trajectories occur for
  n_flux in roughly [1.4, 3] -- consistent with T3-A's truth-picking search,
  which independently landed on n_flux ~ 1.5 as the flattest point.
- But forward chi2 in that qualitatively-flat region is WORSE than the
  const closure (chi2 rises from 25.7 at n_flux=0 past 35-70 across [1.4, 3]),
  because the flux-decay mechanism flattens growth by SLOWING it at ALL
  three exposures roughly proportionally, not by selectively suppressing
  the 168->480 h segment the way the real data's dip requires.
- The lowest forward chi2 (22.9, only a modest improvement over 25.7) sits
  at n_flux ~ 0.4 -- a monotonically-growing trajectory, not a flat one --
  because it merely slows the endpoint growth (W_480h: 44.3 -> 36.2,
  landing close to the data's 36 nm) without touching the 72->168 h rise
  the data does not show either.
- **Conclusion**: no single n_flux value is simultaneously shape-correct
  and quantitatively better than the T2-E baseline. The M-A closure is not
  just unfittable (T3-A) -- even inspected purely by eye across its full
  physically plausible range, it cannot reproduce the data's actual
  shape (rise-then-partial-fall) with a single time-independent coupling
  exponent. This strengthens (rather than merely restates) the T2-E/T3-A
  conclusion: the flat width is evidence for a qualitatively different
  mechanism (e.g. non-monotonic mobility, or the interface-coupled
  decaying process T2-E's docstring already flagged), not a parameter this
  closure family is simply mis-tuned on.

Since neither closure passed T3-A, there is no
identified parameter to fit -- the honest deliverable is a forward envelope,
matching how T2-D handled the analogous Tier-2 situation for the in-layer
defect gradient.

- Deliverable: `src/tier3_forward_envelope.py`. Scan n_flux (M-A -- the
  closer-to-identifiable closure; M-B is not worth carrying forward given
  its outright S_cap non-identifiability) over a physically motivated range
  at the T2-E base kinetics (D0, r_Ni, phi_ol held at their T2-E fitted
  values -- never re-opened), plotting predicted W(t)/A(72h) bands against
  the real Veile Ni observables with their population-SD error bars.
- Reports, not fits: the n_flux range for which the predicted width
  trajectory is qualitatively flat/non-monotonic (like the data) vs.
  monotonically growing (like the T2-E const closure) -- a qualitative
  consistency check, explicitly NOT a point estimate or an uncertainty
  interval on n_flux (T3-A showed that isn't defensible).
- Acceptance: none in the fit-quality sense -- this is a descriptive figure
  for the paper, analogous to T2-D's forward Cr-plateau comparison. The
  "result" is the qualitative statement about which closure family is
  physically consistent with a flat zone, with the explicit caveat that its
  parameters are not separately identifiable from the current data.
- Paper framing: pairs with T2-A as a second, independent identifiability
  NO-GO in this system -- both point at the same missing ingredient (more
  exposure time points), not at a modeling failure.

### (superseded by the T3-A NO-GO) T3-B — differentiable hard-mode PDE solver ("real PINN inverse")

Not built. Retained for the record; would be revisited only if future data
(the >2000 h designed experiment) moves either closure to a T3-A GO.

Port the winning-closure Wagner PDE (and the constant-D baseline) to a
differentiable torch implementation: method of lines on the fixed xi grid +
implicit time stepping (CN or BDF2), with gradients through the solve either
by unrolled autograd or implicit-function-theorem adjoints on the per-step
linear solves. Physics exact by construction — zero residual slack possible —
so the F-1 failure mode is structurally eliminated for the spatial inverse,
exactly as RK4 hard mode did for the Tier-1 ODEs.

- Deliverable: `src/tier3_hard_inverse.py` (torch forward + Adam -> L-BFGS on
  the kinetic parameters only, mirroring `inverse_trainer.py` hard mode).
- **Acceptance**:
  - parity: torch forward vs the Radau MOL engine < 1e-6 relative on
    c(xi, t) at the observation times, all three closure variants;
  - gradients: autograd dchi2/dtheta vs central finite differences of the
    Radau engine < 1e-4 relative, every parameter;
  - inverse parity: on the T2-E constant-D problem, the hard inverse
    reproduces the T2-E multi-start global optimum (r_Ni = 0.29,
    D = 3.8 nm^2/h, phi_ol = 0.57) to <= 1 % from a neutral start —
    demonstrating the gradient path replaces multi-start.
- Tests: `test_torch_radau_parity`, `test_torch_gradcheck`,
  `test_hard_inverse_reproduces_t2e`.

### (superseded by the T3-A NO-GO) T3-C — refit the real Veile Ni observables

Fit the T3-A-winning closure with the T3-B hard solver to the real data
(same observables and weighting as T2-E). Multi-start retained as a check
that the gradient path finds the global optimum, not as the primary method.

- Deliverable: updated `outputs/tier2_ni_inverse.json` -> `tier3_ni_inverse.json`
  + profile-likelihood scan for every parameter referenced to the global
  minimum (T2-E lesson: single-start local minima exist).
- **Acceptance (the tension is "resolved")**:
  - chi2/dof <= 3 (vs 25.7) — i.e. within the s^2-calibrated heterogeneity;
  - shape: predicted widths at 72/168/480 h are non-monotone-increasing
    within data SD (the flat-width signature reproduced, not just chi2 luck);
  - self-consistency: parameters re-solved with the independent Radau engine
    reproduce the torch trajectory chi2 to < 1 % (closed-loop F-1 check);
  - mass budget: the phi_ol > 0 finding (Delta-chi2 >= ~6 when dropped)
    survives under the new closure — if it does not, report that honestly
    (it is currently a headline claim).
- If acceptance FAILS (chi2/dof stays >> 3): the tension is physical beyond
  single-mechanism closures — document as in T2-E, keep the forward envelope,
  and the paper scopes Ni to predict-and-compare. This is a valid outcome.

### (superseded by the T3-A NO-GO, folded into T3-B') T3-D — uncertainty + artifact refresh

- D3-style parametric bootstrap (50 resamples of widths/amplitude at their
  population SDs, each refit with the fast T3-B hard inverse) + comparison to
  the T3-C profile likelihoods — same cross-validation logic as D3 (agreement
  for data-identified params; divergence flags prior/structure dominance).
- Regenerate: `tier2_composition --fitted` on the new closure,
  `make_paper_figs.py` / `make_paper_tables.py` (new fig: measured vs
  predicted W(t) and A(t) for constant-D vs winning closure — the
  before/after exhibit; new table row: closure comparison chi2 ladder,
  the Tier-3 analogue of table2's M1-M5 ladder).
- Acceptance: full pytest suite green (existing 12 + new Tier-3 cases);
  IMPL_REPORT.md gains T3-A..D sections in the established format.

### T3-E (optional, only if T3-C passes) — NSEM surrogate of the new physics

Space-time SCEN solve of the winning-closure Wagner PDE (T2-C architecture:
CN in time, DVR in xi, BRDR + TwoPhaseOptimizer), verified against the T3-B
torch solver. Acceptance: rel Linf <= 0.5 % (the standing NSEM bar). Value:
completes the pattern "NSEM carries every model in the paper" and provides
the fast forward map for any (T, [O2]) parametric extension of the Ni story.
Skip without regret if the schedule is tight — T3-B/C are the substance.

---

## Effort + order

Strictly sequential T3-A -> T3-B -> T3-C -> T3-D (each gates the next);
T3-E parallel-izable after T3-C. Rough scale: T3-A one session (the engine
refactor is mechanical, the gate reuses T2-A machinery); T3-B the largest
single item (differentiable implicit stepping + gradcheck); T3-C/D fast once
T3-B exists (~seconds per solve on the 150-node grid).

## What is explicitly out of scope

- Re-opening any Tier-1 (M4) or Tier-2 frozen parameter.
- Poisson-field upgrades, per-species defect transport inverses (T2-A verdict
  stands: forward-only below the ~0.9 abs % gradient detection floor).
- New experimental data. The >2000 h designed-experiment recommendation and
  the time-resolved early-exposure (< 72 h) recommendation both remain paper
  discussion items, now sharpened: the two closures differ most strongly
  in the 0-72 h window where there is no data.
