# IMPL_REPORT — x6crninb18_pdm

Run log and findings. Plan: `../X6CrNiNb18/PLAN_PDM_X6CrNiNb18.md`. Derivation: `pdm_eqns.md`.

## Phase A — derivation + data (2026-07-05) DONE

- `pdm_eqns.md` written: full HTW_PDM derivation from the Li 2020 SCW_PDM, reduced to the
  6-parameter apparent system {A_bl, b3, C_bl, PBR_eff, C_x, L0} (+ L_ol0 added in Phase B).
  During the derivation the constant-volume substitution was found to collapse exactly:
  dL_ol/dt = PBR_eff*dL_bl/dt - Omega_ol*k11*C_O^r, so L_ol(t) = PBR_eff*(L_bl(t)-L0) + C_x*t
  in closed form (simpler than Li Eq. 42; both implemented and cross-checked).
- Veile Fig. 9 digitized: `data/veile2024_fig9_means.csv` (authoritative, from text) +
  `data/veile2024_fig9_scans.csv` (scan-level read-offs).
- **Acceptance PASSED**: scan-level linear LSM refit of L = k t^n reproduces Veile exactly for
  Cr (k = 6.522 vs 6.521, n = 0.4964 vs 0.4964). Fe read-offs are digitization-limited
  (k = 54.6 vs 64.51, n = 0.249 vs 0.221) — consistent with the enormous physical scatter of the
  discrete-crystal outer layer (SD ~ 50-80 % of mean); means file carries the authoritative values.

## Phase B1-B2 — Radau baseline + parity (2026-07-05) DONE

- `src/baseline_ode.py`: Radau vs closed forms agree to ~4e-11 nm (demo set) and ~7e-9 nm
  absolute on µm-scale Li Table 5 curves (>= 10 significant digits). The affine L_ol closed form
  and Li's Eq. 42 form agree to 8.5e-14 nm.
- Closed form stabilized in log space (log1p/logaddexp) — exact at t = 0 and t -> infinity
  (L_bl(1e9 h) = L_bl,ss to 12 digits).
- Li Table 5 regression: HCM12A and 316L thickness curves at the paper's tens-of-µm scale.
  **Open check**: the printed Table 5 values give 316L L_bl,ss = 64.8 µm, which exceeds the
  paper's narrative "~37 µm total after 10 y". The extraction had already flagged the printed
  table as possibly inconsistent. Does not affect our fits (we fit our own parameters).

## Phase B3 — deterministic inverse + identifiability (2026-07-05) DONE

`src/baseline_fit.py`, weighted LSQ on layer means (sigma = SD/sqrt(n)), soft log-prior on
L0 (~2 nm, sigma 0.6) only. (The PBR_eff prior was removed 2026-08-07 — see below and the
"PBR_eff prior" section of README.md.)

Model ladder (chi2_data / total incl. priors), refreshed after the prior removal:
- M1 core (4p): 7.15 / 7.15 — cannot reproduce the thick early Fe layer with strict
  constant-volume coupling from L_ol(0)=0; reaches PBR_eff = 2.43 instead.
- M2 (+C_bl): identical to M1; C_bl driven to 0.
- M3 (+C_bl, +C_x): chi2 4.11, but at the degenerate ridge A_bl -> 956 nm/h, b3 -> -4.4e-6
  (linear-growth escape direction) — lowest chi2 in the ladder and rejected on physics
  (155 nm vs 535 nm at 10 y). The classic PDM identifiability trap, demonstrated explicitly.
- **M4 (+L_ol0, the accepted model): chi2 5.79.**
  A_bl = 0.7269 nm/h, b3 = -0.012487 1/nm, PBR_eff = 1.0962, L0 = 1.9237 nm, L_ol0 = 115.6 nm.
  L_ol0 ~ 116 nm quantifies rapid initial outer-crystal precipitation (Veile see 268 nm
  crystals already at 72 h).
- M5 (+C_bl): pure interpolation (chi2 1.2 with L0 = 33 nm, 3.6 sigma against the air-film
  prior) — rejected by the prior, kept as the overfitting exhibit.
- M6 (+L_ol0, +C_x): chi2 5.51 at PBR_eff = 5.15, C_x = -0.94 nm/h. **Diagnostic only, never
  a candidate.** Added 2026-08-07 to settle whether the fitted PBR_eff conflicts with the
  chromium mass balance. It does not, for two reasons (`analyse_pbr_closure.py`):
  (a) the stoichiometric value is phase-dependent, spanning [0.52, 3.79] over the candidate
  barrier spinels (`src/mass_balance.py`), which contains M4's 1.096; and (b) M4 pins C_x=0
  so its PBR_eff is a net-accumulation rate, not the production rate the mass balance fixes.
  Imposing the FeCr2O4 value 2.05 with C_x free gives chi2 5.671 — an improvement of 0.12 —
  at C_x = -0.24 nm/h (~9 ug Fe/(dm2 h) release). Profiling PBR_eff over [0.5, 8] moves chi2
  by 0.89 total, so the delta-chi2 = 1 interval is the entire range: with L_bl(t) fixed the
  outer-layer solution has 3 parameters against 3 observations and is saturated.

Physical recovery (alpha3 = 0.12 assumed, chi = 8/3, gamma = 22.6 /V at 513 K):
**field strength eps_f ~ 1.7e4 V/cm** — between Li's 112 V/cm (500 C SCW, thermally conductive
film) and ~1e6 V/cm (room-T passive films). Sensible ordering with temperature.

Identifiability (profile likelihood, M4): A_bl, b3, L_ol0 sharply identifiable; PBR_eff weakly
(1-sigma [0.23, 2.3], prior-assisted); **L0 unidentifiable** from 72-480 h data (needs the
air-film prior). Dissolution: **C_bl < 0.71 nm/h (95 %)** — i.e. the data only exclude
destruction rates comparable to the growth prefactor; k_d < 1.3e-12 mol cm^-2 s^-1
(cf. Li's SCW value 2.4e-13). Ultrapure-water physics, not the data, motivates C_bl ~ 0.

Long-time extrapolation (the paper's "so-what"): at 10 y the PDM (M4) predicts L_bl = 535 nm
(M5 saturating branch: 511 nm) vs 1853 nm from power-law extrapolation — a 3.5x difference in
predicted barrier-layer thickness, and the two PDM branches are indistinguishable on 480 h data
(motivates a >2000 h exposure as the designed experiment).

Fit quality vs the empirical law: PDM M4 chi2 = 5.8 (5 free) vs power-law chi2 = 20.1 (4 free,
same weights) — the mechanistic model *outperforms* the power law on the joint Cr+Fe data
because the constant-volume constraint couples the layers.

## Phase C — NSEM forward (2026-07-06) DONE

`src/physics.py` (nondim groups t_c = 480 h, L_c = 100 nm; CN residuals; exp clamp) +
`src/trainer.py` (two time-only SCENElementNetworks, softplus positivity, DVR log-clustered
time grid Nt = 24, IC pretrain to <1e-12, BRDR over {ode_bl, ode_ol, ic},
TwoPhaseOptimizer Adam(2000) -> L-BFGS(3)).

Acceptance (rel Linf vs closed form <= 0.5 %): **MLP PASS** (L_bl 0.14 %, L_ol 0.12 %,
final loss 1e-7, ~30 s CPU); **KAN PASS** (0.11 % / 0.04 %, loss 7.5e-8). Note: kte time
mapping requires |alpha| < 1 in DVRMapper — use mapping_t: log with alpha_t = 1.5.

## Phase D — NSEM inverse (2026-07-06) DONE

`src/inverse_trainer.py`, two modes (conf/inverse_config.yaml `inverse.mode`):

**Finding (soft mode / F-1): joint field+parameter PINN inversion leaks misfit into ODE
slack on sparse data.** With 3 time points the BRDR equilibrium sits at data chi2 ~ 0 but ODE
residual ~1e-3 (~4 nm/step of CN slack); the recovered kinetics FAIL the closed-form
self-consistency check (chi2 34 vs the network's 0.1); raising fixed lambda_phys to 1e3 does
not move the equilibrium (BRDR renormalizes). Root cause: on the "data fits exactly" manifold
the physics cannot tighten, and its gradient there is smaller than the data gradient off it.
This is the sparse-data analogue of rpdm FAILURES #3 (free variables with escape directions).

**Fix (hard mode, the reference)**: trajectory generated by differentiable RK4 through the
learnable kinetics — physics exact by construction, zero slack possible; only the kinetic
parameters are optimized (Adam 500 @ lr 1e-2 + L-BFGS). The `chi2 (closed form at recovered
params)` self-consistency check is now a first-class output of both modes.

Results (hard mode):
- **Real Veile data**: recovers the deterministic M4 optimum to <= 0.6 % on every parameter
  (A_bl 0.7309 nm/h, b3 = -0.012551 1/nm, PBR_eff 1.0962, L0 1.924 nm, L_ol0 115.57 nm;
  eps_f = 1.73e4 V/cm), chi2 = 5.79 = baseline, self-consistency PASS.
- **Synthetic recovery (2 % noise)**: A_bl 1.4 %, b3 3.4 %, PBR_eff 0.6 %, L_ol0 0.16 %;
  L0 11 % (prior-dominated — exactly the parameter the identifiability analysis flags).
  At 10 % noise the A_bl/b3 deviations grow to ~45-60 % along their correlation ridge —
  statistical, not methodological (soft-mode PINN at same noise shows the same ridge).

The soft mode is retained (documented) as the architecture required for the Tier-2 spatial
model, where no integrator shortcut exists; there the fix is dense collocation in the data
dimension + the self-consistency check against a Newton BVP solve.

## Validation matrix (2026-08-07) — `python -m pytest tests/ -q`: 38/38 PASS

| Check | Result |
|---|---|
| Radau vs closed form (generic + Li Table 5) | < 1e-8 nm |
| Affine L_ol vs Li Eq. 42 form | < 1e-10 nm |
| t=0 / t->inf limits; C_bl->0 continuity (log/expm1-stable form) | exact |
| Veile Cr power-law refit (digitization) | k 6.522/6.521, n 0.4964/0.4964 |
| CN residual of exact solution | < 1e-10 |
| Nondim roundtrip | 1e-12 |
| Hard RK4 integrator vs closed form | < 1e-9 |
| NSEM forward MLP / KAN vs closed form | 0.14 % / 0.11 % (PASS <= 0.5 %) |
| Inverse hard: real data vs deterministic baseline | <= 0.6 % all params, chi2 5.79 = 5.79 |
| Inverse hard: synthetic recovery | PASS (see above) |
| Inverse soft: self-consistency | FAIL — documented finding F-1 |

## Phase E1 — result figures (2026-07-06) DONE

`plot_results.py`: three figures from the deterministic M4/M5 fits (no NSEM checkpoint
needed).

- `plot_fits_vs_data.png`: Cr/Fe layer thickness vs t, M4 vs power law, linear + log-log.
- `plot_long_time.png`: M4 (unbounded log growth, C_bl = 0) vs M5 (C_bl -> 0+, saturating
  at L_bl,ss = 2122 nm) vs power-law extrapolation, out to 1e5 y. **Correction to the
  Phase B3 note above**: the unconstrained M5 fit does not separate from M4 within any
  practical horizon — both branches track within ~5% of each other out to 1e5 y (the
  saturating time constant scales as 1/(b3*C_bl), and C_bl is driven to ~2e-12 nm/h by
  the 3-point data, giving a saturation timescale far beyond 1e5 y). The two PDM branches
  are therefore not just indistinguishable on the 72-480 h data — they stay
  indistinguishable over any physically relevant extrapolation horizon, while the
  power-law fit diverges unphysically (10x the PDM value by 10 y). The dissolution-rate
  95% upper-bound scan in `baseline_fit.py` (`C_bl < ...`) currently reports "no bound
  found below 1 nm/h" with the present code/data — the earlier "C_bl < 0.71 nm/h" figure
  in this file's Phase B3 section is stale and should not be reused without rerunning
  the scan.
- `plot_profile_likelihood.png`: Delta-chi2 vs log-parameter for all 5 M4 parameters,
  reproducing the Stage-3 identifiability table (A_bl/b3/L_ol0 sharply identifiable,
  PBR_eff weak, L0 unidentifiable) as a figure.

## Phase D3 — ensemble over seeds for uncertainty (2026-07-06) DONE

`src/uncertainty_ensemble.py`: parametric bootstrap of the real Veile fit. 50 resamples
of the Cr/Fe means (Gaussian noise at each point's own reported SD-of-the-mean), each
refit with the hard-mode NSEM inverse (differentiable RK4, kinetics-only — no field
networks, ~14 s/fit), giving an empirical posterior for the 5 M4 parameters. Compared
against the B3 deterministic profile-likelihood 1-sigma intervals (`baseline_fit.py`).

| param | ensemble mean | ensemble std | ensemble [16,84]% | profile 1-sigma |
|---|---|---|---|---|
| A_bl | 0.756 | 0.137 | [0.618, 0.866] | [0.626, 0.845] |
| b3 | -0.01284 | 0.00207 | [-0.0148, -0.0106] | [-0.0145, -0.0108] |
| PBR_eff | 1.385 | 0.975 | [6.5e-11, 2.271] | [0.245, 2.32] |
| L0 | 1.919 | 0.036 | [1.883, 1.950] | [0.429, 8.20] |
| L_ol0 | 86.1 | 75.5 | [3.8e-10, 161.4] | [25.8, 210.6] |

**Finding** (rerun 2026-08-07, after the PBR_eff prior was removed): the two
data-identified parameters (A_bl, b3 — "IDENTIFIABLE" in the B3 profile scan) show
close agreement between the bootstrap ensemble and the profile likelihood,
cross-validating both methods. That part is unchanged.

What changed is instructive, and is the cleanest demonstration in this study of what a
prior was actually doing. With its prior in place, PBR_eff had a bootstrap spread of
0.036 against a profile interval of [0.235, 2.34] — a factor of ~60 narrower, because
the bootstrap perturbs only the *data*, so when the data has no leverage every resampled
refit collapsed onto the prior. With the prior removed, the same 50 resamples give a
spread of 0.975 over [6.5e-11, 2.27], which now *matches* the profile interval
[0.245, 2.32]. The narrow spread was never a property of the data; it was the prior,
reported as if it were a measurement. The same reading applies to L_ol0 (44 -> 75.5),
which trades off against PBR_eff. L0 keeps its prior and still shows the collapsed
signature (spread 0.036 against a profile interval spanning [0.429, 8.20]) — this is
what a prior-set parameter looks like, and it is now the only one.

Several resampled fits drive PBR_eff and L_ol0 to ~0, which is why the 16th percentile
sits at 1e-11 rather than at a physically sensible value: with three exposure times and
no prior, some resamples genuinely contain no information distinguishing a thin
outer layer that grows fast from a thick one that barely grows.

**Practical conclusion**: quote the profile-likelihood interval for L0 (data alone
cannot constrain it — the narrow bootstrap spread is a property of the assumed prior,
not of the data), and the bootstrap (or profile, they now agree) interval for
A_bl/b3/PBR_eff/L_ol0. `plot_uncertainty_ensemble.png` shows the per-parameter
histograms against the profile-likelihood band.

## Phase E2 — parametric (T, [O2]) sensitivity study (2026-07-06) DONE

Forward sensitivity study under sourced assumptions (NOT a re-fit — the scoping note
below the D3 section still holds and is now encoded in the module docstrings).

- `src/physics_parametric.py`: (T, [O2]) -> apparent parameters, anchored at M4.
  Channels, each traced to the Li 2020 structure: (1) Arrhenius on k3^00 (Li Eq. 9)
  with dG0_R as the ONE explicitly assumed scan parameter (not identifiable from
  single-temperature Veile data); (2) b3 ~ gamma(T) ~ 1/T exactly (eps_f T-independent);
  (3) optional ideal-Nernst ECP shift of A_bl with [O2] (lower bound on the true
  mixed-potential response — swap in a measured ECP map when available); (4) dissolution
  C_bl ~ C_O^(1/2) (Li Table 4), inactive at the M4 default C_bl = 0. Water density /
  pKw from embedded IAPWS tables. Validity T in [200, 285] C: at 7 MPa water boils at
  285.8 C, so the plan's nominal 300 C endpoint is unreachable in the liquid phase.
  Self-check: reference condition (240 C, 0.4 ppm) reproduces M4 exactly.
- `src/parametric_trainer.py` + `conf/parametric_config.yaml`: 9 Sobol conditions over
  the box (reference always prepended), per-condition SCEN net pairs, joint
  TwoPhaseOptimizer with one shared BRDR (rpdm_nsem parametric discipline). Note: the
  joint 27-term BRDR needs a longer schedule than the single-condition forward run —
  2000 Adam / 3x400 L-BFGS lands at ~1e-2 rel Linf; the config default 5000 / 5x600
  passes at **worst-condition rel Linf 3.0e-3 <= 0.5 %** (all 9 conditions vs their own
  closed forms).
- `plot_sensitivity_maps.py` -> `plot_sensitivity_maps.png`: L_bl(480 h) and L_bl(10 y)
  contour maps over the box for dG0_R in {25, 50, 100} kJ/mol, NSEM validation points
  overlaid. Headline numbers: across the whole BWR operating box the barrier-layer
  spread is modest — max/min ratio 1.20-1.42 at 480 h and 1.19-1.25 at 10 y depending
  on dG0_R; temperature dominates (Arrhenius + weaker field attenuation at high T),
  [O2] contributes only the mild Nernstian ECP tilt while C_bl = 0. The long-time
  prediction is therefore robust to operating-condition uncertainty within the
  liquid-phase envelope — the npj "so-what" of this phase.
- Tests: 3 new cases in `tests/test_ode_parity.py` (reference reproduction, channel
  signs/scalings incl. sqrt-C_O dissolution and ECP-off invariance, liquid-range
  guard). Suite: **12/12 PASS**.

## Phase E3 — Tier-2 spatial model design (2026-07-06) DONE (design only)

`TIER2_DESIGN.md`: spatial HTW_PDM vs the Veile EDX depth profiles (Figs. 4/6/8).
Key design decisions: Landau-transformed two-moving-boundary NSEM reusing the
rpdm_nsem component stack; Ni enrichment via a k_Ni ~ 0 exclusion closure; Tier-1
ODEs as the mandatory flux-integrated regression limit; F-1 mitigation is mandatory
(no RK4 hard mode exists for PDEs) — dense collocation + Newton-BVP self-consistency
+ Tier-1 parameters frozen at M4 with D3 bounds. Gate before any implementation:
a synthetic-profile identifiability study (T2-A); if in-layer diffusivities prove
unidentifiable (plausible — the Cr-rich layer is compositionally flat), Tier 2
re-scopes to forward predict-and-compare.

## Phase E4 — paper figures + tables archive (2026-07-06) DONE

`make_paper_figs.py` + `make_paper_tables.py` -> `outputs/paper/` (regenerate with two
commands; outputs/ stays untracked):
- fig1_model_schematic.png (duplex-oxide schematic + potential distribution — new),
  fig2_baseline_verification.png (Radau/closed-form parity + Li Table 5 regression —
  new), fig3-fig7 copied from the existing result scripts.
- table1_parameters.csv (M4 fit + profile 1-sigma + bootstrap columns from
  ensemble_summary.csv + eps_f), table2_model_ladder.csv (M1-M5 + power law, the
  degenerate-ridge exhibit), table3_nsem_accuracy.csv (MLP/KAN parity + hard-inverse
  recovery from checkpoint metadata), table4_f1_exhibit.csv (trajectory chi2 vs
  closed-form chi2 at recovered params: hard 5.79/5.79 PASS; soft 0.125/34.2 and
  0.004/47.3 FAIL — the F-1 smoking gun, recomputed fresh from stored parameters).
- `src/uncertainty_ensemble.py` now also writes ensemble_members.csv /
  ensemble_summary.csv for the paper tables.

## T2-A — Tier-2 identifiability gate (2026-07-06) DONE — PARTIAL GO

`src/tier2_identifiability.py`; full result table + scope consequence recorded in
TIER2_DESIGN.md Section 5.1. Headline: with the Veile instrument model AND scan-level
physical heterogeneity (the decisive ingredient — pure instrument noise gives
fictitious identifiability), interface/zone/composition parameters are recoverable
(2-19 % replicate SD) but the in-layer defect-transport gradient is WEAK (47 % SD at
a 1 abs% gradient; minimum detectable ~0.9 abs%). Tier-2 inverse scope is therefore
restricted to the identifiable set; defect transport stays forward-only.

## T2-B — steady spatial solve (2026-07-06) DONE — PASS

`src/tier2_physics.py` + `src/tier2_steady_trainer.py` + `conf/tier2_config.yaml`.
Two defect species (OV z=+2 outward, CV z=-8/3 inward) under the Tier-1 constant
field, growth flux anchored to M4 (J_OV = (chi/2)k3), Robin annihilation closures.
Newton on the DVR nodes vs the analytic closed form: 2.2e-13; NSEM (first-integral
flux residuals, BRDR + TwoPhaseOptimizer) vs Newton: worst 2.5e-4 (<= 0.5% PASS).
**Structural finding**: D cancels from Pe = z*gamma*E*L — the steady nondim profile
shape carries NO diffusivity information (independent corroboration of T2-A). The
predicted defect scales: c0_OV = 2.5e20 cm^-3, c0_CV = 9.3e20 cm^-3 (~2% of spinel
cation sites — the physical origin of the g_bl ~ 1% scale used in T2-A).

## T2-C — moving-boundary transient (2026-07-06) DONE — PASS

`src/tier2_transient_trainer.py`: Landau-transformed transient with fixed reference
scales, time-varying Pe(tau) and Tier-1 flux anchor jhat(tau); space-time MLP on the
(Nt x Nx) tensor grid, CN in time, DVR in space, quasi-steady warm start. Regression:
R2 (Tier-1 reduction) m/bl flux = Tier-1 growth flux at tau=1 to <= 0.08% both
species; R1 (quasi-steadiness) measured lag 0.65% for OV (tau_mig ~ 5 h) vs 5.7% for
CV (tau_mig ~ 36 h) — a quantification of the quasi-steady error the classical PDM
assumes away, matching the tau_mig/t_growth scaling.

## T2-D — composition map + EDX comparison (2026-07-06) DONE

`src/tier2_composition.py` (nominal + --fitted modes). Forward, no fit to line scans:
Cr fraction from FeCr2O4 stoichiometry = 46.7 wt% (measured plateaus ~45%); Cr widths
track the data at Tier-1 fit quality (+22/-20/+3% at 72/168/480 h); predicted defect
gradient 1.45 -> 0.46 abs% straddles the T2-A detection floor (0.93%) — early
exposures marginally detectable, late undetectable. Outputs:
plot_tier2_composition.png, tier2_composition_comparison.csv.

## T2-E — restricted inverse (Ni exclusion) (2026-07-06) DONE — with documented tension

`src/tier2_inverse.py`: exact moving-frame Wagner enrichment PDE (method of lines +
Radau; the quasi-steady closure misfits at chi2 ~ 100 and was replaced), fitted to
the real Veile Ni observables (zone widths at 3 exposures + 72 h peak amplitude,
population-SD weighting). Multi-start required — single-start landed in a local
minimum ABOVE its own phi=0 sub-fit; profile intervals are referenced to the global
minimum across all evaluations.

Results: **r_Ni = 0.29** (Ni incorporated at ~30% of the Fe/Cr rate — partial, not
total, exclusion), D_Ni_eff = 3.8 nm^2/h (~1e-17 cm^2/s, grain-boundary/defect-
enhanced vs bulk lattice), phi_ol = 0.57 (weak, [0.53, 0.90] at dchi2<=1).
**Mass-budget finding**: dropping the outer-layer supply term worsens chi2 by 6.4 —
evidence that the Fe-rich outer crystals draw cations from the base metal through
the barrier layer, not from re-precipitation alone. **Documented tension**:
chi2/dof = 25.7/1; predicted widths grow monotonically (16/24/44 nm) while the data
is flat (31/43/36 nm) — no constant-D exclusion transport reproduces a zone that
forms by 72 h and then stops growing; the formal dchi2<=1 intervals for r_Ni/D_Ni
are therefore optimistic (s^2 ~ 26 calibration applies) and the shape misfit points
to interface-coupled (defect-injection-enhanced, decaying) effective Ni mobility —
future work, not fitted. tier2_composition --fitted uses the same Wagner engine
(mixing fitted params into the quasi-steady closure overpredicts wildly): fitted
amplitudes flat at 23-25 wt% across exposures (as observed), widths -39/-35/+47%.

## T3-A — Tier-3 Ni-closure identifiability gate (2026-07-10) DONE — NO-GO

`TIER3_PLAN.md` (new; scoped in response to the T2-E tension). Two candidate
closures for the T2-E chi2/dof = 25.7 shape misfit (constant-D predicts
monotonically growing widths; the data is flat): M-A defect-flux-coupled
mobility D_Ni_eff(t) = D0 (J(t)/J_ref)^n (n reuses the frozen Tier-1 growth
flux, zero new physics input), M-B finite-capacity interfacial trapping
(saturating rejection BC, capacity S_cap). Both implemented in
`src/tier3_wagner.py` with exact/asymptotic regression limits back to the
committed T2-E trajectory (verified in `tests/test_tier3.py`, 9/9 pass,
without modifying `tier2_inverse.py`).

Gate (`src/tier3_identifiability.py`): 20-replicate synthetic recovery per
closure at the flattest-trajectory truth, real Veile population-SD noise,
anchored 5x3 multi-start, acceptance referenced to a freshly refined T2-E
profile-likelihood scan (D_Ni_eff half-width 0.607, r_Ni 0.0804, phi_ol
0.352 — the T2-E script's own 20-pt grid gives degenerate zero-width
intervals for D_Ni_eff/r_Ni, too coarse to resolve the true minimum width).

**Both closures NO-GO.** M-A (flux): its own new parameter n_flux IS
recoverable (27% replicate rel SD, under the 30% bar), but D_Ni_eff
destabilizes just past the bar (37% rel SD, 1.43 vs the 1.21 threshold) —
the closer call. M-B (trap): fails decisively — S_cap itself is not
recoverable (131% rel SD, 4x the bar) and destabilizes every other
parameter; a free-standing saturation capacity has too little independent
leverage on 4 scalar observables (3 widths + 1 amplitude) against 4 free
parameters. Practical note during the gate: the naive S_cap search range
(1-500) saturates the trap within hours at the T2-E kinetics, giving W = 0
everywhere (a `RuntimeError: no viable truth found`, not a physics failure)
— fixed by widening to geomspace(2e2, 2e4).

**Consequence**: Tier 3 re-scopes to forward predict-and-compare (T3-B' in
TIER3_PLAN.md) — no differentiable hard-mode inverse solver is built, since
there is no identified parameter left to fit. This is the same conclusion
as T2-A for a different sub-problem: the in-layer defect gradient (T2-A) and
now the Ni-zone time-dependence (T3-A) are both unidentifiable from the
current 3-exposure Veile summary statistics, for the same root cause (too
few independent observables) — both point at the same fix, more exposure
time points, not a more flexible model.

## T3-B' — forward predict-and-compare envelope (2026-07-10) DONE — Tier 3 complete

`src/tier3_forward_envelope.py`: n_flux scanned over [-2, 3] at the frozen
T2-E base kinetics (no fitting, per the T3-A NO-GO). Reference point n_flux=0
reproduces T2-E exactly. Finding: the n_flux range that looks qualitatively
right (non-monotonic/flat, [1.4, 3]) is quantitatively WORSE (forward chi2
35-70+) than the near-const region (best chi2 22.9 at n_flux~0.4, only a
modest improvement over T2-E's 25.7, and itself still monotonically
growing) — the flux-decay mechanism slows growth roughly uniformly across
all three exposures rather than selectively suppressing the 168->480 h
segment the data's dip requires. **No single n_flux value is both
shape-correct and quantitatively competitive.** This strengthens the T2-E/
T3-A conclusion beyond "unfittable": the M-A closure family cannot
reproduce the data's rise-then-partial-fall shape at all, pointing at a
qualitatively different mechanism (non-monotonic or interface-coupled
decaying mobility) as necessary future work, not a mistuned parameter.
Outputs: `outputs/plot_tier3_forward_envelope.png`,
`outputs/tier3_forward_envelope.csv`.

## Next
- Paper drafting per PAPER_PLAN.md (venue decision first). Tier-2 adds
  candidate content: the quasi-steady-error quantification (T2-C), the
  D-cancellation argument (T2-B), and the Ni mass-budget/partial-exclusion
  result (T2-E). Tier 3 adds a second, independent identifiability NO-GO
  (T3-A/T3-B') that pairs with T2-A's — both trace to too few exposure time
  points, strengthening the paper's case for the >2000 h designed experiment.
- Open modeling question from T2-E/T3-A/T3-B': the Ni zone's flat/dipping
  width vs. time is not reproduced by any single time-independent-mechanism
  closure tested (constant D, flux-coupled D, finite-capacity trapping).
  Future work, contingent on new data (more exposure time points), not a
  fit to the current 3-point summary statistics.
