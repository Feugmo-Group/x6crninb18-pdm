# TIER2_DESIGN — spatial HTW_PDM vs EDX depth profiles (Phase E3 design note)

Status: DESIGN + the T2-A gate EXECUTED (see Section 5.1). Implementation of
milestones T2-B onward remains a separate effort of roughly rpdm_nsem scale.

## 1. Goal and observables

Tier 1 (implemented here) predicts two scalars per exposure: L_bl(t), L_ol(t).
Tier 2 adds a spatial dimension x across the barrier layer to predict quantities the
0-D model cannot touch, targeting the composition-vs-depth data actually available in
Veile 2024:

- EDX line scans (Figs. 4/6/8): element fractions O/Cr/Fe/Ni/Nb vs depth, 3-4 scans per
  exposure at 72/168/480 h. Instrument accuracy: composition +-10 % relative, position
  +-0.945 nm, spatial resolution ~2 nm (extraction Section 3.3/3.7) — these set the
  data-loss weights.
- The Ni interfacial enrichment zone (31-43 nm, roughly constant in t) — in Tier 1 an
  inert consistency check; in Tier 2 an actual prediction of the slow-oxidizing noble
  constituent accumulating at the receding m/bl interface.
- Inner-layer element fractions (Fig. 10).

## 2. Model equations (sketch)

Fields on the barrier layer 0 <= x <= L_bl(t) (Landau-transformed to xi = x/L_bl(t),
as in rpdm_nsem):

1. Defect transport (Nernst-Planck, migration-dominated under the high field):
   cation vacancies c_CV(xi, t), oxygen vacancies c_OV(xi, t), and — the Tier-2-specific
   addition — per-cation-species interstitial/vacancy fluxes for Fe, Cr, Ni to resolve
   composition. Minimal viable set: 3 cation species x 1 defect type + O vacancies
   = 4 transport PDEs.
2. Poisson or fixed-field closure for phi(xi): start with the classical PDM constant
   field eps_f (consistent with Tier 1's b3); upgrade to Poisson only if profiles
   demand it (rpdm_nsem lesson: the constant-field limit is the stable starting point).
3. Two moving boundaries: m/bl (recedes into metal, Ni accumulates) and bl/ol (grows
   with PBR coupling). The bl/ol front carries the Tier-1 affine constraint as its
   consistency limit; the Tier-1 ODEs are the spatial model's exact 0-D reduction and
   MUST be recovered in the flux-integrated limit (regression test).
4. Composition map: element fraction profiles f_Fe/f_Cr/f_Ni(xi, t) from the cation
   sub-lattice site balance (transport numbers x flux divergence), convolved with a
   ~2 nm Gaussian to compare with EDX resolution.

Ni enrichment closure: Ni oxidizes much slower than Fe/Cr (extraction Section 3.1);
model as a species whose interfacial reaction rate constant k_Ni ~ 0, so its site
fraction at the receding interface grows by exclusion — one new parameter (k_Ni ratio),
directly identified by the ~35 nm enrichment-zone width.

## 3. NSEM architecture mapping (what rpdm_nsem already provides)

| Need | rpdm_nsem component to reuse |
|---|---|
| Space-time SCEN fields on (xi, tau) | element networks + DVRMapper grids (dynamic_trainer.py) |
| Landau transform / moving boundary | dynamic trainer's L(t)-scaled residuals |
| Scale-spanning concentrations | log-variable reformulation (hole-concentration recipe) |
| Interface BC coefficients | flux_R2/flux_R3-style boundary residual terms |
| Verification standard | steady_newton.py-style Newton BVP per time slice |
| Loss balancing / optimizer | BRDR + TwoPhaseOptimizer (unchanged) |

## 4. Inverse problem and the F-1 constraint

Finding F-1 (IMPL_REPORT Phase D) applies with force: joint field+parameter PINN
inversion leaks misfit into PDE slack on sparse data. Tier 2 has no RK4 shortcut (the
hard mode does not generalize to PDEs), so the documented mitigation is mandatory, not
optional:

1. dense collocation in the data dimension (the line scans provide ~100s of points per
   profile — far richer than the 3-point thickness data, which helps);
2. the closed-loop self-consistency check: recovered parameters re-solved with the
   Newton BVP must reproduce the data chi2 of the network trajectory;
3. freeze all Tier-1-identified parameters (A_bl, b3, PBR_eff, L_ol0) at their M4
   values with the D3 ensemble spread as hard bounds — Tier 2 fits only the NEW
   parameters (diffusivities/transport numbers, k_Ni ratio), never re-opens the
   Tier-1 fit.

## 5. Identifiability risk (go/no-go before coding)

The EDX profiles are element fractions, not defect concentrations; composition inside
a uniform Cr-rich spinel is nearly flat (Veile: "primary layer uniform Cr-rich"), so
the profile shape information lives almost entirely in (a) the interface widths and
(b) the Ni enrichment zone. Before implementing, run the design's cheapest test:
generate synthetic EDX profiles from an assumed parameter set + instrument noise and
check with the deterministic Newton solver whether the new parameters are recoverable
at all. If the synthetic study shows a flat likelihood (plausible for the in-layer
diffusivities), Tier 2 should be re-scoped to predict-and-compare (forward validation
against profiles) rather than a fit — which is still a publishable consistency check.

### 5.1 T2-A gate result (2026-07-06) — EXECUTED, verdict: PARTIAL GO

`src/tier2_identifiability.py` (analytic constant-field profile shapes replace the
Newton BVP — they coincide in the designed starting limit). 20 replicates of the
full synthetic experiment (3 scans x 4 traces, Veile instrument model: 2 nm
resolution, 2 % point noise, 0.945 nm position jitter, AND scan-level physical
heterogeneity at the Fig. 9-observed CVs — the decisive ingredient; without it the
gate returns fictitious 1e5-scale Delta-chi2 for everything). Verdict from the
replicate scatter of the recovered parameters (effective N = 3 scans, not 2040
points):

| shape parameter | maps to Tier-2 physics | replicate rel. SD | verdict |
|---|---|---|---|
| L_bl | (Tier-1 output) | 2 % | identifiable |
| f_Cr_bl | interfacial flux ratios | 2 % | identifiable |
| A_Ni | k_Ni exclusion ratio | 7 % | identifiable |
| w_Ni | k_Ni ratio + D_Ni | 8 % | identifiable |
| w_int | solid-state interdiffusion D_s | 17 % | identifiable |
| w_out | outer-interface kinetics | 19 % | identifiable |
| g_bl | defect-transport in-layer gradient | 47 % | **WEAK** |

Minimum detectable in-layer gradient: ~0.9 abs % across the layer (2x replicate
SD). Systematic floor: EDX composition accuracy +-10 % relative caps f_Cr_bl and
A_Ni regardless of statistics.

**Consequence for scope (per Section 5's re-scoping rule):** Tier 2 proceeds with
the inverse limited to the interface/zone/composition parameters (k_Ni ratio, D_s,
outer kinetics, flux ratios). The defect-transport parameters — the quantities the
spatial PDE machinery uniquely adds — are fittable only if the true gradient
exceeds ~1 abs %; below that, they are forward predict-and-compare only. The full
PDE solve retains value as the *forward* map from PDM kinetics to the (weak)
expected gradient and to the interface widths, but a defect-diffusivity inverse
from the Veile line scans alone is not defensible.

## 6. Staged milestones — ALL EXECUTED (2026-07-06); results in IMPL_REPORT.md

1. T2-A: synthetic identifiability gate — **DONE, partial go (Section 5.1)**.
2. T2-B: steady solve — **DONE, PASS** (Newton vs analytic 2e-13; NSEM vs Newton
   2.5e-4). Bonus structural result: D cancels from the steady profile shape.
3. T2-C: moving-boundary transient — **DONE, PASS** (Tier-1 flux reduction <=0.08%;
   quasi-steady lag measured: 0.65% OV / 5.7% CV).
4. T2-D: composition map + EDX comparison — **DONE** (stoichiometric Cr plateau
   matches; Cr widths at Tier-1 fit quality; predicted defect gradient straddles
   the T2-A detection floor).
5. T2-E: restricted inverse — **DONE with documented tension** (Wagner-PDE engine;
   r_Ni = 0.29 partial exclusion, mass budget requires matrix-supplied ol Fe at
   Delta-chi2 = 6.4; chi2/dof = 25.7 shape misfit — flat measured widths vs
   monotone model — flags time-decaying effective Ni mobility as the open item).
