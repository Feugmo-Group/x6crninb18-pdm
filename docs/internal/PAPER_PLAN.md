# Paper plan — HTW_PDM parameter identification for X6CrNiNb18-10 (paper 4 of the series)

## 0. Position in the series

This is the **fourth paper** of the NSEM/PDM program, and the first *application*
paper — the first time the toolkit meets real experimental data on a real reactor
material rather than a published benchmark:

| # | Paper | Venue / status | Role in the series |
|---|---|---|---|
| 1 | NSEM method paper (SCEN spectral-element networks, DVR operators) | *Machine Learning: Science and Technology* — submitted | Introduces the architecture |
| 2 | PINNACLE PDM paper (autodiff MLP-PINN, passive PDM, failure modes, FEM branch anchor) | *APL Machine Learning* — under revision | Establishes the PINN failure-mode taxonomy on the PDM family |
| 2.5 | Farooqi, Bösing, Tetsassi Feugmo (2025), "A physics-informed neural network approach to the point defect model for electrochemical oxide film growth," arXiv:2510.02872 | preprint | **Directly adjacent companion paper (found via lit review, `paper/literature_review.md` Thread 5) — same PDM/PINN lineage, likely overlapping authorship.** Validates forward/inverse PDM-PINN machinery against synthetic FEM benchmarks; catalogues 4 training-stability failure modes (imbalanced losses, scale disparity, BC enforcement, spurious convergence). Never touches real alloy data, identifiability, or extrapolation. |
| 3 | rpdm_nsem (hard-constrained NSEM, full Bösing Fig. 4, Newton verification) | *Journal of Computational Physics* — planned | Proves the toolkit on the hardest published PDM benchmark, pure-forward |
| 4 | **This paper** (HTW_PDM for AISI 347 in BWR water, inverse from Veile 2024 data) | **to decide — see Section 2** | First inverse identification from real data; first engineering-relevant material/environment; completes the *applied half* of 2.5's methodology (real data, identifiability, a 5th failure mode specific to sparse-real-data inversion, long-time/operating-envelope prediction) |

**CONFIRMED (2026-07-11)**: arXiv:2510.02872 is our own prior work — required
self-citation, Introduction frames paper 4 as "completing the applied half"
of it (see `paper/literature_review.md` Thread 5 synthesis), not as a first
PINN-PDM attempt.

Framing consequence: papers 1–3 carry the methods burden. Paper 4 **must not** be
another methods paper — the NSEM machinery is cited infrastructure ("the tool"),
and the contribution is corrosion science: mechanism, parameters, identifiability,
and prediction for an Nb-stabilized stainless steel in BWR hydrothermal water.
This is the inverse of the JCP paper's framing, and it is what makes the two
submissions non-overlapping: same toolkit, disjoint claims.

**Scope decision (2026-07-11): Tier 2 is IN, Tier 3 is a compact discussion
item, not its own section.** Tier 2 (spatial duplex-oxide model validated
against Veile's EDX depth profiles) is complete and adds genuinely new,
zero-free-parameter validation claims (the Cr-plateau composition prediction,
the D-cancellation structural identifiability result) that directly
strengthen contributions 1–3 below — it turns the paper from a parameter-
fitting exercise into an EDX-validated mechanistic model, which is a
materially stronger claim for both target audiences. Tier 3 (an attempt to
resolve the T2-E Ni-zone shape tension with two candidate time-dependent
mobility closures) returned a clean NO-GO on both closures at the
identifiability gate — a genuine second data point on the "3 exposure times
is not enough" theme, but not substantial enough to carry its own section;
it belongs in Discussion/Limitations alongside the T2-A gate, both pointing
at the same designed-experiment ask.

Continuity requirements (reviewers will cross-read):
- Notation consistent with papers 2–3 (χ, γ = F/RT, k_i^0, non-dimensionalization
  symbols; cross-check against `aip_paper.tex` and the rpdm_nsem draft).
- The F-1 finding here (soft joint inversion leaks misfit into ODE slack on sparse
  data) is the inverse-problem sibling of paper 2's failure-mode taxonomy and
  paper 3's free-parameter escape routes — cite both, present F-1 as the
  sparse-data completion of that taxonomy.

## 1. What this paper contributes (all computed, in IMPL_REPORT.md)

1. **First PDM parameter identification for Nb-stabilized AISI 347 in BWR HTW.**
   Veile 2024 provide only the empirical L = k·tⁿ; this paper derives the HTW_PDM
   from the Li 2020 SCW_PDM (high-density/EO limit, duplex oxide) and identifies
   its kinetics: A_bl = 0.727 nm/h, b3 = −0.0125 1/nm (ε_f ≈ 1.7×10⁴ V/cm —
   physically ordered between Li's 500 °C SCW value ~10² and room-T passive films
   ~10⁶ V/cm), PBR_eff = 1.051 with the *data*, not the prior, landing on the
   stoichiometric spinel/magnetite value, L_ol0 ≈ 119 nm quantifying rapid initial
   outer-crystal precipitation.
2. **Mechanistic reinterpretation of Veile's power-law exponents.** Parabolic Cr
   layer (n ≈ 0.50) ⇒ near-zero barrier-layer dissolution in ultrapure water;
   cubic-like Fe layer (n ≈ 0.22) ⇒ constant-volume-constrained deposition from a
   large initial precipitate population. The mechanistic model *outperforms* the
   power law on the joint Cr+Fe data (χ² 5.8 vs 20.1, comparable parameter count)
   because the constant-volume constraint couples the layers — the empirical fit
   treats them as independent.
3. **Quantified identifiability from 3 time points** — the systematic weakness of
   literature PDM fits, never previously made explicit. Profile likelihoods:
   A_bl/b3/L_ol0 sharply identifiable; PBR_eff weak; L0 unidentifiable (air-film
   prior required); C_bl bounded only above ~1 nm/h. Cross-validated by a
   50-member parametric bootstrap through the NSEM inverse: data-identified
   parameters agree with the profiles; prior-dominated parameters collapse onto
   their priors (the bootstrap perturbs only data — an honest-reporting point in
   itself: quote profile intervals for prior-dominated parameters).
4. **Finding F-1 + the hard-mode fix.** Joint field+parameter PINN inversion on
   sparse data equilibrates at data-χ² ≈ 0 with ~1e-3 ODE slack; recovered
   kinetics fail the closed-form self-consistency check, and raising the physics
   weight does not move the equilibrium (BRDR renormalizes). Fix: differentiable
   RK4 through the learnable kinetics — physics exact by construction — which
   recovers the deterministic optimum to ≤0.6 % on every parameter. The
   *self-consistency check* (re-solve at recovered parameters, compare χ²) is
   proposed as a reporting standard for PINN inverse problems, extending paper 3's
   Newton-verification standard from forward to inverse.
5. **Prediction with consequences.** At 10 y the PDM predicts L_bl = 535 nm vs
   1853 nm from power-law extrapolation — a 3.5× divergence in predicted
   barrier-layer thickness (and hence metal loss) — while the two PDM branches
   (unbounded-log vs saturating) remain indistinguishable to 10⁵ y. This yields a
   concrete designed-experiment proposal: a >2000 h exposure discriminates PDM
   from power law; no feasible exposure discriminates the PDM branches, so the
   dissolution rate must come from water chemistry, not thickness data.
6. **Operating-envelope robustness.** Condition-mapped sensitivity maps
   (Arrhenius/1/T-field/Nernst-ECP channels anchored at the fit, ΔG⁰_R as an
   explicit scan): across the whole BWR box (200–285 °C at 7 MPa, 0.01–8 ppm O₂)
   L_bl varies by only 1.2–1.4×, with temperature dominant and O₂ nearly inert
   while barrier dissolution ≈ 0. The long-time prediction is robust to
   operating-condition uncertainty — the engineering "so-what."
7. **MLP vs KAN inside the same harness** (deliverable from the project scope):
   both pass the 0.5 % forward acceptance (0.14 % vs 0.11 %); parity, not
   superiority, is the honest headline.
8. **Spatial (Tier-2) validation against independent EDX composition-depth
   data — zero free parameters.** The Tier-1 kinetics, frozen, predict the
   Cr-rich barrier-layer plateau composition from spinel stoichiometry alone
   (46.7 wt%, vs ~45% measured) and layer widths at Tier-1 fit quality
   (+22/-20/+3% across the three exposures) — a genuine forward cross-check
   against data never used in the Tier-1 fit, not a re-fit.
9. **Structural identifiability result: diffusivity cancels from the steady
   defect-transport profile shape.** The nondimensional steady profile depends
   only on Pe = z·γ·E·L (all Tier-1-fixed quantities); D only sets the absolute
   concentration scale, invisible to EDX at the % level. Independently
   corroborates the T2-A synthetic identifiability gate (item 10) by a
   completely different (analytic/structural, not statistical) argument —
   two methods agreeing is itself evidence worth reporting.
10. **Quantified quasi-steady-state error** — the approximation every classical
    PDM application makes silently. Comparing the transient spatial solve to
    the flux-integrated Tier-1 reduction: 0.65% lag for the fast species (O
    vacancies, τ_mig ~ 5 h) vs 5.7% for the slow species (cation vacancies,
    τ_mig ~ 36 h), scaling as τ_mig/t_growth as expected from the physics —
    the first explicit quantification of this error we are aware of in the
    PDM literature (Thread 1/2 of the lit review found none).
11. **A second, independent identifiability NO-GO (compact, Discussion-level
    item, not a full section).** Extending Tier 2 to resolve the Ni-exclusion
    zone's flat/dipping width vs. time (unexplained by the constant-diffusivity
    closure, χ²/dof = 25.7) with two physically motivated time-dependent
    mobility closures both fail a synthetic identifiability gate before
    fitting real data — the same "3-4 exposure times cannot support this many
    parameters" theme as item 3, now demonstrated on a second, independent
    sub-problem. Strengthens rather than dilutes the paper's core
    identifiability message; belongs in Discussion/Limitations, not its own
    Results section.

**Sharpened gap statement** (from `paper/literature_review.md`, informed by a
full lit review — see that file for the complete reference list and
per-thread analysis): PDM kinetics have been fit to sparse data (3-6 points,
whether EIS spectra or depth profiles) for four decades without ever
formally checking whether the data constrains the parameters being estimated
(no prior PDM/MCM paper found applies profile-likelihood or bootstrap
identifiability analysis, despite the technique being mature and routine in
systems biology and battery P2D modeling); the one prior PINN-PDM paper
(Farooqi, Bösing, Tetsassi Feugmo 2025, arXiv:2510.02872 — see Section 0
above) validated the forward/inverse machinery only against synthetic FEM
benchmarks, leaving open both the real-data identifiability question and a
distinct sparse-real-data inversion failure mode (F-1, item 4) that only
appears once real noisy few-point data is the inversion target. This paper
closes both gaps at once, on AISI 347 in BWR HTW — the first alloy/
environment where real EDX/SIMS data, PDM inversion, an identifiability
audit, and long-time/operating-envelope extrapolation are all carried out
together, with a spatial EDX cross-check (items 8-10) as an independent
validation layer no prior PDM paper attempts.

## 2. Target venue — DECIDED (2026-07-11): *npj Materials Degradation*

**Venue is locked: *npj Materials Degradation*, primary framing.** The
derivation (Section "Model") goes to SI, not main text, per the npj MD
column below. Lead the Introduction with the nuclear/BWR materials framing
(§0/§1 sharpened gap statement, nuclear-audience half); the
electrochemistry/PDM framing stays as the secondary thread woven in via the
identifiability contribution, not the lead. *Corrosion Science* is no longer
under consideration unless npj MD rejects outright.

**arXiv:2510.02872 is CONFIRMED our own prior work (2026-07-11).** It is now
a required self-citation. The Introduction states directly, without
hedging, that this paper completes the applied half of that companion
methodology paper (real data replaces synthetic FEM benchmarks; adds
identifiability analysis, the F-1 sparse-real-data failure mode, and
long-time/operating-envelope prediction, none of which 2510.02872
attempts). See §0 table and `paper/writing_plan.md` Introduction para 4 —
remove the "if NOT our own work, soften to..." conditional there, it no
longer applies.

| | *npj Materials Degradation* | *Corrosion Science* (Elsevier) |
|---|---|---|
| Fit | Mechanistic degradation studies with modern methods; Nature-portfolio visibility; the plan's original target | The home journal of the source model (Li 2020) and the halide extension (Alexiadis 2025); the PDM community reads here |
| Audience sees | A degradation-prediction story for a reactor material, ML-assisted | A PDM-lineage paper extending SCW_PDM to subcritical HTW |
| Novelty framing | Foreground items 1, 2, 5, 6 (mechanism + prediction); identifiability (3) as the rigor differentiator | Foreground items 1–3 with the Li-2020 derivation chain explicit; NSEM inverse (4) as supporting method |
| Review risk | "Only 3 time points / one temperature" — answer with the identifiability analysis and the designed-experiment proposal (that *is* the contribution) | "Why a neural inverse at all for a 2-ODE model?" — answer: the deterministic baseline is included and agrees; NSEM is the bridge to the Tier-2 spatial problem where no integrator shortcut exists |
| Series logic | Best: keeps paper 4 clearly an *application* paper, maximally distinct from JCP paper 3 | Also fine, slightly more model-derivation-heavy overlap with paper 3's framing |

**Recommendation: *npj Materials Degradation* primary, *Corrosion Science*
fallback.** Both are genuine domain venues, which is the series-level requirement.
Third options if scope shifts: *Journal of Nuclear Materials* (if the BWR/HWC
chemistry discussion grows), *npj Computational Materials* (only if the editors
at npj Mater. Degrad. redirect — it would drag the framing back toward methods,
which papers 1–3 already cover).

**Audience-specific gap framing** (from `paper/literature_review.md` synthesis
— both share the same evidence base, differ only in what leads the
Introduction):
- **Nuclear/BWR materials framing (npj MD lead)**: current LWR-internals
  predictive practice is either an operational ECP/dissolved-O₂ threshold
  criterion (HWC/NWC guidance — not mechanistic) or purely data-driven ML of
  crack-growth rate (cf. the 2025 npj MD IGSCC-UQ paper, which shows the
  venue already rewards UQ-aware modeling but never touches oxide chemistry).
  AISI 347 had no mechanistic oxide-growth model at all before this work;
  Veile et al. (2024) stopped at per-element empirical power laws. Lead with
  the 3.5× long-time divergence and operating-envelope robustness as directly
  actionable for surveillance-program design.
- **Electrochemistry/PDM framing (Corrosion Science fallback)**: forty years
  of PDM parameter estimation (Macdonald 1992 → Li 2020 → the 316L/316LN/321
  literature) has never been paired with formal identifiability analysis,
  despite routinely fitting 4-6 kinetic parameters to a handful of data
  points — exactly the regime profile-likelihood/bootstrap methods (mature in
  systems biology, now battery P2D modeling) were built to diagnose.
  Simultaneously, the one PINN-PDM precedent (arXiv:2510.02872) validated
  only against synthetic FEM benchmarks. Lead with the identifiability audit
  and the F-1 finding/self-consistency-check as a methodological template.

Both framings interleave naturally in one Introduction: open with the
material/application motivation, pivot to the methodological gap as the
mechanism that makes the answer trustworthy, close by naming both the
empirical antecedent (Veile et al. 2024) and the methodological antecedent
(Farooqi/Bösing/Tetsassi Feugmo 2025) as the two papers this work completes.

**Action:** confirm the venue before drafting — it decides whether the Li-2020
derivation goes in the main text (Corrosion Science) or Methods/SI (npj MD).

## 3. Proposed title

*"Mechanistic Identification of Passive-Film Growth Kinetics on Nb-Stabilized
Stainless Steel in Boiling-Water-Reactor Hydrothermal Water: A Point Defect
Model Study with Physics-Informed Inverse Analysis"*

(alt., shorter, npj-style: *"What Three Exposure Times Can and Cannot Tell You:
Point Defect Model Identifiability for Duplex Oxide Growth on AISI 347 in
High-Temperature Water"* — the identifiability-first framing is riskier but more
distinctive.)

## 4. Proposed section outline (npj MD framing)

1. **Introduction** — AISI 347 in BWR HTW (Veile 2024 system); the gap between
   empirical power laws and mechanistic prediction; PDM lineage
   (Macdonald → Li 2020 SCW_PDM); series positioning (one paragraph, papers 1–3
   AND 2.5 [arXiv:2510.02872] as the methods substrate — explicitly frame this
   paper as completing 2.5's applied half, see Section 0); contributions
   (Section 1 items 1, 2, 3, 5, 8 — lead with the sharpened gap statement).
2. **Model** — HTW_PDM derivation summary (full derivation → SI / `pdm_eqns.md`):
   reaction set in the high-density EO limit, the exact constant-volume collapse
   dL_ol/dt = PBR_eff·dL_bl/dt − Ω·k₁₁·C_O^r (simpler than Li Eq. 42; both
   verified to 8.5e-14 nm), reduced 6-parameter apparent system, closed forms.
3. **Data and deterministic inverse** — Veile Fig. 9 digitization + acceptance
   (power-law refit reproduces k, n exactly); model ladder M1–M5 with the
   degenerate-ridge exhibit (M3: A_bl ~ C_bl → ∞, b3 → 0 — the classic PDM
   identifiability trap demonstrated explicitly); M4 as the physical model;
   profile-likelihood identifiability map.
4. **Physics-informed inverse (NSEM)** — hard mode (RK4-through-kinetics) as the
   reference; finding F-1 with the soft mode as the documented failure case; the
   self-consistency check as a reporting standard; synthetic-recovery validation
   (2 % and 10 % noise); bootstrap ensemble vs profile likelihoods.
5. **Spatial validation (Tier 2) — new section.** Two defect species (O
   vacancies, cation vacancies) under the frozen Tier-1 constant field; steady
   NSEM vs Newton BVP (structural D-cancellation result, item 9); moving-
   boundary transient vs the Tier-1 flux-integrated reduction (quasi-steady
   error quantification, item 10); zero-free-parameter composition-map
   prediction vs Veile's EDX line scans (item 8, headline: 46.7 wt% predicted
   Cr plateau vs ~45% measured). Frame as the independent cross-check that
   the 0-D kinetics (Section 3-4) are not merely curve-fit but structurally
   consistent with the spatial compositional data.
6. **Predictions** — long-time extrapolation (3.5× PDM-vs-power-law divergence,
   branch indistinguishability, the >2000 h designed experiment); (T, [O₂])
   sensitivity maps with the ΔG⁰_R scan; field-strength ordering across
   temperature regimes.
7. **Discussion** — mechanistic reading of Veile's exponents; Ni enrichment as
   an exclusion zone (T2-E: partial exclusion r_Ni=0.29, mass-budget evidence
   for matrix-supplied outer-layer Fe, documented shape tension); COMPACT
   paragraph on the Tier-3 identifiability NO-GO (item 11: two candidate
   time-dependent Ni-mobility closures both fail the synthetic gate before
   touching real data — a second independent demonstration of the "3-4
   exposure times cannot support this many parameters" theme, not a separate
   result); limitations: single temperature, open-circuit potential absorbed
   into A_bl (Nernst channel is a lower bound on the true ECP response), 3
   time points, EDX-thickness ≠ total-oxide caveats; the >2000 h /
   more-exposure-times designed-experiment recommendation as the single fix
   that would resolve BOTH the T2-A and T3-A identifiability gates.
8. **Methods** — NSEM/SCEN + DVR summary (cite papers 1, 3), BRDR,
   TwoPhaseOptimizer, MLP/KAN parity table, Radau/closed-form parity, test suite.
9. **Data/Code availability** — this example dir + PhysicsNeMo
   `experimental.models.scen`; digitized Veile data CSVs with provenance notes.

## 5. Figures/tables (status)

- Fig. 1: model schematic — duplex oxide (Cr-spinel bl / Fe-crystal ol / Ni zone),
  reactions and interfaces, mapped onto the Veile TEM picture. `make_paper_figs.py`
  -> `outputs/paper/fig1_model_schematic.png` — hand-built (matplotlib patches/
  arrows, not AI-generated; deliberately, for credibility in a rigorous corrosion-
  science venue), DONE.
- Fig. 2: baseline verification — Radau vs closed forms + Li Table 5 regression.
  `make_paper_figs.py` -> `outputs/paper/fig2_baseline_verification.png`, DONE.
  Candidate for SI depending on venue (Section 2 action item).
- Fig. 3: fits vs data — `outputs/plot_fits_vs_data.png` DONE (publication redo:
  add SD bands, panel letters — still open).
- Fig. 4: identifiability — `outputs/plot_profile_likelihood.png` DONE.
- Fig. 5: inverse posterior vs baseline —
  `outputs/plot_uncertainty_ensemble.png` DONE.
- Fig. 6: long-time prediction — `outputs/plot_long_time.png` DONE (the 3.5×
  divergence + branch-indistinguishability figure).
- Fig. 7: sensitivity maps — `outputs/plot_sensitivity_maps.png` DONE.
- Fig. 8 (NEW — Tier 2, item 9): steady spatial solve — D-cancellation
  demonstration (nondim profile shape invariant under 2-3 order-of-magnitude
  D variation, only c0 scale changes) + Newton-vs-analytic parity exhibit.
  `outputs/plot_tier2_physics_steady.png` (to build from `src/tier2_physics.py`
  + `src/tier2_steady_trainer.py` NSEM checkpoint).
- Fig. 9 (NEW — Tier 2, item 10): quasi-steady lag — transient NSEM profile vs
  the instantaneous quasi-steady shape at final time, per species, with the
  0.65%/5.7% (OV/CV) lag annotated. From `src/tier2_transient_trainer.py`.
- Fig. 10 (NEW — Tier 2, item 8): composition-map/EDX comparison —
  `outputs/plot_tier2_composition.png` (from `src/tier2_composition.py --fitted`),
  DONE and already publication-quality (3-panel, per-exposure Cr/Fe/Ni/O
  profiles vs measured Cr width band). Headline figure for the Tier-2 section.
- Table 1: parameter table (M4 values, profile 1σ, bootstrap [16, 84] %,
  physical recovery ε_f) — assemble from IMPL_REPORT numbers.
- Table 2: model ladder (M1–M5 χ², the ridge exhibit, power-law comparison).
- Table 3: MLP vs KAN forward parity + hard-inverse recovery accuracy.
- Table 4 (or SI): F-1 exhibit — soft-mode trajectory χ² vs closed-form χ² at
  recovered parameters (0.1 vs 34), the smoking gun.
- Table 5 (NEW — Tier 2, item ~item 3 sibling): T2-A identifiability gate
  summary — shape parameter, replicate rel. SD, verdict (identifiable/weak),
  from `outputs/tier2_identifiability.csv`.
- Table 6 (NEW — Tier 2, item 11 support): T2-E Ni-exclusion exhibit —
  r_Ni, D_Ni_eff, phi_ol_supply fit + profile intervals, chi2/dof, the
  mass-budget Delta-chi2 finding, from `outputs/tier2_inverse_fit.csv`.
- Table 7 (NEW — Tier 3, compact Discussion item 11): closure NO-GO summary
  — 2 closures x 4 parameters, replicate rel. SD vs acceptance threshold,
  verdict, from `outputs/tier3_identifiability.csv`. Small table or inline
  in Discussion text rather than a numbered main-text table, given its
  Discussion-level (not Results-level) role per the Section 4 outline.

## 6. Before drafting — checklist

- [x] **Venue decision** (Section 2) — DECIDED 2026-07-11: *npj Materials
      Degradation*. Derivation goes to SI.
- [x] **arXiv:2510.02872 authorship** — CONFIRMED 2026-07-11: our own prior
      work. Required self-citation; Introduction frames this paper as
      completing its applied half.
- [ ] Sign-offs: David (project owner per the Winter-2026 scope) and any Veile
      2024 authors if data reuse beyond published-figure digitization is
      contemplated — current use is digitized published figures with provenance
      notes, which needs citation, not permission, but flag it explicitly.
- [x] Draw Fig. 1 (model schematic) — DONE via `make_paper_figs.py`.
- [ ] Decide Fig. 2's fate (SI vs cut, depends on venue decision).
- [ ] Publication-quality pass on Figs. 3–7 (fonts, SD bands, panel letters).
- [ ] Build Figs. 8-9 (Tier-2 D-cancellation/Newton parity, quasi-steady lag —
      Section 5 above); Fig. 10 (composition map) already publication-quality.
- [ ] Re-run the full pipeline once end-to-end at fixed seed and archive the
      exact outputs used in the paper (current runs are validated but scattered
      across `outputs/` subdirs; one clean reproduction run → one archived dir).
- [ ] Notation cross-check against `aip_paper.tex` and the rpdm_nsem draft
      (χ, γ, k_i^0, hat-variables) so papers 2–4 read as a series.
- [ ] Decide how to phrase the stale-C_bl-bound correction (IMPL_REPORT Phase E1
      note): the paper should present the E1-corrected statement (no bound below
      1 nm/h from thickness data alone; branches indistinguishable to 10⁵ y),
      not the earlier 0.71 nm/h figure.
- [x] Literature review complete — `paper/literature_review.md` (24 references,
      6 threads, full synthesis + audience-specific gap framings). Recommend a
      citation-management pass to verify DOIs/page ranges before submission
      (several entries reconstructed from secondary listings).
- [ ] References: Li 2020 (Corros. Sci. 163:108280), Veile 2024 (Materials
      17:4500), Macdonald PDM lineage, papers 1–3 of the series (statuses as of
      submission), Farooqi/Bösing/Tetsassi Feugmo 2025 (arXiv:2510.02872),
      IAPWS-97 for water properties — full candidate list in
      `paper/literature_review.md` Section "Full reference list".
- [ ] Confirm the [O₂]/ECP caveat wording with a BWR water-chemistry reference
      (measured ECP-vs-O₂ curves are steeper than ideal-Nernst in the ppb
      transition region — the maps are a lower bound on O₂ sensitivity).

## 7. Suggested immediate next steps

1. Venue + sign-off decisions, and the arXiv:2510.02872 authorship
   confirmation (Section 6, first boxes) — the latter materially changes how
   the Introduction can be framed.
2. I draft the abstract + introduction for review in the chosen framing,
   using `paper/literature_review.md`'s synthesis as the citation backbone.
3. Build Figs. 8-9 (Tier-2 D-cancellation/Newton parity, quasi-steady lag);
   publication pass on Figs. 3-7; assemble Tables 1-7 (1-4 Tier-1, 5-7 new
   Tier-2/3) from IMPL_REPORT + the `outputs/tier2_*`/`tier3_*` CSVs.
4. One clean end-to-end reproduction run, archived.
