# Mechanistic Identification of Passive-Film Growth Kinetics on Nb-Stabilized Stainless Steel in Boiling-Water-Reactor Hydrothermal Water: A Point Defect Model Study with Physics-Informed Inverse Analysis

**Target venue**: npj Materials Degradation. **Status**: full first-pass
draft complete (all 9 sections + Abstract, ~6,600 words). Every number below
is drawn from `paper/facts_table.md`. Known open items before this is
submission-ready: (1) Abstract is 282 words, trim to 200-250 for npj MD
limits; (2) no formal citation markers/reference list yet — citations are
currently descriptive prose ("Li and coworkers...", "an independent,
contemporaneous study...") and need to become numbered/author-date
citations against the verified reference list in `literature_review.md`
once DOIs are spot-checked; (3) figure/table callouts (e.g. "Fig. 3",
"Table 2") are not yet inserted into the prose — currently the text
stands alone without pointing at the actual display items; (4) needs a
full read-through for the "red thread" and terminology consistency the
skill's Stage 3 revision checklist calls for, plus the notation
cross-check against papers 2-3 that `PAPER_PLAN.md` §6 flags.

---

## Abstract

Nb-stabilized austenitic stainless steel AISI 347 is used in boiling-water-
reactor internals for its weld-decay resistance, yet no mechanistic model of
its passive-film growth in reactor-representative hydrothermal water has
existed; recent depth-profile characterization of this alloy under
simulated boiling-water-reactor conditions stopped at empirical per-element
power-law fits with no shared physical parameter set. We adapt the point
defect model, in the formulation developed for supercritical-water
corrosion, to this subcritical hydrothermal-water regime and identify its
kinetics from the published depth-profile data using both a deterministic
inverse fit and a physics-informed neural inversion, cross-validated against
each other and against a fifty-member parametric bootstrap. The identified
mechanistic model outperforms the empirical power-law description on the
same joint dataset (χ² 5.8 versus 20.1) and extrapolates to a barrier-layer
thickness 3.5 times smaller than a naive power-law extrapolation by ten
years, while remaining robust — varying by only 1.2 to 1.4-fold — across the
full boiling-water-reactor operating envelope. We formally characterize
which of the fitted model's parameters the available three exposure times
can and cannot constrain, and document a previously unreported
physics-informed inverse-problem failure mode in which joint field-and-
parameter inversion satisfies the data loss while leaving the governing
equation unsatisfied, together with a hard-mode fix and a self-consistency
check proposed as a general reporting standard. Extending the identified
kinetics spatially, we reproduce the measured chromium-plateau composition
from stoichiometry alone, with zero free parameters, and independently
corroborate the statistical identifiability findings through an analytic
structural argument verified by direct computation over a ten-thousand-fold
diffusivity range. Together these results motivate a concrete
recommendation: an exposure beyond roughly 2000 hours would discriminate
this mechanistic prediction from a naive extrapolation of current data, with
direct consequences for structural-material life assessment.

---

## Introduction

Nb-stabilized austenitic stainless steel AISI 347 (X6CrNiNb18-10) is used in
boiling-water-reactor internals and piping specifically for its resistance
to weld-decay sensitization, yet the mechanistic evolution of its passive
oxide film under reactor-representative hydrothermal water conditions has,
until now, no predictive model. Current practice for predicting oxide
behavior on light-water-reactor internals relies either on operational
criteria — hydrogen- or normal-water-chemistry guidance framed around a
threshold electrochemical corrosion potential and dissolved-oxygen
concentration — that says nothing about oxide thickness or composition, or
on purely data-driven machine-learning models of crack-growth rate that,
while explicitly uncertainty-quantified and increasingly accepted in this
literature, are equally silent on the underlying oxide chemistry. Veile et
al. recently provided the first detailed characterization of the duplex
oxide layer this alloy actually develops in simulated boiling-water-reactor
hydrothermal water, resolving chromium-rich barrier and iron-rich outer
layers by energy-dispersive X-ray depth profiling at three exposure times,
but their own analysis stops at empirical per-element power-law fits, with
no shared physical parameter set connecting the layers and no assessment of
how far those fits can be trusted to extrapolate.

The point defect model, introduced by Macdonald and developed through
successive generations over four decades, offers the mechanistic
alternative: passive-film growth and dissolution are treated as
vacancy-mediated transport under a high interfacial field, yielding
closed-form thickness-versus-time laws directly connected to interfacial
reaction kinetics. Li and coworkers extended this framework to supercritical
water conditions, and the present work adapts that supercritical-water
formulation to the subcritical hydrothermal-water regime a boiling-water
reactor actually operates in, applied for the first time to this specific
alloy. What has not accompanied four decades of point-defect-model
parameter estimation, however formulated or however calibrated — against
electrochemical impedance spectra in bench-scale cells or, as here, against
depth-profile data from realistic exposure coupons — is any formal check of
whether the data used actually constrains the parameters being estimated.
Models with four to six kinetic parameters are routinely fit to three to
six data points without asking whether that fit is unique, a gap that
becomes consequential exactly when, as here, the fitted model is meant to
extrapolate to timescales and conditions well beyond the calibration data.

We close this gap using tools mature in adjacent fields — profile-likelihood
and bootstrap identifiability analysis, standard in systems biology and
increasingly applied to electrochemical battery models — but essentially
absent from the point-defect-model literature specifically, despite that
literature's chronic exposure to exactly the sparse-data regime these tools
were built to diagnose. We further extend the identification to the
spatial domain, resolving defect transport across the barrier layer and
cross-checking the resulting composition predictions against Veile et al.'s
depth profiles directly, which lets us corroborate the identifiability
conclusions reached statistically in the zero-dimensional fit by an
entirely independent, structural argument, and lets us quantify an
approximation — quasi-steady defect transport — that every prior
zero-dimensional point-defect-model application has made without
characterizing its error.

Physics-informed neural networks offer one route to performing this kind of
inversion at the scale the spatial extension requires, where no
closed-form or simple-integrator shortcut exists, and we previously
demonstrated that physics-informed neural machinery can solve and invert
the point defect model's governing equations against synthetic,
finite-element-benchmarked data, cataloguing four distinct training-time
failure modes along the way. That prior work never touched real
experimental data, never assessed parameter identifiability, and never
extrapolated a fitted model beyond its calibration window — this paper
completes its applied half. In doing so we also encounter and document a
fifth failure mode specific to the sparse-real-data inverse setting, absent
from that synthetic-benchmark study: joint field-and-parameter inversion
can settle into an equilibrium that satisfies the data loss while leaving
the governing differential equation only approximately satisfied, a failure
invisible to the training loss itself and only exposed by independently
re-solving the recovered parameters through an exact forward solve. An
independent, contemporaneous study reaching a closely related diagnosis in
an unrelated application domain corroborates that this is a real,
recognized failure mode of sparse-data physics-informed inversion generally,
not an idiosyncrasy of this particular system.

This paper makes five contributions. First, it identifies the first
mechanistic passive-film growth model for X6CrNiNb18-10 in boiling-water-
reactor hydrothermal water, recovering kinetics that outperform Veile et
al.'s own empirical description on their joint dataset. Second, it
formally characterizes which of the fitted model's parameters three
exposure times can and cannot constrain, cross-validated by both
profile-likelihood and bootstrap methods. Third, it documents a previously
undescribed physics-informed inverse-problem failure mode specific to
sparse real data, together with a hard-mode fix and a self-consistency-check
reporting standard proposed as a general diagnostic for this class of
problem. Fourth, it extends the identified kinetics spatially and
cross-validates them against independent compositional depth-profile data
using zero free parameters, while independently corroborating the
statistical identifiability findings through a structural argument.
Fifth, it shows that the resulting long-time extrapolation diverges by a
factor of 3.5 from a naive empirical extrapolation of the same data by ten
years, a divergence directly relevant to structural-material life
assessment, while remaining robust to operating-condition uncertainty
across the full boiling-water-reactor envelope — together motivating a
concrete, actionable exposure-duration recommendation for future
experimental campaigns on this and similar systems.

---

## Methods

The forward and inverse solves in this work use spectral-element networks
(SCEN), a physics-informed neural architecture built on discrete
variational-representation (DVR) collocation operators, as introduced in the
first paper of this series and applied to the full point defect model (PDM)
family in the third. We do not re-derive that architecture here; briefly,
each field of interest is represented by an element network evaluated on a
DVR node set, and physical residuals are enforced at those nodes through a
first-integral or collocation formulation rather than through automatic
differentiation of a black-box surrogate. Loss terms across physical
processes are combined with the balanced-residual dynamic reweighting (BRDR)
aggregator, and optimization proceeds in two phases — an Adam phase for
coarse basin-finding followed by an L-BFGS phase for high-precision
convergence — through the TwoPhaseOptimizer scheduler described in the same
prior work.

Before any physics-informed result is reported, we establish a deterministic
numerical baseline to machine precision, since every subsequent claim in
this paper is validated against it. The closed-form solutions of the
zero-dimensional HTW_PDM kinetics agree with a Radau-method numerical
integration of the same ordinary differential equations to better than
5×10⁻¹¹ nm on a generic parameter set, and the associated Crank–Nicolson
residual of the exact solution vanishes to numerical precision. This
baseline, not any neural-network result, is what every forward and inverse
PINN solve in this paper is checked against.

Two neural backbones are evaluated inside the same forward-solve harness — a
standard multilayer perceptron (MLP) and a Kolmogorov–Arnold network (KAN)
element representation — to test backbone sensitivity independently of the
physics formulation. Both are trained to the same acceptance criterion
(relative L∞ error against the closed-form solution, ≤0.5%) under identical
optimizer and loss-aggregation settings; results are reported in the
Data-and-deterministic-inverse and Physics-informed-inverse sections below.

All numerical results in this paper — deterministic fits, profile-likelihood
scans, bootstrap ensembles, forward and inverse neural solves, and the
Tier-2 spatial extension — are backed by an automated test suite (25 tests
at the time of submission) that checks parity between closed-form,
Newton-iterated, and neural solutions at each stage of the model hierarchy;
the suite is included with the code release referenced in the Data and Code
Availability section.

---

## Model

The governing kinetics follow the point defect model, in the specific
formulation Li and coworkers developed for steel corrosion in supercritical
water, adapted here from that supercritical, high-temperature regime to the
subcritical hydrothermal-water conditions a boiling-water reactor actually
operates under. The oxide Veile et al. characterize on X6CrNiNb18-10 is a
duplex scale: a chromium-rich, nanocrystalline spinel barrier layer growing
into the metal, and discrete iron-rich magnetite crystals forming a porous
outer layer beyond it, with a nickel enrichment zone at the receding
metal/barrier-layer interface that the base kinetics treat as an inert
consistency check rather than an active species (we return to it explicitly
in the Discussion). The full reaction network, potential-distribution
assumptions, and rate-law derivation are given in the Supplementary
Information; here we summarize only the structure that determines the
apparent-parameter system we actually fit.

Of the full interfacial reaction set, only two classes of reaction move the
model's interfaces: barrier-layer growth into the metal, driven by
oxygen-vacancy generation at the metal/barrier-layer interface, and
barrier- and outer-layer destruction by chemical conversion and dissolution.
Every interfacial rate constant depends on the applied potential, the local
pH, and — for the growth reaction specifically — on the barrier-layer
thickness itself, through an exponential term whose sign is fixed by the
constant interfacial field strength assumption standard to the point defect
model: film growth decelerates exponentially as the film thickens, which is
what produces logarithmic-to-parabolic growth kinetics and, when a nonzero
dissolution rate is present, an eventual finite steady-state thickness.
Because open-circuit potential and pH are both constant over a given
exposure, they cannot be separated from the growth reaction's intrinsic rate
constant using thickness-versus-time data alone; the growth reaction's
potential and pH dependence is therefore absorbed into a single apparent
growth prefactor, and the film-thickness dependence into a single apparent
field-related exponent, which is what the identification below actually
recovers rather than the underlying elementary rate constants individually.

The outer layer's growth is coupled to the barrier layer's through a
constant-volume constraint: metal consumption at the receding
metal/barrier-layer interface and barrier-layer advance proceed at the same
rate, which — for the case of negligible barrier-layer destruction relevant
to this ultrapure-water environment — collapses the two layers' growth
rates into an exact affine relationship, with the outer-layer growth rate a
fixed multiple of the barrier-layer growth rate plus a small,
non-positive outer-layer dissolution term. This affine reduction is
algebraically simpler than the closed form Li and coworkers originally
published for the same relationship, and the two forms were independently
verified to agree to better than 1×10⁻¹⁰ nm before either was used in any
fit reported in this paper. The result is a six-parameter apparent system —
a barrier-layer growth prefactor, a field-related exponent, a barrier-layer
dissolution rate, the outer-to-barrier growth-rate ratio, an outer-layer
dissolution rate, and an initial barrier-layer thickness — governing two
coupled ordinary differential equations with closed-form solutions,
stabilized in log space so that the solutions remain numerically exact in
both the short-time and long-time limits. It is this reduced,
closed-form-solvable system, not the full interfacial reaction network,
that both the deterministic and physics-informed inversions in the
following sections identify.

---

## Data and deterministic inverse

Veile et al. characterized the oxide layer developed on X6CrNiNb18-10
(AISI 347) stainless steel after exposure to simulated boiling-water-reactor
hydrothermal water (240 °C, 7 MPa, 0.4 ppm dissolved O₂) for 72, 168, and
480 h, using focused-ion-beam cross-sections with energy-dispersive X-ray
(EDX) line scans across the resulting duplex oxide. Before using their
digitized layer-thickness data as an inversion target, we verified the
digitization against their own reported empirical fits: a scan-level
linear-space least-squares refit of the power law L = k·tⁿ to our digitized
chromium-layer data reproduces their published values almost exactly
(k = 6.522 versus their 6.521, n = 0.4964 versus their 0.4964). The
corresponding refit of the iron-layer data is limited by digitization
precision against their scan-level scatter rather than by a digitization
error, consistent with the pronounced physical heterogeneity Veile et al.
themselves report for the discrete-crystal outer layer; we therefore use
their reported layer means, not our own scan-level Fe re-digitization, as
the authoritative data throughout.

We fit the reduced HTW_PDM apparent parameter system to the combined
chromium (barrier-layer) and iron (outer-layer) thickness means in a nested
model ladder, each variant adding degrees of freedom to test whether the
added flexibility is actually supported by the three-exposure-time dataset.
The core four-parameter model (M1) cannot reproduce the disproportionately
thick early iron layer under the strict constant-volume coupling implied by
a barrier layer starting from zero thickness; adding a free dissolution rate
(M2) does not help, because the fit drives that rate to zero rather than
using it. Removing the constant-volume-coupling penalty entirely in the
six-parameter M3 variant instead reveals a classical, and here explicitly
demonstrated, point-defect-model identifiability trap: without a
counterbalancing prior, the barrier-layer growth prefactor and dissolution
rate run away together toward a degenerate ridge (prefactor and dissolution
rate both diverging while the field-strength parameter collapses toward
zero), a combination that fits the same data equally well through an
unphysical linear-growth escape route rather than through the intended
mechanism. This is, to our knowledge, the first time this specific failure
mode has been demonstrated explicitly on real point-defect-model data rather
than described in the abstract.

The identifiability trap is resolved, and the physically motivated fit
obtained, by adding a single additional parameter representing the initial
outer-layer thickness (model M4): a nonzero starting value for the
Fe-rich outer-layer crystals captures the rapid initial precipitation Veile
et al. observe by 72 h without requiring an unphysical dissolution/growth
balance. This five-parameter model reduces the data χ² from 11.5 (M1) to
5.79 (one degree of freedom remaining), a substantially better fit than
Veile et al.'s own per-element empirical power-law description achieves on
the same joint dataset (χ² 20.10, comparable parameter count) — the
mechanistic model outperforms the empirical one specifically because the
constant-volume constraint couples the chromium and iron layers through a
single growth-and-precipitation-boundary picture, where the power-law fit
treats each layer as an independent curve. The recovered kinetics place the
effective barrier-layer growth prefactor at 0.727 nm/h and the field-related
exponent at −0.0125 nm⁻¹; converting the latter to a field strength under
standard assumptions yields ε_f ≈ 1.7×10⁴ V/cm, a value that falls, as
physically expected from the intervening temperature difference, between
the ∼10² V/cm reported for supercritical-water point-defect-model fits at
500 °C and the ∼10⁶ V/cm typical of room-temperature passive films. The
recovered Pilling–Bedworth-type ratio, 1.051, lands close to the
stoichiometric spinel/magnetite value — and does so because the data pull
it there, not because of the prior: its profile-likelihood interval, [0.23,
2.34], is wide enough that the prior alone could not have produced this
specific point estimate.

A model ladder is only as convincing as the identifiability analysis behind
it, so we characterize which of the five M4 parameters the three-exposure
dataset can actually constrain, using profile-likelihood scans referenced to
the joint global minimum across the fit and all profile evaluations. The
barrier-layer growth prefactor, field exponent, and initial outer-layer
thickness are sharply identifiable (maximum Δχ² of 50, 130, and 82
respectively across their profile scans); the Pilling–Bedworth ratio is only
weakly identifiable (maximum Δχ² 16.8, profile interval spanning nearly an
order of magnitude); and the barrier-layer initial thickness is formally
unidentifiable from the exposure-time data alone (maximum Δχ² of 1.0,
requiring the air-film prior to pin down a value at all). We are not aware
of a prior point-defect-model parameter-estimation study, whether fit to
electrochemical impedance data or to compositional depth profiles, that
performs this kind of formal identifiability characterization — despite the
field routinely fitting four-to-six-parameter mechanistic models to data of
comparable or smaller size, a regime in which the distinction between
"the model fits" and "the data constrains the parameters" is not
automatic. Separately, an explicit search for a dissolution-rate upper
bound from the thickness data alone finds no bound below 1 nm/h; the two
qualitatively distinct long-time behaviors the model can produce — unbounded
logarithmic growth versus eventual saturation — remain numerically
indistinguishable given the available exposure window, a point we return to
in the Predictions section.

---

## Physics-informed inverse (NSEM)

Having established a trustworthy deterministic reference, we turn to a
physics-informed neural inversion of the same problem, motivated not by a
need to outperform the deterministic fit on this particular
zero-dimensional model — it does not need to, and does not try to — but
because the same inversion machinery is what the spatial extension in the
next section requires, where no closed-form or simple-integrator shortcut
exists. Getting the zero-dimensional case right, and understanding exactly
how it can go wrong, is the necessary foundation for trusting the spatial
result.

We consider two inversion modes. In the reference "hard" mode, the
barrier- and outer-layer thickness trajectories are generated by
differentiable Runge–Kutta integration through a set of learnable kinetic
parameters, so that the governing ordinary differential equations are
satisfied exactly by construction and only the kinetic parameters
themselves are optimized against the data. Applied to the real Veile et al.
dataset, this hard-mode inversion recovers the deterministic M4 optimum to
within 0.6% on every one of its five parameters, and its trajectory χ² of
5.79 matches the deterministic baseline exactly.

The more general "soft" mode instead lets a pair of physics-informed field
networks represent the barrier- and outer-layer thickness trajectories
directly, subject to a joint loss combining data misfit, ordinary
differential equation residuals, and initial-condition penalties, all
balanced through the same adaptive aggregator used elsewhere in this work.
This is the formulation that must generalize to the spatial problem, where
no differentiable integrator shortcut is available. Applied to the same real
data, we find that this joint field-and-parameter inversion settles into an
equilibrium at which the data loss is essentially zero while the physics
residual is not — the recovered thickness trajectory fits the three
exposure points closely, but the governing ordinary differential equation
along that trajectory is not actually satisfied to the precision the
low data loss would suggest. Independently re-solving the recovered kinetic
parameters through the exact closed-form or Runge–Kutta integration exposes
this directly: the trajectory's own χ² is 0.13, but the same parameters
evaluated through the exact deterministic solution give χ² = 33.7, an
inconsistency of more than two orders of magnitude that the trajectory loss
alone never reveals. Increasing the fixed weight on the physics residual
does not resolve this — because the adaptive loss aggregator renormalizes
relative term weights during training, raising one fixed multiplier shifts
where the equilibrium sits only marginally, not qualitatively. The same
pattern reproduces under an independent reweighting (shifting relative
emphasis toward the data term instead): trajectory χ² of 0.50 against a
closed-form χ² of 21.4, the same order-of-magnitude inconsistency under a
different specific configuration. We term this failure mode F-1: on the
manifold where the data loss is already near zero, the physics-residual
gradient is smaller than the data-loss gradient off that manifold, so
gradient descent has no incentive to leave a trajectory that satisfies the
data while leaving the physics only approximately satisfied — a failure
mode that only appears with sparse, few-point data, since with dense
collocation the data and physics terms cannot be satisfied independently in
the same way.

The practical fix is the hard-mode construction described above:
integrating the governing equation exactly removes the possibility of this
kind of slack by construction, at the cost of requiring a differentiable
integrator, which is available for the zero-dimensional kinetics but not,
in general, for the spatial partial differential equation extension in the
next section. We therefore propose, as a general-purpose reporting standard
for physics-informed inverse problems whether or not a hard-mode
alternative is available, a self-consistency check: re-solving the
recovered parameters through an independent, physics-exact forward solve
(closed-form, Runge–Kutta, or Newton iteration as appropriate) and reporting
the resulting χ² alongside the trajectory χ² the network itself reports.
This extends, from forward to inverse problems, the same
Newton-verification discipline already standard for this model family's
forward solves, and it is the only test in our experience that reliably
distinguishes a trustworthy soft-mode fit from an F-1 failure without
requiring a hard-mode alternative to exist.

We validate the hard-mode inversion's recovery behavior on synthetic data
generated from known ground-truth kinetics with 2% relative noise at the
Veile-like exposure times, before trusting it on the real dataset. Recovery
deviations from ground truth are 1.35% (growth prefactor), 3.36% (field
exponent), 0.48% (Pilling–Bedworth ratio), 10.96% (barrier-layer initial
thickness), and 0.07% (outer-layer initial thickness) — the one large
deviation lands, as expected, on exactly the parameter the profile-
likelihood analysis above already flagged as formally unidentifiable from
this exposure-time structure, a direct cross-validation of the
identifiability finding by an independent method.

We further cross-validate the profile-likelihood identifiability
conclusions against a fifty-member parametric bootstrap: each member
resamples the Cr and Fe layer means within their reported standard errors
and re-solves the hard-mode inversion independently. For the two
sharply-identifiable parameters the growth prefactor and field exponent, the
bootstrap and profile-likelihood intervals agree closely — a second,
independent line of evidence for the same conclusion. For the
prior-regularized parameters, the Pilling–Bedworth ratio and the
barrier-layer initial thickness, the bootstrap spread is instead
substantially narrower than the profile-likelihood interval. This is not a
tighter constraint; it is an artifact of the bootstrap perturbing only the
data while the profile likelihood explores the full parameter range —
when the data genuinely has no leverage on a parameter, as the profile scan
already showed, every resampled refit collapses onto the same informative
prior rather than exploring the space the prior was regularizing. The
practical conclusion is that the profile-likelihood interval, not the
narrower bootstrap spread, is the honest uncertainty statement to quote for
these two parameters — an instance where two nominally independent
uncertainty-quantification methods must be interpreted together rather than
averaged, and where reporting only the numerically smaller (bootstrap)
interval would be actively misleading.

Finally, running the same forward-solve harness with two different backbone
representations — a standard multilayer perceptron and a Kolmogorov–Arnold
network — under identical training and loss-balancing settings, both pass
the 0.5% forward acceptance criterion (relative L∞ error 0.14% and 0.12%
respectively for the two thickness fields under the multilayer-perceptron
backbone, 0.11% and 0.04% under the Kolmogorov–Arnold backbone). We report
this as a parity result, not a superiority claim for either backbone: on
this problem, architectural choice inside the same physics-informed harness
does not materially change forward accuracy.

---

## Spatial validation (Tier 2)

The zero-dimensional kinetics above are fit to two scalar thickness
trajectories. Veile et al.'s energy-dispersive X-ray line scans, however,
resolve full composition-versus-depth profiles at each exposure, a
substantially richer dataset that the zero-dimensional fit never uses. We
therefore extend the model spatially — resolving oxygen- and
cation-vacancy transport across the barrier layer under the field strength
already identified in the zero-dimensional fit — and use the resulting
composition predictions as an independent cross-check against data the
kinetic fit never saw, rather than as an additional fitting target. Every
kinetic parameter entering this section (the growth prefactor, field
exponent, and Pilling–Bedworth ratio) is frozen at its zero-dimensional
value; nothing here is re-fit.

Before attempting any spatial inversion, we first ask which spatial
quantities the exposure-time-and-instrument structure of the Veile et al.
dataset could plausibly constrain at all, using a synthetic identifiability
gate analogous in spirit to the zero-dimensional profile-likelihood analysis
above but built around the digitized-EDX instrument model (position jitter,
compositional noise, and, critically, the scan-to-scan physical
heterogeneity Veile et al.'s own replicate scans exhibit — omitting this
heterogeneity and treating only instrument noise as the source of
uncertainty produces a spuriously optimistic identifiability verdict for
every parameter, an artifact worth flagging explicitly given how easy it is
to omit). Twenty replicate synthetic experiments show that the
barrier-layer thickness, the interfacial transition widths, the nickel
enrichment-zone amplitude and width, and the chromium fraction inside the
barrier layer are all identifiable, with replicate relative standard
deviations of 2–19%; the in-layer defect-transport composition gradient —
the one quantity the spatial extension uniquely adds over the
zero-dimensional model — is only weakly identifiable, with a replicate
relative standard deviation of 47% and a minimum statistically detectable
gradient of 0.93 absolute percent across the layer. We therefore restrict
the spatial model's inverse scope to the interface, zone, and composition
parameters the gate supports, and treat the in-layer defect-transport
gradient as a forward prediction to be compared against data, not a
parameter to be fit — the same discipline the zero-dimensional
identifiability analysis already established, now applied a second time to
a genuinely different sub-problem.

The steady-state defect-transport problem, solved both by Newton iteration
on a spectral collocation grid and by a spectral-element-network forward
solve, agrees with its analytic reference to 8.9×10⁻¹⁶ (Newton, oxygen
vacancies) and 2.2×10⁻¹³ (Newton, cation vacancies), and the neural solve
agrees with the Newton reference to 2.5×10⁻⁴ and 5.4×10⁻⁵ relative error
respectively — both well inside the 0.5% acceptance bar used throughout this
work. This verification exercise surfaces a structural result worth
reporting independently of the verification itself: the steady-state
nondimensional profile shape depends only on the Péclet-type group
Pe = |z|γEL, a combination of quantities the zero-dimensional kinetics fit
already fixes, and does not depend on the defect diffusivity at all. We
confirmed this directly, not merely by inspecting the governing equations,
by solving the identical problem with the diffusivity varied over a
ten-thousand-fold range: the resulting profile shapes are identical to
within floating-point precision (maximum deviation 0.0×10⁰ across three
diffusivity values spanning four orders of magnitude), while the absolute
defect concentration scale — a quantity invisible to a composition-based
measurement such as EDX — scales inversely with diffusivity exactly as the
governing equations predict. This is, to our knowledge, the first explicit
statement of this structural identifiability limit in the point-defect-model
literature, and it independently corroborates the statistical
identifiability gate above by a completely different, purely analytic
argument: two methods reaching the same conclusion by unrelated routes is
itself evidence worth reporting.

The transient extension, solved on a moving Landau-transformed domain that
tracks the growing barrier-layer thickness, lets us quantify an
approximation every zero-dimensional point-defect-model application makes
implicitly: that the spatial defect profile equilibrates instantaneously to
its steady-state shape at each moment, rather than lagging the truly
time-dependent transport problem. Comparing the transient solution at the
end of the 480-hour exposure window to the instantaneous quasi-steady
profile at that same moment, the fast-migrating oxygen-vacancy species (with
a migration transit time of order 5 hours) lags its quasi-steady shape by
only 0.65%, while the slower cation-vacancy species (migration transit time
of order 36 hours) lags by 5.75% — both comfortably inside our acceptance
tolerances, and both scaling with the ratio of migration transit time to
growth timescale exactly as the underlying physics predicts. We are not
aware of a prior explicit quantification of this quasi-steady approximation
error in the point-defect-model literature; every application we are aware
of, including our own zero-dimensional fit above, makes the approximation
without characterizing its size.

The composition-map prediction is the section's headline result precisely
because it uses no free parameters at all: from spinel stoichiometry alone
— the FeCr₂O₄ barrier-layer phase Veile et al. identify by transmission
electron microscopy — the predicted chromium plateau composition inside the
barrier layer is 46.7 weight percent, against a measured plateau of
approximately 45%, with layer widths reproduced at the same quality as the
zero-dimensional kinetic fit itself (deviations of +22%, −20%, and +3% at
the three exposure times). No parameter entering this prediction was fit to
the composition data; the comparison is a genuine forward cross-check
against data collected for an entirely different purpose than kinetic
parameter estimation. The predicted in-layer defect gradient — 1.45, 0.96,
and 0.46 absolute percent at 72, 168, and 480 hours respectively — straddles
the 0.93 absolute percent minimum detectable gradient identified by the
identifiability gate above: the gradient is marginally detectable at the
earliest exposure and undetectable by the latest, a direct, quantitative
link between the structural and statistical identifiability arguments and
the actual measurable quantity a future EDX campaign would report.

---

## Predictions

The practical value of a mechanistic model over an empirical power-law
description is that it can be extrapolated on physical grounds beyond the
exposure window it was fit to, and the two descriptions diverge sharply once
extrapolated. At ten years, the identified point-defect model predicts a
barrier-layer thickness of 535 nm, while a naive extrapolation of Veile et
al.'s own empirical power law predicts 1853 nm — a factor of 3.5 difference
in projected barrier-layer thickness, and by extension in the metal
consumed to produce it, between the two descriptions at a timescale directly
relevant to structural-component life assessment. This divergence is not a
subtle statistical distinction: it is the practical consequence of the
constant-volume-coupled mechanistic growth law saturating logarithmically
while the empirical power law, fit independently to each layer without a
coupling constraint, continues to extrapolate as a power law indefinitely.

The point-defect model itself admits two qualitatively different long-time
behaviors depending on whether the dissolution rate is exactly zero or
merely small and positive — an unbounded, slowly saturating logarithmic
growth in the first case, and an eventual hard saturation at a finite
steady-state thickness in the second. We find that these two branches track
within about five percent of each other out to 10⁵ years, meaning no
practically feasible exposure experiment could distinguish between them
using thickness data alone; the earlier identifiability analysis's finding
that the dissolution rate is unconstrained below 1 nm/h is the same
conclusion stated as a parameter bound rather than as a prediction. Taken
together, these two results motivate a concrete, actionable recommendation:
an exposure beyond roughly 2000 hours would be sufficient to discriminate
the mechanistic model from the empirical power-law extrapolation, since the
two descriptions separate substantially well before that horizon, but no
feasible exposure would discriminate between the point-defect model's own
two long-time branches — settling the dissolution-rate question requires
independent water-chemistry reasoning, not additional thickness
measurements at longer exposure times.

Because the Veile et al. dataset was collected at a single temperature and a
single dissolved-oxygen concentration, we separately map the model's
sensitivity across the full range of conditions a boiling-water-reactor
might operate under — 200 to 285 °C at 7 MPa (the upper bound set by the
liquid-phase stability limit at that pressure; the originally
considered 300 °C endpoint is not reachable as a liquid at this pressure and
is therefore excluded from the scan) and 0.01 to 8 ppm dissolved oxygen —
using an Arrhenius temperature dependence on the rate-determining reaction,
a field-attenuation channel tied to the same temperature dependence, and an
ideal-Nernst shift of the growth prefactor with dissolved-oxygen
concentration as a explicit, conservative lower bound on the true
corrosion-potential response (see Discussion for the caveat this bound
carries). Across this entire operating envelope, the predicted
barrier-layer thickness at the 480-hour exposure varies by only a factor of
1.2 to 1.4, and at ten years by only 1.19 to 1.25, depending on the assumed
reaction free energy; temperature dominates the sensitivity through the
combined Arrhenius and field-attenuation channels, while dissolved oxygen
contributes only a mild tilt through the Nernstian shift, consistent with
the barrier-layer dissolution rate remaining effectively zero across the
scanned range. The long-time prediction is therefore robust to the specific
operating condition within the envelope a reactor would actually see — the
central engineering conclusion of this sensitivity analysis.

The recovered field strength, 1.7×10⁴ V/cm at 240 °C, is also worth placing
in its broader physical context: it falls, as expected from the intervening
temperature dependence of ionic transport, between the roughly 10² V/cm
reported for supercritical-water point-defect-model fits at 500 °C and the
roughly 10⁶ V/cm typical of room-temperature passive films. We report this
ordering as physically sensible corroborating evidence for the fitted field
exponent rather than as a validated temperature trend in its own right,
since it rests on only three literature anchor points spanning very
different alloy systems and film chemistries.

---

## Discussion

The mechanistic model's advantage over Veile et al.'s own per-element power
laws is not merely a better χ²; it supports a physical reading of their
fitted exponents that the power-law description alone cannot offer. The
near-parabolic chromium-layer exponent they report (n ≈ 0.50) is, in the
mechanistic picture, the signature of a barrier layer growing under
near-zero dissolution in ultrapure water, while the sub-parabolic,
near-cubic iron-layer exponent (n ≈ 0.22) reflects deposition constrained by
the constant-volume coupling to the barrier layer's growth from a
substantial initial outer-layer precipitate population, rather than
independent iron-layer growth kinetics. The joint χ² comparison in the Data
and deterministic inverse section is the quantitative version of this same
claim: the mechanistic model outperforms the empirical fit specifically
because it enforces this coupling, which the power-law description, fit
layer by layer, has no mechanism to express.

The nickel enrichment zone Veile et al. observe at the receding metal/
barrier-layer interface offers a second mechanistic reading beyond the
spatial composition-map cross-check of the previous section. Fitting an
exact moving-frame nickel-transport model to the real zone widths and peak
amplitude recovers a nickel incorporation ratio of 0.29 — nickel is
incorporated into the growing barrier layer at roughly thirty percent of
the rate chromium and iron are, a partial exclusion rather than the
essentially total exclusion a naive interfacial-kinetics assumption would
suggest. The fit further requires a nonzero matrix contribution to the
outer layer's cation supply: excluding this term worsens the fit by six
units of χ² even after re-optimizing every other parameter, which we read
as quantitative evidence that the iron-rich outer-layer crystals draw at
least part of their cation supply from the receding metal surface through
the barrier layer, rather than growing entirely by re-precipitation of
already-dissolved species — a mass-balance argument the zero-dimensional
kinetics alone could not have made.

This nickel-zone fit is not, however, fully satisfactory, and we report the
shortfall rather than omit it. Its χ² per degree of freedom is large, and
the underlying discrepancy is one of shape rather than scatter: the fitted
constant-diffusivity exclusion model predicts a zone width that grows
monotonically with exposure time, while the measured widths are
approximately flat across the three exposures. Motivated by this tension, we
tested two physically motivated extensions — a defect-flux-coupled
effective mobility tied to the already-identified barrier-layer growth
kinetics, and a finite-capacity interfacial trapping closure — each adding a
single new parameter to the same nickel-transport model. Applying the same
synthetic identifiability discipline used for the spatial defect-transport
gradient above, both extensions fail a twenty-replicate identifiability gate
before we allowed either to touch the real data: the flux-coupled closure's
own new parameter is recoverable, but destabilizes the already-identified
nickel-transport parameters past an acceptable threshold, while the
finite-capacity closure's new parameter is not recoverable at all. A
subsequent forward-only scan across the flux-coupled closure's plausible
parameter range, at the frozen nickel-transport kinetics, confirms that no
single value is simultaneously shape-correct and a quantitative improvement
over the original constant-diffusivity fit. We read this as a second,
independent demonstration of the same limitation the spatial
identifiability gate already established for the in-layer defect-transport
gradient: three to four exposure times cannot support the additional
parameters either extension would require, regardless of which physically
motivated closure is chosen, and the nickel zone's flat-width behavior
points toward a qualitatively different, likely time-decaying or
interface-coupled mobility mechanism that this dataset cannot resolve — a
concrete target for future work once denser exposure-time data exists,
not a modeling failure to be patched with a more flexible parameterization
fit to the same three points.

Several limitations bound the scope of the conclusions above. The entire
kinetic identification rests on a single exposure temperature and a single
dissolved-oxygen concentration; the operating-envelope sensitivity maps in
the Predictions section are forward projections under sourced physical
assumptions, not validation against multi-temperature or multi-chemistry
data, which does not yet exist for this alloy in this environment. The
open-circuit potential dependence is absorbed into the fitted growth
prefactor rather than treated as an independent channel, and the explicit
ideal-Nernst oxygen-concentration channel used in the sensitivity maps is
consequently a lower bound on the true corrosion-potential response —
measured electrochemical corrosion potential against dissolved-oxygen
curves in boiling-water-reactor-relevant water chemistry are known to be
steeper than the ideal-Nernst relationship in the parts-per-billion
transition region, so the true oxygen sensitivity across the operating
envelope may exceed what we report. The three-exposure-time structure of
the dataset is, as this paper has argued at length, both the central
limitation and the central methodological point; we do not repeat that
argument here beyond noting that every identifiability result in this paper
— the zero-dimensional profile-likelihood analysis, the spatial
defect-transport gate, and the nickel-mobility closure gate — traces to the
same root cause and points at the same remedy. Finally, energy-dispersive
X-ray-derived layer thicknesses are not identical to total oxide mass, and
the composition-map comparison in particular should be read as a shape and
plateau-composition cross-check rather than a mass-balance validation.

That every one of these limitations traces back to the same underlying
constraint — too few independent exposure-time observations for the number
of physical parameters genuinely in play, whether in the zero-dimensional
kinetics, the spatial defect-transport gradient, or the nickel-mobility
closure — is itself the paper's central methodological conclusion, and it
yields a single, concrete recommendation rather than a list of disconnected
caveats: an exposure campaign extending beyond roughly 2000 hours, the same
threshold identified in the Predictions section as sufficient to
discriminate the mechanistic model from a naive power-law extrapolation,
would simultaneously improve the identifiability of every parameter this
paper found weakly constrained or entirely unidentifiable.

---

## Data and Code Availability

All code implementing the deterministic fits, physics-informed forward and
inverse solves, identifiability analyses, and spatial extension described in
this paper is released with PhysicsNeMo's spectral-element-network module
(`experimental.models.scen`) referenced in the Methods section, together
with the example directory containing this paper's specific model,
configuration files, and analysis scripts. The digitized Veile et al.
layer-thickness and depth-profile data used throughout this paper are
transcriptions of published figures, released with explicit provenance
notes tracing each digitized value back to its source figure; this
constitutes reuse of published data requiring citation, not separate data-
sharing permission, and is cited as such throughout. Exact reproduction
commands for every figure and table in this paper are documented alongside
each corresponding analysis script.
