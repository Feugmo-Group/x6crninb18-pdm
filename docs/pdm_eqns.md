# HTW_PDM — Point Defect Model for Oxide Growth on X6CrNiNb18-10 in High-Temperature Water

Complete derivation of the theoretical model implemented in this example. The mechanistic
framework is the SCW_PDM of Li & Macdonald (Corros. Sci. 163 (2020) 108280), specialized from
supercritical to subcritical high-temperature water (HTW) and applied to the duplex oxide on
Nb-stabilized X6CrNiNb18-10 (AISI 347) under the conditions of Veile et al. (Materials 17 (2024)
4500). Equation numbers `(Ln)` refer to the Li 2020 paper; unnumbered boxed equations are the
working forms implemented in `src/`.

---

## 1. System, geometry, and species

Duplex scale, coordinates with origin at the **barrier-layer/outer-layer (bl/ol) interface**:

```
 metal (m) | barrier layer (bl), MO_{chi/2} | outer layer (ol), MO_{delta/2} | HTW (e)
           x = L_bl                x = 0                      x = -L_ol
```

Microstructural identification for X6CrNiNb18-10 in BWR HTW (Veile 2024):

| PDM object | Observed layer | Phase (XPS/EDX) | Valences |
|---|---|---|---|
| barrier layer (bl) | inner nanocrystalline Cr-rich layer | FeCr2O4 / NiCr2O4 spinel | chi = 8/3 (spinel-averaged cation) |
| outer layer (ol) | discrete Fe-rich crystals | Fe3O4 (magnetite), partly Fe2O3 | delta = 8/3, theta = 3 |
| — (inert) | Ni enrichment zone at bl/matrix interface | metallic/NiO traces | not modeled (dL_Ni/dt ≈ 0) |

Kröger–Vink species in the bl: cation vacancy V_M^chi', cation interstitial M_i^chi+, oxygen
vacancy V_O**, oxide ion O_O, cation M_M, electron e'. Constants: gamma = F/(RT);
Omega_bl, Omega_ol = molar volumes per cation of bl and ol oxides; Omega_m = molar volume of
metal; PBR = Omega_bl/Omega_m (Pilling–Bedworth ratio).

Environment (Veile 2024 conditions):
- T = 513.15 K (240 °C), P = 7 MPa → liquid water, density rho ≈ 0.8137 kg/L (IAPWS-97).
- rho >> 0.1 g/cm3 ⇒ the **electrochemical-oxidation (EO), high-density branch** of the SCW_PDM
  applies: this is the classical condensed-water PDM limit of the model.
- Dissolved oxygen [O2] = 0.4 ppm; ultrapure water, conductivity 0.055 µS/cm.
- Neutral pH at temperature: pKw(240 °C) ≈ 11.2 ⇒ pH ≈ 5.6.
- Dimensionless relative volumetric concentrations (L13, L14), reference C0 = 1 mol/L:

      C_H2O = 1000*rho / (18.015 * C0_H2O)              ≈ 45.2
      C_O   = [O2]_ppm * rho * 1e-3 / (32.00 * C0_O)    ≈ 1.02e-5

- Potential V: open circuit (no polarization applied in the autoclave). V is set by the
  O2/H2O redox couple (ECP) and is constant over an exposure; it is therefore absorbed into
  effective rate constants (Section 6). The OCP closure i_pol = 0 (L46) becomes relevant only
  if V is made explicit (parametric studies in ECP).

---

## 2. Interfacial reactions

The complete SCW_PDM reaction set (L Fig. 5), with the high-density/EO significance retained for
HTW. chi and delta are the cation valences in bl and ol; theta in the dissolved/further-oxidized
product.

At the **m/bl interface** (x = L_bl):

| # | Reaction | Role | Lattice |
|---|---|---|---|
| 1 | m + V_M^chi' -> M_M + v_m + chi e' | annihilates cation vacancy; cation feed for ol (vacancy path) | conservative |
| 2 | m -> M_i^chi+ + v_m + chi e' | generates cation interstitial; cation feed for ol (interstitial path) | conservative |
| 3 | m -> M_M + (chi/2) V_O** + chi e' | generates oxygen vacancies: **bl growth into the metal** | **non-conservative** |

At the **bl/ol interface** (x = 0):

| # | Reaction | Role | Lattice |
|---|---|---|---|
| 4 | M_M + (delta/4) O2 + chi e' -> V_M^chi' + MO_{delta/2} | generates cation vacancy; deposits ol (direct O2 route) | conservative |
| 4' | M_M -> V_M^chi' + M^delta+ + (delta-chi) e' | generates cation vacancy; cation ejected (dissolution/re-precipitation route; favored in condensed water) | conservative |
| 5 | M_i^chi+ + (delta/4) O2 + chi e' -> MO_{delta/2} | annihilates interstitial; deposits ol | conservative |
| 5' | M_i^chi+ -> M^delta+ + (delta-chi) e' | annihilates interstitial; cation ejected | conservative |
| 6 | V_O** + 1/2 O2 + 2e' -> O_O | annihilates oxygen vacancy (O2 present) | conservative |
| 7 | V_O** + H2O + 2e' -> O_O + H2 | annihilates oxygen vacancy (water route) | conservative |
| 8 | V_O** + H2O -> O_O + 2H+ | annihilates oxygen vacancy, proton release (classical PDM; H+ available in condensed water) | conservative |
| 9 | H2O + 2e' -> 1/2 H2 + OH- | cathodic partial reaction | (no defect) |
| 10 | MO_{chi/2} + ((delta-chi)/4) O2 -> MO_{delta/2} | converts bl to ol: **bl destruction** (chemical) | **non-conservative** |
| 10' | MO_{chi/2} + chi H+ -> M^delta+ + (chi/2) H2O + (delta-chi) e' | proton-assisted **bl dissolution** | **non-conservative** |

At the **ol/water interface** (x = -L_ol):

| # | Reaction | Role | Lattice |
|---|---|---|---|
| 11 | MO_{delta/2} + ((theta-delta)/4) O2 -> MO_{theta/2}(d) | **ol destruction**: dissolution / further oxidation (magnetite -> hematite, seen in Veile's XPS) | **non-conservative** |

Only Reactions 3 (growth) and 10/10' (destruction) move the bl interfaces; only Reaction 11
destroys the ol. These four define the thickness ODEs.

---

## 3. Potential distribution

Postulates (L §2.1): potential drops exist at m/bl and bl/e; the field strength eps_f in the bl is
constant (buffered, independent of V and L_bl); the bl/e drop is linear in V and pH and
independent of L_bl:

    phi_ble = alpha*V + beta*pH + phi0                                   (L1)
    V = phi_mbl + eps_f*L_bl + phi_ble                                   (L2)
    phi_mbl = (1 - alpha)*V - eps_f*L_bl - beta*pH - phi0                (L3)

alpha = polarizability of the bl/e interface (d phi_ble / dV); beta = d phi_ble / d pH (< 0,
typically -0.005 to -0.05 V); phi0 = drop at V = 0, pH = 0, L_bl = 0. The porous ol carries no
potential drop (cathodic reactions occur at bl/ol via O2/H2O transport through pores; no remote
cathode current in an autoclave at OCP).

---

## 4. Rate constants (activated complex theory with partial charges)

Every interfacial rate constant takes the form (L8, derivation in Li Appendix A):

    k_i = k_i^0 * exp(a_i*V) * exp(b_i*L_bl) * exp(c_i*pH)

with temperature dependence carried by the base standard rate constant (L9):

    k_i^00 = k_i^00' * exp[ -(alpha_i * dG0_Ri / R) * (1/T - 1/T0) ]

Coefficients (L Table 1); alpha_i = transfer coefficient of Reaction i:

| i | a_i (1/V) | b_i (1/cm) | c_i |
|---|---|---|---|
| 1, 2, 3 (m/bl) | alpha_i*chi*gamma*(1-alpha) | **-alpha_i*chi*gamma*eps_f** | -alpha_i*chi*gamma*beta |
| 4, 5, 6, 7, 10, 11 | 0 | 0 | 0 |
| 4', 5' | alpha_i*delta*gamma*alpha | 0 | alpha_i*delta*gamma*beta |
| 8, 9 | 2*alpha_i*gamma*alpha | 0 | 2*alpha_i*gamma*beta |
| 10' | alpha_10p*(delta-chi)*gamma*alpha | 0 | alpha_10p*(delta-chi)*gamma*beta |

The m/bl reactions inherit V-, L_bl-, and pH-dependence from phi_mbl (L3); the bl/ol
electrochemical reactions from phi_ble (L1), hence b_i = 0 there. The single most important
structural fact: **b_3 = -alpha_3*chi*gamma*eps_f < 0** — the growth reaction decelerates
exponentially with film thickness. This is what generates direct-logarithmic/pseudo-parabolic
kinetics and a finite steady-state thickness.

Steady-state defect flux balances (rates equal at both interfaces per defect, L10–L12):

    k_1 * C_VM^L      = k_4*C_O^n + k_4'                                  (L10)
    k_2               = (k_5*C_O^n + k_5') * C_Mi^0                       (L11)
    (chi/2) * k_3     = (k_6*C_O^m + k_7*C_H2O^p + k_8*C_H2O^p) * C_VO^0  (L12)

Kinetic orders: n (R4, R5 in C_O), m (R6 in C_O), p (R7–9 in C_H2O), q (R10 in C_O),
r (R11 in C_O), s (R10' in H+ activity ratio). Li assume n = m = q = r = 1/2.

---

## 5. Thickness evolution equations (the governing ODEs)

### 5.1 Barrier layer

Non-conservative reactions only (L16, with porosity correction rho0_bl/rho_bl):

    dL_bl/dt = Omega_bl * [ k_3 - k_10*C_O^q - k_10'*(C_H+/C0_H+)^s ] * rho0_bl/rho_bl

Substituting k_3 = k_3^0 exp(a_3 V) exp(b_3 L_bl) exp(c_3 pH) at constant V, pH (L21–L23):

    ┌───────────────────────────────────────────────────────────┐
    │  dL_bl/dt = A_bl * exp(b_3 * L_bl) - C_bl                  │
    └───────────────────────────────────────────────────────────┘

    A_bl = Omega_bl * k_3^0 * exp(a_3*V) * exp(c_3*pH) * rho0_bl/rho_bl      (L22)
    C_bl = Omega_bl * [ k_10*C_O^q + k_10'*(C_H+/C0_H+)^s ] * rho0_bl/rho_bl (L23; the paper's
           printed "k_9" in L23 is a typographical error for k_10 — required by L16/L17/L44)

**Closed-form solution** for constant A_bl, b_3, C_bl and L_bl(0) = L0 (L24/L39):

    L_bl(t) = L0 - (1/b_3) * ln[ 1 + (A_bl/C_bl) * exp(b_3*L0) * (exp(-b_3*C_bl*t) - 1) ] - C_bl*t

**Steady state** (dL_bl/dt = 0, L17–L20):

    L_bl,ss = ((1-alpha)/eps_f)*V - (beta/eps_f)*pH
              - (1/(alpha_3*chi*eps_f*gamma)) * ln[ (k_10*C_O^q + k_10'*(.)^s) / k_3^0 ]

L_bl,ss is linear in V and pH — the PDM signature. In ultrapure HTW (0.055 µS/cm, pH ≈ 5.6,
C_O ≈ 1e-5) the destruction terms are very small: C_bl -> 0, the steady state moves to very large
L, and over a finite window L_bl(t) ≈ L0 - (1/b_3)*ln(1 - b_3*A_bl*exp(b_3*L0)*t) — direct
logarithmic growth, which over 72–480 h is numerically indistinguishable from the parabolic
t^0.5 law Veile fitted (their Cr layer: n = 0.4964, no saturation at 480 h). Quantifying this
equivalence is one of the results of the study.

**Limiting-case map** (what the data can select between):
- C_bl = 0, |b_3|*L << 1: linear growth L ≈ L0 + A_bl*t.
- C_bl = 0, |b_3|*L = O(1): direct-logarithmic (pseudo-parabolic over a decade of t).
- C_bl > 0: logarithmic -> saturation at L_bl,ss.

### 5.2 Outer layer and the constant-volume constraint

The ol grows from cations transmitted through the bl (fluxes k_1*C_VM^L via vacancies and k_2 via
interstitials, deposited by R4/4'/5/5'), plus bl->ol conversion (R10/10'), minus dissolution
(R11) (L26):

    dL_ol/dt = Omega_ol * [ k_1*C_VM^L + k_2 + k_10*C_O^q + k_10'*(.)^s - k_11*C_O^r ] * rho0_ol/rho_ol

**Constant-volume growth** (marker experiments: the bl/ol interface stays at the original metal
surface). Total metal consumption rate = bl advance rate into the metal (L29–L32):

    dL_m/dt  = Omega_m * (k_1*C_VM^L + k_2 + k_3)          (metal destruction, R1+R2+R3)
    dL_bl/dt = Omega_bl * (k_3 - k_d),  k_d := k_10*C_O^q + k_10'*(.)^s

    dL_m/dt = dL_bl/dt  =>
    ┌─────────────────────────────────────────────────────────────┐
    │  k_1*C_VM^L + k_2 + k_3 = PBR * (k_3 - k_d)                  │   (L32)
    └─────────────────────────────────────────────────────────────┘

For an n-type bl (interstitials/oxygen vacancies dominate; k_1*C_VM^L << k_2) and k_d ≈ 0:

    k_2 = (PBR - 1) * k_3                                          (L33)

Substituting L32 into the ol equation — the k_d terms combine exactly,
(PBR-1)k_3 - PBR*k_d + k_d = (PBR-1)*(k_3 - k_d) — giving (L41):

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  dL_ol/dt = Omega_ol * [ (PBR-1)*(k_3 - k_d) - k_11*C_O^r ]               │
    │           = (PBR-1)*(Omega_ol/Omega_bl) * dL_bl/dt                        │
    │             - Omega_ol * k_11*C_O^r * rho0_ol/rho_ol                      │
    └──────────────────────────────────────────────────────────────────────────┘

so with negligible bl destruction the thickness ratio is fixed by stoichiometry alone (L36/L37):

    dL_ol/dL_bl = (PBR - 1) * Omega_ol / Omega_bl  =: PBR_eff

**Closed-form.** Li write it (L42–L45) with A_ol = (PBR-1)*(Omega_ol/Omega_bl)*A_bl and
C_ol = Omega_ol*[(PBR-1)*k_d + k_11*C_O^r] (so that dL_ol/dt = A_ol*exp(b_3*L_bl) - C_ol):

    L_ol(t) = (A_ol/(A_bl*b_3)) * ln[ A_bl*exp(b_3*L_bl(t)) - C_bl ] - C_ol*t + D_ol
    D_ol    = -(A_ol/(A_bl*b_3)) * ln[ A_bl*exp(b_3*L0) - C_bl ] + L_ol(0)

Because dL_ol/dt is an exact affine function of dL_bl/dt, the reduced system integrates trivially:

    L_ol(t) = PBR_eff * ( L_bl(t) - L0 ) + C_x * t + L_ol(0),
    PBR_eff := (PBR-1)*Omega_ol/Omega_bl,   C_x := -Omega_ol*k_11*C_O^r  (<= 0)

— an independent verification target alongside L42.

Total scale: L_tot = L_bl + L_ol. Weight gain (not measurable for Veile's cubes, kept for
completeness, L28): d(dw)/dt = r_bl*rho_bl*dL_bl/dt + r_ol*rho_ol*dL_ol/dt, with r = oxygen mass
fraction of each oxide; closed 7-parameter form in L48–L56.

### 5.3 Interfacial current (for ECP-explicit extensions)

Partial currents (Li Appendix B): i_1 = chi*F*k_1*C_VM^L, i_2 = chi*F*k_2, i_3 = chi*F*k_3,
i_4 = -chi*F*k_4*C_O^n, i_5 = -chi*F*k_5*C_O^n*C_Mi^0, i_6 = -2F*k_6*C_O^m*C_VO^0,
i_7 = -2F*k_7*C_H2O^p*C_VO^0, i_9 = -2F*k_9*C_H2O^p, i_4' = (delta-chi)*F*k_4', etc.
At open circuit: sum of partial currents = 0 (L46) — fixes V = V_ocp given the k_i^0.

---

## 6. Reduction to the identifiable HTW_PDM (what we actually fit)

At open circuit V and pH are constant over an exposure, so exp(a_3*V + c_3*pH) is a constant
multiplier: it cannot be separated from k_3^0 by thickness-vs-time data alone. The observable
model is the **apparent-parameter system** (identical in structure to Li Table 5):

    dL_bl/dt = A_bl * exp(b_3 * L_bl) - C_bl                      L_bl(0) = L0
    dL_ol/dt = PBR_eff * (A_bl * exp(b_3 * L_bl) - C_bl) + C_x    L_ol(0) = 0

with C_x := -Omega_ol*k_11*C_O^r <= 0 (outer-layer dissolution / further oxidation only).

**Parameter vector** theta = { A_bl [nm/h], b_3 [1/nm] (< 0), C_bl [nm/h] (>= 0),
PBR_eff [-] (> 0), C_x [nm/h], L0 [nm] (> 0) } — 6 parameters, 2 ODEs, data at 3 times
for 2 layers (+ Ni consistency check).

Physical recovery after fitting:

    eps_f   = -b_3 / (alpha_3 * chi * gamma * (1))     [from b_3 = -alpha_3*chi*gamma*eps_f]
    k_3^0   = A_bl / (Omega_bl * exp(a_3*V + c_3*pH))  [order of magnitude, given alpha_3, V]
    PBR     = 1 + PBR_eff * Omega_bl/Omega_ol
    k_d     = C_bl / Omega_bl

with alpha_3 in [0.1, 0.5] (Li fit alpha_3 = 0.12 for HCM12A) and chi = 8/3, gamma = F/RT =
22.63 1/V at 513.15 K. Molar volumes per cation: Omega_bl(FeCr2O4, 3 cations) = 46.7/3 =
15.6 cm3/mol; Omega_ol(Fe3O4) = 45.0/3 = 15.0 cm3/mol; Omega_m(fcc SS) ≈ 7.1 cm3/mol
=> theoretical PBR ≈ 2.05–2.10 (Li Table 3), PBR_eff ≈ 1.0–1.1.

**Priors / constraints** (3 time points cannot determine 6 parameters freely):
- L0: air-formed film on high-Cr austenitic SS, 1–5 nm (Li: 1.01 nm for 316L). Prior L0 in [0.5, 10] nm.
- PBR_eff: prior [0.8, 1.5] centered on the stoichiometric 1.0–1.1.
- C_bl: >= 0, expected ≈ 0 (ultrapure water); test both C_bl = 0 (5-parameter model) and free.
- b_3: < 0 strictly. |b_3| ~ alpha_3*chi*gamma*eps_f; with eps_f in [1e2, 1e6] V/cm this spans
  |b_3| in [~7e-5, ~0.7] 1/nm — the data must localize it.
- C_x: free sign, small.

**Units convention for implementation:** thickness in nm, time in h. Conversions from Li's
cgs (cm, s): 1 cm/s = 3.6e10 nm/h; 1 1/cm = 1e-7 1/nm.

Sign conventions to respect everywhere: b_3 < 0, C_bl >= 0, beta < 0, D_ol < 0.

---

## 7. Nondimensionalization for the NSEM solver

Characteristic scales: t_c = 480 h (window), L_c = 100 nm (observed Cr-layer scale). With
tau = t/t_c, lam = L/L_c:

    dlam_bl/dtau = Ahat * exp(bhat * lam_bl) - Chat          lam_bl(0) = lam0
    dlam_ol/dtau = PBR_eff * (Ahat*exp(bhat*lam_bl) - Chat) + Cxhat

    Ahat = A_bl * t_c / L_c,   bhat = b_3 * L_c,   Chat = C_bl * t_c / L_c,
    Cxhat = C_x * t_c / L_c,   lam0 = L0 / L_c

For the expected regime (L_bl: 37 -> 134 nm over 72 -> 480 h with near-parabolic shape) the
groups are O(0.1–10): e.g. a pure-logarithmic fit gives bhat ≈ -2 ... -4, Ahat ≈ O(10),
Chat ≈ 0. All learnable parameters are O(1) — the rpdm_nsem lesson (a badly scaled parameter is
an unlearnable parameter) applied from the start.

Residuals are enforced in Crank–Nicolson integral form on the GLL time grid (conditioning of the
spectral first-derivative matrix is the dominant error floor otherwise):

    R_j = lam[j] - lam[j-1] - (dt_j/2) * ( f(lam[j-1]) + f(lam[j]) ),   j = 1..Nt-1

plus IC residuals lam_bl(0) - lam0 and lam_ol(0).

---

## 8. Verification targets

1. **Closed-form parity**: Radau integration of §6 must match L24/L39 and L42/L45 to >= 10
   significant digits for generic (A_bl, b_3, C_bl, PBR_eff, C_x, L0).
2. **Li Table 5 regression**: with HCM12A (L0 = 4.98e-5 cm, b_3 = -6.08e2 1/cm,
   A_bl = 3.15e-10 cm/s, C_bl = 3.66e-12 cm/s, A_ol = 3.66e-10, C_ol = 4.25e-12,
   D_ol = -4.18e-2) and 316L values, reproduce the published thickness-vs-time behavior.
3. **Veile power-law refit**: least squares of L = k*t^n on the digitized data must recover
   Cr (k = 6.521 nm, n = 0.4964) and Fe (k = 64.51 nm, n = 0.2209) within digitization error.
4. **Ni consistency**: fitted model implies no Ni-layer dynamics; data SD band (31–43 nm,
   ~constant) must contain the assumed-constant line.

## 9. Known errata in the sources (documented, not repeated here)

- Li 2020 Eq. (23): prints k_9 where k_10 is required (consistency with Eqs. 16/17/44).
- Veile 2024: exposure times are 72/168/480 h ("78 h"/"178 h" in Results are typos);
  Fig. 9 legend swaps the n values between Fe and Cr — body text is authoritative
  (Fe: n = 0.2209; Cr: n = 0.4964).
