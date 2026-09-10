"""Phase E2 — (T, [O2]) -> apparent HTW_PDM parameters for the sensitivity study.

Maps an operating condition (temperature T, dissolved oxygen [O2]) to the apparent
parameters {A_bl, b3, C_bl} of the reduced HTW_PDM, anchored at the Phase-B3 M4 fit
(T0 = 240 C, [O2]0 = 0.4 ppm, 7 MPa). Every channel is a documented consequence of
The Li 2020 rate-constant structure (see ../pdm_eqns.md and the extraction's Eq. 9 /
Table 1 / Table 4); the ONE quantity that cannot be sourced from the single-temperature
Veile data is the standard reaction Gibbs energy dG0_R of the growth reaction R3 —
it is therefore an explicit scan parameter, never a hidden default.

Channels:

1. A_bl(T)  — Arrhenius factor on k3^00 (Li Eq. 9):
       k3^00(T) = k3^00(T0) * exp[-(alpha3*dG0_R/R) * (1/T - 1/T0)]
   plus the pH channel exp(c3(T)*pH(T)) with c3 = -alpha3*chi*gamma(T)*beta and
   pH(T) = pKw(T)/2 (neutral ultrapure water at temperature) — a ~2 % effect across
   The range, included for completeness.

2. A_bl([O2]) — optional ECP channel: at open circuit exp(a3*V) is absorbed into
   A_bl at the reference condition; changing [O2] shifts the corrosion potential.
   Modeled as the IDEAL Nernst slope of the O2 electrode, dV = (RT/4F)*ln(C_O/C_O_ref),
   entering through exp(a3*dV) with a3 = alpha3*chi*gamma*(1-alpha). This is a lower
   bound on the true mixed-potential ECP response (real BWR ECP-vs-O2 curves are much
   steeper in the 1-100 ppb transition region); disable with ecp_channel=False when a
   measured ECP(T,[O2]) map is available to substitute.

3. b3(T) — explicit 1/T dependence: b3 = -alpha3*chi*gamma(T)*eps_f with
   gamma = F/RT and eps_f (field strength) taken T-independent over 200-285 C.

4. C_bl(T,[O2]) — barrier-layer dissolution, C_bl ~ Omega*k10*C_O^q with q = 1/2
   (Li Table 4) and C_O = [O2]*rho(T)*1e-3/32 (Li Eq. 14). The M4 fit gives
   C_bl(ref) = 0 (undetermined in ultrapure water), so by default this channel is
   inactive; pass C_bl_ref > 0 to study the dissolution scenario (e.g. a value at the
   Phase-B3 95 % bound). The k10 Arrhenius factor reuses the same dG0_R scan value.

Validity domain: T in [200, 285] C — at 7 MPa water boils at 285.8 C, so the plan's
nominal 300 C endpoint is not reachable in the liquid phase at BWR pressure.

Self-check:  python -m htw_pdm.physics_parametric   (reference condition must reproduce M4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from htw_pdm.physics import Parameters

R_GAS = 8.314462  # J/(mol K)
F_CONST = 96485.332  # C/mol
CHI = 8.0 / 3.0  # spinel-averaged cation valence (barrier layer)
ALPHA3 = 0.12  # transfer coefficient, Li Table 4 (HCM12A)
ALPHA_POL = 0.78  # bl/e interface polarizability, Li Table 4
BETA_PH = -0.005  # V, d(phi_ble)/d(pH), Li Table 4

# Reference condition = the Veile/Phase-B3 anchor.
T0_K = 513.15  # 240 C
O2_REF_PPM = 0.4

# IAPWS-97 liquid water density at 7 MPa, kg/L (240 C anchored to the Phase-A value).
_RHO_T_C = np.array([200.0, 220.0, 240.0, 260.0, 285.0])
_RHO_KG_L = np.array([0.8687, 0.8434, 0.8137, 0.7854, 0.7415])

# Ionization product of water pKw(T) near saturation (IAPWS); neutral pH = pKw/2.
_PKW_T_C = np.array([200.0, 220.0, 240.0, 260.0, 285.0])
_PKW = np.array([11.29, 11.22, 11.19, 11.19, 11.24])


def rho_water(T_C: np.ndarray) -> np.ndarray:
    """Liquid water density at 7 MPa in kg/L, valid 200-285 C."""
    T_C = np.asarray(T_C, dtype=float)
    if np.any(T_C < 200.0) or np.any(T_C > 285.0):
        raise ValueError("T outside the 7 MPa liquid range [200, 285] C")
    return np.interp(T_C, _RHO_T_C, _RHO_KG_L)


def neutral_pH(T_C: np.ndarray) -> np.ndarray:
    """pH of neutral ultrapure water at temperature: pKw(T)/2."""
    return np.interp(np.asarray(T_C, dtype=float), _PKW_T_C, _PKW) / 2.0


def C_O(T_C: np.ndarray, O2_ppm: np.ndarray) -> np.ndarray:
    """Dimensionless dissolved-O2 concentration, Li Eq. 14."""
    return np.asarray(O2_ppm, dtype=float) * rho_water(T_C) * 1e-3 / 32.00


@dataclass
class ConditionScan:
    """Scan configuration: the assumed dG0_R (J/mol) and the optional channels."""

    dG0_R: float = 50e3  # J/mol — Li's representative sensitivity scale; SCAN THIS
    ecp_channel: bool = True
    C_bl_ref: float = 0.0  # nm/h at the reference condition (M4: 0)


def apparent_parameters(
    T_C: float, O2_ppm: float, ref: Parameters, scan: ConditionScan
) -> Parameters:
    """Apparent HTW_PDM parameters at (T, [O2]), anchored at `ref` = the M4 fit."""
    T = T_C + 273.15
    gamma = F_CONST / (R_GAS * T)
    gamma0 = F_CONST / (R_GAS * T0_K)

    # Channel 1: Arrhenius on k3^00 + pH drift.
    arrh = np.exp(-(ALPHA3 * scan.dG0_R / R_GAS) * (1.0 / T - 1.0 / T0_K))
    c3 = -ALPHA3 * CHI * gamma * BETA_PH  # > 0 since beta < 0
    c3_0 = -ALPHA3 * CHI * gamma0 * BETA_PH
    ph_factor = np.exp(c3 * neutral_pH(T_C) - c3_0 * neutral_pH(240.0))

    # Channel 2: ideal-Nernst ECP shift with [O2].
    ecp_factor = 1.0
    if scan.ecp_channel:
        dV = (R_GAS * T / (4.0 * F_CONST)) * np.log(
            C_O(T_C, O2_ppm) / C_O(240.0, O2_REF_PPM)
        )
        a3 = ALPHA3 * CHI * gamma * (1.0 - ALPHA_POL)
        ecp_factor = np.exp(a3 * dV)

    A_bl = ref.A_bl * arrh * ph_factor * ecp_factor

    # Channel 3: b3 ~ gamma(T) ~ 1/T (eps_f T-independent).
    b3 = ref.b3 * (T0_K / T)

    # Channel 4: dissolution C_bl ~ C_O^(1/2), same Arrhenius scale on k10.
    C_bl = scan.C_bl_ref * arrh * np.sqrt(
        C_O(T_C, O2_ppm) / C_O(240.0, O2_REF_PPM)
    )

    return Parameters(
        A_bl=float(A_bl), b3=float(b3), C_bl=float(C_bl), PBR_eff=ref.PBR_eff,
        C_x=ref.C_x, L0=ref.L0, L_ol0=ref.L_ol0,
    )


if __name__ == "__main__":
    ref = Parameters()  # M4 defaults
    scan = ConditionScan(dG0_R=50e3)
    p0 = apparent_parameters(240.0, 0.4, ref, scan)
    print("Reference-condition self-check (must reproduce M4):")
    for n in ("A_bl", "b3", "C_bl"):
        v, r = getattr(p0, n), getattr(ref, n)
        ok = abs(v - r) <= 1e-12 * max(abs(r), 1.0)
        print(f"  {n:6s}: {v:.6g} vs {r:.6g}  {'OK' if ok else 'MISMATCH'}")

    print("\nSensitivity preview (dG0_R = 50 kJ/mol):")
    for T_C in (200.0, 240.0, 285.0):
        for O2 in (0.01, 0.4, 8.0):
            p = apparent_parameters(T_C, O2, ref, scan)
            print(f"  T = {T_C:5.1f} C  [O2] = {O2:5.2f} ppm:  "
                  f"A_bl = {p.A_bl:7.4f} nm/h  b3 = {p.b3:9.6f} 1/nm")
