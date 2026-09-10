"""Stoichiometric bound on PBR_eff from a chromium/iron mass balance.

PBR_eff is defined by the outer-layer equation of the reduced model,

    dL_ol/dt = PBR_eff * dL_bl/dt + C_x,

so it is the ratio at which barrier-layer growth *produces* outer-layer oxide.
This module computes that ratio from stoichiometry and molar volumes, which is
The only quantity the fitted PBR_eff may legitimately be compared against.

A caution that motivated writing this down.  `composition_map.PBR_BL = 2.05`
is the barrier layer's *classical* Pilling-Bedworth ratio -- oxide volume per
unit volume of metal consumed -- and for an FeCr2O4 barrier on this alloy it
happens to come out at 2.05, within a few percent of the production ratio
computed here (2.05).  The two are conceptually unrelated and their agreement is
a coincidence of this particular phase pair.  Both are reported by
`summary()` so the coincidence is visible rather than load-bearing.

Derivation (per unit area, barrier phase with n_Cr, n_Fe, n_Ni cations per
formula unit and molar volume V_bl; outer layer Fe3O4 with molar volume V_ol):

  growth of the barrier by dL_bl forms   dn = dL_bl / V_bl   moles of barrier,
  consuming n_Cr*dn mol Cr and n_Fe*dn mol Fe.

  The premise of the duplex structure is that chromium is retained in the
  barrier, so all chromium dissolved at the metal interface reports there.
  Supplying n_Cr*dn mol Cr therefore dissolves n_Cr*dn/x_Cr mol of alloy, which
  liberates (x_Fe/x_Cr)*n_Cr*dn mol Fe.  Of that, n_Fe*dn is taken back up by
  The barrier and the remainder is ejected:

      n_Fe,excess = dn * (n_Cr * x_Fe/x_Cr - n_Fe)

  Precipitating as Fe3O4 gives dL_ol = (V_ol/3) * n_Fe,excess, hence

      R_prod = dL_ol/dL_bl = (V_ol / (3 V_bl)) * (n_Cr * x_Fe/x_Cr - n_Fe)

Molar volumes are derived from lattice parameters rather than tabulated bulk
densities, because natural-mineral densities carry impurity corrections that do
not apply to a thin reaction-grown film.

Run:  python -m htw_pdm.mass_balance
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_A = 6.02214076e23  # 1/mol
ANG3_PER_CM3 = 1e24

# Atomic masses (g/mol)
M = {"Fe": 55.845, "Cr": 51.996, "Ni": 58.693, "O": 15.999,
     "Mn": 54.938, "Si": 28.086, "Nb": 92.906}


def molar_volume_from_cell(a_ang: float, z: int, c_ang: float | None = None,
                           hexagonal: bool = False) -> float:
    """Molar volume (cm^3/mol of formula units) from lattice parameters.

    Cubic unless `hexagonal`, in which case the hexagonal cell volume
    (sqrt(3)/2) a^2 c is used.  `z` is formula units per unit cell.
    """
    if hexagonal:
        if c_ang is None:
            raise ValueError("hexagonal cell needs c_ang")
        v_cell = (np.sqrt(3.0) / 2.0) * a_ang**2 * c_ang
    else:
        v_cell = a_ang**3
    return (v_cell / z) * N_A / ANG3_PER_CM3


@dataclass(frozen=True)
class Phase:
    """An oxide phase: cations per formula unit and its molar volume."""

    name: str
    n_Fe: float
    n_Cr: float
    n_Ni: float
    V: float  # cm^3 per mol formula units

    @property
    def n_cation(self) -> float:
        return self.n_Fe + self.n_Cr + self.n_Ni


# Spinels are cubic Fd-3m with Z = 8; eskolaite is hexagonal R-3c with Z = 6.
MAGNETITE = Phase("Fe3O4", 3.0, 0.0, 0.0, molar_volume_from_cell(8.396, 8))

BARRIER_CANDIDATES = (
    # The four chromium-rich phases the manuscript names as consistent with the
    # reported X-ray photoelectron spectroscopy.
    Phase("FeCr2O4", 1.0, 2.0, 0.0, molar_volume_from_cell(8.377, 8)),
    Phase("Cr2O3", 0.0, 2.0, 0.0, molar_volume_from_cell(4.9587, 6, c_ang=13.599,
                                                         hexagonal=True)),
    Phase("Fe2CrO4", 2.0, 1.0, 0.0, molar_volume_from_cell(8.390, 8)),
    Phase("NiCr2O4", 0.0, 2.0, 1.0, molar_volume_from_cell(8.248, 8)),
)

# Nominal X6CrNiNb18-10 (1.4550 / AISI 347) composition, weight percent.
NOMINAL_WT = {"Cr": 18.0, "Ni": 10.5, "Mn": 1.5, "Si": 0.5, "Nb": 0.7}


def atom_fractions(wt: dict[str, float] | None = None) -> dict[str, float]:
    """Alloy atom fractions from weight percent, iron by difference."""
    wt = dict(NOMINAL_WT if wt is None else wt)
    wt.setdefault("Fe", 100.0 - sum(wt.values()))
    moles = {el: w / M[el] for el, w in wt.items()}
    total = sum(moles.values())
    return {el: m / total for el, m in moles.items()}


def production_ratio(barrier: Phase, x: dict[str, float] | None = None,
                     outer: Phase = MAGNETITE, r_Ni: float = 0.0) -> float:
    """R_prod = dL_ol/dL_bl from stoichiometry, for one barrier phase.

    `r_Ni` is the fraction of liberated nickel that reports to the oxide rather
    than remaining in the metallic enrichment zone (the spatial model fits 0.295).  Nickel
    routed to the oxide precipitates in the outer layer and adds volume there;
    nickel consumed by the barrier itself is debited first.  Setting r_Ni = 0
    recovers the iron-only balance.

    Returns 0.0 if the barrier phase is iron-richer than the alloy can feed,
    which means that phase cannot grow by this mechanism at all.
    """
    x = atom_fractions() if x is None else x
    fe_per_cr = x["Fe"] / x["Cr"]
    ni_per_cr = x["Ni"] / x["Cr"]

    # Per mole of barrier formed.
    fe_liberated = barrier.n_Cr * fe_per_cr
    fe_excess = fe_liberated - barrier.n_Fe
    if fe_excess <= 0.0:
        return 0.0

    ni_liberated = barrier.n_Cr * ni_per_cr
    ni_to_oxide = max(r_Ni * ni_liberated - barrier.n_Ni, 0.0)

    # Outer layer: Fe3O4 plus any nickel substituting on its cation sublattice,
    # i.e. cations share the same sites, so volume scales with cation count.
    cations_out = fe_excess + ni_to_oxide
    return (outer.V / outer.n_cation) * cations_out / barrier.V


def alloy_volume_per_mol_atoms(x: dict[str, float] | None = None,
                               rho_alloy: float = 7.95) -> float:
    """cm^3 per mole of alloy atoms, from the mean atomic mass and density."""
    x = atom_fractions() if x is None else x
    return sum(x[el] * M[el] for el in x) / rho_alloy


def barrier_pbr(barrier: Phase, x: dict[str, float] | None = None,
                rho_alloy: float = 7.95) -> float:
    """Classical Pilling-Bedworth ratio of the barrier: V_oxide / V_metal.

    The standard definition, counting only the metal atoms that end up *in* the
    oxide (its cations).  For FeCr2O4 this reproduces the 2.05 hardcoded in
    `composition_map.PBR_BL`.

    Included only so it can be told apart from `production_ratio`; it is NOT the
    quantity the fitted PBR_eff should be compared against.  The distinction is
    easy to lose because a duplex film has a second, different "metal consumed":
    The alloy dissolved to *supply* the barrier's chromium, most of whose iron
    never joins the barrier at all (see `metal_recession_ratio`).
    """
    v_metal = alloy_volume_per_mol_atoms(x, rho_alloy)
    return barrier.V / (barrier.n_cation * v_metal)


def metal_recession_ratio(barrier: Phase, x: dict[str, float] | None = None,
                          rho_alloy: float = 7.95) -> float:
    """Barrier thickness grown per unit thickness of alloy actually dissolved.

    Differs from `barrier_pbr` whenever the barrier rejects a cation: supplying
    n_Cr chromium dissolves n_Cr/x_Cr alloy atoms, not n_cation of them.  This is
    The ratio that converts a measured barrier thickness into metal loss.
    """
    x = atom_fractions() if x is None else x
    v_metal = alloy_volume_per_mol_atoms(x, rho_alloy)
    atoms_dissolved = barrier.n_Cr / x["Cr"]
    return barrier.V / (atoms_dissolved * v_metal)


def release_flux(C_x_nm_per_h: float, outer: Phase = MAGNETITE) -> dict[str, float]:
    """Convert an outer-layer loss rate (nm/h, C_x < 0) to a release flux.

    Returns the oxide and iron fluxes in ug/(dm^2 h), the units coolant
    corrosion-product measurements are normally reported in.
    """
    loss_nm_per_h = abs(C_x_nm_per_h)
    m_formula = outer.n_Fe * M["Fe"] + (4.0 * M["O"] if outer is MAGNETITE else 0.0)
    rho = m_formula / outer.V  # g/cm^3
    # nm/h -> cm/h -> g/(cm^2 h) -> ug/(dm^2 h)   [1 dm^2 = 100 cm^2]
    g_per_cm2_h = loss_nm_per_h * 1e-7 * rho
    oxide = g_per_cm2_h * 1e6 * 100.0
    return {"oxide_ug_dm2_h": oxide,
            "Fe_ug_dm2_h": oxide * outer.n_Fe * M["Fe"] / m_formula}


def summary(r_Ni: float = 0.0) -> dict:
    """Stoichiometric bound as a range over the candidate barrier phases."""
    x = atom_fractions()
    rows = []
    for ph in BARRIER_CANDIDATES:
        rows.append({
            "phase": ph.name,
            "V_cm3_mol": ph.V,
            "R_prod": production_ratio(ph, x, r_Ni=r_Ni),
            "PBR_barrier": barrier_pbr(ph, x),
            "recession_ratio": metal_recession_ratio(ph, x),
        })
    feasible = [r["R_prod"] for r in rows if r["R_prod"] > 0.0]
    return {"atom_fractions_pct": {el: 100.0 * v for el, v in x.items()},
            "r_Ni": r_Ni,
            "phases": rows,
            "R_prod_range": (min(feasible), max(feasible))}


def main():
    x = atom_fractions()
    print("Alloy atom fractions (%):")
    for el in sorted(x, key=lambda e: -x[e]):
        print(f"  {el:3s} {100 * x[el]:6.2f}")
    print(f"  Fe/Cr = {x['Fe'] / x['Cr']:.3f}   Ni/Cr = {x['Ni'] / x['Cr']:.3f}")
    print(f"\nOuter layer {MAGNETITE.name}: V = {MAGNETITE.V:.2f} cm3/mol")

    for r_ni in (0.0, 0.2954):
        s = summary(r_Ni=r_ni)
        print(f"\n=== r_Ni = {r_ni:.4f} "
              f"({'iron-only balance' if r_ni == 0 else 'the spatial model fitted Ni routing'})")
        print(f"{'barrier':10s} {'V (cm3/mol)':>12s} {'R_prod':>9s} "
              f"{'PBR_barrier':>12s} {'recession':>10s}")
        for row in s["phases"]:
            rp = f"{row['R_prod']:.3f}" if row["R_prod"] > 0 else "infeasible"
            print(f"{row['phase']:10s} {row['V_cm3_mol']:12.2f} {rp:>9s} "
                  f"{row['PBR_barrier']:12.3f} {row['recession_ratio']:10.3f}")
        lo, hi = s["R_prod_range"]
        print(f"  stoichiometric range for PBR_eff: [{lo:.2f}, {hi:.2f}]")

    print("\nImplied release at the C_x that buys PBR_eff = 2.05 "
          "(see scripts/analyse_pbr_closure.py):")
    for cx in (-0.0249, -0.2493, -0.9409):
        f = release_flux(cx)
        print(f"  C_x = {cx:7.4f} nm/h -> {f['oxide_ug_dm2_h']:8.2f} ug oxide/(dm2 h), "
              f"{f['Fe_ug_dm2_h']:8.2f} ug Fe/(dm2 h)")


if __name__ == "__main__":
    main()
