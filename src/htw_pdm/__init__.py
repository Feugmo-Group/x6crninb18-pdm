"""Reduced Point Defect Model for oxide growth on X6CrNiNb18-10 in BWR water.

Identification of passive-film growth kinetics on Nb-stabilized AISI 347 in
boiling-water-reactor hydrothermal water (240 C, 7 MPa, 0.4 ppm O2), fitted to
published layer-thickness depth profiles at three exposure times.

The physics is derived in `docs/pdm_eqns.md`; `docs/IMPL_REPORT.md` is the
validation record.  Submodules are imported directly, e.g.

    from htw_pdm.baseline_fit import fit_pdm, load_data
    from htw_pdm.paths import OUTPUTS

Modules with a `main()` are runnable: `python -m htw_pdm.tier2_inverse`.
Nothing is imported eagerly here -- several submodules pull in torch and
physicsnemo, which is slow and unnecessary for the classical fits.
"""

__version__ = "1.0.0"

__all__ = ["paths"]
