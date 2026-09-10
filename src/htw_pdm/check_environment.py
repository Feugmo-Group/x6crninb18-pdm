"""Report which half of the pipeline this environment can run.

    python -m htw_pdm.check_environment

The package splits cleanly in two: a classical half that needs only NumPy,
SciPy and Matplotlib and carries every headline claim in the manuscript, and a
neural half that needs PyTorch and a pinned PhysicsNeMo revision. Rather than
let someone discover which half they have by hitting an ImportError twenty
minutes into a run, this prints it up front.
"""

from __future__ import annotations

import importlib.metadata as md
import sys

from htw_pdm._optional import have

CLASSICAL = ("numpy", "scipy", "matplotlib")
NEURAL = ("torch", "hydra", "omegaconf", "physicsnemo")

CLASSICAL_ENTRY_POINTS = (
    "htw_pdm.baseline_fit",
    "htw_pdm.mass_balance",
    "htw_pdm.pbr_closure",
    "htw_pdm.spatial_identifiability",
    "htw_pdm.composition_map",
    "htw_pdm.nickel_identifiability",
    "htw_pdm.nickel_forward_envelope",
)
NEURAL_ENTRY_POINTS = (
    "htw_pdm.trainer",
    "htw_pdm.inverse_trainer",
    "htw_pdm.uncertainty_ensemble",
    "htw_pdm.parametric_trainer",
    "htw_pdm.field_closure_solve",
    "htw_pdm.transient_inverse",
    "htw_pdm.nickel_mobility_closure",
)


# Import name -> distribution name, where they differ.
DIST_NAME = {"physicsnemo": "nvidia-physicsnemo", "hydra": "hydra-core"}


def _version(name: str) -> str:
    try:
        return md.version(DIST_NAME.get(name, name))
    except md.PackageNotFoundError:
        return "?"


def _report(title: str, mods, entry_points) -> bool:
    ok = all(have(m) for m in mods)
    print(f"\n{title}: {'available' if ok else 'NOT available'}")
    for m in mods:
        present = have(m)
        mark = "ok     " if present else "MISSING"
        print(f"  [{mark}] {m:<14s} {_version(m) if present else ''}")
    print(f"  {len(entry_points)} entry points:")
    for ep in entry_points:
        print(f"    python -m {ep}")
    return ok


def main() -> int:
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    classical = _report("Classical path", CLASSICAL, CLASSICAL_ENTRY_POINTS)
    neural = _report("Neural (NSEM) path", NEURAL, NEURAL_ENTRY_POINTS)

    print()
    if classical and neural:
        print("Everything is installed; the full pipeline in docs/REPRODUCE.md will run.")
    elif classical:
        print("Every headline number in the manuscript is reachable from this")
        print("environment. To add the neural half:  uv sync --extra nsem")
        print("(see docs/NSEM_DEPENDENCY.md for what that installs and why).")
    else:
        print("Core dependencies are missing. Install with:  uv sync")
    return 0 if classical else 1


if __name__ == "__main__":
    raise SystemExit(main())
