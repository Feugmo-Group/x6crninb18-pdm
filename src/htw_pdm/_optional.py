"""Optional heavy dependencies, imported on use rather than on import.

The classical half of this package -- the deterministic fits, the model ladder,
The profile likelihoods, the bootstrap, the mass balance and the Wagner/the nickel-zone closure
solves -- needs only NumPy, SciPy and Matplotlib, and every headline number in
The manuscript is reachable from that half alone.  The neural half needs PyTorch
and PhysicsNeMo, and PhysicsNeMo is not resolvable from a package index in the
form this work uses.

Importing those at module scope would have made the classical half unusable
without them, which is what this module exists to prevent: the dependency is
taken when a neural code path actually runs, and the failure, when it comes,
says what to install rather than raising a bare ModuleNotFoundError from three
frames down.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_TORCH_HINT = (
    "PyTorch is required for the neural (NSEM) code paths.\n"
    "  uv sync --extra nsem        # or: pip install -e '.[nsem]'\n"
    "The classical fits, the identifiability analysis and the the nickel-zone closure solves do "
    "not need it."
)

_PHYSICSNEMO_HINT = (
    "PhysicsNeMo is required for the neural (NSEM) code paths and is not\n"
    "installable from PyPI in the form this work uses. See docs/NSEM_DEPENDENCY.md\n"
    "for the exact revision and an install recipe.\n"
    "The classical fits, the identifiability analysis and the the nickel-zone closure solves do "
    "not need it."
)


def _require(name: str, hint: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - exercised by absence
        raise ImportError(f"{name} is not installed.\n\n{hint}") from exc


def require_torch() -> ModuleType:
    """Return the `torch` module, or raise with an actionable message."""
    return _require("torch", _TORCH_HINT)


def require_physicsnemo() -> ModuleType:
    """Return the `physicsnemo` module, or raise with an actionable message."""
    return _require("physicsnemo", _PHYSICSNEMO_HINT)


def have(name: str) -> bool:
    """True if `name` can be imported. Used by tests to skip rather than fail."""
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


class _LazyModule:
    """Attribute-forwarding stand-in for a module imported on first use.

    `spatial_physics` reaches for `torch` in a dozen places across six functions
    but is also the home of the classical steady/composition helpers that the
    The spatial model and the nickel-zone closure entry points need.  Binding the real module inside every
    function would have been six edits and six chances to miss one; this
    forwards attribute access instead, so the import happens on the first
    `torch.<something>` a neural code path evaluates and not before.

    Annotations are unaffected: every module using this also does
    `from __future__ import annotations`, so `torch.Tensor` in a signature is
    a string and is never evaluated.
    """

    __slots__ = ("_name", "_hint", "_mod")

    def __init__(self, name: str, hint: str) -> None:
        self._name = name
        self._hint = hint
        self._mod: ModuleType | None = None

    def __getattr__(self, attr: str):
        if self._mod is None:
            object.__setattr__(self, "_mod", _require(self._name, self._hint))
        return getattr(self._mod, attr)


def lazy_torch() -> _LazyModule:
    """A `torch` stand-in that imports on first attribute access."""
    return _LazyModule("torch", _TORCH_HINT)
