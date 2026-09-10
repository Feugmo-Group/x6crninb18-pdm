# SPDX-FileCopyrightText: Copyright (c) 2026 Conrard Giresse Tetsassi Feugmo
# SPDX-License-Identifier: Apache-2.0

"""The classical tier is a promise; this checks it is kept.

docs/REPRODUCE.md labels every command `classical` or `neural`, derived from a
module-level import graph rather than a hand-kept list. The derivation can still
be wrong -- a lazily-imported dependency that turns out not to be lazy, a new
module that pulls torch in through a helper -- and the failure would be silent:
the doc keeps saying `classical` while a reader on `uv sync` hits an
ImportError.

These tests are the ground truth behind that label. They only mean anything
where torch is genuinely absent, so they skip on a neural install and run in
CI's classical job, which is the environment the claim is about.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from htw_pdm._optional import have
from htw_pdm.paths import ROOT

pytestmark = pytest.mark.skipif(
    have("torch"),
    reason="only meaningful without the nsem extra; run under a plain `uv sync`",
)

NEURAL_ROOTS = {"torch", "physicsnemo", "hydra", "omegaconf"}
HTW_DIR = ROOT / "src" / "htw_pdm"
SCRIPTS_DIR = ROOT / "scripts"
_HTW = {p.stem: p for p in HTW_DIR.glob("*.py")}


def _top_level_imports(path: pathlib.Path) -> set[str]:
    """Dotted names PATH imports at module scope. Mirrors make_reproduce_doc."""
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _needs_neural(path: pathlib.Path, seen: frozenset[str] = frozenset()) -> bool:
    imports = _top_level_imports(path)
    if any(n.split(".")[0] in NEURAL_ROOTS for n in imports):
        return True
    for name in imports:
        if not name.startswith("htw_pdm."):
            continue
        stem = name.split(".")[1]
        if stem in seen or stem not in _HTW:
            continue
        if _needs_neural(_HTW[stem], seen | {stem}):
            return True
    return False


CLASSICAL_MODULES = sorted(
    stem for stem, p in _HTW.items()
    if not stem.startswith("_") and not _needs_neural(p)
)


@pytest.mark.parametrize("name", CLASSICAL_MODULES)
def test_classical_modules_import_without_torch(name):
    """Anything the doc calls classical must import on a NumPy/SciPy install."""
    importlib.import_module(f"htw_pdm.{name}")


def test_the_figure_scripts_are_all_classical():
    """Every manuscript figure must be drawable without the neural extra.

    This is the specific promise REPRODUCE.md makes: the two exhibits that
    report neural results read committed artefacts, so a classical clone
    rebuilds every float. If a figure script ever grows a module-level torch
    import, that promise quietly stops being true.
    """
    figure_scripts = sorted(
        p for p in SCRIPTS_DIR.glob("*.py")
        if p.stem.startswith(("plot_", "make_paper_fig")) or p.stem == "make_ni_curve"
    )
    assert figure_scripts, "no figure scripts found — glob is wrong"
    neural = [p.name for p in figure_scripts if _needs_neural(p)]
    assert not neural, f"figure scripts that now need the nsem extra: {neural}"


def test_the_artefact_producers_that_do_need_it_are_the_expected_ones():
    """Pins the neural tier so it cannot grow silently.

    A new module needing torch is fine; it needing torch *unnoticed* is not,
    because REPRODUCE.md's command list and the install instructions in the
    README are both written against this set.
    """
    expected = {
        "inverse_trainer", "parametric_trainer", "tier2_steady_trainer",
        "tier2_transient_trainer", "tier4_ni_closure", "tier4_pnp",
        "tier4_pnp_solve", "tier4_transient_inverse", "trainer",
        "uncertainty_ensemble",
    }
    actual = {stem for stem, p in _HTW.items()
              if not stem.startswith("_") and _needs_neural(p)}
    assert actual == expected
