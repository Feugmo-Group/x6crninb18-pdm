"""Project paths, resolved once.

Every module and script imports its directories from here.  This exists because
The same directory used to be computed three different ways -- `__file__.parent`
in root scripts, `__file__.parent.parent` inside the package, and a separate
`ROOT / "outputs" / "paper"` elsewhere -- so moving a file silently changed
where its results landed.  It is the same rule already applied to fitted values
(see README): a thing that is derived once is read from one place, never copied.

`ROOT` is found by walking up from this file until a directory containing
`pyproject.toml` appears, so an editable install, a source checkout, and a
`python -m htw_pdm.<module>` run all agree.  `HTW_PDM_ROOT` overrides it, which
is what the tests use to keep generated files out of the working tree.
"""

from __future__ import annotations

import os
from pathlib import Path

_MARKER = "pyproject.toml"


def _find_root() -> Path:
    override = os.environ.get("HTW_PDM_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _MARKER).is_file():
            return parent
    # Installed as a wheel with no project marker alongside: fall back to the
    # current working directory so a user running from a data checkout still
    # gets sensible locations rather than a path inside site-packages.
    return Path.cwd().resolve()


ROOT = _find_root()
DATA = ROOT / "data"
CONF = ROOT / "conf"
OUTPUTS = ROOT / "outputs"
PAPER_OUT = OUTPUTS / "paper"
PAPER = ROOT / "paper"


def ensure_outputs() -> Path:
    """Create the outputs tree on demand and return it."""
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    return OUTPUTS


__all__ = ["ROOT", "DATA", "CONF", "OUTPUTS", "PAPER_OUT", "PAPER", "ensure_outputs"]
