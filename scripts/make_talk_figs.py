"""Refresh talk/figures/ from paper/figures/.

The deck keeps REAL copies of its figures rather than reaching into
paper/figures/, so talk/ can be handed to someone, opened on another
machine, or zipped for a host without dragging the manuscript along.

That duplication is only safe if something refreshes it.  This script is
that something: it reads the \\includegraphics names out of slides.tex, so
the set copied is exactly the set the deck asks for, and it fails loudly
if a requested figure is missing rather than leaving a stale copy in place.
A silent no-op here is precisely the bug that froze outputs/paper/ for a
month, so there is no `if src.exists()` guard anywhere below.

Run after regenerating the paper figures:

    python scripts/make_talk_figs.py
"""

import re
import shutil
import sys

from htw_pdm.paths import PAPER, ROOT

TALK = ROOT / "talk"
SRC = PAPER / "figures"
DST = TALK / "figures"


def wanted():
    """Figure filenames the deck actually includes, in first-use order."""
    tex = (TALK / "slides.tex").read_text()
    names = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    seen = {}
    for n in names:
        seen.setdefault(n, None)
    return list(seen)


def main():
    names = wanted()
    if not names:
        sys.exit("no \\includegraphics found in slides.tex; has the deck moved?")

    missing = [n for n in names if not (SRC / n).exists()]
    if missing:
        sys.exit(
            "missing source figures in paper/figures/: "
            + ", ".join(missing)
            + "\nRegenerate them first; see docs/REPRODUCE.md."
        )

    DST.mkdir(parents=True, exist_ok=True)

    # Drop copies the deck no longer uses, so the folder cannot accumulate
    # figures that were quietly dropped from a slide.
    for old in DST.glob("*.png"):
        if old.name not in names:
            old.unlink()
            print(f"removed unused: talk/figures/{old.name}")

    for n in names:
        shutil.copy(SRC / n, DST / n)
        print(f"copied: paper/figures/{n} -> talk/figures/{n}")

    print(f"\n{len(names)} figures refreshed in talk/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
