"""Check the Corrosion Science highlights against the publisher's limits.

Elsevier asks for 3 to 5 highlights, each at most 85 characters including
spaces.  This reads the bullets out of ``highlights.tex``, expands the LaTeX
that occupies fewer columns on the page than in the source, and reports any
bullet that would not fit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HIGHLIGHTS = Path(__file__).resolve().parents[1] / "paper-corrosion-science" / "highlights.tex"

MAX_CHARS = 85
MIN_BULLETS, MAX_BULLETS = 3, 5


def rendered_length(bullet: str) -> int:
    """Length of a bullet as the reader sees it, not as LaTeX spells it."""
    text = bullet
    text = text.replace("---", "\u2014").replace("--", "\u2013")
    text = text.replace("~", " ")
    text = re.sub(r"\\[a-zA-Z]+\s*", "", text)  # \% , \emph and friends
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    return len(" ".join(text.split()))


def main() -> int:
    source = HIGHLIGHTS.read_text()
    bullets = [b.strip() for b in re.findall(r"\\item\s+(.*?)(?=\n\\item|\n\\end\{itemize\})", source, re.S)]

    problems = []
    if not MIN_BULLETS <= len(bullets) <= MAX_BULLETS:
        problems.append(f"{len(bullets)} highlights; Corrosion Science wants {MIN_BULLETS} to {MAX_BULLETS}")

    for i, bullet in enumerate(bullets, start=1):
        n = rendered_length(bullet)
        flag = "  <-- too long" if n > MAX_CHARS else ""
        print(f"  {i}. {n:>3} chars{flag}")
        if n > MAX_CHARS:
            problems.append(f"highlight {i} is {n} characters, limit is {MAX_CHARS}")

    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1
    print(f"OK: {len(bullets)} highlights, all within {MAX_CHARS} characters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
