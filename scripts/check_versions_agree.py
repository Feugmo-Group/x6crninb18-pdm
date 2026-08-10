"""Assert the two venue manuscripts carry identical science.

paper-corrosion-science/main-cas.tex is generated from paper/main-snjnl.tex by
reordering sections and swapping the front matter, so every number in the body
must survive the transformation unchanged.  This compares the multiset of
numeric tokens in the two bodies and fails loudly if any differ.

Run:  python scripts/check_versions_agree.py

Only the *body* is compared.  The preamble and the closing bibliography block
legitimately differ -- different document class, different .bst, different
front-matter macros -- and several of those names contain digits
(cas-model2-names, sn-nature).  Comparing whole files reports those as science
differences, which is a false alarm that trains you to ignore the check.
"""

import re
import sys
from collections import Counter

from htw_pdm.paths import PAPER, ROOT

NPJ = PAPER / "main-snjnl.tex"
CAS = ROOT / "paper-corrosion-science" / "main-cas.tex"


def body(path):
    """Text from the Introduction up to the bibliography, i.e. the science."""
    t = path.read_text()
    t = t.split("\\section{Introduction}", 1)[1]
    for marker in ("\\bibliographystyle{", "\\bibliography{"):
        t = t.split(marker, 1)[0]
    return t


def numbers(path):
    return Counter(re.findall(r"\d+\.?\d*", body(path)))


def main():
    for p in (NPJ, CAS):
        if not p.exists():
            sys.exit(f"missing {p}; run scripts/make_corrosion_science_version.py")

    a, b = numbers(NPJ), numbers(CAS)
    only_npj, only_cas = a - b, b - a
    if not only_npj and not only_cas:
        print(f"OK: {sum(a.values())} numeric tokens identical in both versions")
        return 0

    print("MISMATCH -- the two versions disagree on the science:")
    if only_npj:
        print(f"  in {NPJ.name} only: {dict(only_npj)}")
    if only_cas:
        print(f"  in {CAS.name} only: {dict(only_cas)}")
    print("\nRegenerate with scripts/make_corrosion_science_version.py; if the "
          "mismatch survives, the generator's line ranges are stale.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
