"""Generate the Corrosion Science version from the npj manuscript.

The two submissions carry identical science; they differ in section order
(Elsevier puts Methods second, Nature Portfolio puts it at the back), in
document class (cas-sc vs sn-jnl), and in the venue front matter.  Rather
than maintain two copies of 1600 lines, this script derives one from the
other, so a correction to the science can only be made in one place.

Edit paper/main-snjnl.tex, then run:

    python scripts/make_corrosion_science_version.py

and rebuild.  Verify with a numeric-token diff of the two main files; they
must agree exactly.
"""

import re
from pathlib import Path

from htw_pdm.paths import PAPER as P, ROOT
src = (P / "main-snjnl.tex").read_text().split("\n")
# Blocks are taken by line range and each is asserted to begin with its own
# \\section below; if main-snjnl.tex is restructured these numbers must move,
# and the assertions will say so rather than silently mis-slicing.
L = lambda a, b: "\n".join(src[a-1:b])   # 1-indexed inclusive

intro       = L(44, 162)
results     = L(163, 1042)
discussion  = L(1043, 1359)
conclusions = L(1360, 1435)
methods     = L(1436, 1579)
backmatter  = L(1580, 1608)

# sanity: each block must start with its own \section
for name, blk in [("intro", intro), ("results", results), ("disc", discussion),
                  ("conc", conclusions), ("meth", methods)]:
    assert blk.lstrip().startswith("\\section{"), (name, blk[:60])
assert backmatter.lstrip().startswith("\\backmatter"), backmatter[:60]

# --- Introduction: the roadmap sentence describes the npj section order ----
# ("results first ... a discussion and a methods section follow"), which is
# false once Methods moves to section 2.  Rewrite it for the Elsevier order.
NPJ_ROADMAP = """The remainder of the paper presents results first: the reduced model and
its differentiable identification, the identifiability and uncertainty
analysis, the physics-informed inversion failure mode and its remedy, the
spatial extension and zero-parameter composition predictions, the nickel
enrichment zone, and the long-time predictions and operating-envelope
sensitivity. A discussion and a methods section follow."""
CAS_ROADMAP = """The remainder of the paper sets out the model and the inversion
methodology, and then the results: the reduced model and its
differentiable identification, the identifiability and uncertainty
analysis, the physics-informed inversion failure mode and its remedy, the
spatial extension and zero-parameter composition predictions, the nickel
enrichment zone, and the long-time predictions and operating-envelope
sensitivity. A discussion and conclusions follow."""
assert NPJ_ROADMAP in intro, "roadmap sentence moved; update this replacement"
intro = intro.replace(NPJ_ROADMAP, CAS_ROADMAP, 1)

# --- Methods: retitle for its new position as section 2 -------------------
methods = methods.replace(
    "\\section{Methods}\\label{sec:methods}",
    "\\section{Model and methods}\\label{sec:methods}", 1)

# --- backmatter: sn-jnl \bmhead -> Elsevier starred sections --------------
back = (backmatter
        .replace("\\backmatter\n", "")
        .replace("\\bmhead{Data and code availability}",
                 "\\section*{Data availability}")
        .replace("\\bmhead{Author contributions}",
                 "\\section*{CRediT authorship contribution statement}")
        .replace("\\bmhead{Competing interests}",
                 "\\section*{Declaration of competing interest}")
        .replace("\\bibliography{references}", "")
        .replace("\\end{document}", "")
        .strip())
# Elsevier's competing-interest wording is prescribed
back = back.replace(
    "The author declares no competing interests.",
    "The author declares that they have no known competing financial interests\n"
    "or personal relationships that could have appeared to influence the work\n"
    "reported in this paper.")

PREAMBLE = r"""%% Corrosion Science (Elsevier) -- CAS single-column template (cas-sc)
%% Companion version of the npj Materials Degradation submission
%% (paper/main-snjnl.tex).  Same science, same numbers; restructured to the
%% Elsevier order (Methods as section 2 rather than at the back) and
%% reformatted from sn-jnl to cas-sc.
%%
%% cas-sc is the single-column member of Elsevier's CAS bundle.  Being one
%% column, the figures keep the \textwidth sizing they have in the npj version
%% and no float needs promoting to a starred environment.
\documentclass[a4paper,fleqn]{cas-sc}

%% Corrosion Science numbers its references.  The CAS bundle ships only an
%% author-year .bst (cas-model2-names), so the numbered style comes from natbib.
\usepackage[numbers,sort&compress]{natbib}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}

%% Long unbreakable tokens (X6CrNiNb18-10) overrun the measure; let TeX stretch
%% a little rather than leave an overfull line.
\emergencystretch=1em

\graphicspath{{figures/}}

\begin{document}

\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

\shorttitle{Identifiability of a point defect model for AISI~347}
\shortauthors{C. G. Tetsassi Feugmo}

\title[mode=title]{Identifiability of a point defect model for oxide growth on AISI~347}

\author[1]{Conrard Giresse Tetsassi Feugmo}[orcid=0000-0002-8992-4335]
\cormark[1]
\ead{giresse.feugmo@gmail.com}

\address[1]{Department of Chemistry and Department of Physics \& Astronomy,
            University of Waterloo, 200 University Avenue West,
            Waterloo, ON N2L 3G1, Canada}

\cortext[1]{Corresponding author.}

\begin{abstract}
Mechanistic corrosion models are routinely fitted with four to six
parameters to a handful of measurements and then extrapolated across
service lifetimes, yet whether the data determine those parameters is
almost never tested. We identify a reduced point defect model for the
duplex oxide grown on Nb-stabilized AISI~347 (X6CrNiNb18-10) in
simulated boiling-water-reactor hydrothermal water
(240~$^\circ$C, 7~MPa, 0.4~ppm dissolved O$_2$) from published
depth-profile data at three exposure times, by differentiable
physics-informed inversion validated against an exact closed-form
reference. Profile likelihood, cross-checked against a 1000-member
bootstrap and a prior-relaxation test, leaves one of the five fitted
parameters undetermined and a second only weakly determined, while a
residual decomposition puts 95\% of the fit statistic on three chromium
points and 80\% on one. The kinetics fit the calibration data as well as
an empirical power law but extrapolate very differently: at ten years
the power law predicts a 3.5 to 6.3 times thicker barrier layer than the
mechanistic 535~nm (bootstrap 95\% interval 447--704~nm), a prediction
varying 1.19 to 1.56-fold across the scanned operating envelope. A
designed exposure beyond roughly 3000~hours would discriminate the two.
We further document and remedy a failure mode specific to sparse-data
inversion.
\end{abstract}

%% Corrosion Science classifies keywords: A. materials, B. techniques
%% and methods, C. processes and phenomena.
\begin{keywords}
A. Stainless steel \sep
B. Modelling studies \sep
C. Oxidation \sep
C. Passive films \sep
C. High temperature corrosion
\end{keywords}

\maketitle
"""

TAIL = r"""
\bibliographystyle{unsrtnat}
\bibliography{references}

\end{document}
"""

# --- table sizing ----------------------------------------------------------
# cas-sc sets one column on a measure narrower than the npj version's, so the
# tables are stepped down once.  \small is enough for all of them here -- it
# was not under elsarticle's 12pt preprint, where four needed two or three
# steps -- so the size is applied uniformly rather than per-table.  This is a
# property of the class, not of the science, which is why it lives here and
# not in main-snjnl.tex.
def size_tables(t):
    out=[]; buf=None
    for line in t.split("\n"):
        if re.match(r"\s*\\begin\{table\}", line):
            buf=[line]; continue
        if buf is not None:
            buf.append(line)
            if re.match(r"\s*\\end\{table\}", line):
                blk="\n".join(buf)
                blk=blk.replace("\\begin{tabular}", "\\small\n\\begin{tabular}", 1)
                out.append(blk); buf=None
            continue
        out.append(line)
    return "\n".join(out)

results = size_tables(results)
discussion = size_tables(discussion)
methods = size_tables(methods)

doc = "\n".join([
    PREAMBLE,
    intro, "",
    methods, "",          # Elsevier order: Methods second, not at the back
    results, "",
    discussion, "",
    conclusions, "",
    back,
    TAIL,
])

DEST = ROOT / "paper-corrosion-science"

# The Corrosion Science folder holds REAL files, not symlinks into paper/, so a
# reviewer can be handed the directory as-is.  That duplication is only safe if
# something refreshes it: these three are copied from paper/ on every run, so
# paper/ stays the single place the science is edited and the copies cannot
# drift the way a hand-maintained second manuscript would.
import shutil
shutil.copy(P / "references.bib", DEST / "references.bib")
shutil.copy(P / "supplementary-body.tex", DEST / "supplementary-body.tex")
(DEST / "figures").mkdir(exist_ok=True)
n_fig = 0
for fig in sorted((P / "figures").glob("*.png")):
    shutil.copy(fig, DEST / "figures" / fig.name)
    n_fig += 1
print(f"refreshed: references.bib, supplementary-body.tex, {n_fig} figures")

out = DEST / "main-cas.tex"
out.write_text(doc)
print("wrote", out, len(doc.split(chr(10))), "lines")
