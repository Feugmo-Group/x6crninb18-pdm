"""Generate the Corrosion Science version from the npj manuscript.

The two submissions carry identical science and, as the npj file is currently
ordered, identical section order: both run Methods before Results.  Note that
npj Materials Degradation actually specifies Methods LAST for Articles, which
the npj file will have to adopt at acceptance (see the VENUE NOTE above its
\\section{Methods}); when that happens, the block boundaries below must slice
the npj source in npj order while the assembly stays in Elsevier order.  The
two also differ in document class (cas-sc vs sn-jnl), citation style, and the
venue front matter.  Rather
than maintain two copies of 1600 lines, this script derives one from the
other, so a correction to the science can only be made in one place.

Edit paper/main-snjnl.tex, then run:

    python scripts/make_corrosion_science_version.py

and rebuild.  Verify with a numeric-token diff of the two main files; they
must agree exactly.
"""

import re

from htw_pdm.paths import PAPER as P
from htw_pdm.paths import ROOT

SRC = (P / "main-snjnl.tex").read_text()

# Blocks are located by their own \\section marker rather than by line number.
# An earlier version sliced by hardcoded line ranges, which silently went stale
# the first time a subsection was added; boundaries are now derived, so editing
# the manuscript cannot desynchronise this generator.
def block(start, end):
    """Text from the start marker up to (not including) the end marker."""
    i = SRC.index(start)
    j = len(SRC) if end is None else SRC.index(end, i)
    return SRC[i:j].rstrip() + "\n"

# npj file order: Introduction -> Methods -> Results -> Discussion -> backmatter.
intro       = block("\\section{Introduction}",  "\\section{Methods}")
methods     = block("\\section{Methods}",       "\\section{Results}")
results     = block("\\section{Results}",       "\\section{Discussion}")
discussion  = block("\\section{Discussion}",    "\\backmatter")
backmatter  = block("\\backmatter", None)

# npj Articles permit no conclusions section, so the npj Discussion closes with
# the synthesis as running prose, marked by a CLOSING-BLOCK comment.  Corrosion
# Science expects a numbered Conclusions, so the same text is promoted here.
# Identical text, different heading: the numeric-token parity check still holds.
_MARK = "%% CLOSING-BLOCK"
assert _MARK in discussion, "CLOSING-BLOCK marker missing from the Discussion"
_head, _tail = discussion.split(_MARK, 1)
discussion  = _head.rstrip() + "\n"
conclusions = ("\\section{Conclusions}\\label{sec:conclusions}\n\n"
               + re.sub(r"\A(?:%%.*\n)+", "", _tail.split("\n", 1)[1]).lstrip())

# The abstract is read from the npj source too, so the two versions cannot
# drift on the one paragraph an editor is guaranteed to read.
_abs = re.search(r"\\abstract\{(.*?)\}\s*\n\s*\n\\keywords", SRC, re.S)
assert _abs, "could not locate \\abstract{...} in main-snjnl.tex"
ABSTRACT = _abs.group(1).strip()

# --- Introduction: no roadmap rewrite is needed --------------------------
# Both files currently run Methods before Results, so the npj roadmap
# paragraph describes the Elsevier order correctly as written.  If the npj
# file moves Methods last at acceptance, reinstate a rewrite here.

# --- author-year citations -------------------------------------------------
# The npj version cites numerically, where "Li et al.~[18]" reads correctly.
# Under cas-model2-names (name-date) the same markup renders "Li et al. (Li
# et al., 2020)", so the narrative forms become \citet.  The rest of the
# \citep calls are parenthetical and read correctly either way.
def narrative_citations(t):
    # "Chao, Lin, and Macdonald" is a plain name list, not an "et al." form:
    # keep the names and make the citation parenthetical-with-year instead.
    t = t.replace("Macdonald~\\citep{", "Macdonald~\\citeyearpar{")
    # "et al." is broken across lines in several places, so normalise the
    # newline first rather than listing every wrapped variant.
    for name in ("Li", "Veile"):
        t = t.replace(f"{name} et\nal.~\\citep{{", "\\citet{")
        t = t.replace(f"{name} et al.~\\citep{{", "\\citet{")
    return t

intro       = narrative_citations(intro)
results     = narrative_citations(results)
discussion  = narrative_citations(discussion)
conclusions = narrative_citations(conclusions)
methods     = narrative_citations(methods)

# --- Methods: Elsevier convention prefers a combined "Model and methods" ---
methods = methods.replace(
    "\\section{Methods}\\label{sec:methods}",
    "\\section{Model and methods}\\label{sec:methods}", 1)

# --- backmatter: sn-jnl \bmhead -> Elsevier starred sections --------------
back = (backmatter
        .replace("\\backmatter\n", "")
        .replace("\\bmhead{Data availability}",
                 "\\section*{Data availability}")
        .replace("\\bmhead{Code availability}",
                 "\\section*{Code availability}")
        .replace("\\bmhead{Author contributions}",
                 "\\section*{CRediT authorship contribution statement}")
        # npj uses the US spelling in its own headings; Elsevier the British one
        .replace("\\bmhead{Acknowledgments}",
                 "\\section*{Acknowledgements}")
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
%% Elsevier front matter and reformatted from sn-jnl to cas-sc; the section
%% order is the same in both.
%%
%% cas-sc is the single-column member of Elsevier's CAS bundle.  Being one
%% column, the figures keep the \textwidth sizing they have in the npj version
%% and no float needs promoting to a starred environment.
\documentclass[a4paper,fleqn]{cas-sc}

%% cas-model2-names is the CAS bundle's name-date style, so natbib runs in
%% authoryear mode.  Note this makes the references author-year rather than
%% numbered; Elsevier restyles at production and accepts either at submission.
\usepackage[authoryear,longnamesfirst]{natbib}
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
\ead{cgtetsas@uwaterloo.ca}

\address[1]{Department of Chemistry and Department of Physics \& Astronomy,
            University of Waterloo, 200 University Avenue West,
            Waterloo, ON N2L 3G1, Canada}

\cortext[1]{Corresponding author.}

\begin{abstract}
%%ABSTRACT%%
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

PREAMBLE = PREAMBLE.replace("%%ABSTRACT%%", ABSTRACT)

TAIL = r"""
\bibliographystyle{cas-model2-names}
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
                # A table that already declares its own size was sized by hand
                # in main-snjnl.tex because it needed more than one step down.
                # Inserting \small after that declaration would step it back
                # *up*, so leave those alone.
                if not re.search(r"\\(?:footnotesize|scriptsize|tiny|small)\b", blk):
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
    methods, "",          # Methods second, as in the npj version
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
for fig in sorted(list((P / "figures").glob("*.png")) + list((P / "figures").glob("*.pdf"))):
    shutil.copy(fig, DEST / "figures" / fig.name)
    n_fig += 1
print(f"refreshed: references.bib, supplementary-body.tex, {n_fig} figures")

out = DEST / "main-cas.tex"
out.write_text(doc)
print("wrote", out, len(doc.split(chr(10))), "lines")
