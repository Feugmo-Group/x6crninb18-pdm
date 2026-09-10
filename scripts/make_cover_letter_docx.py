# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cover_letter.docx -- the cover letter as Word, for editors who want one.

Nature Portfolio submission takes the cover letter as an editable file, so a
.docx is wanted alongside the PDF.  The LaTeX source stays the single source of
truth: this script derives the Word version from it rather than keeping a second
copy of the letter that would drift the first time a sentence changed.

Writes paper/cover_letter.md (readable intermediate) and paper/cover_letter.docx.

Needs pandoc.

Run:  python scripts/make_cover_letter_docx.py
"""

from __future__ import annotations

import re
import subprocess

from htw_pdm.paths import PAPER as P

SRC = P / "cover_letter.tex"
MD = P / "cover_letter.md"
DOCX = P / "cover_letter.docx"

tex = SRC.read_text(encoding="utf-8")


def body_of(env: str, s: str) -> str:
    i = s.index(f"\\begin{{{env}}}")
    i = s.index("}", s.index("{", i)) + 1 if env == "letter" else s.index("\n", i)
    return s[i:s.index(f"\\end{{{env}}}")]


def arg_of(cmd: str, s: str) -> str:
    """The braced argument of \\cmd{...}, brace-balanced."""
    i = s.index(f"\\{cmd}{{") + len(cmd) + 2
    d = 1
    for j in range(i, len(s)):
        if s[j] == "{":
            d += 1
        elif s[j] == "}":
            d -= 1
            if d == 0:
                return s[i:j]
    raise ValueError(cmd)


def detex(t: str) -> str:
    t = re.sub(r"(?<!\\)%.*", "", t)
    # inline maths and units that appear in this letter
    t = t.replace("$^\\circ$C", "\u00b0C").replace("O$_2$", "O\u2082")
    t = t.replace("$b_3$", "b\u2083").replace("$\\chi^2$", "\u03c7\u00b2")
    t = re.sub(r"\\,", "\u202f", t)          # thin space before units
    t = t.replace("~", " ")
    # innermost-first, looped, so nested \textbf{\textit{...}} resolves
    for _ in range(6):
        before = t
        t = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", t)
        t = re.sub(r"\\(?:textit|emph)\{([^{}]*)\}", r"*\1*", t)
        t = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", t)
        if t == before:
            break
    t = t.replace("``", "\u201c").replace("''", "\u201d")
    t = t.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    t = t.replace("--", "\u2013")
    t = re.sub(r"\\\\", "  \n", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


address = detex(arg_of("address", tex))
to = detex(body_of("letter", tex).split("\n\n")[0]) if False else detex(
    tex[tex.index("\\begin{letter}{") + 15: tex.index("}\n\n\\opening")]
)
opening = detex(arg_of("opening", tex))
closing = detex(arg_of("closing", tex))
signature = detex(arg_of("signature", tex))

body = tex[tex.index("\\opening"):tex.index("\\closing")]
body = body[body.index("\n"):]
body = re.sub(r"\\begin\{itemize\}", "", body)
body = re.sub(r"\\end\{itemize\}", "", body)
body = re.sub(r"\\item\s+", "\n\n- ", body)

paras = []
for chunk in re.split(r"\n\s*\n", body):
    c = detex(chunk)
    if c:
        paras.append(c)

md = [address, "", to, "", opening, ""]
md += [p + "\n" for p in paras]
md += ["", closing, "", signature, ""]
MD.write_text("\n".join(md), encoding="utf-8")

subprocess.run(
    ["pandoc", str(MD), "-o", str(DOCX), "--from", "markdown", "--to", "docx"],
    check=True, capture_output=True,
)
print(f"Saved: {MD}")
print(f"Saved: {DOCX}")
