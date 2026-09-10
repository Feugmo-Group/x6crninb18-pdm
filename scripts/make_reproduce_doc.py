"""Regenerate docs/REPRODUCE.md: which command produces which artefact, and
which environment it needs.

Two things are derived rather than hand-kept, so neither can drift from the
code.  The artefact column comes from scanning each module and script for the
output filenames it writes.  The environment column comes from a module-level
import graph: a command is "neural" if it, or anything it imports from
htw_pdm, imports torch, PhysicsNeMo, Hydra or OmegaConf at module scope.
Module scope is the operative part -- htw_pdm.physics and htw_pdm.tier2_physics
reach torch through htw_pdm._optional inside function bodies, which is what
lets a classical install import them, and an AST walk over top-level statements
alone sees that distinction where a text search would not.

Run:  python scripts/make_reproduce_doc.py
"""
import ast
import pathlib

from htw_pdm.paths import ROOT

OUT = ROOT / "outputs"
srcs = sorted(pathlib.Path(ROOT/"src/htw_pdm").glob("*.py")) + \
       sorted(pathlib.Path(ROOT/"scripts").glob("*.py"))
# This script names every artefact in its own FIG table, so it would match
# all of them; exclude it from the scan.
srcs = [s for s in srcs if s.name != "make_reproduce_doc.py"]

def cmd_for(p):
    return (f"python -m htw_pdm.{p.stem}" if p.parent.name == "htw_pdm"
            else f"python scripts/{p.name}")


# ── environment classification ───────────────────────────────────────────────
NEURAL_ROOTS = {"torch", "physicsnemo", "hydra", "omegaconf"}


def top_level_imports(path: pathlib.Path) -> set[str]:
    """Dotted names imported by PATH at module scope.

    Only ast.Module.body is walked, so imports guarded by `if TYPE_CHECKING:`
    or deferred into a function do not count -- which is the whole point: those
    are exactly the ones a classical install never executes.
    """
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


_HTW = {p.stem: p for p in pathlib.Path(ROOT / "src/htw_pdm").glob("*.py")}


def needs_neural(path: pathlib.Path, _seen: frozenset[str] = frozenset()) -> bool:
    """True if PATH cannot be imported without the `nsem` extra installed."""
    imports = top_level_imports(path)
    if any(n.split(".")[0] in NEURAL_ROOTS for n in imports):
        return True
    for name in imports:
        if not name.startswith("htw_pdm."):
            continue
        stem = name.split(".")[1]
        if stem in _seen or stem not in _HTW:
            continue
        if needs_neural(_HTW[stem], _seen | {stem}):
            return True
    return False


ENV = {cmd_for(s): ("neural" if needs_neural(s) else "classical") for s in srcs}


# Two artefacts are produced by a classical script but read out of training
# checkpoints, so the command's own tier understates what they need. The script
# says so at run time; the table has to as well.
ARTEFACT_ENV = {
    "paper/table3_nsem_accuracy.csv": "neural",
    "paper/table4_f1_exhibit.csv": "neural",
}


def env_cell(cmds, artefact: str | None = None) -> str:
    if artefact in ARTEFACT_ENV:
        return ARTEFACT_ENV[artefact]
    kinds = {ENV.get(c, "classical") for c in cmds}
    return "neural" if "neural" in kinds else "classical"

arts = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
owner = {}
for s in srcs:
    t = s.read_text()
    for a in arts:
        if a.split("/")[-1] in t:
            owner.setdefault(a, set()).add(cmd_for(s))

# A file named by several scripts is produced by the one that is not merely
# reading it; prefer the writer, resolved by hand for the handful of cases.
PRIMARY = {
    "ensemble_members_1000.csv": "python -m htw_pdm.uncertainty_ensemble 1000",
    "paper/fig6_band.npz": "python scripts/plot_results.py",
    "paper/referee_stats.json": "python scripts/make_paper_stats.py",
    "paper/table1_parameters.csv": "python scripts/make_paper_tables.py",
    "plot_fits_vs_data.png": "python scripts/plot_results.py",
    "plot_long_time.png": "python scripts/plot_results.py",
    "plot_profile_likelihood.png": "python scripts/plot_results.py",
    "plot_sensitivity_maps.png": "python scripts/plot_sensitivity_maps.py",
    "plot_envelope_lines.png": "python scripts/plot_sensitivity_maps.py",
    "paper/ni_zone_curve.npz": "python scripts/make_ni_curve.py",
    # The trajectory artefact needs the neural extra; the figure that reads it
    # does not, which is why they are two steps.
    "paper/f1_trajectory.npz":
        "python scripts/make_f1_trajectory.py   (needs --extra nsem)",
    "paper/fig11_f1_failure.png": "python scripts/make_paper_fig_f1.py",
    "paper/fig12_field_closure.png": "python scripts/make_paper_fig_closure.py",
    "paper/tier4_pnp.json": "python -m htw_pdm.tier4_pnp_solve",
    "plot_uncertainty_ensemble.png": "python -m htw_pdm.uncertainty_ensemble",
    "plot_tier2_composition.png": "python -m htw_pdm.tier2_composition",
    "tier2_identifiability.csv": "python -m htw_pdm.tier2_identifiability",
    "tier2_inverse_fit.csv": "python -m htw_pdm.tier2_inverse",
    "tier2_inverse_fit.json": "python -m htw_pdm.tier2_inverse",
    "tier3_identifiability.csv": "python -m htw_pdm.tier3_identifiability",
    "tier3_identifiability_verdict.json": "python -m htw_pdm.tier3_identifiability",
    "ensemble_summary.csv": "python -m htw_pdm.uncertainty_ensemble",
}
for a in list(owner):
    for k, v in PRIMARY.items():
        if a.endswith(k):
            owner[a] = {v}

FIG = {  # manuscript float -> artefact
    "Fig. 1":  "paper/fig1_model_schematic.png",
    "Fig. 2":  "paper/fig2_baseline_verification.png",
    "Fig. 3":  "paper/fig3_fits_vs_data.png",
    "Fig. 4":  "paper/fig4_profile_likelihood.png",
    "Fig. 5":  "paper/fig5_uncertainty_ensemble.png",
    "Fig. 6":  "paper/fig6_long_time.png",
    "Fig. 7":  "paper/fig7_envelope_lines.png",
    "Fig. 8":  "paper/fig8_steady_spatial.png",
    "Fig. 9":  "paper/fig9_transient_qs.png",
    "Fig. 10": "paper/fig10_composition_edx.png",
    "Fig. 11 (F-1 exhibit)": "paper/fig11_f1_failure.png",
    "Fig. 12 (field closure)": "paper/fig12_field_closure.png",
    "Fig. S3 (envelope maps)": "paper/figS3_sensitivity_maps.png",
    "Fig. S-workflow": "paper/fig_nsem_workflow.png",
    "Table 1":  "paper/table1_parameters.csv",
    "Table 2":  "paper/table2_model_ladder.csv",
    "Table 3":  "paper/table3_nsem_accuracy.csv",
    "Table 4":  "paper/table4_f1_exhibit.csv",
    "Table 5":  "paper/table5_t2a_identifiability.csv",
    "Table 6":  "paper/table6_t2e_ni_exhibit.csv",
    "Table 7":  "paper/table7_t3_closure_nogo.csv",
    "Table 8":  "paper/table8_envelope_factors.csv",
    "Table 10": "paper/table10_pbr_closure.csv",
}

L = []
L.append("# Reproducing every number in the paper\n")
L.append("Each manuscript float and every committed artefact, with the exact command\n"
         "that writes it. **This file is generated** by `scripts/make_reproduce_doc.py`,\n"
         "which scans the source for output filenames, so it cannot drift from the code.\n")
L.append("Everything lands under `outputs/`, resolved by `htw_pdm.paths` from the\n"
         "`pyproject.toml` marker — never relative to the working directory. Set\n"
         "`HTW_PDM_ROOT` to redirect it.\n")
L.append("## Ordering\n")
L.append("""Three dependencies are real; the rest is free order.

1. `make_referee_analyses.py` must follow `make_paper_stats.py`, whose 1000-member
   ensemble it propagates.
2. `tier3_forward_envelope` must follow `tier2_inverse`, whose fit it reads from
   `outputs/tier2_inverse_fit.json`.
3. `make_paper_figs.py` must follow `plot_results.py` and `plot_sensitivity_maps.py`,
   whose PNGs it copies into `outputs/paper/`. It now fails loudly if they are absent;
   it used to skip silently, leaving figures 3, 4, 6 and 7 stale.
""")
_neural_cmds = sorted({c for c, k in ENV.items() if k == "neural"})
L.append("## Two environments\n")
L.append(f"""Commands are labelled `classical` or `neural`.

`classical` needs only `uv sync` — NumPy, SciPy and Matplotlib. **Every figure
in the manuscript is drawn by a classical command**, the two that report neural
results included: the F-1 exhibit and the field-closure exhibit read committed
artefacts rather than checkpoints, so a classical clone rebuilds every float in
the paper. One partial exception runs the other way. Tables 3 and 4 are read
out of training checkpoints, so `make_paper_tables.py` on a classical install
regenerates tables 1, 2, 8 and 10 and leaves those two as committed — printing
that it did, rather than writing two headers and no rows.

`neural` needs `uv sync --extra nsem`, which adds torch, Hydra and the pinned
PhysicsNeMo revision (see `docs/NSEM_DEPENDENCY.md`). {len(_neural_cmds)} commands are in this
tier. They are what *produced* the neural artefacts, and re-running them is
only necessary to regenerate those artefacts from scratch:

{chr(10).join('- `%s`' % c for c in _neural_cmds)}

The seed-0 runs are deterministic on CPU in float64, so re-running reproduces
the committed values rather than merely resembling them —
`tests/test_paper_numbers.py` pins the artefacts against the numbers printed in
the manuscript, and that test runs in the classical tier.
""")
L.append("## Manuscript floats\n")
L.append("| Float | Artefact | Env | Command |")
L.append("|---|---|---|---|")
for k, a in FIG.items():
    cmds = sorted(owner.get(a, {"— not regenerated by a tracked script —"}))
    L.append(f"| {k} | `outputs/{a}` | {env_cell(cmds, a)} | "
             f"{'<br>'.join('`%s`' % c for c in cmds)} |")

L.append("\n## Every other tracked artefact\n")
L.append("| Artefact | Env | Command |")
L.append("|---|---|---|")
for a in arts:
    if a in FIG.values():
        continue
    cmds = sorted(owner.get(a, []))
    if not cmds:
        continue
    L.append(f"| `outputs/{a}` | {env_cell(cmds, a)} | "
             f"{'<br>'.join('`%s`' % c for c in cmds)} |")

unowned = [a for a in arts if a not in owner and a not in FIG.values()]
L.append("\n## Written under a run directory, not by name\n")
L.append("Hydra run directories and network checkpoints: the trainers name these at\n"
         "run time, so they are listed by the command that produces the directory.\n")
L.append("| Path | Command |")
L.append("|---|---|")
seen = set()
for a in unowned:
    top = a.split("/")[0]
    if top in seen:
        continue
    seen.add(top)
    c = ("`python -m htw_pdm.trainer`" if top.startswith("forward")
         else "`python -m htw_pdm.inverse_trainer`" if top.startswith(("inverse", "nsem", "20"))
         else "`python -m htw_pdm.tier2_identifiability`" if top == "tier2"
         else "—")
    L.append(f"| `outputs/{top}/` | {c} |")

(ROOT/"docs"/"REPRODUCE.md").write_text("\n".join(L) + "\n")
print("wrote docs/REPRODUCE.md")
