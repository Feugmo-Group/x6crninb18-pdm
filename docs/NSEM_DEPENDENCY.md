# The PhysicsNeMo dependency

The classical half of this package needs nothing unusual. The neural half needs
one thing that is not on PyPI, and this document says exactly what, why, and how
it is pinned.

## What is needed, and why it is not upstream

The NSEM solver imports three things that exist in a fork of PhysicsNeMo and not
in the NVIDIA upstream:

| Import | Provides |
|---|---|
| `physicsnemo.experimental.models.scen` | `DVRMapper` — the spectral-element / discrete-variational-representation networks |
| `physicsnemo.optim` | `TwoPhaseOptimizer` (Adam → L-BFGS), `build_aggregator` |
| `physicsnemo.utils` | `save_checkpoint`, `set_default_dtype`, `PythonLogger` |

`scen` and the two-phase optimizer are the implementation of the neural spectral
element method (Tetsassi Feugmo & Pankaczy, *Mach. Learn.: Sci. Technol.* **7**,
045053, 2026; doi:10.1088/2632-2153/ae8e2f). They have not been merged upstream,
so the fork is the only source. That is the entire reason a fork appears here.

## How it is pinned

`pyproject.toml` pins it by **immutable commit**, not by branch:

```toml
[tool.uv.sources]
nvidia-physicsnemo = {
  git = "https://github.com/Feugmo-Group/physicsnemo.git",
  rev = "a76d11b06ff89bdaff510722af0364824f744ea5",
}
```

A branch can move after publication. The revision behind a paper's results must
not, so the commit hash is what is recorded, and `uv.lock` records it a second
time. The pinned commit is the tip of the fork's `electrochem` branch at the time
of submission; its `experimental/models/scen` and `optim` trees are byte-identical
to the state this package was developed against.

**This package is standalone.** It is not a fork of PhysicsNeMo and does not need
to live inside one. PhysicsNeMo is an ordinary pinned dependency of the optional
`nsem` extra, and nothing in the classical path touches it.

## Installing

```bash
uv sync                 # classical path: NumPy/SciPy/Matplotlib + pytest, ~300 MB
uv sync --extra nsem    # adds PyTorch and PhysicsNeMo at the pinned revision
```

With pip, the pin has to be given explicitly, because `[tool.uv.sources]` is not
part of the published metadata:

```bash
pip install -e .
pip install -e '.[nsem]' \
  'nvidia-physicsnemo @ git+https://github.com/Feugmo-Group/physicsnemo.git@a76d11b06ff89bdaff510722af0364824f744ea5'
```

## What still works without it

Everything the manuscript's headline claims rest on:

- the model ladder M1–M6, the accepted M4 fit and its $\chi^2$;
- the profile likelihoods, the 1000-member bootstrap, the prior-relaxation test
  and the residual-leverage decomposition;
- the mass balance, the `PBR_eff` closure and the long-time predictions;
- the Wagner/the nickel-zone closure solves and the composition map;
- **every figure in the paper**, the F-1 exhibit and the constant-field closure
  exhibit included. Those two report neural results, but the figures are drawn
  from committed artefacts (`outputs/paper/f1_trajectory.npz` and
  `outputs/paper/field_closure.json`) rather than from checkpoints, so producing them
  needs this dependency and *plotting* them does not.

Table 3 is the exception: it is read out of training checkpoints, so
`scripts/make_paper_tables.py` regenerates tables 1, 2, 4, 8 and 10 on a
classical install and leaves that one as committed, printing that it did rather
than writing an empty file. Table 4, the F-1 self-consistency exhibit, used to
be in the same position; it is now built from `outputs/paper/f1_runs.json`,
which `scripts/make_f1_trajectory.py` writes from the same runs that produce the
figure, so the table and the figure cannot disagree and neither needs this
dependency to be rebuilt.

The committed artefacts are pinned against the numbers printed in the manuscript
by `tests/test_paper_numbers.py::TestNeuralArtefacts`, which runs in the
classical tier — so drift in an artefact is caught without installing any of
this. `tests/test_classical_install.py` checks the other direction: that
everything `docs/REPRODUCE.md` labels `classical` really does import without
PyTorch.

Absent PyTorch or PhysicsNeMo, the neural modules raise an `ImportError` naming
this document rather than a bare `ModuleNotFoundError`, and the neural tests skip
rather than fail. Run `python -m htw_pdm.check_environment` to see which half of
The pipeline the current environment can run.

## Updating the pin

Change `rev`, run `uv lock`, and re-run the acceptance checks in
`docs/IMPL_REPORT.md`. The forward-solve parity tests (`tests/test_ode_parity.py`,
`tests/test_field_closure.py`) are what would catch a behavioural change in the solver;
`tests/test_paper_numbers.py` is what would catch it reaching the manuscript.
