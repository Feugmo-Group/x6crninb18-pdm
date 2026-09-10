# Reproducing every number in the paper

Each manuscript float and every committed artefact, with the exact command
that writes it. **This file is generated** by `scripts/make_reproduce_doc.py`,
which scans the source for output filenames, so it cannot drift from the code.

Everything lands under `outputs/`, resolved by `htw_pdm.paths` from the
`pyproject.toml` marker — never relative to the working directory. Set
`HTW_PDM_ROOT` to redirect it.

## Ordering

Three dependencies are real; the rest is free order.

1. `make_referee_analyses.py` must follow `make_paper_stats.py`, whose 1000-member
   ensemble it propagates.
2. `nickel_forward_envelope` must follow `spatial_inverse`, whose fit it reads from
   `outputs/spatial_inverse_fit.json`.
3. `make_paper_figs.py` must follow `plot_results.py` and `plot_sensitivity_maps.py`,
   whose PNGs it copies into `outputs/paper/`. It now fails loudly if they are absent;
   it used to skip silently, leaving figures 3, 4, 6 and 7 stale.

## Two environments

Commands are labelled `classical` or `neural`.

`classical` needs only `uv sync` — NumPy, SciPy and Matplotlib. **Every figure
in the manuscript is drawn by a classical command**, the two that report neural
results included: the F-1 exhibit and the field-closure exhibit read committed
artefacts rather than checkpoints, so a classical clone rebuilds every float in
The paper. One partial exception runs the other way. The NSEM accuracy table is
read out of training checkpoints, so `make_paper_tables.py` on a classical
install regenerates the others and leaves that one as committed — printing that
it did, rather than writing a header and no rows.

`neural` needs `uv sync --extra nsem`, which adds torch, Hydra and the pinned
PhysicsNeMo revision (see `docs/NSEM_DEPENDENCY.md`). 11 commands are in this
tier. They are what *produced* the neural artefacts, and re-running them is
only necessary to regenerate those artefacts from scratch:

- `python -m htw_pdm.field_closure`
- `python -m htw_pdm.field_closure_solve`
- `python -m htw_pdm.inverse_trainer`
- `python -m htw_pdm.nickel_mobility_closure`
- `python -m htw_pdm.parametric_trainer`
- `python -m htw_pdm.spatial_steady_trainer`
- `python -m htw_pdm.spatial_transient_trainer`
- `python -m htw_pdm.trainer`
- `python -m htw_pdm.transient_inverse`
- `python -m htw_pdm.uncertainty_ensemble`
- `python scripts/make_f1_trajectory.py`

The seed-0 runs are deterministic on CPU in float64, so re-running reproduces
The committed values rather than merely resembling them —
`tests/test_paper_numbers.py` pins the artefacts against the numbers printed in
The manuscript, and that test runs in the classical tier.

## Manuscript floats

| Float | Artefact | Env | Command |
|---|---|---|---|
| Fig. 1 | `outputs/paper/fig1_model_schematic.png` | classical | `python scripts/make_paper_figs.py` |
| Fig. 2 | `outputs/paper/fig_nsem_workflow.png` | classical | `python scripts/make_paper_fig_nsem.py` |
| Fig. 3 | `outputs/paper/fig3_fits_vs_data.png` | classical | `python scripts/make_paper_figs.py` |
| Fig. 4 | `outputs/paper/fig4_profile_likelihood.png` | classical | `python scripts/make_paper_figs.py` |
| Fig. 5 | `outputs/paper/fig5_uncertainty_ensemble.png` | classical | `python scripts/make_paper_fig5.py` |
| Fig. 6 | `outputs/paper/fig6_long_time.png` | classical | `python scripts/make_paper_figs.py` |
| Fig. 7 | `outputs/paper/fig7_envelope_lines.png` | classical | `python scripts/make_paper_figs.py` |
| Fig. 8 | `outputs/paper/fig10_composition_edx.png` | classical | `python scripts/make_paper_figs_spatial.py` |
| Fig. 9 | `outputs/paper/fig11_f1_failure.png` | classical | `python scripts/make_paper_fig_f1.py` |
| Fig. 10 | `outputs/paper/fig12_field_closure.png` | classical | `python scripts/make_paper_fig_closure.py` |
| Fig. S1 | `outputs/paper/fig2_baseline_verification.png` | classical | `python scripts/make_paper_figs.py` |
| Fig. S2 | `outputs/paper/fig9_transient_qs.png` | classical | `python scripts/make_paper_figs_spatial.py` |
| Fig. S3 | `outputs/paper/figS3_sensitivity_maps.png` | classical | `python scripts/make_paper_figs.py` |
| Fig. S4 | `outputs/paper/fig8_steady_spatial.png` | classical | `python scripts/make_paper_figs_spatial.py` |
| Table 3 | `outputs/paper/table2_model_ladder.csv` | classical | `python scripts/make_paper_tables.py` |
| Table 5 | `outputs/paper/table1_parameters.csv` | classical | `python scripts/make_paper_tables.py` |
| Table 7 | `outputs/paper/table8_envelope_factors.csv` | classical | `python scripts/make_envelope_table.py` |
| Table 8 | `outputs/paper/table10_pbr_closure.csv` | classical | `python -m htw_pdm.pbr_closure` |
| Table S3 | `outputs/paper/table_nickel_zone_fit.csv` | classical | `python scripts/make_paper_tables_spatial.py` |
| Table S4 | `outputs/paper/table_nickel_closure_gate.csv` | classical | `python scripts/make_paper_tables_spatial.py` |
| Table S6 | `outputs/paper/table4_f1_exhibit.csv` | classical | `python scripts/make_paper_tables.py` |
| Table S7 | `outputs/paper/table_spatial_identifiability.csv` | classical | `python -m htw_pdm.spatial_identifiability` |

## Every other tracked artefact

| Artefact | Env | Command |
|---|---|---|
| `outputs/ensemble_members.csv` | neural | `python -m htw_pdm.uncertainty_ensemble` |
| `outputs/ensemble_members_1000.csv` | classical | `python -m htw_pdm.uncertainty_ensemble 1000` |
| `outputs/ensemble_summary.csv` | neural | `python -m htw_pdm.uncertainty_ensemble` |
| `outputs/forward/checkpoint.0.2000.pt` | classical | `python scripts/make_paper_tables.py` |
| `outputs/forward_kan/checkpoint.0.2000.pt` | classical | `python scripts/make_paper_tables.py` |
| `outputs/inverse/kinetics_hard.pt` | neural | `python -m htw_pdm.inverse_trainer`<br>`python scripts/make_paper_tables.py` |
| `outputs/inverse_hard_synth/kinetics_hard.pt` | neural | `python -m htw_pdm.inverse_trainer`<br>`python scripts/make_paper_tables.py` |
| `outputs/nickel_forward_envelope.csv` | classical | `python -m htw_pdm.nickel_forward_envelope` |
| `outputs/nickel_identifiability.csv` | classical | `python -m htw_pdm.nickel_identifiability` |
| `outputs/nickel_identifiability_verdict.json` | classical | `python -m htw_pdm.nickel_identifiability` |
| `outputs/paper/f1_runs.json` | neural | `python scripts/make_f1_trajectory.py`<br>`python scripts/make_paper_tables.py` |
| `outputs/paper/f1_trajectory.npz` | classical | `python scripts/make_f1_trajectory.py   (needs --extra nsem)` |
| `outputs/paper/field_closure.json` | neural | `python -m htw_pdm.field_closure_solve` |
| `outputs/paper/fig1_model_schematic.pdf` | classical | `python scripts/make_paper_figs.py` |
| `outputs/paper/fig6_band.npz` | classical | `python scripts/plot_results.py` |
| `outputs/paper/fig_nsem_workflow.pdf` | classical | `python scripts/make_paper_fig_nsem.py` |
| `outputs/paper/ni_zone_curve.npz` | classical | `python scripts/make_ni_curve.py` |
| `outputs/paper/nickel_mobility_closure.json` | neural | `python -m htw_pdm.nickel_mobility_closure` |
| `outputs/paper/pbr_closure.json` | classical | `python -m htw_pdm.pbr_closure` |
| `outputs/paper/referee_response.json` | classical | `python scripts/make_referee_analyses.py` |
| `outputs/paper/referee_stats.json` | classical | `python scripts/make_paper_stats.py` |
| `outputs/paper/table3_nsem_accuracy.csv` | neural | `python scripts/make_paper_tables.py` |
| `outputs/paper/table8_o2_only_spread.csv` | classical | `python scripts/make_envelope_table.py` |
| `outputs/paper/transient_inverse.json` | neural | `python -m htw_pdm.transient_inverse` |
| `outputs/plot_composition_map.png` | classical | `python -m htw_pdm.composition_map` |
| `outputs/plot_envelope_lines.png` | classical | `python scripts/plot_sensitivity_maps.py` |
| `outputs/plot_fits_vs_data.png` | classical | `python scripts/plot_results.py` |
| `outputs/plot_long_time.png` | classical | `python scripts/plot_results.py` |
| `outputs/plot_nickel_forward_envelope.png` | classical | `python -m htw_pdm.nickel_forward_envelope` |
| `outputs/plot_nickel_identifiability.png` | classical | `python -m htw_pdm.nickel_identifiability` |
| `outputs/plot_pbr_closure.png` | classical | `python -m htw_pdm.pbr_closure` |
| `outputs/plot_profile_likelihood.png` | classical | `python scripts/plot_results.py` |
| `outputs/plot_sensitivity_maps.png` | classical | `python scripts/plot_sensitivity_maps.py` |
| `outputs/plot_spatial_inverse.png` | classical | `python -m htw_pdm.spatial_inverse` |
| `outputs/plot_uncertainty_ensemble.png` | neural | `python -m htw_pdm.uncertainty_ensemble` |
| `outputs/spatial_composition_comparison.csv` | classical | `python -m htw_pdm.composition_map` |
| `outputs/spatial_identifiability.csv` | classical | `python -m htw_pdm.spatial_identifiability` |
| `outputs/spatial_inverse_fit.csv` | classical | `python -m htw_pdm.spatial_inverse` |
| `outputs/spatial_inverse_fit.json` | classical | `python -m htw_pdm.spatial_inverse` |

## Written under a run directory, not by name

Hydra run directories and network checkpoints: the trainers name these at
run time, so they are listed by the command that produces the directory.

| Path | Command |
|---|---|
| `outputs/forward/` | `python -m htw_pdm.trainer` |
| `outputs/forward_kan/` | `python -m htw_pdm.trainer` |
| `outputs/paper/` | — |
| `outputs/spatial/` | `python -m htw_pdm.spatial_identifiability` |
