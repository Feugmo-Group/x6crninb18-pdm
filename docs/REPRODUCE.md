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
2. `tier3_forward_envelope` must follow `tier2_inverse`, whose fit it reads from
   `outputs/tier2_inverse_fit.json`.
3. `make_paper_figs.py` must follow `plot_results.py` and `plot_sensitivity_maps.py`,
   whose PNGs it copies into `outputs/paper/`. It now fails loudly if they are absent;
   it used to skip silently, leaving figures 3, 4, 6 and 7 stale.

## Manuscript floats

| Float | Artefact | Command |
|---|---|---|
| Fig. 1 | `outputs/paper/fig1_model_schematic.png` | `python scripts/make_paper_figs.py` |
| Fig. 2 | `outputs/paper/fig2_baseline_verification.png` | `python scripts/make_paper_figs.py` |
| Fig. 3 | `outputs/paper/fig3_fits_vs_data.png` | `python scripts/make_paper_figs.py` |
| Fig. 4 | `outputs/paper/fig4_profile_likelihood.png` | `python scripts/make_paper_figs.py` |
| Fig. 5 | `outputs/paper/fig5_uncertainty_ensemble.png` | `python scripts/make_paper_fig5.py` |
| Fig. 6 | `outputs/paper/fig6_long_time.png` | `python scripts/make_paper_figs.py` |
| Fig. 7 | `outputs/paper/fig7_sensitivity_maps.png` | `python scripts/make_paper_figs.py` |
| Fig. 8 | `outputs/paper/fig8_steady_spatial.png` | `python scripts/make_paper_figs_tier2.py` |
| Fig. 9 | `outputs/paper/fig9_transient_qs.png` | `python scripts/make_paper_figs_tier2.py` |
| Fig. 10 | `outputs/paper/fig10_composition_edx.png` | `python scripts/make_paper_figs_tier2.py` |
| Fig. S-workflow | `outputs/paper/fig_nsem_workflow.png` | `python scripts/make_paper_fig_nsem.py` |
| Table 1 | `outputs/paper/table1_parameters.csv` | `python scripts/make_paper_tables.py` |
| Table 2 | `outputs/paper/table2_model_ladder.csv` | `python scripts/make_paper_tables.py` |
| Table 3 | `outputs/paper/table3_nsem_accuracy.csv` | `python scripts/make_paper_tables.py` |
| Table 4 | `outputs/paper/table4_f1_exhibit.csv` | `python scripts/make_paper_tables.py` |
| Table 5 | `outputs/paper/table5_t2a_identifiability.csv` | `python scripts/make_paper_tables_tier2.py` |
| Table 6 | `outputs/paper/table6_t2e_ni_exhibit.csv` | `python scripts/make_paper_tables_tier2.py` |
| Table 7 | `outputs/paper/table7_t3_closure_nogo.csv` | `python scripts/make_paper_tables_tier2.py` |
| Table 8 | `outputs/paper/table8_envelope_factors.csv` | `python scripts/make_envelope_table.py` |
| Table 10 | `outputs/paper/table10_pbr_closure.csv` | `python -m htw_pdm.pbr_closure` |

## Every other tracked artefact

| Artefact | Command |
|---|---|
| `outputs/ensemble_members.csv` | `python -m htw_pdm.uncertainty_ensemble` |
| `outputs/ensemble_members_1000.csv` | `python -m htw_pdm.uncertainty_ensemble 1000` |
| `outputs/ensemble_summary.csv` | `python -m htw_pdm.uncertainty_ensemble` |
| `outputs/forward/checkpoint.0.2000.pt` | `python scripts/make_paper_tables.py` |
| `outputs/forward_kan/checkpoint.0.2000.pt` | `python scripts/make_paper_tables.py` |
| `outputs/inverse/checkpoint.0.6000.pt` | `python scripts/make_paper_tables.py` |
| `outputs/inverse/kinetics_hard.pt` | `python -m htw_pdm.inverse_trainer`<br>`python scripts/make_paper_tables.py` |
| `outputs/inverse_hard/kinetics_hard.pt` | `python -m htw_pdm.inverse_trainer`<br>`python scripts/make_paper_tables.py` |
| `outputs/inverse_hard_synth/kinetics_hard.pt` | `python -m htw_pdm.inverse_trainer`<br>`python scripts/make_paper_tables.py` |
| `outputs/inverse_ld10/checkpoint.0.6000.pt` | `python scripts/make_paper_tables.py` |
| `outputs/paper/fig6_band.npz` | `python scripts/plot_results.py` |
| `outputs/paper/pbr_closure.json` | `python -m htw_pdm.pbr_closure` |
| `outputs/paper/referee_response.json` | `python scripts/make_referee_analyses.py` |
| `outputs/paper/referee_stats.json` | `python scripts/make_paper_stats.py` |
| `outputs/paper/tier4_ni_closure.json` | `python -m htw_pdm.tier4_ni_closure` |
| `outputs/paper/tier4_pnp.json` | `python -m htw_pdm.tier4_pnp_solve` |
| `outputs/paper/tier4_transient_inverse.json` | `python -m htw_pdm.tier4_transient_inverse` |
| `outputs/plot_fits_vs_data.png` | `python scripts/plot_results.py` |
| `outputs/plot_long_time.png` | `python scripts/plot_results.py` |
| `outputs/plot_pbr_closure.png` | `python -m htw_pdm.pbr_closure` |
| `outputs/plot_profile_likelihood.png` | `python scripts/plot_results.py` |
| `outputs/plot_sensitivity_maps.png` | `python scripts/plot_sensitivity_maps.py` |
| `outputs/plot_tier2_composition.png` | `python -m htw_pdm.tier2_composition` |
| `outputs/plot_tier2_inverse.png` | `python -m htw_pdm.tier2_inverse` |
| `outputs/plot_tier3_forward_envelope.png` | `python -m htw_pdm.tier3_forward_envelope` |
| `outputs/plot_tier3_identifiability.png` | `python -m htw_pdm.tier3_identifiability` |
| `outputs/plot_uncertainty_ensemble.png` | `python -m htw_pdm.uncertainty_ensemble` |
| `outputs/tier2_composition_comparison.csv` | `python -m htw_pdm.tier2_composition` |
| `outputs/tier2_identifiability.csv` | `python -m htw_pdm.tier2_identifiability` |
| `outputs/tier2_inverse_fit.csv` | `python -m htw_pdm.tier2_inverse` |
| `outputs/tier2_inverse_fit.json` | `python -m htw_pdm.tier2_inverse` |
| `outputs/tier3_forward_envelope.csv` | `python -m htw_pdm.tier3_forward_envelope` |
| `outputs/tier3_identifiability.csv` | `python -m htw_pdm.tier3_identifiability` |
| `outputs/tier3_identifiability_verdict.json` | `python -m htw_pdm.tier3_identifiability` |

## Written under a run directory, not by name

Hydra run directories and network checkpoints: the trainers name these at
run time, so they are listed by the command that produces the directory.

| Path | Command |
|---|---|
| `outputs/2026-08-07/` | `python -m htw_pdm.inverse_trainer` |
| `outputs/forward/` | `python -m htw_pdm.trainer` |
| `outputs/forward_kan/` | `python -m htw_pdm.trainer` |
| `outputs/inverse/` | `python -m htw_pdm.inverse_trainer` |
| `outputs/inverse_ld10/` | `python -m htw_pdm.inverse_trainer` |
| `outputs/nsem_real/` | `python -m htw_pdm.inverse_trainer` |
| `outputs/nsem_soft_data/` | `python -m htw_pdm.inverse_trainer` |
| `outputs/nsem_soft_phys/` | `python -m htw_pdm.inverse_trainer` |
| `outputs/nsem_synth/` | `python -m htw_pdm.inverse_trainer` |
| `outputs/tier2/` | `python -m htw_pdm.tier2_identifiability` |
