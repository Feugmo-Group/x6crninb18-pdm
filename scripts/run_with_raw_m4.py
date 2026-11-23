"""Run a script with htw_pdm.physics.Parameters defaulting to the M4 fit of the
current dataset (HTW_PDM_DATASET, t <= HTW_PDM_TMAX) instead of the digitized-data
values.  Used to rerun SI analyses on the raw line scans.

Run:  HTW_PDM_DATASET=veile_raw HTW_PDM_TMAX=480 python scripts/run_with_raw_m4.py <script or -m module>
"""

import dataclasses
import runpy
import sys

import htw_pdm.physics as physics
from htw_pdm.baseline_fit import build_fit_data, fit_pdm, load_data

q = fit_pdm(build_fit_data(load_data()[0]), 4)[1]
fields = {f: float(getattr(q, f)) for f in ("A_bl", "b3", "C_bl", "PBR_eff", "C_x", "L0", "L_ol0")}
physics.Parameters = dataclasses.make_dataclass(
    "Parameters", [(k, float, dataclasses.field(default=v)) for k, v in fields.items()])
print("Parameters defaults:", fields)
if sys.argv[1] == "-m":
    sys.argv = sys.argv[2:]
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
else:
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
