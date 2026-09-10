"""The composition map — composition-vs-depth map + EDX convolution, forward vs Veile features.

Assembles element-fraction depth profiles at the three Veile exposures from the
the zero-dimensional fit/the spatial model forward chain — NO fitting to the line scans:

  geometry     L_bl(t) from the the zero-dimensional fit closed form (M4 kinetics)
  bl chemistry f_Cr_bl from spinel stoichiometry: FeCr2O4 -> Cr = 2*52.0/222.7
               = 46.7 wt% of the (metals+O) EDX total (Veile normalize ignoring
               light elements; O is measured) — a zero-parameter forward value
  Ni zone      exclusion closure with nominal (pre-the spatial inverse) parameters r_Ni, D_Ni_eff:
               w_Ni(t) = D_Ni_eff / v_m(t), v_m = (dL_bl/dt)/PBR (recession speed),
               excess(t) = Ni_matrix*(1-r_Ni)*d_consumed(t), A_Ni = Ni_m + excess/w
               Pass --fitted after running spatial_inverse.py to use the the spatial inverse values.
  gradient     g_bl predicted from the steady spatial solve cation-vacancy profile: site-fraction
               modulation = c0_CV * (chat(1)-chat(0)) / N_cation(spinel) — compared
               against the the spatial identifiability gate minimum-detectable gradient (the no-go check).

The profiles reuse the the spatial identifiability gate shape builder (spatial_identifiability.profiles) with the
2 nm EDX convolution, then layer widths are extracted with Veile's own convention
(rise off the matrix baseline) and compared against the Fig. 9 means.

Run:  python -m htw_pdm.composition_map [--fitted]
"""

from __future__ import annotations

import csv
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htw_pdm.baseline_fit import load_data  # noqa: E402
from htw_pdm.baseline_ode import HTWPDMParams, L_bl_closed  # noqa: E402
from htw_pdm.paths import OUTPUTS as OUT  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402
from htw_pdm.spatial_identifiability import NI_M, profiles  # noqa: E402
from htw_pdm.spatial_physics import analytic_steady, species_groups  # noqa: E402

OUT.mkdir(exist_ok=True)

PBR_BL = 2.05  # FeCr2O4 (Li Table 3) — metal depth consumed = L_bl / PBR
N_CATION_SPINEL = 4.05e22  # cm^-3 (magnetite-structure cation sites)
F_CR_BL_STOICH = 100.0 * 2 * 52.00 / (2 * 52.00 + 55.85 + 4 * 16.00)  # 46.7 wt%

# Nominal (pre-the spatial inverse) Ni-exclusion parameters; overwritten by --fitted.
NI_NOMINAL = {"r_Ni": 0.0, "D_Ni_eff_nm2_h": 2.3, "phi_ol_supply": 0.0}

EXPOSURES_H = [72.0, 168.0, 480.0]


def zero_d_geometry(p1: Parameters, t_h: float):
    bp = HTWPDMParams(A_bl=p1.A_bl, b3=p1.b3, C_bl=p1.C_bl, PBR_eff=p1.PBR_eff,
                      C_x=p1.C_x, L0=p1.L0, L_ol0=p1.L_ol0)
    L = float(L_bl_closed(bp, np.array([t_h]))[0])
    dLdt = p1.A_bl * np.exp(p1.b3 * L) - p1.C_bl
    return L, dLdt


def ni_zone(p1: Parameters, t_h: float, ni: dict, fitted: bool = False):
    """Ni zone (decay length nm, amplitude wt%) at exposure t.

    fitted=True: solve the same Wagner moving-frame PDE the the spatial inverse parameters were
    fitted through (mixing them into the quasi-steady closure is inconsistent and
    wildly overpredicts the zone). fitted=False: nominal quasi-steady closure.
    """
    if fitted:
        from htw_pdm.spatial_inverse import wagner_solve

        theta = np.array([ni["D_Ni_eff_nm2_h"], ni["r_Ni"], ni["phi_ol_supply"]])
        c, xi = wagner_solve(theta, p1, t_eval=np.array([t_h]))
        excess = c[0] - NI_M
        A = NI_M + float(excess[0])
        # 1/e decay length of the solved enrichment profile.
        below = np.where(excess <= excess[0] / np.e)[0]
        w = float(xi[below[0]]) if len(below) else float(xi[-1])
        return w, A
    L, dLdt = zero_d_geometry(p1, t_h)
    v_m = dLdt / PBR_BL  # recession speed into the metal, nm/h
    w = min(ni["D_Ni_eff_nm2_h"] / v_m, np.sqrt(2.0 * ni["D_Ni_eff_nm2_h"] * t_h))
    d_consumed = (L - p1.L0) / PBR_BL + ni["phi_ol_supply"] * t_h  # nm of metal
    excess = NI_M * (1.0 - ni["r_Ni"]) * d_consumed / 100.0  # nm of pure-Ni equiv
    A = NI_M + 100.0 * excess / w
    return w, A


def predicted_gradient(p1: Parameters, t_h: float) -> float:
    """g_bl (abs %) from the steady spatial solve cation-vacancy profile at exposure t."""
    L, _ = zero_d_geometry(p1, t_h)
    g = species_groups(p1, L)["CV"]
    chat = analytic_steady(g, np.array([0.0, 1.0]))
    return 100.0 * g.c0_cm3 * abs(chat[1] - chat[0]) / N_CATION_SPINEL


def extract_width(x: np.ndarray, f: np.ndarray, baseline: float,
                  rise: float = 3.0) -> float:
    """Veile's convention: extent where the trace sits above baseline + rise."""
    above = f > baseline + rise
    if not np.any(above):
        return 0.0
    idx = np.where(above)[0]
    return float(x[idx[-1]] - x[idx[0]])


def main(fitted: bool = False):
    p1 = Parameters()
    ni = dict(NI_NOMINAL)
    tag = "nominal"
    fit_file = OUT / "spatial_inverse_fit.json"
    if fitted and fit_file.exists():
        ni.update(json.loads(fit_file.read_text()))
        tag = "the spatial inverse fitted"
    print(f"Ni-exclusion parameters ({tag}): {ni}")

    means, _ = load_data()
    obs = {el: {r[0]: (r[1], r[2]) for r in means[el]} for el in ("Cr", "Fe", "Ni")}

    plt.rcParams.update({
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ipanel, (ax, t_h) in enumerate(zip(axes, EXPOSURES_H)):
        L, _ = zero_d_geometry(p1, t_h)
        w_ni, a_ni = ni_zone(p1, t_h, ni, fitted=(tag == "the spatial inverse fitted"))
        g_pred = predicted_gradient(p1, t_h)
        shape = dict(L_bl=L, w_int=3.0, w_out=3.0, A_Ni=a_ni, w_Ni=w_ni,
                     f_Cr_bl=F_CR_BL_STOICH, g_bl=g_pred)
        x = np.arange(-140.0, L + 90.0, 0.5)
        f = profiles(x, shape)

        w_cr = extract_width(x, f["Cr"], 18.0, rise=3.0)
        w_ni_x = extract_width(x[x <= 0], f["Ni"][x <= 0], NI_M, rise=2.0)
        for el, pred in (("Cr", w_cr), ("Ni", w_ni_x)):
            m, sd = obs[el][t_h]
            rows.append([int(t_h), el, f"{pred:.1f}", f"{m:.1f}", f"{sd:.1f}",
                         f"{(pred - m) / m:+.1%}"])
        rows.append([int(t_h), "Ni_peak_wt%", f"{a_ni:.1f}",
                     "23.7" if t_h == 72.0 else "", "7.0" if t_h == 72.0 else "",
                     f"{(a_ni - 23.7) / 23.7:+.1%}" if t_h == 72.0 else ""])
        rows.append([int(t_h), "g_bl_pred_abs%", f"{g_pred:.2f}",
                     "min detectable 0.93 (the spatial identifiability gate)", "", ""])

        for k, c in (("O", "tab:red"), ("Cr", "tab:green"), ("Fe", "tab:blue"),
                     ("Ni", "tab:purple")):
            ax.plot(x, f[k], c=c, label=k if t_h == 72.0 else None)
        m_cr, sd_cr = obs["Cr"][t_h]
        ax.axvspan(0, m_cr, color="green", alpha=0.10,
                   label="measured Cr-layer width" if t_h == 72.0 else None)
        # Measured feature overlays (Veile 2024): the Cr plateau level inside
        # The barrier layer (~45 wt%) and, at 72 h, the reported Ni-peak
        # amplitude (23.7 +- 7.0 wt%). These are the features actually
        # compared -- full digitized EDX traces are not available.
        ax.axhline(45.0, color="tab:green", ls=":", lw=1.2,
                   label="measured Cr plateau (~45 wt%)" if t_h == 72.0 else None)
        if t_h == 72.0:
            ax.errorbar([-15.0], [23.7], yerr=[7.0], fmt="s", color="tab:purple",
                        ms=6, capsize=4, label="measured Ni peak (72 h)")
        ax.set_title(f"{int(t_h)} h — predicted Cr width {w_cr:.0f} nm "
                     f"(measured {m_cr:.0f}±{sd_cr:.0f})", fontsize=10)
        ax.set_xlabel("depth from m/bl interface (nm)")
        ax.text(0.02, 0.98, f"({'abc'[ipanel]})", transform=ax.transAxes,
                fontweight="bold", va="top", fontsize=12)
    # wt% of the EDX-normalized (metals + O) total — mass-based stoichiometry
    # (F_CR_BL_STOICH uses atomic masses), matching Veile's normalization.
    axes[0].set_ylabel("composition (wt%)")
    # Single figure-level legend above the panels, clear of the 72-h Ni-peak
    # error bar it used to occlude.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=8,
               frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    out_png = OUT / "plot_composition_map.png"
    plt.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")

    with open(OUT / "spatial_composition_comparison.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["t_h", "feature", "predicted", "measured_mean", "measured_sd",
                    "rel_dev"])
        w.writerows(rows)
    print(f"Saved: {OUT / 'spatial_composition_comparison.csv'}")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main(fitted="--fitted" in sys.argv)
