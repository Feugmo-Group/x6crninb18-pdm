"""The nickel forward envelope -- forward predict-and-compare envelope for the Ni-zone closure.

the nickel-closure notes.md the nickel-closure gate found BOTH candidate closures (M-A flux-coupled mobility,
M-B finite-capacity trapping) fail the synthetic identifiability gate --
neither is safe to fit to the real Veile Ni observables (M-B decisively so;
M-A is the closer call, with its own new parameter recoverable but the base
kinetics destabilized just past the acceptance bar). Per the plan's own
contingency, the nickel-zone closure re-scopes from "refit the spatial inverse" to forward predict-and-compare
-- the same treatment the composition map gave the unidentifiable in-layer defect gradient.

This script does NOT fit anything. It scans n_flux (M-A; M-B is not carried
forward given its outright S_cap non-identifiability) over a physically
plausible range at the the spatial inverse base kinetics (D_Ni_eff, r_Ni, phi_ol_supply --
frozen at their the spatial inverse fitted values, never re-opened) and reports:

  1. the predicted W(t)/A(72h) envelope across the scan, plotted against the
     real Veile Ni observables with their population-SD error bars;
  2. which part of the n_flux range gives a qualitatively flat/non-monotonic
     width trajectory (like the data) vs. a monotonically growing one (like
     The the spatial inverse const closure, n_flux = 0) -- a descriptive statement, NOT a
     point estimate or an uncertainty interval (the nickel-closure gate showed neither is
     defensible for this parameter given the current data).

Run:  python -m htw_pdm.nickel_forward_envelope
"""

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htw_pdm.baseline_fit import load_data  # noqa: E402
from htw_pdm.nickel_wagner import A72_MEAN, A72_SIG, T_OBS, model  # noqa: E402
from htw_pdm.paths import OUTPUTS as OUT  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402

OUT.mkdir(exist_ok=True)

# The spatial inverse fitted optimum (D_Ni_eff, r_Ni, phi_ol_supply) -- frozen for this scan,
# never re-opened.  Read from spatial_inverse.py's own output rather than copied:
# a hardcoded duplicate silently goes stale whenever the M4 kinetics move (this
# one did, when the PBR_eff prior was removed), and then the envelope is frozen
# at kinetics no longer in the repository.
_FIT_JSON = OUT / "spatial_inverse_fit.json"
if not _FIT_JSON.exists():
    raise SystemExit(f"missing {_FIT_JSON}; run src/spatial_inverse.py first")
_fit = json.loads(_FIT_JSON.read_text())
BASE = np.array([_fit["D_Ni_eff_nm2_h"], _fit["r_Ni"], _fit["phi_ol_supply"]])

# Physically plausible n_flux range: the the nickel-closure gate gate found the flattest
# trajectory (closest to the data's flat/non-monotonic shape) near n_flux ~
# 1.5, and demonstrated recovery is unreliable outside roughly [-1, 2] at
# this data's noise level; scan a bit wider than the gate's own tested range
# to show the full qualitative behaviour, including the const-closure limit.
N_FLUX_SCAN = np.linspace(-2.0, 3.0, 51)


def flatness(W: np.ndarray) -> float:
    return float(np.std(W) / np.mean(W))


def is_non_monotone_increasing(W: np.ndarray, tol: float = 1.0) -> bool:
    """True if W is NOT monotonically non-decreasing within `tol` nm slack
    (the qualitative "flat/saturating-like-the-data" signature)."""
    return bool(np.any(np.diff(W) < -tol))


def main():
    p1 = Parameters()
    means, _ = load_data()
    ni = {r[0]: (r[1], r[2], r[3]) for r in means["Ni"]}
    W_obs = np.array([ni[t][0] for t in T_OBS])
    W_sig = np.array([ni[t][1] for t in T_OBS])
    print(f"Real Veile Ni observables: widths {W_obs} +- {np.round(W_sig, 2)} nm, "
          f"A(72h) = {A72_MEAN} +- {A72_SIG} wt%")
    print(f"Frozen the spatial inverse base kinetics (D_Ni_eff, r_Ni, phi_ol_supply): {BASE}")

    rows = []
    W_all = np.empty((len(N_FLUX_SCAN), len(T_OBS)))
    A72_all = np.empty(len(N_FLUX_SCAN))
    for i, n in enumerate(N_FLUX_SCAN):
        theta = np.concatenate([BASE, [n]])
        W, A72 = model(theta, p1, "flux")
        W_all[i], A72_all[i] = W, A72
        cv = flatness(W)
        non_mono = is_non_monotone_increasing(W)
        chi2 = float(np.sum(((W - W_obs) / W_sig) ** 2)
                     + ((A72 - A72_MEAN) / A72_SIG) ** 2)
        rows.append([n, *W, A72, cv, non_mono, chi2])

    chi2_col = np.array([r[7] for r in rows])
    non_mono_mask = np.array([bool(r[6]) for r in rows])
    n0_idx = int(np.argmin(np.abs(N_FLUX_SCAN - 0.0)))  # const-closure reference

    print(f"\nconst closure (n_flux=0) reference: W={np.round(W_all[n0_idx],1)}, "
          f"A72={A72_all[n0_idx]:.1f}, chi2={chi2_col[n0_idx]:.1f} "
          f"(matches the spatial inverse: chi2/dof = 25.7/1)")
    if non_mono_mask.any():
        lo, hi = N_FLUX_SCAN[non_mono_mask].min(), N_FLUX_SCAN[non_mono_mask].max()
        print(f"\nQualitatively flat/non-monotonic width trajectories "
              f"(like the data) occur for n_flux in roughly "
              f"[{lo:.2g}, {hi:.2g}] out of the scanned "
              f"[{N_FLUX_SCAN.min():.2g}, {N_FLUX_SCAN.max():.2g}].")
        best_i = int(np.argmin(chi2_col))
        print(f"Best forward chi2 (descriptive only, NOT a fit) at "
              f"n_flux = {N_FLUX_SCAN[best_i]:.3g}: W = {np.round(W_all[best_i],1)}, "
              f"A72 = {A72_all[best_i]:.1f}, chi2 = {chi2_col[best_i]:.1f}")
    else:
        print("\nNo scanned n_flux gives a non-monotonic width trajectory -- "
              "the M-A closure cannot qualitatively reproduce the flat/dipping "
              "real data shape at these base kinetics, only slow its growth.")

    # --- figure: envelope + real data ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.cm.viridis
    norm = plt.Normalize(N_FLUX_SCAN.min(), N_FLUX_SCAN.max())
    for i, n in enumerate(N_FLUX_SCAN):
        ax1.plot(T_OBS, W_all[i], "-", color=cmap(norm(n)), alpha=0.5, lw=1)
    ax1.plot(T_OBS, W_all[n0_idx], "k--", lw=2, label="n_flux = 0 (the spatial inverse const)")
    ax1.errorbar(T_OBS, W_obs, yerr=W_sig, fmt="o", color="red", ms=7,
                 capsize=4, zorder=5, label="Veile data (population SD)")
    ax1.set_xlabel("exposure time (h)")
    ax1.set_ylabel("Ni zone width (nm)")
    ax1.set_title("the nickel forward envelope: forward n_flux envelope vs real Ni widths\n"
                  "(descriptive only -- the nickel-closure gate showed n_flux is not fittable)")
    ax1.legend(fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax1, label="n_flux")

    ax2.plot(N_FLUX_SCAN, chi2_col, "b-")
    ax2.axhline(25.71, color="grey", ls="--", lw=1, label="the spatial inverse const chi2 (25.7)")
    ax2.axvline(0.0, color="k", lw=0.8)
    ax2.fill_between(N_FLUX_SCAN, 0, chi2_col.max() * 1.05,
                     where=non_mono_mask, color="green", alpha=0.1,
                     label="non-monotonic (flat-like) region")
    ax2.set_xlabel("n_flux")
    ax2.set_ylabel("forward chi2 (descriptive, not a fit statistic)")
    ax2.set_title("Forward chi2 vs n_flux (NOT an identifiability claim)")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "plot_nickel_forward_envelope.png", dpi=150)
    print(f"\nSaved: {OUT / 'plot_nickel_forward_envelope.png'}")

    with open(OUT / "nickel_forward_envelope.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_flux", "W_72h", "W_168h", "W_480h", "A72", "width_CV",
                    "non_monotonic", "forward_chi2"])
        for r in rows:
            w.writerow(r)
    print(f"Saved: {OUT / 'nickel_forward_envelope.csv'}")


if __name__ == "__main__":
    main()
