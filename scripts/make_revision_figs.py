"""Figures added in the npj revision, drawn from outputs/revision/<dataset>.json.

    fig_profiles_wide.png     profiles over +-6 ln units, data-only and with the L0 prior
    fig_dissolution_ridge.png (a) chi2 along the fixed-C_bl ridge, (b) barrier layer to
                              10 years for M4, ridge members and the power law
    (fig_blind() is kept but no longer drawn: the 104-day exposure is not a forecast test.)

Run after scripts/make_revision_analyses.py:
    python scripts/make_revision_figs.py [--dataset veile_raw]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from htw_pdm.baseline_fit import DEFAULT_DATASET, build_fit_data, load_data, unpack  # noqa: E402
from htw_pdm.baseline_ode import L_bl_closed  # noqa: E402
from htw_pdm.paths import OUTPUTS, ROOT  # noqa: E402

OUT = OUTPUTS / "revision"
PAPER_FIGS = ROOT / "paper" / "npj-revision" / "figures"
HY = 8766.0
PL_PUBLISHED = (6.521, 0.4964)  # Veile et al. Cr power law (scan level)
BLUE, GREEN, RED, GREY = "#1f5fa8", "#2c8c4a", "#c0392b", "#555555"
# Same print scaling as scripts/plot_results.py: figures sit at \textwidth (5.15 in),
# so fontsize=N*PT prints at N pt whatever the authored figure width.
TEXTWIDTH_IN = 5.15
PT = 1.0


def set_scale(fig_w_in, tick=7.0, label=8.0, legend=6.5):
    global PT
    PT = fig_w_in / TEXTWIDTH_IN
    plt.rcParams.update({"axes.labelsize": label * PT, "xtick.labelsize": tick * PT,
                         "ytick.labelsize": tick * PT, "legend.fontsize": legend * PT,
                         "font.size": tick * PT})


def panel_letter(ax, letter, y=0.98):
    ax.text(0.02, y, f"({letter})", transform=ax.transAxes, fontweight="bold",
            va="top", fontsize=9 * PT)


def fig_profiles(res, path):
    names = list(res["profiles"]["prior"])
    labels = {"A_bl": r"$A_\mathrm{bl}$ (nm/h)", "-b3": r"$-b_3$ (nm$^{-1}$)",
              "PBR_eff": r"PBR$_\mathrm{eff}$", "L0": r"$L_0$ (nm)",
              "L_ol0": r"$L_\mathrm{ol,0}$ (nm)"}
    set_scale(12)
    fig, axs = plt.subplots(1, len(names), figsize=(12, 2.8), sharey=True)
    for ax, n in zip(axs, names):
        for mode, st, c in (("prior", "-", BLUE), ("data", "--", RED)):
            v = res["profiles"][mode][n]
            ax.semilogx(v["grid"], v["dchi2"], st, color=c,
                        label="with $L_0$ prior" if mode == "prior" else "data only")
        ax.axhline(1, color=GREY, lw=0.7, ls=":")
        ax.axhline(4, color=GREY, lw=0.7, ls="-.")
        ax.set_ylim(0, 12)
        ax.set_xlabel(labels.get(n, n))
    axs[0].set_ylabel(r"$\Delta\chi^2$")
    axs[3].legend(loc="center left")
    for k, ax in enumerate(axs):
        panel_letter(ax, "abcde"[k])
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def fig_ridge(res, d, path):
    rows = res["dissolution_scan"]["rows"]
    set_scale(9.5)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    c = np.array([r["C_bl"] for r in rows[1:]])
    ax[0].semilogx(c, [r["chi2_data"] for r in rows[1:]], "o-", color=GREEN, ms=3)
    ax[0].axhline(rows[0]["chi2_data"], color=BLUE, lw=1)
    ax[0].text(c[-1], rows[0]["chi2_data"] - 0.08, "M4 ($C_\\mathrm{bl}=0$)", color=BLUE, fontsize=7 * PT, ha="right")
    ax[0].set(xlabel=r"barrier dissolution rate $C_\mathrm{bl}$ (nm/h), fixed",
              ylabel=r"$\chi^2$ (data)")
    t = np.geomspace(10, 10 * HY, 300)
    for r, a in zip(rows[1::4], np.linspace(0.35, 1, len(rows[1::4]))):
        ax[1].loglog(t / HY, L_bl_closed(unpack(np.array(r["x"]), 5), t), color=GREEN, alpha=a, lw=1)
    ax[1].loglog(t / HY, L_bl_closed(unpack(np.array(rows[0]["x"]), 5), t), color=BLUE, lw=2, label="M4")
    ax[1].loglog([], [], color=GREEN, label=r"M5, fixed $C_\mathrm{bl}$")
    ax[1].loglog(t / HY, PL_PUBLISHED[0] * t ** PL_PUBLISHED[1], "--", color=RED, label="power law (Veile)")
    ax[1].errorbar(d.t / HY, d.L_cr, yerr=d.sig_cr, fmt="ko", ms=4, label="Cr layer, mean ± s.e.")
    ax[1].axvspan(72 / HY, 480 / HY, color="0.9")
    ax[1].axvline(2496 / HY, color=GREY, ls=":", lw=1)  # 104-day exposure
    ax[1].set(xlabel="exposure time (years)", ylabel="barrier-layer thickness (nm)", ylim=(2, 4000))
    ax[1].legend(loc="lower right")
    for k, a in enumerate(ax):
        panel_letter(a, "ab"[k], y=0.15 if k == 0 else 0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def fig_blind(res, path, scans=None):
    b = res["blind_test"]
    if b is None:
        return False
    h = b["held_out"]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for k, e in enumerate(h):
        x = k
        ax.errorbar(x - 0.24, e["measured_Cr"], yerr=e["sigma_Cr"], fmt="ko", capsize=3,
                    label="measured mean $\\pm$ SE" if k == 0 else None)
        if scans:
            v = [L for t, L in scans["Cr"] if t == e["t_h"]]
            ax.plot([x - 0.32] * len(v), v, "o", mfc="none", mec="k", ms=4,
                    label="single scans" if k == 0 else None)
        ax.errorbar(x - 0.04, e["M4_Cr"], yerr=e.get("M4_Cr_sd", 0), fmt="s", color=BLUE,
                    capsize=3, label="M4" if k == 0 else None)
        ax.vlines(x + 0.12, *e["M5_Cr_range_1sigma"], color=GREEN, lw=4,
                  label="M5, 1$\\sigma$ set of $C_\\mathrm{bl}$" if k == 0 else None)
        ax.errorbar(x + 0.28, e["power_law_Cr"], yerr=e["power_law_Cr_sd"], fmt="^", color=RED,
                    capsize=3, label="power law" if k == 0 else None)
    ax.set_xticks(range(len(h)), [f"{e['t_h']:.0f} h" for e in h])
    ax.set_xlim(-0.7, len(h) - 0.3)
    ax.set_ylim(0, None)
    ax.set_ylabel("barrier-layer thickness (nm)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("HTW_PDM_DATASET", DEFAULT_DATASET))
    a = ap.parse_args()
    res = json.loads((OUT / f"{a.dataset}.json").read_text())
    d = build_fit_data(load_data(a.dataset, t_max=480.0)[0])
    made = []
    for name, fn in (("fig_profiles_wide.png", lambda p: fig_profiles(res, p)),
                     ("fig_dissolution_ridge.png", lambda p: fig_ridge(res, d, p))):
        path = OUT / name
        if fn(path) is not False:
            made.append(path)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    for p in made:
        shutil.copy(p, PAPER_FIGS / p.name)
        print("wrote", p, "->", PAPER_FIGS / p.name)


if __name__ == "__main__":
    main()
