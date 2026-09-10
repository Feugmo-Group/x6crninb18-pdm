# SPDX-FileCopyrightText: Copyright (c) 2026 Conrard Giresse Tetsassi Feugmo
# SPDX-License-Identifier: Apache-2.0

"""Phase E figures for the HTW_PDM fit to Veile 2024 (X6CrNiNb18-10 duplex oxide).

Three figures, all built from the deterministic baseline_fit/baseline_ode fits
(no NSEM checkpoint needed — these are the "paper" figures for the classical
inverse solution, M4 being the accepted physical model):

  1. plot_fits_vs_data.png   — Cr/Fe layer thickness vs t: data (mean+-SD),
     M4 (PDM) curve, power-law curve, on linear and log-log axes.
  2. plot_long_time.png      — M4 vs M5 vs power-law extrapolation to 40 y,
     the "so-what" divergence figure from IMPL_REPORT.md Phase B3.
  3. plot_profile_likelihood.png — Delta-chi2 profile per M4 parameter with
     the 1-sigma (Delta-chi2=1) and identifiability (Delta-chi2=4) lines.

Run:  python scripts/plot_results.py
"""


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htw_pdm.baseline_fit import (  # noqa: E402
    build_fit_data,
    fit_pdm,
    fit_power_law,
    load_data,
    param_names,
    profile_likelihood,
)
from htw_pdm.baseline_ode import L_bl_closed, L_bl_steady_state, L_ol_closed  # noqa: E402
from htw_pdm.paths import OUTPUTS as OUT_DIR  # noqa: E402

OUT_DIR.mkdir(exist_ok=True)

# Figures are placed at width=\textwidth in the manuscript, where
# \textwidth = 372 pt = 5.15 in.  A figure authored FIG_W inches wide is
# scaled by 5.15/FIG_W on the page, so a font specified as N pt in matplotlib
# lands at N*5.15/FIG_W pt in print.  set_scale() inverts that: it sets PT so
# that `fontsize=N*PT` renders at N pt, keeping in-figure text at or just
# below the \footnotesize (~8 pt) caption size.
TEXTWIDTH_IN = 5.15
PT = 1.0


def set_scale(fig_w_in, tick=7.0, label=8.0, legend=6.5):
    """Set the render-scale factor and rcParams for a figure FIG_W_IN wide."""
    global PT
    PT = fig_w_in / TEXTWIDTH_IN
    plt.rcParams.update({
        "axes.labelsize": label * PT,
        "xtick.labelsize": tick * PT,
        "ytick.labelsize": tick * PT,
        "legend.fontsize": legend * PT,
        "font.size": tick * PT,
    })
    return PT


def panel_letter(ax, letter):
    ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=9 * PT)

means, scans = load_data()
d = build_fit_data(means)
pl = fit_power_law(scans)

sol4, p4, chi2_4, dof_4, _ = fit_pdm(d, 4)
# M5's own optimum: C_bl is driven to ~0 by the 3-point data (same chi2 as M4), but at
# a tiny nonzero value it implies a finite (very large) steady state -- L_bl_steady_state
# below -- so the two branches are numerically indistinguishable on the 72-480 h data
# yet asymptote differently (M4 unbounded log growth vs M5 saturating at L_bl,ss).
sol5, p5, chi2_5, dof_5, _ = fit_pdm(d, 5)
print(f"M5 steady state L_bl,ss = {L_bl_steady_state(p5):.1f} nm")

# ── Figure 1: fits vs data ───────────────────────────────────────────────────
t_fine = np.linspace(0.1, 500.0, 400)
t_fine_log = np.logspace(-1, np.log10(500.0), 400)

# Gridspec: 2x2 top (Cr linear / Cr log-log, Fe linear / Fe log-log) plus one
# centered bottom panel (Ni, linear only).
# 9.5 x 11.0 in -> aspect 0.864, so at \textwidth the float is 431 pt tall and
# fits the page with its caption (the earlier 11 x 15 overflowed by 53 pt).
set_scale(9.5)
fig = plt.figure(figsize=(9.5, 11.0))
gs = fig.add_gridspec(4, 4)
axes = [
    [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4])],
    [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4])],
]
ax_res = fig.add_subplot(gs[2, 0:2])
ax_lev = fig.add_subplot(gs[2, 2:4])
ax_ni = fig.add_subplot(gs[3, 0:4])

# Weighted power-law refit coefficients (same objective as M4), if the referee
# statistics have been generated -- makes the model comparison apples-to-apples
# in the figure as well as the table.
import json  # noqa: E402

_stats_file = OUT_DIR / "paper" / "referee_stats.json"
pl_weighted = None
if _stats_file.exists():
    _s = json.loads(_stats_file.read_text())
    pl_weighted = _s["power_law_weighted_refit"]

layer_data = [
    ("Cr", "barrier layer (Cr-rich inner spinel)", L_bl_closed),
    ("Fe", "outer layer (Fe-rich crystals)", L_ol_closed),
]

letters = iter("abcd")
for (el, title, model_fn), (ax_lin, ax_log) in zip(layer_data, axes):
    rows = sorted(means[el])
    t_data = np.array([r[0] for r in rows])
    L_data = np.array([r[1] for r in rows])
    sd_data = np.array([r[2] for r in rows])
    k_pl, n_pl = pl[el]

    for ax, t_grid, xscale in ((ax_lin, t_fine, "linear"), (ax_log, t_fine_log, "log")):
        ax.errorbar(
            t_data, L_data, yerr=sd_data, fmt="ko", ms=6, capsize=3, label="Veile 2024 data"
        )
        ax.plot(t_grid, model_fn(p4, t_grid), "b-", lw=2, label="point defect model (M4)")
        ax.plot(t_grid, k_pl * t_grid**n_pl, "r--", lw=1.5,
                label="power law (published coeffs)")
        if pl_weighted is not None:
            kw, nw = pl_weighted[el]["k"], pl_weighted[el]["n"]
            ax.plot(t_grid, kw * t_grid**nw, color="darkorange", ls=":", lw=1.5,
                    label="power law (weighted refit)")
        ax.set_xscale(xscale)
        if xscale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("t (h)")
        ax.set_ylabel("L (nm)")
        ax.set_title(f"{el} — {title}", fontsize=8 * PT)
        panel_letter(ax, next(letters))
# Single shared legend for the four repeated-entry panels. It used to sit
# inside panel (c), which looked empty in the source PNG but is not: the Fe
# error bars at 168 and 480 h span most of the panel height, so the box hid two
# of the three iron measurements in the typeset figure. A figure-level legend
# above the panels cannot cover data.
_h, _l = axes[0][0].get_legend_handles_labels()
# Two columns, not four: four entries on one row overrun the figure width and
# The outer two get clipped.
fig.legend(_h, _l, fontsize=6.5 * PT, loc="upper center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, 1.0))

# ── Panels (e, f): weighted residuals and chi2 leverage ──────────────────────
# The fit statistic is not distributed evenly over the six data points: the Fe
# layer's scan-to-scan scatter is so large that it carries almost no weight, so
# The M4 optimum -- and therefore the long-time extrapolation that depends on
# The fitted curvature -- is set almost entirely by the three Cr points. Making
# that explicit is the honest way to present a five-parameter fit to six means.
res_r, res_labels, res_colors, res_chi2 = [], [], [], []
for el, model_fn, color in (("Cr", L_bl_closed, "tab:blue"),
                            ("Fe", L_ol_closed, "tab:orange")):
    rows_el = sorted(means[el])
    t_el = np.array([r[0] for r in rows_el])
    L_el = np.array([r[1] for r in rows_el])
    sig_el = np.array([r[2] / np.sqrt(r[3]) for r in rows_el])
    r_el = (model_fn(p4, t_el) - L_el) / sig_el
    for t_i, r_i in zip(t_el, r_el):
        res_r.append(r_i)
        res_labels.append(f"{el}\n{t_i:.0f} h")
        res_colors.append(color)
        res_chi2.append(r_i**2)

x_pos = np.arange(len(res_r))
ax_res.axhspan(-1, 1, color="grey", alpha=0.15, label="$\\pm1\\sigma$")
ax_res.axhline(0, color="k", lw=0.8)
ax_res.bar(x_pos, res_r, color=res_colors, edgecolor="k", lw=0.6)
for xi, ri in zip(x_pos, res_r):
    # 5.5 pt, not 6.5: at the larger size the "+0.15" and "+0.24" labels on the
    # adjacent Cr 480 h and Fe 72 h bars overlap.
    ax_res.text(xi, ri + (0.12 if ri >= 0 else -0.12), f"{ri:+.2f}",
                ha="center", va="bottom" if ri >= 0 else "top", fontsize=5.5 * PT)
ax_res.set_xticks(x_pos)
ax_res.set_xticklabels(res_labels, fontsize=7 * PT)
ax_res.set_ylabel("weighted residual ($\\sigma$)")
ax_res.set_ylim(-3.1, 2.5)
ax_res.set_title("M4 weighted residuals", fontsize=8 * PT)
ax_res.legend(fontsize=6.5 * PT, loc="lower right")
panel_letter(ax_res, "e")

chi2_tot = float(np.sum(res_chi2))
frac = 100 * np.array(res_chi2) / chi2_tot
ax_lev.bar(x_pos, frac, color=res_colors, edgecolor="k", lw=0.6)
for xi, fi in zip(x_pos, frac):
    ax_lev.text(xi, fi + 1.5, f"{fi:.0f}%", ha="center", fontsize=6.5 * PT)
ax_lev.set_xticks(x_pos)
ax_lev.set_xticklabels(res_labels, fontsize=7 * PT)
ax_lev.set_ylabel("share of total $\\chi^2$ (%)")
ax_lev.set_ylim(0, 100)
cr_share = 100 * sum(res_chi2[:3]) / chi2_tot
ax_lev.set_title(f"$\\chi^2$ leverage (Cr: {cr_share:.0f}% of "
                 f"{chi2_tot:.2f})", fontsize=8 * PT)
panel_letter(ax_lev, "f")

# Bottom panel: Ni enrichment zone thickness vs. exposure time, data only -- no
# PDM curve, since the zero-dimensional kinetics explicitly do not model the
# Ni zone as a growing layer (it is treated as an interfacial exclusion
# feature, not a barrier/outer-layer species). Included so this figure
# reproduces all three element traces of Veile et al.'s own Figure 9, not
# just the two the zero-dimensional PDM fits.
rows_ni = sorted(means["Ni"])
t_ni = np.array([r[0] for r in rows_ni])
L_ni = np.array([r[1] for r in rows_ni])
sd_ni = np.array([r[2] for r in rows_ni])
ax_ni.errorbar(t_ni, L_ni, yerr=sd_ni, fmt="ks", ms=6, capsize=3, zorder=3,
               label="Veile 2024 data (population SD)")
ax_ni.axhline(L_ni.mean(), color="grey", ls=":", lw=1.5,
              label=f"measured mean = {L_ni.mean():.1f} nm")
# The constant-mobility transport closure of Section 2.6, fitted to these same
# widths. Its rejection is a statement about SHAPE -- it can only rise, and the
# data do not -- which chi2 = 25.7 alone does not convey and the three fitted
# points do not show either. Hence the dense curve.
_ni_curve = OUT_DIR / "paper" / "ni_zone_curve.npz"
if _ni_curve.exists():
    _nc = np.load(_ni_curve)
    ax_ni.plot(_nc["t_h"], _nc["W_nm"], "b-", lw=2, zorder=2,
               label=("constant-mobility closure, best fit "
                      f"($\\chi^2 = {float(_nc['chi2']):.1f}$ on 1 dof)"))
    ax_ni.set_ylim(0, max(float(_nc["W_nm"].max()), float((L_ni + sd_ni).max())) * 1.55)
ax_ni.set_xlim(0, 500)
ax_ni.set_xlabel("t (h)")
ax_ni.set_ylabel("Ni zone width (nm)")
ax_ni.set_title("Ni — enrichment zone: measured widths are flat, "
                "a constant mobility can only rise", fontsize=8 * PT)
# Upper right, with headroom added above: the panel letter holds the top left,
# and both the data and the rising curve stay in the lower two thirds.
ax_ni.legend(fontsize=6.0 * PT, loc="upper right", framealpha=0.9)
panel_letter(ax_ni, "g")

plt.tight_layout(rect=(0, 0, 1, 0.962))
out1 = OUT_DIR / "plot_fits_vs_data.png"
plt.savefig(out1, dpi=300)
print(f"Saved: {out1}")
print(f"  M4: chi2/dof = {chi2_4:.2f}/{dof_4}")

# ── Figure 2: long-time extrapolation ────────────────────────────────────────
years = np.logspace(-2, 5, 60)  # 0.01 y to 100,000 y -- far enough to show the M4/M5 split
t_years = years * 8766.0

set_scale(7.5)
fig2, ax2 = plt.subplots(figsize=(7.5, 5.2))

# Bootstrap uncertainty band (1000-member deterministic parametric bootstrap,
# generated by make_paper_stats.py) -- propagated through the M4 closed form.
_band_file = OUT_DIR / "paper" / "fig6_band.npz"
if _band_file.exists():
    band = np.load(_band_file)
    yb = band["t_h"] / 8766.0
    ax2.fill_between(yb, band["p2p5"], band["p97p5"], color="b", alpha=0.10,
                     label="M4 bootstrap 68/95% bands (n=1000)")
    ax2.fill_between(yb, band["p16"], band["p84"], color="b", alpha=0.18)

ax2.plot(years, L_bl_closed(p4, t_years), "b-", label="M4 (log growth, $C_{bl}=0$)")
ax2.plot(years, L_bl_closed(p5, t_years), "g--", label="M5 (saturating, $C_{bl}\\to 0^+$)")
ax2.axhline(
    L_bl_steady_state(p5), color="g", ls=":", lw=1,
    label=f"M5 steady state = {L_bl_steady_state(p5):.0f} nm",
)
k_cr, n_cr = pl["Cr"]
ax2.plot(years, k_cr * t_years**n_cr, "r--",
         label="power law (published coeffs)")
if pl_weighted is not None:
    kw, nw = pl_weighted["Cr"]["k"], pl_weighted["Cr"]["n"]
    ax2.plot(years, kw * t_years**nw, color="darkorange", ls=":",
             label="power law (weighted refit)")
ax2.axvspan(72 / 8766.0, 480 / 8766.0, color="grey", alpha=0.2, label="data range (72-480 h)")
# The six measurements everything rests on were represented only by the grey
# band, so the figure showed extrapolations with no visible anchor. Plot the
# three Cr points themselves.
_rows_cr = sorted(means["Cr"])
_t_cr_y = np.array([r[0] for r in _rows_cr]) / 8766.0
ax2.errorbar(_t_cr_y, [r[1] for r in _rows_cr], yerr=[r[2] for r in _rows_cr],
             fmt="ko", ms=4, capsize=2, lw=1, zorder=5, label="Veile 2024 Cr data")
# Headline-prediction marker: 10-year horizon.
ax2.axvline(10.0, color="k", ls="-.", lw=0.8)
ax2.text(10.0 * 1.15, 0.03, "t = 10 y", rotation=90, fontsize=6.5 * PT, va="bottom",
         transform=ax2.get_xaxis_transform())
# The paper's headline claim is the SIZE of the gap at ten years, which the
# reader previously had to estimate by eye off the t = 10 y rule. Annotate it.
_t10 = np.array([10.0 * 8766.0])
_m4_10 = float(L_bl_closed(p4, _t10)[0])
_pl_10 = [float(k_cr * _t10[0] ** n_cr)]
if pl_weighted is not None:
    _pl_10.append(float(pl_weighted["Cr"]["k"] * _t10[0] ** pl_weighted["Cr"]["n"]))
_lo, _hi = min(_pl_10) / _m4_10, max(_pl_10) / _m4_10
ax2.annotate("", xy=(10.0, max(_pl_10)), xytext=(10.0, _m4_10),
             arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
ax2.text(10.0 * 1.9, np.sqrt(_m4_10 * max(_pl_10)),
         f"$\\times${_lo:.1f}–{_hi:.1f}\nat 10 y",
         fontsize=7.0 * PT, va="center", ha="left")

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("time (years)")
ax2.set_ylabel("barrier-layer thickness $L_\\mathrm{bl}$ (nm)")
# Two columns and no frame: the single-column box filled the upper-left quadrant.
ax2.legend(fontsize=6.0 * PT, ncol=2, loc="upper left", frameon=False)
plt.tight_layout()
out2 = OUT_DIR / "plot_long_time.png"
plt.savefig(out2, dpi=300)
print(f"Saved: {out2}")
print(
    f"  at 10 y: M4 = {L_bl_closed(p4, np.array([10 * 8766.0]))[0]:.0f} nm, "
    f"M5 = {L_bl_closed(p5, np.array([10 * 8766.0]))[0]:.0f} nm, "
    f"power law = {k_cr * (10 * 8766.0) ** n_cr:.0f} nm"
)

# ── Figure 3: profile likelihoods (M4) ───────────────────────────────────────
# The internal fit vector is the NATURAL log of each parameter's magnitude
# (baseline_fit.unpack: A_bl = exp(x0), b3 = -exp(x1), ...), so the profile
# grid returned by profile_likelihood is ln(parameter) — labeled accordingly.
names = param_names(4)
PRETTY = {
    "A_bl": "$A_\\mathrm{bl}$ (nm h$^{-1}$)",
    "-b3": "$-b_3$ (nm$^{-1}$)",
    "PBR_eff": "$\\mathrm{PBR}_\\mathrm{eff}$ (–)",
    "L0": "$L_0$ (nm)",
    "L_ol0": "$L_\\mathrm{ol,0}$ (nm)",
    "C_bl": "$C_\\mathrm{bl}$ (nm h$^{-1}$)",
    "-C_x": "$-C_x$ (nm h$^{-1}$)",
}
# 2 x 3 rather than 1 x 5: the single row rendered ~1 in tall at \textwidth,
# which put every tick and axis label near 2 pt on the page.
set_scale(11.5)
fig3, axes3_grid = plt.subplots(2, 3, figsize=(11.5, 7.6))
axes3 = axes3_grid.ravel()
for _ax in axes3[len(names):]:
    _ax.remove()

# Verdict per parameter, from the same artifact the parameter table is built
# from, so the figure and Table 2 cannot drift apart. The verdict IS the result
# of this figure; leaving the reader to derive it from the curve and the
# Delta-chi2 = 4 rule buries it.
VERDICT = {}
_par_csv = OUT_DIR / "paper" / "table1_parameters.csv"
if _par_csv.exists():
    import csv as _csv
    with _par_csv.open() as _fh:
        for _row in _csv.DictReader(_fh):
            VERDICT[_row["param"]] = (_row["identifiability"],
                                      _row["profile_max_dchi2"])
VERDICT_KEY = {"A_bl": "A_bl", "-b3": "b3", "PBR_eff": "PBR_eff",
               "L0": "L0", "L_ol0": "L_ol0"}

for k, (ax, i, name) in enumerate(zip(axes3, range(len(sol4.x)), names)):
    grid, chi2s = profile_likelihood(d, 4, sol4.x, i)
    dchi = chi2s - chi2s.min()
    ax.plot(grid, dchi, "b-")
    ax.axhline(1.0, color="grey", ls="--", lw=1, label="$\\Delta\\chi^2=1$ (1$\\sigma$)")
    ax.axhline(4.0, color="grey", ls=":", lw=1, label="$\\Delta\\chi^2=4$ (2$\\sigma$)")
    ax.axvline(sol4.x[i], color="k", ls="-", lw=0.8, label="M4 optimum")
    ax.set_title(PRETTY.get(name, name), fontsize=8 * PT)
    ax.set_xlabel("$\\ln$(parameter)")
    ax.set_ylabel("$\\Delta\\chi^2$")
    ax.set_ylim(0, 10)
    ax.text(0.04, 0.96, f"({'abcde'[k]})", transform=ax.transAxes,
            fontweight="bold", va="top", fontsize=9 * PT)
    # Second abscissa in the parameter's own units. The fit vector is
    # ln(magnitude), so a reader of the log axis alone cannot recover a value
    # in nm/h or nm from any of these five panels. Ticks are placed by hand:
    # exp() of a linear axis is log-spaced, and the automatic locator crowds
    # four labels into the right-hand third of the panel.
    _sec = ax.secondary_xaxis("top", functions=(np.exp, np.log))
    _lo, _hi = float(np.exp(grid.min())), float(np.exp(grid.max()))
    _ticks = sorted({float(f"{v:.2g}") for v in np.geomspace(_lo * 1.05, _hi * 0.95, 4)})
    _sec.set_xticks(_ticks)
    _sec.set_xticklabels([f"{v:g}" for v in _ticks])
    _sec.tick_params(labelsize=6.0 * PT, pad=1.0)
    # The verdict is the result of this figure. Carrying it in the title, rather
    # than leaving the reader to apply the Delta-chi2 = 4 rule panel by panel,
    # is the difference between showing the analysis and showing its input.
    _v = VERDICT.get(VERDICT_KEY.get(name, ""), None)
    _ttl = PRETTY.get(name, name)
    if _v is not None:
        # Verdict and statistic on separate lines: run together they overflow
        # The panel width and collide with the neighbouring title.
        _ttl += (f"\n{_v[0].replace('unidentifiable', 'not identifiable')}"
                 f"\nmax $\\Delta\\chi^2$ = {float(_v[1]):.1f}")
    ax.set_title(_ttl, fontsize=6.8 * PT, pad=13 * PT, linespacing=1.25)
    if k == 0:
        ax.legend(fontsize=6.5 * PT, loc="lower left")

plt.tight_layout()
out3 = OUT_DIR / "plot_profile_likelihood.png"
plt.savefig(out3, dpi=300)
print(f"Saved: {out3}")
