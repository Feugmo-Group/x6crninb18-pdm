"""Layer thicknesses from the raw EDX line scans supplied by Veile et al.

Input: one CSV per line scan, exported from AISI347_TEM_Linescans.opju into
data/raw/veile_linescans/, with columns "Distance (nm)" and element signals
(at least Cr, Fe, Ni; O, Pt, Nb are used if present).  The exposure time is read
from the file name: 3Tage -> 72 h, 7Tage -> 168 h, 20Tage -> 480 h, 100Tage -> 2496 h (104 days, near-stagnant autoclave; not used in fits).
A file may also carry several scans side by side ("Distance (nm)", "Cr", ...,
"Distance (nm).1", "Cr.1", ...); each block is treated as one scan.

Layer width: the signals are first converted to cation fractions,
c_X = X / (Cr + Fe + Ni) in percent, so that the oxygen in the oxide does not
dilute them, and smoothed over SMOOTH points.  The Cr layer is the contiguous run
around the Cr maximum in the oxide where c_Cr exceeds its metal value (median of
the last 10% of the scan) by CR_RISE points; the Ni zone is the same rule around
the Ni maximum on the metal side of the Cr layer with NI_RISE.  Runs are clipped
at the surface (Pt below 50 wt%).  The two thresholds were chosen with --calibrate,
which uses only the 72-480 h scans and the per-scan widths read from Veile et al.
Fig. 9; the 2400 h scans play no part in the choice.

Writes data/veile_raw_scans.csv and data/veile_raw_means.csv in the schema of
the veile2024_fig9_* files, so htw_pdm.baseline_fit.load_data("veile_raw") reads them.

Run:  python scripts/extract_layer_thickness.py [--check] [--calibrate]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from htw_pdm.paths import DATA

RAW = DATA / "raw" / "veile_linescans"
DAYS_TO_H = {3: 72.0, 7: 168.0, 20: 480.0, 100: 2496.0}  # "100 Tage" folder: 104 days (Veile, pers. comm.)
ELEMENTS = ("Cr", "Fe", "Ni")
CR_RISE = 1.5  # cation-% points above the metal value (from --calibrate)
NI_RISE = 1.0
SMOOTH = 7  # points in the moving average of the cation fractions
O_RISE = 5.0  # wt% O above metal baseline marks oxide


def exposure_hours(name: str) -> float:
    m = re.search(r"(\d+)\s*Tage", name, re.I)
    if not m or int(m.group(1)) not in DAYS_TO_H:
        raise ValueError(f"cannot read exposure time from {name!r}")
    return DAYS_TO_H[int(m.group(1))]


def read_blocks(path: Path):
    """Yield (distance, {element: signal}) for each scan block in one CSV."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    # Origin exports may put long name / units / comments in the first rows; the
    # header is the first row that contains a distance column.
    h = next(i for i, r in enumerate(rows) if any("distance" in c.lower() or "position" in c.lower() for c in r))
    head = [c.strip() for c in rows[h]]
    data = []
    for r in rows[h + 1:]:
        try:
            data.append([float(c) if c.strip() else np.nan for c in r[:len(head)]])
        except ValueError:
            continue  # unit or comment row
    a = np.array(data, float)
    dist_cols = [j for j, c in enumerate(head) if "distance" in c.lower() or "position" in c.lower()]
    for k, j0 in enumerate(dist_cols):
        j1 = dist_cols[k + 1] if k + 1 < len(dist_cols) else len(head)
        sig = {}
        for j in range(j0 + 1, j1):
            name = re.split(r"[\s_.(]", head[j].replace("K", " ").strip())[0]
            for el in ELEMENTS + ("O", "Pt", "Nb"):
                if name.lower() == el.lower() or head[j].lower().startswith(el.lower() + " "):
                    sig[el] = a[:, j]
        x = a[:, j0]
        ok = np.isfinite(x)
        yield x[ok], {el: s[ok] for el, s in sig.items()}


def orient(x, sig):
    """Return (depth, signals) with the surface at depth 0 and the metal at the end.

    The metal end is the one with no oxygen and alloy composition (Fe ~70 wt%);
    the Pt cap alone is not enough because some scans start inside the oxide.
    """
    n = len(x)
    k = max(5, n // 12)
    def metal_score(sl):
        return sig["Fe"][sl].mean() - 3.0 * sig["O"][sl].mean() - sig.get("Pt", np.zeros(n))[sl].mean()
    if metal_score(slice(0, k)) > metal_score(slice(n - k, n)):
        x, sig = x[::-1], {e: v[::-1] for e, v in sig.items()}
    order = np.argsort(np.abs(x - x[0]))
    d = np.abs(x - x[0])[order]
    return d, {e: v[order] for e, v in sig.items()}


def _run_around(mask, i):
    lo = hi = i
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    while hi < len(mask) - 1 and mask[hi + 1]:
        hi += 1
    return lo, hi


def cation_fraction(s, el, k=SMOOTH):
    met = s["Cr"] + s["Fe"] + s["Ni"]
    c = np.where(met > 20.0, 100.0 * s[el] / np.maximum(met, 1e-9), 0.0)
    return np.convolve(c, np.ones(k) / k, mode="same")


def widths(x, sig, cr_rise=CR_RISE, ni_rise=NI_RISE, o_rise=O_RISE):
    """Cr layer, Fe layer and Ni zone widths (nm) of one scan.

    Cr layer and Ni zone: see the module docstring.  Fe layer: total length of
    oxide (O above its metal baseline by `o_rise` wt%) between the surface and the
    outer edge of the Cr layer.
    """
    d, s = orient(x, sig)
    n = len(d)
    tail = slice(int(0.9 * n), n)
    pt = s.get("Pt", np.zeros(n))
    surf = int(np.argmax(pt < 50.0)) if (pt >= 50.0).any() else 0
    oxide = s["O"] >= float(np.median(s["O"][tail])) + o_rise
    ox_idx = np.where(oxide)[0]
    last_ox = int(ox_idx.max()) if len(ox_idx) else n - 1
    out = {}
    c = cation_fraction(s, "Cr")
    mask = c >= float(np.median(c[tail])) + cr_rise
    mask[:surf] = False
    i_cr = surf + int(np.argmax(c[surf:min(n, last_ox + 30)]))
    lo_cr, hi_cr = _run_around(mask, i_cr)
    out["Cr"] = float(d[hi_cr] - d[lo_cr])
    step = float(np.median(np.diff(d)))
    out["Fe"] = float(oxide[surf:lo_cr].sum() * step)
    c = cation_fraction(s, "Ni")
    mask = c >= float(np.median(c[tail])) + ni_rise
    mask[:surf] = False
    i_ni = lo_cr + int(np.argmax(c[lo_cr:min(n, hi_cr + 60)]))
    lo_ni, hi_ni = _run_around(mask, i_ni)
    out["Ni"] = float(d[hi_ni] - d[lo_ni])
    return out


def read_reference():
    ref = {}
    with open(DATA / "veile2024_fig9_scans.csv") as f:
        for r in csv.DictReader(x for x in f if not x.startswith("#")):
            ref.setdefault((float(r["t_h"]), r["element"]), []).append(float(r["L_nm"]))
    return ref


def calibrate(files):
    """Sum of squared differences between sorted per-scan widths and Fig. 9, 72-480 h only."""
    ref = read_reference()
    scans = [(exposure_hours(f.name), x, sig) for f in files for x, sig in read_blocks(f)]
    scans = [s for s in scans if s[0] <= 480.0]
    for el, rises in (("Cr", (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)), ("Ni", (0.5, 1.0, 1.5, 2.0, 3.0))):
        for r in rises:
            kw = {"cr_rise": r} if el == "Cr" else {"ni_rise": r}
            by = {}
            for t, x, sig in scans:
                by.setdefault(t, []).append(widths(x, sig, **kw)[el])
            ss = sum(float(np.sum((np.sort(by[t]) - np.sort(ref[(t, el)])) ** 2)) for t in by)
            print(f"  {el} rise {r:4.1f}  SS {ss:7.0f} nm^2  "
                  + "  ".join(f"{int(t)} h {[round(v) for v in by[t]]}" for t in sorted(by)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true",
                    help="threshold sweep against Fig. 9 per-scan widths (72-480 h only)")
    ap.add_argument("--check", action="store_true",
                    help="compare the 72/168/480 h means with the digitized Fig. 9 means")
    a = ap.parse_args()
    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV files in {RAW}; export the Origin workbooks first")
    if a.calibrate:
        calibrate(files)
        return

    rows = []
    for f in files:
        t = exposure_hours(f.name)
        for k, (x, sig) in enumerate(read_blocks(f)):
            w = widths(x, sig)
            for el in ELEMENTS:
                rows.append((t, el, w[el], f"{f.stem}#{k}"))
            print(f"  {f.stem:45s} Cr {w['Cr']:6.1f}  Fe {w['Fe']:6.1f}  Ni {w['Ni']:6.1f}")
    with open(DATA / "veile_raw_scans.csv", "w", newline="") as fo:
        fo.write(f"# Layer widths from raw EDX line scans (Veile et al., supplied by the authors);\n"
                 f"# criterion: cation fraction X/(Cr+Fe+Ni) above its metal value by {CR_RISE} (Cr) / {NI_RISE} (Ni) points;\n"
                 f"# Fe = oxide length outside the Cr layer\n")
        w = csv.writer(fo)
        w.writerow(["t_h", "element", "L_nm", "source"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.1f}", r[3]])
    means = []
    for t in sorted({r[0] for r in rows}):
        for el in ELEMENTS:
            L = np.array([r[2] for r in rows if r[0] == t and r[1] == el])
            if len(L):
                means.append((t, el, L.mean(), L.std(ddof=1) if len(L) > 1 else 0.0, len(L)))
    with open(DATA / "veile_raw_means.csv", "w", newline="") as fo:
        fo.write("# Means +- sample SD (ddof = 1) over raw EDX line scans (see veile_raw_scans.csv)\n")
        w = csv.writer(fo)
        w.writerow(["t_h", "element", "mean_nm", "sd_nm", "n_scans"])
        for m in means:
            w.writerow([int(m[0]), m[1], f"{m[2]:.2f}", f"{m[3]:.2f}", m[4]])
    for m in means:
        print(f"  {m[0]:6.0f} h {m[1]:2s}  {m[2]:7.1f} +- {m[3]:6.1f} nm  (n = {m[4]})")

    if a.check:
        ref = {}
        with open(DATA / "veile2024_fig9_means.csv") as f:
            for r in csv.DictReader(x for x in f if not x.startswith("#")):
                ref[(float(r["t_h"]), r["element"])] = (float(r["mean_nm"]), float(r["sd_nm"]))
        print("check against digitized Fig. 9 means (difference in units of the digitized SD):")
        for t, el, mu, sd, n in means:
            if (t, el) in ref:
                m0, s0 = ref[(t, el)]
                print(f"  {t:6.0f} h {el:2s}  raw {mu:7.1f}  fig9 {m0:7.1f}  "
                      f"diff {(mu - m0) / max(s0, 1e-9):+.2f} SD")


if __name__ == "__main__":
    main()
