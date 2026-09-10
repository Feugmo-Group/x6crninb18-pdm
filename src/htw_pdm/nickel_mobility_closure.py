"""The extended solves the nickel mobility closure — a nickel-zone mobility closure of unknown functional form.

The nickel-zone closure tested three closures for the effective nickel mobility -- constant,
growth-flux-coupled, and finite-capacity trapping -- and the identifiability gate
could not choose among them. That is the signature of a question posed the wrong
way round: each closure pre-commits to a functional form for D(t) and then asks
whether four observations prefer it. The quantity actually wanted is the function
itself.

This module carries D(t) as an unknown function rather than an unknown
parameter. A small network represents ln D(t) subject to the moving-frame
transport equation, pinned to D0 at the reference exposure so D0 keeps its
The nickel-zone closure meaning:

    D(t) = D0 * exp( g(t) - g(t_ref) ),    g a neural function of ln t

This is the one construction in the project that a classical solver cannot
supply: least squares can fit any parametric D(t) one writes down, but it cannot
fit a D(t) one has not written down.

The differentiable solver. the nickel-zone closure integrates the moving-frame PDE with
scipy/Radau, which cannot be differentiated through. The PDE is linear in c at
fixed (D(t), v(t)), so a Crank-Nicolson march with one linear solve per step is
both differentiable and exact to second order; `wagner_torch` implements it and
is verified against the the nickel-zone closure Radau solver at matched parameters (check C0).

Zone widths are extracted with a smooth surrogate

    W = integral_0^ximax sigmoid( (c(xi) - c_thr) / eps ) dxi

which converges to the the nickel-zone closure hard threshold-crossing width as eps -> 0 and is
differentiable for eps > 0. Check C0 also verifies this surrogate against the
hard width.

What this can and cannot establish. Four observations cannot determine a free
function; the expected outcome is that many different D(t) fit equally well. The
deliverable is therefore not a fitted curve but the *ensemble*: independently
initialized fits are run and the spread of D(t) is reported as a function of
time. Where the ensemble is narrow, the data determine the mobility; where it
flares, they do not. That map is the honest answer to "what does the nickel
profile tell us about mobility?", and it is only obtainable by letting the
functional form float.

Run:
    python -m htw_pdm.nickel_mobility_closure
    python -m htw_pdm.nickel_mobility_closure --smoke
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from htw_pdm.baseline_fit import load_data  # noqa: E402
from htw_pdm.nickel_wagner import (  # noqa: E402
    A72_MEAN,
    A72_SIG,
    N_XI,
    NI_M,
    PBR_BL,
    PBR_OL,
    RISE,
    T_OBS,
    T_REF,
    XI_MAX,
    dL_bl_dt,
)
from htw_pdm.nickel_wagner import (  # noqa: E402
    model as wagner_model_scipy,
)
from htw_pdm.paths import PAPER_OUT  # noqa: E402
from htw_pdm.physics import Parameters  # noqa: E402

N_STEPS = 200  # CN steps over the 480 h window (A72 converged to 1e-4 relative)
T_START = 1e-3
EPS_W = 0.25  # wt% smoothing width for the differentiable zone width
N_ENSEMBLE = 12
D_FLOOR, D_CEIL = 1e-2, 1e2  # nm^2/h, the the nickel-zone closure bounds


def _tgrid(n_steps: int = N_STEPS) -> torch.Tensor:
    """Log-spaced time nodes: the early transient sets the peak amplitude."""
    return torch.tensor(
        np.geomspace(T_START, float(T_OBS[-1]), n_steps + 1), dtype=torch.float64
    )


def v_of_t(t_h, p1: Parameters, phi: float) -> float:
    """Interface recession speed, nm/h (the nickel-zone closure definition, unchanged)."""
    dbl = dL_bl_dt(float(t_h), p1)
    return dbl / PBR_BL + phi * (p1.PBR_eff * dbl + p1.C_x) / PBR_OL


def wagner_torch(D_of_t, r: torch.Tensor, phi: float, p1: Parameters,
                 t_nodes: torch.Tensor, t_obs=T_OBS):
    """Differentiable Crank-Nicolson march of the moving-frame Wagner PDE.

        dc/dt = D(t) c'' + v(t) c'
        D c'(0) = -v (1 - r) c(0)          (rejection BC, ghost-node form)
        c(xi_max) = c_m                    (far field pinned)

    Returns c(xi) at each observation time, shape (len(t_obs), N_XI).
    """
    dtype = t_nodes.dtype
    xi = torch.linspace(0.0, XI_MAX, N_XI, dtype=dtype)
    h = float(xi[1] - xi[0])
    I = torch.eye(N_XI, dtype=dtype)

    # Interior second-difference and centred first-difference operators.
    L2 = torch.zeros(N_XI, N_XI, dtype=dtype)
    L1 = torch.zeros(N_XI, N_XI, dtype=dtype)
    idx = torch.arange(1, N_XI - 1)
    L2[idx, idx - 1] = 1.0 / h**2
    L2[idx, idx] = -2.0 / h**2
    L2[idx, idx + 1] = 1.0 / h**2
    L1[idx, idx - 1] = -1.0 / (2 * h)
    L1[idx, idx + 1] = 1.0 / (2 * h)

    def A_of(t):
        """Spatial operator at time t, with the rejection BC folded into row 0.

        Ghost node c_{-1} = c_1 + 2h*rej/D with rej = v (1-r) c_0. Substituting
        into the centred stencils at node 0 gives

            dc_0/dt = 2D (c_1 - c_0)/h^2 + (2/h) rej - (v/D) rej,

        so the coefficient on c_0 is -2D/h^2 + 2v(1-r)/h - v^2(1-r)/D.
        """
        Dt, vt = D_of_t(t), v_of_t(t, p1, phi)
        A = (Dt * L2 + vt * L1).clone()
        coup = vt * (1.0 - r)
        A[0] = torch.zeros(N_XI, dtype=dtype)
        A[0, 1] = 2.0 * Dt / h**2
        A[0, 0] = -2.0 * Dt / h**2 + (2.0 / h) * coup - vt * coup / Dt
        return A

    c = torch.full((N_XI,), NI_M, dtype=dtype)
    out, k = [], 0
    for j in range(len(t_nodes) - 1):
        t0, t1 = t_nodes[j], t_nodes[j + 1]
        dt = t1 - t0
        A0, A1 = A_of(float(t0)), A_of(float(t1))
        M = I - 0.5 * dt * A1
        b = (I + 0.5 * dt * A0) @ c
        M = M.clone()
        M[-1] = I[-1]
        b = b.clone()
        b[-1] = torch.as_tensor(NI_M, dtype=dtype)
        c = torch.linalg.solve(M, b)
        while k < len(t_obs) and float(t1) >= t_obs[k] - 1e-9:
            out.append(c)
            k += 1
    while len(out) < len(t_obs):
        out.append(c)
    return torch.stack(out), xi


def soft_width(c: torch.Tensor, xi: torch.Tensor, eps: float = EPS_W) -> torch.Tensor:
    """Differentiable zone width: smoothed measure of {xi : c(xi) > c_m + RISE}."""
    ind = torch.sigmoid((c - (NI_M + RISE)) / eps)
    return torch.trapz(ind, xi)


class LogDNet(nn.Module):
    """g(ln t) -> ln-mobility offset; D(t) = D0 * exp(g(t) - g(t_ref))."""

    def __init__(self, hidden: int = 16, n_layers: int = 3, dtype=torch.float64):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.to(dtype)

    def g(self, t_h) -> torch.Tensor:
        u = (torch.log(torch.as_tensor(t_h, dtype=torch.float64)
                       .clamp_min(1e-4)) - np.log(T_REF)) / np.log(10.0)
        return self.net(u.reshape(-1, 1)).squeeze(-1).squeeze()


def c0_verification(p1: Parameters) -> dict:
    """The differentiable solver and the smooth width must match the nickel-zone closure.

    Compares `wagner_torch` at constant D against the the nickel-zone closure scipy/Radau
    integrator at the same parameters, on both the hard and the smooth width.
    """
    theta = np.array([2.0, 0.30, 0.30])
    W_ref, A_ref = wagner_model_scipy(theta, p1, "const")
    t_nodes = _tgrid()
    c, xi = wagner_torch(lambda t: torch.tensor(theta[0], dtype=torch.float64),
                         torch.tensor(theta[1], dtype=torch.float64),
                         float(theta[2]), p1, t_nodes)
    W_hard = []
    for j in range(len(T_OBS)):
        above = (c[j] > NI_M + RISE).nonzero()
        W_hard.append(float(xi[above[-1]]) if len(above) else 0.0)
    W_soft = [float(soft_width(c[j], xi)) for j in range(len(T_OBS))]
    return {
        "theta": theta.tolist(),
        "W_scipy": W_ref.tolist(),
        "W_torch_hard": W_hard,
        "W_torch_soft": W_soft,
        "A72_scipy": float(A_ref),
        "A72_torch": float(c[0, 0]),
        "rel_err_W_hard": float(np.max(np.abs(np.array(W_hard) - W_ref)
                                       / np.maximum(W_ref, 1e-9))),
        "rel_err_A72": float(abs(float(c[0, 0]) - A_ref) / A_ref),
    }


def fit_once(p1, W_obs, W_sig, t_nodes, seed: int, n_steps: int,
             smooth_pen: float = 1e-3) -> dict:
    """One neural-closure fit from an independent random initialization."""
    torch.manual_seed(seed)
    net = LogDNet()
    # Disperse the starting mobility across the the nickel-zone closure bounds so the ensemble
    # probes the basin structure rather than re-running one basin N times.
    rng = np.random.default_rng(seed)
    D0_init = float(np.exp(rng.uniform(np.log(0.3), np.log(20.0))))
    log_D0 = torch.tensor(np.log(D0_init), dtype=torch.float64, requires_grad=True)
    logit_r = torch.tensor(np.log(0.3 / 0.7), dtype=torch.float64,
                           requires_grad=True)
    logit_phi = torch.tensor(np.log(0.3 / 0.7), dtype=torch.float64,
                             requires_grad=True)
    params = list(net.parameters()) + [log_D0, logit_r, logit_phi]
    opt = torch.optim.Adam(params, lr=5e-3)

    t_probe = torch.tensor(np.geomspace(1.0, float(T_OBS[-1]), 24),
                           dtype=torch.float64)
    g_ref = None

    def D_of(t):
        val = torch.exp(log_D0) * torch.exp(net.g(t) - g_ref)
        return val.clamp(D_FLOOR, D_CEIL)

    last = {}
    for step in range(n_steps):
        opt.zero_grad()
        g_ref = net.g(T_REF)
        r = torch.sigmoid(logit_r)
        phi = float(torch.sigmoid(logit_phi).detach())
        c, xi = wagner_torch(D_of, r, phi, p1, t_nodes)
        W = torch.stack([soft_width(c[j], xi) for j in range(len(T_OBS))])
        A72 = c[0, 0]
        chi2 = (((W - torch.tensor(W_obs)) / torch.tensor(W_sig)) ** 2).sum() \
            + ((A72 - A72_MEAN) / A72_SIG) ** 2
        # Smoothness prior on the free function: penalize curvature of g in ln t.
        gp = net.g(t_probe)
        curv = ((gp[2:] - 2 * gp[1:-1] + gp[:-2]) ** 2).mean()
        (chi2 + smooth_pen * curv).backward()
        opt.step()
        last = {"chi2": float(chi2), "curv": float(curv)}

    with torch.no_grad():
        g_ref = net.g(T_REF)
        D_curve = [float(D_of(float(t))) for t in t_probe]
        r = float(torch.sigmoid(logit_r))
        phi = float(torch.sigmoid(logit_phi))
        c, xi = wagner_torch(D_of, torch.tensor(r, dtype=torch.float64), phi,
                             p1, t_nodes)
        W_hard = []
        for j in range(len(T_OBS)):
            above = (c[j] > NI_M + RISE).nonzero()
            W_hard.append(float(xi[above[-1]]) if len(above) else 0.0)
        chi2_hard = float(
            (((np.array(W_hard) - W_obs) / W_sig) ** 2).sum()
            + ((float(c[0, 0]) - A72_MEAN) / A72_SIG) ** 2
        )
    return {
        "seed": seed, "chi2_soft": last["chi2"], "chi2_hard": chi2_hard,
        "D0": float(torch.exp(log_D0)), "r_Ni": r, "phi_ol_supply": phi,
        "t_probe": t_probe.tolist(), "D_curve": D_curve,
        "W_hard": W_hard, "A72": float(c[0, 0]),
    }


def main(smoke: bool = False) -> dict:
    torch.set_default_dtype(torch.float64)
    p1 = Parameters()

    res = {"C0_verification": c0_verification(p1)}
    v = res["C0_verification"]
    print(f"[C0] torch vs scipy: hard-width rel err {v['rel_err_W_hard']:.3e}, "
          f"A72 rel err {v['rel_err_A72']:.3e}")
    print(f"     W scipy {np.round(v['W_scipy'],2)}  torch hard "
          f"{np.round(v['W_torch_hard'],2)}  torch soft "
          f"{np.round(v['W_torch_soft'],2)}")

    means, _ = load_data()
    ni = {r[0]: (r[1], r[2], r[3]) for r in means["Ni"]}
    W_obs = np.array([ni[t][0] for t in T_OBS])
    W_sig = np.array([ni[t][1] for t in T_OBS])
    print(f"[C1] Ni zone widths {W_obs} +- {W_sig} nm")

    n_steps = 60 if smoke else 700
    n_ens = 3 if smoke else N_ENSEMBLE
    fits = []
    for s in range(n_ens):
        f = fit_once(p1, W_obs, W_sig, _tgrid(), seed=s, n_steps=n_steps)
        fits.append(f)
        print(f"[C2] seed {s}: chi2(soft) {f['chi2_soft']:.3f}  "
              f"chi2(hard) {f['chi2_hard']:.3f}  D0 {f['D0']:.3f}  "
              f"r_Ni {f['r_Ni']:.3f}  W {np.round(f['W_hard'],1)}")

    # Identifiability of the free function: spread of D(t) across the ensemble,
    # restricted to fits that actually reached an acceptable chi^2.
    good = [f for f in fits if f["chi2_hard"] < 4.0 * len(W_obs)]
    curves = np.array([f["D_curve"] for f in good]) if good else np.zeros((0, 24))
    t_probe = np.array(fits[0]["t_probe"])
    band = []
    if len(curves) >= 2:
        lo, hi = curves.min(axis=0), curves.max(axis=0)
        med = np.median(curves, axis=0)
        for i, t in enumerate(t_probe):
            band.append({"t_h": float(t), "D_lo": float(lo[i]),
                         "D_med": float(med[i]), "D_hi": float(hi[i]),
                         "spread_decades": float(np.log10(max(hi[i], 1e-30)
                                                          / max(lo[i], 1e-30)))})
    res.update({
        "W_obs": W_obs.tolist(), "W_sig": W_sig.tolist(),
        "n_ensemble": n_ens, "n_steps": n_steps,
        "fits": fits, "n_acceptable": len(good), "D_band": band,
    })
    if band:
        sp = [b["spread_decades"] for b in band]
        i_min = int(np.argmin(sp))
        res["identifiability"] = {
            "min_spread_decades": float(np.min(sp)),
            "max_spread_decades": float(np.max(sp)),
            "t_best_constrained_h": float(t_probe[i_min]),
            "verdict": ("determined" if np.max(sp) < 0.5 else
                        "partially determined" if np.min(sp) < 0.5 else
                        "undetermined"),
        }
        print(f"[C3] {len(good)}/{n_ens} acceptable fits; D(t) ensemble spread "
              f"{np.min(sp):.2f}-{np.max(sp):.2f} decades, narrowest at "
              f"t = {t_probe[i_min]:.0f} h -> {res['identifiability']['verdict']}")

    out_dir = str(PAPER_OUT)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "nickel_mobility_closure.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"[the nickel mobility closure] wrote {path}")
    return res


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
