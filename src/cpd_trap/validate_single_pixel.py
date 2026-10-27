"""
validate_single_pixel.py

Single-pixel checks for cpd_trap_model.py, run before any 2-D study:

1. Square-pulse calibration (spec Eq. 3): I_P = 3 uA for 2.5 ms, then 18 ms
   of darkness, T = i_R = z = 0 initially. Reproduces the table's ON/OFF
   peaks and rho_square for the zero-trap row and the four trap rows.
   u is 0/1 here for CALIBRATION ONLY.
2. One Gaussian pass (sigma = 2 px, v = 2 px/ms, pixel crossed through its
   center): rho_G = |min i_R| / max i_R, for all four Task A electrical
   pairs and the four trap conditions.
3. Timestep convergence: dt = 2 us vs 1 us on the Gaussian pass.

Writes outputs/cpd_trap/single_pixel_validation.png and
outputs/cpd_trap/single_pixel_summary.txt.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cpd_trap_model import (CPDArray, CircuitParams, TASK_A_PAIRS,
                            TASK_B_TRAPS, THETA)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "cpd_trap")
os.makedirs(OUT_DIR, exist_ok=True)

SIGMA, V = 2.0, 0.002          # px, px/us
REF = TASK_A_PAIRS["A-reference"]


def run_pixel(u_of_t, t_end, dt, circuit, traps=None, piecewise=False):
    """Integrate one pixel. If piecewise, u is held at its step-start value."""
    px = CPDArray((1, 1), circuit, traps)
    n = int(round(t_end / dt))
    t = np.arange(n + 1) * dt
    i_R = np.zeros(n + 1)
    for k in range(n):
        if piecewise:
            u_k = np.full((1, 1), u_of_t(t[k]))
            u_fn = lambda _t, u_k=u_k: u_k
        else:
            u_fn = lambda tt: np.full((1, 1), u_of_t(tt))
        i_R[k + 1] = px.step(u_fn, t[k], dt)[0, 0]
    return t, i_R


def square_pulse(t_on=2500.0, t_dark=18000.0, dt=2.0, traps=None):
    t, i = run_pixel(lambda tt: 1.0 if tt < t_on else 0.0, t_on + t_dark, dt,
                     REF, traps, piecewise=True)
    on_pk = i[t <= t_on].min()
    off_pk = i[t > t_on].max()
    return t, i, on_pk, off_pk, abs(on_pk) / off_pk


def gaussian_u(tt, start=-4 * SIGMA):
    """u at pixel x=0 while the spot center moves from `start` at v."""
    x = start + V * tt
    return np.exp(-x * x / (2 * SIGMA ** 2))


def gaussian_pass(circuit, traps=None, dt=2.0, t_end=20000.0):
    t, i = run_pixel(gaussian_u, t_end, dt, circuit, traps)
    return t, i, i.min(), i.max(), abs(i.min()) / i.max()


def main():
    lines = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    lines.append("1) Square-pulse calibration (2.5 ms ON, 18 ms dark, dt = 2 us)")
    lines.append(f"{'condition':<16}{'ON peak (uA)':>14}{'OFF peak (uA)':>15}{'rho_sq':>9}"
                 f"{'spec ON':>10}{'spec OFF':>10}{'spec rho':>9}  check")
    # peaks and ratios exactly as printed in the task file's calibration table
    spec_sq = {"zero-trap": (-0.44007, 0.44007, 1.000),
               "B-trap-sq1.0": (-0.32007, 0.32241, 0.993),
               "B-trap-sq0.7": (-0.25413, 0.36591, 0.695),
               "B-trap-sq0.5": (-0.18636, 0.37584, 0.496),
               "B-trap-sq0.2": (-0.07458, 0.37605, 0.198)}
    rows = [("zero-trap", None)] + [(k, p) for k, (p, _) in TASK_B_TRAPS.items()]
    n_fail = 0
    for name, traps in rows:
        t, i, on_pk, off_pk, rho = square_pulse(traps=traps)
        s_on, s_off, s_rho = spec_sq[name]
        ok = abs(on_pk - s_on) < 5e-5 and abs(off_pk - s_off) < 5e-5 and abs(rho - s_rho) < 1e-3
        n_fail += not ok
        lines.append(f"{name:<16}{on_pk:>14.5f}{off_pk:>15.5f}{rho:>9.3f}"
                     f"{s_on:>10.5f}{s_off:>10.5f}{s_rho:>9.3f}  {'PASS' if ok else 'FAIL'}")
        axes[0].plot(t / 1000, i, label=f"{name} (rho_sq={rho:.3f})", lw=1)
    axes[0].set_xlim(0, 12)
    axes[0].set_title("Square-pulse calibration (u = 0/1, calibration only)")
    axes[0].set_xlabel("time (ms)"); axes[0].set_ylabel("i_R (uA)")
    axes[0].legend(fontsize=7)

    lines.append("")
    lines.append("2) One Gaussian pass through pixel center (sigma=2 px, v=2 px/ms, dt=2 us)")
    lines.append(f"{'condition':<16}{'ON peak (nA)':>14}{'OFF peak (nA)':>15}{'rho_G':>8}{'spec':>8}"
                 f"{'peak/theta':>12}  check")
    spec_g = {"A-fast": 1.000, "A-reference": 1.004, "A-slow": 1.047, "A-very-slow": 1.334,
              "B-trap-sq1.0": 1.12, "B-trap-sq0.7": 1.02, "B-trap-sq0.5": 1.05, "B-trap-sq0.2": 1.27}
    gauss_rows = [(k, c, None) for k, c in TASK_A_PAIRS.items()] + \
                 [(k, REF, p) for k, (p, _) in TASK_B_TRAPS.items()]
    for name, circ, traps in gauss_rows:
        t, i, on_pk, off_pk, rho = gaussian_pass(circ, traps)
        weakest = min(abs(on_pk), off_pk)
        # spec prints Task A ratios to 3 decimals and Task B ratios to 2
        ok = abs(rho - spec_g[name]) <= (0.0005 if name.startswith("A") else 0.005)
        n_fail += not ok
        lines.append(f"{name:<16}{on_pk * 1e3:>14.2f}{off_pk * 1e3:>15.2f}{rho:>8.3f}"
                     f"{spec_g[name]:>8.3f}{weakest / THETA:>12.2f}  {'PASS' if ok else 'FAIL'}")
        ax = axes[1] if traps is None else axes[2]
        ax.plot(t / 1000, i * 1e3, label=f"{name} (rho_G={rho:.3f})", lw=1)
    for ax, title in ((axes[1], "Task A: Gaussian pass, electrical pairs"),
                      (axes[2], "Task B: Gaussian pass, reference circuit + traps")):
        ax.axhline(THETA * 1e3, color="k", ls=":", lw=0.8)
        ax.axhline(-THETA * 1e3, color="k", ls=":", lw=0.8)
        ax.set_xlim(0, 10); ax.set_title(title)
        ax.set_xlabel("time (ms)"); ax.set_ylabel("i_R (nA)"); ax.legend(fontsize=7)
    t_u = np.linspace(0, 10000, 500)
    ax2 = axes[1].twinx()
    ax2.plot(t_u / 1000, gaussian_u(t_u), color="0.7", ls="--", lw=1)
    ax2.set_ylabel("u (grey dashed)")

    lines.append("")
    lines.append("3) Timestep check, Gaussian pass, reference + trap-sq0.2")
    for name, traps in (("A-reference", None), ("B-trap-sq0.2", TASK_B_TRAPS["B-trap-sq0.2"][0])):
        _, _, on2, off2, r2 = gaussian_pass(REF, traps, dt=2.0)
        _, _, on1, off1, r1 = gaussian_pass(REF, traps, dt=1.0)
        lines.append(f"{name:<16} dt=2us rho_G={r2:.5f}  dt=1us rho_G={r1:.5f}  "
                     f"max peak change={max(abs(on2 - on1), abs(off2 - off1)) * 1e6:.3f} pA")

    lines.append("")
    lines.append(f"SPEC CHECK: {13 - n_fail}/13 rows match the task file "
                 f"({'ALL PASS' if n_fail == 0 else f'{n_fail} FAIL'})")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "single_pixel_validation.png"), dpi=130)
    plt.close(fig)
    text = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "single_pixel_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
