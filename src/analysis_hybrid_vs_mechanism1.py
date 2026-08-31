"""
analysis_hybrid_vs_mechanism1.py

Direct test of "can the hybrid beat Mechanism 1's own tracing accuracy?"
This script exists specifically to keep an honest NEGATIVE RESULT in the
repo, not just wins.

A naive comparison (just pick each model's compare_models.py q_thresh and
measure RMSE) makes the Hybrid look like it wins, because it happens to
fire more events at those particular thresholds -- and firing more events
(denser temporal sampling of the same moving spot) lowers RMSE on its own,
regardless of the underlying event-generation physics. That is exactly the
"comparing models at wildly different event rates would confound results"
trap this repo's own docs warn about (see compare_models.py, README.md
section 4) -- it is easy to fall into even when you know to watch for it.

This script controls for that properly: for a series of TARGET event
counts, it bisection-searches each model's q_thresh until both fire
(approximately) the same number of events on the identical stimulus, then
compares RMSE only at matched density. See device_hybrid.py's module
docstring for the full story and what we found instead (a genuine, but
different, advantage under a fixed, un-retuned threshold -- see
analysis_fixed_threshold_range.py).
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from spot_simulator import SpotSimulator
from device_mechanism1 import Mechanism1DualBranch
from device_hybrid import HybridModel
from memory_decode import MemoryDecoder

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def rmse_ignoring_nan(a, b):
    mask = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
    if mask.sum() == 0:
        return np.nan, 0.0
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))), float(mask.mean())


def event_count(make_model, q, L_video):
    return len(make_model(q).run(L_video))


def find_q_for_target(make_model, target, qlo, qhi, L_video, tol=0.03, max_iter=25):
    """Bisection on q_thresh (monotonic: lower q -> more events)."""
    qmid = (qlo + qhi) / 2
    for _ in range(max_iter):
        qmid = (qlo + qhi) / 2
        n = event_count(make_model, qmid, L_video)
        if n == 0:
            qhi = qmid
            continue
        if abs(n - target) / target < tol:
            return qmid, n
        if n > target:
            qlo = qmid
        else:
            qhi = qmid
    return qmid, n


def eval_at_q(make_model, q, L_video, grid_size, dt, n_steps, gt_path, tau=0.005):
    ev = make_model(q).run(L_video)
    dec = MemoryDecoder(grid_size, dt, tau=tau, kappa=1.0)
    traced, _ = dec.run(ev, n_steps)
    rmse, cov = rmse_ignoring_nan(traced, gt_path)
    return len(ev), rmse, cov


def main():
    sim = SpotSimulator(grid_size=24, n_steps=3000, dt=1e-3,
                         path="circle", radius=7.0, period=1.0)
    L_video = sim.generate()
    gt_path = np.stack([sim.cx, sim.cy], axis=1)
    dt, grid_size, n_steps = sim.dt, sim.grid_size, sim.n_steps

    mech1_maker = lambda q: Mechanism1DualBranch(grid_size, dt, tau_slow=5e-3, R=1.0, q_thresh=q)
    hybrid_maker = lambda q: HybridModel(grid_size, dt, tau_slow=5e-3, R=1.0,
                                          alpha=0.4, x0=0.006, q_thresh=q)

    targets = [3000, 6000, 10000, 15000, 20000]
    rows = []
    for target in targets:
        q1, n1 = find_q_for_target(mech1_maker, target, 0.0002, 0.02, L_video)
        q2, n2 = find_q_for_target(hybrid_maker, target, 0.0002, 0.02, L_video)
        _, r1, c1 = eval_at_q(mech1_maker, q1, L_video, grid_size, dt, n_steps, gt_path)
        _, r2, c2 = eval_at_q(hybrid_maker, q2, L_video, grid_size, dt, n_steps, gt_path)
        rows.append((target, n1, r1, c1, n2, r2, c2))

    header = f"{'target N':<10}{'Mech1 N':<10}{'Mech1 RMSE':<13}{'Hybrid N':<10}{'Hybrid RMSE':<13}{'winner'}"
    lines = [header]
    for target, n1, r1, c1, n2, r2, c2 in rows:
        winner = "Hybrid" if r2 < r1 else "Mech1"
        margin = abs(r2 - r1) / max(r1, r2) * 100
        lines.append(f"{target:<10}{n1:<10}{r1:<13.3f}{n2:<10}{r2:<13.3f}"
                     f"{winner} (by {margin:.1f}%)")
    mech1_wins = sum(1 for r in rows if r[2] <= r[5])
    margins = [abs(r[5] - r[2]) / max(r[2], r[5]) * 100 for r in rows]
    summary = "\n".join(lines)
    print(summary)
    print(f"\nConclusion: at MATCHED event density, Mechanism 1 wins or ties "
          f"{mech1_wins}/{len(rows)} of the tested densities, by small margins "
          f"({min(margins):.1f}-{max(margins):.1f}%) -- nowhere close to the "
          f"large, systematic gap seen when comparing at mismatched thresholds "
          f"(the naive, uncontrolled comparison). The hybrid does NOT beat "
          f"Mechanism 1's tracing accuracy on this uniform-contrast, "
          f"noise-free stimulus once event rate is controlled for; if "
          f"anything Mechanism 1's pure linear response has a slight, "
          f"consistent edge here, exactly as expected since there is no real "
          f"dynamic-range problem in this stimulus for compression to solve.")

    with open(os.path.join(OUT_DIR, "hybrid_vs_mechanism1_summary.txt"), "w") as f:
        f.write(summary + "\n")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ns = [r[1] for r in rows]
    r1s = [r[2] for r in rows]
    r2s = [r[5] for r in rows]
    ax.plot(ns, r1s, "o-", label="Mechanism 1", color="tab:blue")
    ax.plot(ns, r2s, "s-", label="Hybrid (v3)", color="tab:green")
    ax.set_xlabel("Event count (density-matched)")
    ax.set_ylabel("Trajectory RMSE (pixels)")
    ax.set_title("Density-matched comparison: Mechanism 1 keeps a small edge\n"
                 "(the earlier apparent hybrid win was an event-rate artifact)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "hybrid_vs_mechanism1_density_matched.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
