"""
compare_models.py

Meeting deliverable: runs a moving-spot ground truth through all three
device models (Mechanism 1, Mechanism 2, Hybrid), decodes each event
stream's trajectory with the physical-observation-memory readout at a
few different fading times (tau), and scores how well each traced path
matches the known ground truth path.

Run with:  python3 compare_models.py
Outputs go to ../outputs/
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
from device_mechanism2 import Mechanism2Pyrophototronic
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


def main():
    # ---- 1. Ground truth stimulus ----
    sim = SpotSimulator(grid_size=24, n_steps=3000, dt=1e-3,
                         path="circle", radius=7.0, period=1.0)
    L_video = sim.generate()
    gt_path = np.stack([sim.cx, sim.cy], axis=1)

    dt = sim.dt
    grid_size = sim.grid_size
    n_steps = sim.n_steps

    # ---- 2. Device models (same dt/grid, comparable q_thresh) ----
    # q_thresh values below were hand-tuned (see tune_thresholds.py) so that
    # each model produces a comparable event rate (~2-5k events over 3000
    # steps) on the SAME stimulus. This matters: comparing models at wildly
    # different event rates would confound "which physics traces better"
    # with "which model just happens to fire more often." Feel free to
    # re-tune for your own stimuli/parameters.
    models = {
        "Mechanism1_WSe2_dual_branch": Mechanism1DualBranch(
            grid_size, dt, tau_slow=5e-3, R=1.0, q_thresh=0.004),
        "Mechanism2_pyro_phototronic": Mechanism2Pyrophototronic(
            grid_size, dt, tau_th=5e-3, alpha_heat=1.0, kappa=1.0,
            alpha_pos=0.27, alpha_neg=0.38, q_thresh=0.08),
        "Hybrid_custom": HybridModel(
            grid_size, dt, tau_slow=5e-3, R=1.0, alpha=0.4, x0=0.006, q_thresh=0.004156),
    }

    # ---- 3. Run each device model to get its event stream ----
    event_streams = {}
    for name, model in models.items():
        print(f"Running {name} ...")
        events = model.run(L_video)
        event_streams[name] = events
        print(f"  -> {len(events)} events generated "
              f"({len(events) / n_steps:.2f} events/step avg)")

    # ---- 4. Decode trajectories at several fading times (tau) ----
    tau_values = [0.005, 0.02, 0.08]  # seconds: fast / medium / slow fading memory

    results = {}  # results[model_name][tau] = dict(signed=..., rect=..., rmse_signed=..., ...)
    for name, events in event_streams.items():
        results[name] = {}
        for tau in tau_values:
            decoder = MemoryDecoder(grid_size, dt, tau=tau, kappa=1.0)
            traced_signed, traced_rect = decoder.run(events, n_steps)

            rmse_signed, coverage_signed = rmse_ignoring_nan(traced_signed, gt_path)
            rmse_rect, coverage_rect = rmse_ignoring_nan(traced_rect, gt_path)

            results[name][tau] = dict(
                traced_signed=traced_signed, traced_rect=traced_rect,
                rmse_signed=rmse_signed, rmse_rect=rmse_rect,
                coverage_signed=coverage_signed, coverage_rect=coverage_rect,
            )

    # ---- 5. Print summary table ----
    print("\n" + "=" * 78)
    print(f"{'Model':<30}{'tau(s)':<10}{'RMSE signed':<14}{'RMSE rect.':<14}{'coverage':<10}")
    print("=" * 78)
    for name in models:
        for tau in tau_values:
            r = results[name][tau]
            print(f"{name:<30}{tau:<10}{r['rmse_signed']:<14.3f}"
                  f"{r['rmse_rect']:<14.3f}{r['coverage_signed']:<10.2f}")
    print("=" * 78)

    # save the table to a text file too
    with open(os.path.join(OUT_DIR, "summary_table.txt"), "w") as f:
        f.write(f"{'Model':<30}{'tau(s)':<10}{'RMSE signed':<14}{'RMSE rect.':<14}{'coverage':<10}\n")
        for name in models:
            for tau in tau_values:
                r = results[name][tau]
                f.write(f"{name:<30}{tau:<10}{r['rmse_signed']:<14.3f}"
                        f"{r['rmse_rect']:<14.3f}{r['coverage_signed']:<10.2f}\n")

    # ---- 6. Plot: traced path vs ground truth, one figure per tau ----
    for tau in tau_values:
        fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5.2))
        for ax, name in zip(axes, models):
            r = results[name][tau]
            ax.plot(gt_path[:, 0], gt_path[:, 1], "k--", linewidth=1.5, label="ground truth")
            ax.plot(r["traced_signed"][:, 0], r["traced_signed"][:, 1],
                     ".", markersize=2, alpha=0.6, label="traced (signed A_M)")
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_aspect("equal")
            ax.set_title(f"{name}\ntau={tau}s  RMSE={r['rmse_signed']:.2f}px  "
                         f"cov={r['coverage_signed']*100:.0f}%", fontsize=10)
            ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f"trajectory_tau_{tau}.png")
        fig.savefig(fname, dpi=130)
        plt.close(fig)
        print(f"Saved {fname}")

    # ---- 7. Plot: RMSE vs fading time, all models on one axis ----
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name in models:
        rmses = [results[name][tau]["rmse_signed"] for tau in tau_values]
        ax.plot(tau_values, rmses, "o-", label=name)
    ax.set_xlabel("Fading time constant, tau (s)")
    ax.set_ylabel("Trajectory RMSE (pixels)")
    ax.set_title("How fading time affects trajectory-tracing accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "rmse_vs_tau.png")
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"Saved {fname}")

    # ---- 8. Plot: raw event rasters for a quick visual sanity check ----
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.5))
    for ax, name in zip(axes, models):
        events = event_streams[name]
        if len(events) > 0:
            arr = np.array(events)  # columns: t_idx, x, y, polarity
            pos = arr[arr[:, 3] > 0]
            neg = arr[arr[:, 3] < 0]
            if len(pos):
                ax.scatter(pos[:, 0] * dt, pos[:, 1], s=1, c="tab:red", label="ON", alpha=0.5)
            if len(neg):
                ax.scatter(neg[:, 0] * dt, neg[:, 1], s=1, c="tab:blue", label="OFF", alpha=0.5)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("pixel y")
        ax.set_title(f"{name}\nevent raster ({len(events)} events)", fontsize=9)
        ax.legend(fontsize=7, markerscale=4)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "event_rasters.png")
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"Saved {fname}")

    print("\nDone. All outputs in:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
