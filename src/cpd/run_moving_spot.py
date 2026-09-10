"""
run_moving_spot.py

Moving Gaussian-spot event imaging (review spec section 13.2), reusing the
existing SpotSimulator (imported unmodified) for direct comparability with
the earlier literature-model study already in this repo. Same three device
configs as run_moving_bar.py: literature Mechanism 1, CPD-symmetric,
CPD-asymmetric, matched tau.

Trajectory reconstruction here is a per-timestep INSTANTANEOUS activity
centroid of |signed map| -- NOT the leaky-integrator memory_decode.py used
elsewhere in this repo. Per Prof. Cheng's explicit instruction, no leaky
integration / fading memory is used for this paper; that comes later. This
still lets us report a trajectory RMSE against the known ground truth
without violating that constraint.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stimuli import SpotSimulator
from cpd_tia_simple import CPDTIAPixel
from device_mechanism1 import Mechanism1DualBranch
from run_moving_bar import mechanism1_signed_map  # reuse, not duplicate

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "cpd")
os.makedirs(OUT_DIR, exist_ok=True)


def instantaneous_centroid(map_video, noise_floor_frac=0.05):
    """Per-timestep activity-weighted centroid of |map_video|, with NO
    temporal state carried between steps (no leaky integration). NaN where
    activity is below noise_floor_frac of the video's peak magnitude."""
    n_steps, grid_size, _ = map_video.shape
    yy, xx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    peak = np.abs(map_video).max()
    floor = noise_floor_frac * peak
    traced = np.full((n_steps, 2), np.nan)
    for t in range(n_steps):
        A = np.abs(map_video[t])
        total = A.sum()
        if total > floor * 4:  # require a handful of active pixels, not just one at the floor
            cx = (A * xx).sum() / total
            cy = (A * yy).sum() / total
            traced[t] = [cx, cy]
    return traced


def rmse_ignoring_nan(a, b):
    mask = ~np.isnan(a[:, 0]) & ~np.isnan(b[:, 0])
    if mask.sum() == 0:
        return np.nan, 0.0
    diff = a[mask] - b[mask]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))), float(mask.mean())


def run_all_models(I_L_video, dt, grid_size, tau, eta=0.5, Rf=1.0, asym_ratio=1.5):
    mech1 = Mechanism1DualBranch(grid_size, dt, tau_slow=tau, R=1.0, q_thresh=1e9)
    mech1_map = mechanism1_signed_map(mech1, I_L_video)

    cpd_sym = CPDTIAPixel(grid_size, dt, tau_cpd=tau, eta=eta, Rf=Rf)
    cpd_sym_I, _ = cpd_sym.run(I_L_video)

    cpd_asym = CPDTIAPixel(grid_size, dt, tau_cpd=tau, eta=eta, Rf=Rf,
                            tau_cpd_off=tau * asym_ratio)
    cpd_asym_I, _ = cpd_asym.run(I_L_video)

    return {
        "Literature (Mechanism 1)": mech1_map,
        "CPD - symmetric": cpd_sym_I,
        "CPD - asymmetric": cpd_asym_I,
    }


def main():
    grid_size = 24
    dt = 2e-6
    period = 0.02  # 20 ms per revolution -- fast enough that a 100us CPD tau is well-exercised
    n_steps = 10000  # 20 ms total = one full revolution at dt=2us
    sim = SpotSimulator(grid_size=grid_size, n_steps=n_steps, dt=dt,
                         baseline=0.5, amplitude=4.0, sigma=1.8,
                         path="circle", radius=7.0, period=period)
    I_L_video = sim.generate()
    gt_path = np.stack([sim.cx, sim.cy], axis=1)

    tau = 100e-6
    maps = run_all_models(I_L_video, dt, grid_size, tau)

    # --- event image snapshot, spec section 13.2 "show: input object; CPD event image" ---
    mid_step = n_steps // 4
    vmax = max(np.abs(m[mid_step]).max() for m in maps.values()) + 1e-9
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(5.2 * (len(maps) + 1), 5))
    axes[0].imshow(I_L_video[mid_step], origin="lower", cmap="gray")
    axes[0].set_title("optical frame (I_L)")
    for ax, (name, m) in zip(axes[1:], maps.items()):
        ax.imshow(m[mid_step], origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=10)
    fig.suptitle(f"Moving spot: signed analog output, tau={tau*1e6:.0f}us", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(OUT_DIR, "moving_spot_event_image.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")

    # --- trajectory reconstruction (instantaneous centroid, no leaky integration) ---
    fig, axes = plt.subplots(1, len(maps), figsize=(5.5 * len(maps), 5.2))
    print("\nTrajectory reconstruction (instantaneous per-frame centroid, no leaky integration):")
    for ax, (name, m) in zip(axes, maps.items()):
        traced = instantaneous_centroid(m)
        rmse, cov = rmse_ignoring_nan(traced, gt_path)
        ax.plot(gt_path[:, 0], gt_path[:, 1], "k--", linewidth=1.2, label="ground truth")
        ax.plot(traced[:, 0], traced[:, 1], ".", markersize=2, alpha=0.5, label="reconstructed")
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nRMSE={rmse:.2f}px  coverage={cov*100:.0f}%", fontsize=10)
        ax.legend(fontsize=7)
        print(f"  {name:<28} RMSE={rmse:.3f}px  coverage={cov*100:.1f}%")
    fig.suptitle(f"Moving-spot trajectory (tau={tau*1e6:.0f}us, instantaneous centroid, "
                 f"NOT leaky-integrated)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(OUT_DIR, "moving_spot_trajectory.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
