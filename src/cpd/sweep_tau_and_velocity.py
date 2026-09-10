"""
sweep_tau_and_velocity.py

Review spec sections 14/15: sweep tau_CPD and target velocity, and report
the quantitative event-imaging metrics Larry's email asks for, across all
three device configs (literature Mechanism 1, CPD-symmetric, CPD-asymmetric)
so they can be systematically compared.

Metrics reported, all vs. tau_CPD and/or velocity:
  - event-trail width (spatial extent of active pixels), and the predicted
    Gamma = v*tau_CPD/sigma scaling (spec section 14)
  - trajectory/localization error (instantaneous centroid, NOT leaky
    integration -- same estimator as run_moving_spot.py)
  - event count / density (thresholded, secondary readout)

Uses a short, FIXED-DURATION moving-spot arc (not a full revolution) for
every sweep point, so runtime doesn't scale with velocity.
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
from event_threshold import threshold_events
from run_moving_bar import mechanism1_signed_map
from run_moving_spot import instantaneous_centroid, rmse_ignoring_nan

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "cpd")
os.makedirs(OUT_DIR, exist_ok=True)

SIGMA = 1.8       # SpotSimulator's default Gaussian spot radius (pixels)
GRID_SIZE = 24
DT = 2e-7         # resolves tau_CPD=10us with dt/tau=0.02
DURATION = 6e-3   # fixed simulation duration for every sweep point (s)


def active_pixel_count(frame, frac=0.10):
    """Number of pixels whose |signal| exceeds frac * that frame's peak --
    a simple, robust proxy for the spatial extent (trail width) of the
    active event region."""
    peak = np.abs(frame).max()
    if peak < 1e-12:
        return 0
    return int((np.abs(frame) > frac * peak).sum())


def run_case(tau, velocity):
    """One (tau, velocity) sweep point: builds a short circular-arc spot
    stimulus at the given velocity, runs all three device models, and
    returns their metrics."""
    period = 2 * np.pi * 7.0 / velocity  # radius=7px circle at this speed
    n_steps = int(DURATION / DT)
    sim = SpotSimulator(grid_size=GRID_SIZE, n_steps=n_steps, dt=DT,
                         baseline=0.5, amplitude=4.0, sigma=SIGMA,
                         path="circle", radius=7.0, period=period)
    I_L_video = sim.generate()
    gt_path = np.stack([sim.cx, sim.cy], axis=1)

    mech1 = Mechanism1DualBranch(GRID_SIZE, DT, tau_slow=tau, R=1.0, q_thresh=1e9)
    mech1_map = mechanism1_signed_map(mech1, I_L_video)

    cpd_sym = CPDTIAPixel(GRID_SIZE, DT, tau_cpd=tau, eta=0.5, Rf=1.0)
    cpd_sym_I, cpd_sym_V = cpd_sym.run(I_L_video)

    cpd_asym = CPDTIAPixel(GRID_SIZE, DT, tau_cpd=tau, eta=0.5, Rf=1.0,
                            tau_cpd_off=tau * 1.5)
    cpd_asym_I, cpd_asym_V = cpd_asym.run(I_L_video)

    results = {}
    mid = n_steps // 2
    # Event counts are thresholded on each device's own NATIVE current
    # (I_net for Mechanism 1, I_cpd for the CPD variants) rather than the
    # TIA-inverted V_o -- Mechanism 1 has no TIA stage in this repo, and
    # comparing native-current sparsity sidesteps the V_o sign-inversion
    # issue documented in event_threshold.py, giving an apples-to-apples
    # "how many transition events does the underlying device physics
    # produce" comparison across all three models.
    for name, m in [("Mechanism 1", mech1_map),
                     ("CPD-symmetric", cpd_sym_I),
                     ("CPD-asymmetric", cpd_asym_I)]:
        traced = instantaneous_centroid(m)
        rmse, cov = rmse_ignoring_nan(traced, gt_path)
        trail_width = active_pixel_count(m[mid])
        v_th = 0.1 * np.abs(m).max()
        n_events = len(threshold_events(m, v_th_pos=v_th, v_th_neg=-v_th, dt=DT))
        results[name] = dict(rmse=rmse, coverage=cov, trail_width=trail_width,
                              n_events=n_events)
    return results


def main():
    tau_values_us = [10, 20, 50, 100, 200, 300, 500]
    fixed_velocity = 2000.0  # px/s

    velocities = [500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    fixed_tau_us = 100

    print("=" * 78)
    print(f"TAU SWEEP (velocity fixed at {fixed_velocity:.0f} px/s)")
    print("=" * 78)
    tau_sweep_results = {name: {"rmse": [], "trail_width": [], "gamma": []}
                          for name in ["Mechanism 1", "CPD-symmetric", "CPD-asymmetric"]}
    for tau_us in tau_values_us:
        tau = tau_us * 1e-6
        gamma = fixed_velocity * tau / SIGMA
        r = run_case(tau, fixed_velocity)
        print(f"\ntau_CPD={tau_us}us (Gamma={gamma:.3f}):")
        for name, vals in r.items():
            print(f"  {name:<16} RMSE={vals['rmse']:.3f}px  trail_width={vals['trail_width']}px  "
                  f"events={vals['n_events']}")
            tau_sweep_results[name]["rmse"].append(vals["rmse"])
            tau_sweep_results[name]["trail_width"].append(vals["trail_width"])
            tau_sweep_results[name]["gamma"].append(gamma)

    print("\n" + "=" * 78)
    print(f"VELOCITY SWEEP (tau_CPD fixed at {fixed_tau_us}us)")
    print("=" * 78)
    vel_sweep_results = {name: {"rmse": [], "trail_width": [], "gamma": []}
                          for name in ["Mechanism 1", "CPD-symmetric", "CPD-asymmetric"]}
    for v in velocities:
        tau = fixed_tau_us * 1e-6
        gamma = v * tau / SIGMA
        r = run_case(tau, v)
        print(f"\nvelocity={v:.0f}px/s (Gamma={gamma:.3f}):")
        for name, vals in r.items():
            print(f"  {name:<16} RMSE={vals['rmse']:.3f}px  trail_width={vals['trail_width']}px  "
                  f"events={vals['n_events']}")
            vel_sweep_results[name]["rmse"].append(vals["rmse"])
            vel_sweep_results[name]["trail_width"].append(vals["trail_width"])
            vel_sweep_results[name]["gamma"].append(gamma)

    # --- Plot: does everything collapse onto Gamma, as spec section 14 predicts? ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"Mechanism 1": "tab:blue", "CPD-symmetric": "tab:green", "CPD-asymmetric": "tab:orange"}
    for name in tau_sweep_results:
        g1 = tau_sweep_results[name]["gamma"]
        w1 = tau_sweep_results[name]["trail_width"]
        g2 = vel_sweep_results[name]["gamma"]
        w2 = vel_sweep_results[name]["trail_width"]
        axes[0].plot(g1, w1, "o-", color=colors[name], label=f"{name} (tau sweep)")
        axes[0].plot(g2, w2, "s--", color=colors[name], alpha=0.6, label=f"{name} (velocity sweep)")

        r1 = tau_sweep_results[name]["rmse"]
        r2 = vel_sweep_results[name]["rmse"]
        axes[1].plot(g1, r1, "o-", color=colors[name], label=f"{name} (tau sweep)")
        axes[1].plot(g2, r2, "s--", color=colors[name], alpha=0.6, label=f"{name} (velocity sweep)")

    axes[0].set_xlabel("Gamma = v * tau_CPD / sigma")
    axes[0].set_ylabel("Event-trail width (active pixels)")
    axes[0].set_title("Trail width vs. Gamma\n(tau-sweep and velocity-sweep points should collapse together)")
    axes[0].legend(fontsize=6.5)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Gamma = v * tau_CPD / sigma")
    axes[1].set_ylabel("Trajectory RMSE (pixels, instantaneous centroid)")
    axes[1].set_title("Localization error vs. Gamma")
    axes[1].legend(fontsize=6.5)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "gamma_scaling.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
