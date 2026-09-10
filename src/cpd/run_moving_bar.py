"""
run_moving_bar.py

Moving-bar event imaging (review spec section 13.1): a bright rectangular
bar sweeps across a dark background. Compares three device configurations
on the IDENTICAL stimulus:
    - the literature baseline (Mechanism1DualBranch, the WSe2 dual-branch
      model already validated elsewhere in this repo)
    - the CPD, symmetric variant (tau_on = tau_off)
    - the CPD, asymmetric variant (tau_off = 1.5 x tau_on)

Primary output is the SIGNED analog map (I_net / I_cpd), per spec section 12
-- thresholded ON/OFF events are shown as an optional fourth panel, not the
main result. No leaky integration / memory decode is used anywhere in this
script, per Prof. Cheng's explicit instruction.

All three models are run at MATCHED nominal time constants (tau_slow for
Mechanism 1, tau_cpd for the CPD variants) so the comparison is apples to
apples -- directly addressing the "functionally the two models are not very
different... useful and fair comparison" framing in Larry's email.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stimuli import MovingBarSimulator
from cpd_tia_simple import CPDTIAPixel
from event_threshold import threshold_events
from device_mechanism1 import Mechanism1DualBranch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "cpd")
os.makedirs(OUT_DIR, exist_ok=True)


def mechanism1_signed_map(model, I_L_video):
    """Mechanism1DualBranch.step() only returns quantized events, not the
    continuous I_net. Re-derive I_net directly here using the exact same
    update rule as device_mechanism1.py, so the literature baseline can be
    shown on the same signed-analog-map footing as the CPD (spec section 12)
    without modifying the existing, already-validated device file.

    IMPORTANT FIX vs. a naive re-derivation: device_mechanism1.py's own
    .step()/.run() start the slow branch at I_slow=0, which is only correct
    if the FIRST frame's illumination is exactly 0 everywhere. Our stimuli
    have a nonzero uniform baseline (e.g. 0.5), so I_slow=0 at t=0 produces
    a large, spurious, spatially-UNIFORM transient (I_net(0) = R*baseline
    everywhere, not 0) that has nothing to do with motion and dominates any
    unsmoothed, whole-frame metric (like the instantaneous centroid in
    run_moving_spot.py/sweep_tau_and_velocity.py) for several tau_slow
    afterward. This is invisible in the original repo's charge-quantized,
    memory-decoded comparisons (RESULTS.md) because that pipeline's leaky
    integration and long run length wash it out, but it must be corrected
    here for a fair, uncontaminated comparison against the CPD -- which has
    no analogous artifact, because CPDTIAPixel is driven by dI_L/dt (well
    defined as 0 before any change is observed) rather than by a state
    variable that needs pre-equilibrating to a nonzero steady state. Fix:
    seed I_slow at its correct steady-state value, R*I_L_video[0], before
    the loop -- this changes nothing about the ongoing per-step DEVICE
    PHYSICS (still R*L - I_slow every step, identical recursion), it only
    removes a numerical cold-start artifact from OUR OWN wrapper. The
    original device_mechanism1.py file is not modified."""
    grid_size = model.grid_size
    R = model.R
    tau_slow = model.tau_slow
    n_steps = I_L_video.shape[0]
    I_slow = R * I_L_video[0].copy()  # start at steady state, not 0
    I_net_trace = np.empty_like(I_L_video)
    for t in range(n_steps):
        L = I_L_video[t]
        I_fast = R * L
        I_slow += (model.dt / tau_slow) * (R * L - I_slow)
        I_net_trace[t] = I_fast - I_slow
    return I_net_trace


def run_all_models(I_L_video, dt, grid_size, tau, eta=0.5, Rf=1.0, asym_ratio=1.5):
    mech1 = Mechanism1DualBranch(grid_size, dt, tau_slow=tau, R=1.0, q_thresh=1e9)
    mech1_map = mechanism1_signed_map(mech1, I_L_video)

    cpd_sym = CPDTIAPixel(grid_size, dt, tau_cpd=tau, eta=eta, Rf=Rf)
    cpd_sym_I, cpd_sym_V = cpd_sym.run(I_L_video)

    cpd_asym = CPDTIAPixel(grid_size, dt, tau_cpd=tau, eta=eta, Rf=Rf,
                            tau_cpd_off=tau * asym_ratio)
    cpd_asym_I, cpd_asym_V = cpd_asym.run(I_L_video)

    return {
        "Literature (Mechanism 1)": mech1_map,
        "CPD - symmetric": cpd_sym_I,
        "CPD - asymmetric": cpd_asym_I,
    }


def static_comparison_grid(sim, tau_values_us, dt):
    """Rows = tau values, columns = the three device configs, each cell =
    signed analog map at a representative timestep (bar's leading edge near
    mid-grid). Spec section 13.1: 'show the actual spatial event image, not
    only scalar metrics,' for each tested tau_CPD."""
    I_L_video = sim.generate()
    grid_size = sim.grid_size
    mid_step = int(sim.n_steps * 0.45)  # bar sweeping through the middle of the grid

    fig, axes = plt.subplots(len(tau_values_us), 4,
                              figsize=(16, 3.6 * len(tau_values_us)))

    for row, tau_us in enumerate(tau_values_us):
        tau = tau_us * 1e-6
        maps = run_all_models(I_L_video, dt, grid_size, tau)

        vmax = max(np.abs(m[mid_step]).max() for m in maps.values()) + 1e-9

        ax = axes[row, 0]
        ax.imshow(I_L_video[mid_step], origin="lower", cmap="gray")
        ax.set_title(f"optical frame (tau={tau_us}us)" if row == 0 else "")
        ax.set_ylabel(f"tau = {tau_us} us", fontsize=10)

        for col, (name, m) in enumerate(maps.items(), start=1):
            ax = axes[row, col]
            im = ax.imshow(m[mid_step], origin="lower", cmap="seismic",
                            vmin=-vmax, vmax=vmax)
            if row == 0:
                ax.set_title(name, fontsize=10)

    fig.suptitle("Moving bar: signed analog output map, matched tau across all three models",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUT_DIR, "moving_bar_comparison_grid.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


def four_panel_single_model_view(sim, tau_us, dt, model_name="CPD - symmetric",
                                  v_th_frac=0.15):
    """The literal spec section 13.1 four-panel view for ONE model:
    optical frame | signed CPD current map | TIA voltage map | optional
    thresholded ON/OFF event image -- at a representative timestep."""
    I_L_video = sim.generate()
    grid_size = sim.grid_size
    tau = tau_us * 1e-6
    mid_step = int(sim.n_steps * 0.45)

    cpd = CPDTIAPixel(grid_size, dt, tau_cpd=tau, eta=0.5, Rf=1.0)
    I_cpd_trace, V_o_trace = cpd.run(I_L_video)

    v_th = v_th_frac * np.abs(V_o_trace).max()
    events = threshold_events(V_o_trace, v_th_pos=v_th, v_th_neg=-v_th, dt=dt)
    on_events = [(x, y) for (t_idx, x, y, p) in events if t_idx == mid_step and p > 0]
    off_events = [(x, y) for (t_idx, x, y, p) in events if t_idx == mid_step and p < 0]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes[0].imshow(I_L_video[mid_step], origin="lower", cmap="gray")
    axes[0].set_title("optical frame (I_L)")

    vmax_i = np.abs(I_cpd_trace[mid_step]).max() + 1e-9
    axes[1].imshow(I_cpd_trace[mid_step], origin="lower", cmap="seismic",
                   vmin=-vmax_i, vmax=vmax_i)
    axes[1].set_title("signed I_cpd map")

    vmax_v = np.abs(V_o_trace[mid_step]).max() + 1e-9
    axes[2].imshow(V_o_trace[mid_step], origin="lower", cmap="seismic",
                   vmin=-vmax_v, vmax=vmax_v)
    axes[2].set_title("TIA voltage V_o map")

    axes[3].imshow(np.zeros((grid_size, grid_size)), origin="lower", cmap="gray", vmin=0, vmax=1)
    if on_events:
        xs, ys = zip(*on_events)
        axes[3].scatter(xs, ys, c="red", s=40, label="ON")
    if off_events:
        xs, ys = zip(*off_events)
        axes[3].scatter(xs, ys, c="cyan", s=40, label="OFF")
    axes[3].set_title(f"optional: thresholded ON/OFF (secondary readout)")
    axes[3].legend(fontsize=8, loc="upper right")

    fig.suptitle(f"{model_name}, tau_CPD={tau_us}us -- full pipeline for one device config",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(OUT_DIR, "moving_bar_four_panel_single_model.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


def event_video(sim, tau_us, dt, v_th_frac=0.15):
    """Event-video animation (spec section 13.1) for one representative
    tau, all three models side by side, showing the primary signed analog
    map directly (not a scatter of thresholded events) per spec section 12's
    primary/secondary distinction."""
    I_L_video = sim.generate()
    grid_size = sim.grid_size
    tau = tau_us * 1e-6
    maps = run_all_models(I_L_video, dt, grid_size, tau)

    vmax = max(np.abs(m).max() for m in maps.values()) + 1e-9
    frame_stride = max(1, sim.n_steps // 150)
    frame_steps = list(range(0, sim.n_steps, frame_stride))

    fig, axes = plt.subplots(1, len(maps), figsize=(5.2 * len(maps), 5))
    ims = []
    for ax, (name, m) in zip(axes, maps.items()):
        im = ax.imshow(m[0], origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax,
                        animated=True)
        ax.set_title(name, fontsize=10)
        ims.append(im)
    time_text = fig.suptitle("", fontsize=11)

    def update(frame_idx):
        step = frame_steps[frame_idx]
        for im, m in zip(ims, maps.values()):
            im.set_data(m[step])
        time_text.set_text(f"t = {step*dt*1e3:.1f} ms  (moving bar, tau_CPD={tau_us}us, "
                            f"signed analog output, matched tau across models)")
        return ims + [time_text]

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ani = animation.FuncAnimation(fig, update, frames=len(frame_steps),
                                   interval=1000 / 20, blit=False)
    out_path = os.path.join(OUT_DIR, f"moving_bar_event_video_tau{tau_us}us.gif")
    ani.save(out_path, writer=animation.PillowWriter(fps=20))
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    grid_size = 24
    dt = 2e-6  # must resolve the smallest tau_CPD (10us) tested elsewhere; fine here too
    n_steps = 4000
    velocity = 3000.0  # pixels/s -- crosses a 24px grid in ~8ms
    sim = MovingBarSimulator(grid_size=grid_size, n_steps=n_steps, dt=dt,
                              baseline=0.5, amplitude=4.0, bar_width=6.0,
                              edge_sigma=0.6, velocity=velocity, axis="x",
                              start_offset=-4.0)

    print("Generating static comparison grid (tau = 10, 100, 500 us) ...")
    static_comparison_grid(sim, [10, 100, 500], dt)

    print("Generating single-model four-panel view (spec section 13.1) ...")
    four_panel_single_model_view(sim, 100, dt)

    print("Generating event-video animation (tau = 100 us) ...")
    event_video(sim, 100, dt)

    print("\nDone.")


if __name__ == "__main__":
    main()
