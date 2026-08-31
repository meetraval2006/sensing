"""
animate.py

Produces an animated GIF of a light spot moving in a circle, watched
simultaneously by all three device models -- the "animation of a light
going around, with 3 models" deliverable.

Each panel shows, for one device model:
  - the ground-truth light intensity field (background heatmap)
  - the dashed ground-truth circular path
  - that model's live ON (red) / OFF (cyan) events in a short trailing
    time window
  - the "physical observation memory" decoder's current traced position
    (green star), at a fixed medium fading time tau

Run with:  python3 animate.py
Output:    ../outputs/animation.gif
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(__file__))

from spot_simulator import SpotSimulator
from device_mechanism1 import Mechanism1DualBranch
from device_mechanism2 import Mechanism2Pyrophototronic
from device_hybrid import HybridModel
from memory_decode import MemoryDecoder

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # One clean loop of the circular path, at a resolution that keeps the
    # GIF a reasonable size while still showing a full revolution.
    sim = SpotSimulator(grid_size=24, n_steps=1000, dt=1e-3,
                         path="circle", radius=7.0, period=1.0)
    L_video = sim.generate()
    dt = sim.dt
    grid_size = sim.grid_size
    n_steps = sim.n_steps

    # Same hand-tuned q_thresh values as compare_models.py, so the event
    # rates shown here match the quantitative comparison elsewhere in the repo.
    models = {
        "Mechanism 1: WSe2 dual-branch": Mechanism1DualBranch(
            grid_size, dt, tau_slow=5e-3, R=1.0, q_thresh=0.004),
        "Mechanism 2: pyro-phototronic": Mechanism2Pyrophototronic(
            grid_size, dt, tau_th=5e-3, alpha_heat=1.0, kappa=1.0,
            alpha_pos=0.27, alpha_neg=0.38, q_thresh=0.08),
        "Hybrid (custom)": HybridModel(
            grid_size, dt, tau_slow=5e-3, R=1.0, alpha=0.4, x0=0.006, q_thresh=0.004156),
    }

    tau_decode = 0.02  # "medium" fading time -- matches the middle case in compare_models.py

    events_by_model = {}
    traced_by_model = {}
    for name, model in models.items():
        events = model.run(L_video)
        events_by_model[name] = events
        decoder = MemoryDecoder(grid_size, dt, tau=tau_decode, kappa=1.0)
        traced_signed, _ = decoder.run(events, n_steps)
        traced_by_model[name] = traced_signed
        print(f"{name}: {len(events)} events over {n_steps} steps")

    # Bucket events per step per model for fast windowed lookup during animation
    events_by_step = {}
    for name, events in events_by_model.items():
        buckets = [[] for _ in range(n_steps)]
        for (t_idx, x, y, p) in events:
            buckets[t_idx].append((x, y, p))
        events_by_step[name] = buckets

    frame_stride = 5    # subsample steps -> frames for a manageable GIF (200 frames)
    trail_window = 25   # steps (25 ms) of recent event history shown per frame
    frame_steps = list(range(0, n_steps, frame_stride))

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 5.4))

    artists = {}
    for ax, name in zip(axes, models):
        im = ax.imshow(L_video[0], origin="lower", cmap="inferno",
                        extent=[0, grid_size, 0, grid_size],
                        vmin=sim.baseline, vmax=sim.baseline + sim.amplitude,
                        animated=True)
        ax.plot(sim.cx, sim.cy, "w--", linewidth=1.0, alpha=0.5, label="ground truth path")
        on_scatter = ax.scatter([], [], s=14, c="tab:red", label="ON event", alpha=0.9)
        off_scatter = ax.scatter([], [], s=14, c="tab:cyan", label="OFF event", alpha=0.9)
        trace_pt, = ax.plot([], [], "*", color="lime", markersize=16,
                             markeredgecolor="black", markeredgewidth=0.6,
                             label="decoded position")
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
        artists[name] = dict(im=im, on=on_scatter, off=off_scatter, trace=trace_pt)

    time_text = fig.suptitle("", fontsize=11)

    def update(frame_idx):
        step = frame_steps[frame_idx]
        updated = []
        for name in models:
            a = artists[name]
            a["im"].set_data(L_video[step])
            updated.append(a["im"])

            lo = max(0, step - trail_window)
            recent = [e for s in range(lo, step + 1) for e in events_by_step[name][s]]
            if recent:
                arr = np.array(recent, dtype=np.float64)
                pos = arr[arr[:, 2] > 0][:, :2]
                neg = arr[arr[:, 2] < 0][:, :2]
            else:
                pos = np.empty((0, 2))
                neg = np.empty((0, 2))
            # +0.5 centers each event marker in its pixel cell for the imshow extent
            a["on"].set_offsets(pos + 0.5 if len(pos) else pos)
            a["off"].set_offsets(neg + 0.5 if len(neg) else neg)
            updated += [a["on"], a["off"]]

            tp = traced_by_model[name][step]
            if not np.isnan(tp[0]):
                a["trace"].set_data([tp[0]], [tp[1]])
            else:
                a["trace"].set_data([], [])
            updated.append(a["trace"])

        time_text.set_text(
            f"t = {step * dt * 1000:.0f} ms  |  moving light spot + live event stream "
            f"+ decoded trace (memory fading time tau = {tau_decode * 1000:.0f} ms)")
        updated.append(time_text)
        return updated

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ani = animation.FuncAnimation(fig, update, frames=len(frame_steps),
                                   interval=1000 / 20, blit=False)

    out_path = os.path.join(OUT_DIR, "animation.gif")
    ani.save(out_path, writer=animation.PillowWriter(fps=20))
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
