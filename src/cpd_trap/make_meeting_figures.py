"""
make_meeting_figures.py

Summary figures for the 10/28 meeting notes (outputs/cpd_trap/meeting/).
Reads summary_basic.csv from run_trajectory_study.py; re-simulates only
single pixels, so it runs in about a minute.

  fig1_pipeline.png        step-by-step pipeline, block diagram + the real
                           signals at one pixel for symmetric vs asymmetric
  fig2_fringe_asymmetry.png ON/OFF peaks vs distance from the spot path:
                           why a single rho_G hides the asymmetry that matters
  fig3_metrics_straight.png symmetric vs asymmetric metrics, straight path
  fig4_error_heatmap.png   mean trajectory error, every condition x path
  fig5_dwell_time.png      |ON|/OFF vs how long a pixel stays lit: the trap
                           asymmetry (|ON| < OFF) needs dwell > ~tau_on
  asymmetry_regimes.txt    the numbers behind fig2/fig5 plus |ON|/OFF vs
                           Gaussian spot speed
"""

import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from cpd_trap_model import CPDArray, TASK_A_PAIRS, TASK_B_TRAPS, THETA, named
from validate_single_pixel import gaussian_u

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "cpd_trap", "meeting")
SUMMARY = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "cpd_trap", "summary_basic.csv")
REF = TASK_A_PAIRS["A-reference"]

# dataviz reference palette, categorical slots 1-3 (validated all-pairs)
C_SYM, C_TRAP, C_CTRL = "#2a78d6", "#eb6834", "#1baf7a"
C_ON, C_OFF = "#e34948", "#2a78d6"
INK, MUTED = "#222222", "#777777"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "lines.linewidth": 2})


def pixel_signals(traps, scale=1.0, dt=2.0, t_end=12000.0):
    """u, T, I_eff, i_R, events and spec tracker state at one pixel for one
    Gaussian pass (sigma 2 px, 2 px/ms); scale < 1 = pixel off the path."""
    px = CPDArray((1, 1), REF, traps)
    n = int(t_end / dt)
    t = np.arange(1, n + 1) * dt
    out = {k: np.zeros(n) for k in ("u", "T", "I_eff", "i_R", "state")}
    ev, on_armed, off_armed, st = [], True, True, 0
    for k in range(n):
        u_fn = lambda tt: np.full((1, 1), scale * gaussian_u(tt))
        T_before = None if px.T is None else px.T[0, 0]
        i = px.step(u_fn, k * dt, dt)[0, 0]
        u = scale * gaussian_u(t[k])
        if traps is None:
            I_eff, T = REF.I_P0 * u, 0.0
        else:
            T = px.T[0, 0]
            I_eff = REF.I_P0 * u - traps.Gamma * (T - T_before) / dt
        if on_armed and i < -THETA:
            ev.append((t[k], 1)); on_armed = False; st = 1
        if off_armed and i > THETA:
            ev.append((t[k], -1)); off_armed = False; st = 0
        on_armed |= i > -THETA / 2
        off_armed |= i < THETA / 2
        for key, v in (("u", u), ("T", T), ("I_eff", I_eff), ("i_R", i), ("state", st)):
            out[key][k] = v
    out["t"] = t / 1000
    out["events"] = ev
    return out


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="white", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=9,
            weight="bold", color=INK)
    ax.text(x + w / 2, y + h / 2 - 0.05, body, ha="center", va="center", fontsize=7.6, color=INK)


def fig_pipeline():
    sym = pixel_signals(None)
    asym = pixel_signals(TASK_B_TRAPS["B-trap-sq0.2"][0])
    fig = plt.figure(figsize=(16, 9.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.25, 1, 1], hspace=0.55, wspace=0.42)

    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off(); ax.set_xlim(-0.012, 1.012); ax.set_ylim(0, 1)
    steps = [
        ("1. Optical input", "Gaussian spot (Eq. 1) or\nvideo frame -> grayscale\nu(x,y,t) in [0,1]\n(never binarized)", MUTED),
        ("2. Photocurrent", "I_P = I_P0 * u\nI_P0 = 3 uA", MUTED),
        ("3. Trap state (Task B)", "dT/dt = u(T_ss-T)/tau_on\n  - (1-u)T/tau_off\nI_eff = I_P - Gamma dT/dt\n(Task A: I_eff = I_P)", C_TRAP),
        ("4. CPD readout circuit", "di_R/dt = z - I_eff/RC_s\ndz/dt = -a z - b i_R + a I_eff/RC_s\ntau1=150, tau2=20, RC_s=100 us\nRK4, dt = 2 us", C_SYM),
        ("5. Events", "ON : i_R crosses -theta\nOFF: i_R crosses +theta\ntheta = 4.5 nA,\none per polarity per passage", INK),
        ("6. Tracker + metrics", "ON sets pixel, OFF resets\nr_est = centroid of active\nvs r_true, u>=0.2 footprint:\nlatency, IoU, error, ...", INK),
    ]
    w, gap = 0.148, (1 - 6 * 0.148) / 5
    for k, (title, body, color) in enumerate(steps):
        x = k * (w + gap)
        box(ax, x, 0.08, w, 0.84, title, body, color)
        if k < 5:
            ax.annotate("", xy=(x + w + gap - 0.004, 0.5), xytext=(x + w + 0.004, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    ax.set_title("Step-by-step: what the simulator does at every pixel, every 2 us",
                 fontsize=12, loc="left", color=INK)

    panels = [
        ("u", "1. input u", None),
        ("I_P", "2. I_P (uA)", None),
        ("T", "3. trap occupancy T", None),
        ("I_eff", "3. I_eff (uA)", None),
        ("i_R", "4. readout i_R (nA)", 1e3),
        ("state", "5-6. events + tracker state", None),
    ]
    for row, (sig, name, color) in enumerate(((sym, "symmetric: A-reference (no traps), rho_G = 1.004", C_SYM),
                                              (asym, "asymmetric: B-trap-sq0.2, rho_sq = 0.198, rho_G = 1.266", C_TRAP))):
        for col, (key, title, sc) in enumerate(panels):
            a = fig.add_subplot(gs[1 + row, col])
            t = sig["t"]
            y = REF.I_P0 * sig["u"] if key == "I_P" else sig[key]
            if sc:
                y = y * sc
            if key == "state":
                a.step(t, y, where="post", color=color)
                for te, p in sig["events"]:
                    a.axvline(te / 1000, color=C_ON if p == 1 else C_OFF, lw=1.2, ls="--")
                a.set_ylim(-0.15, 1.3); a.set_yticks([0, 1])
                a.text(0.02, 0.97, "red ON  blue OFF", transform=a.transAxes, fontsize=7,
                       va="top", color=MUTED)
            else:
                a.plot(t, y, color=color)
            if key == "i_R":
                for s in (1, -1):
                    a.axhline(s * THETA * 1e3, color=INK, lw=0.8, ls=":")
                a.text(0.98, 0.04, "dotted: +-theta", transform=a.transAxes, ha="right",
                       fontsize=7, color=MUTED)
            if key == "T" and row == 0:
                a.text(0.5, 0.5, "no trap state\n(Task A)", transform=a.transAxes,
                       ha="center", va="center", color=MUTED)
            a.set_xlim(2, 8 if key != "T" else 12)
            a.set_title(title, fontsize=9, loc="left")
            if row == 1:
                a.set_xlabel("time (ms)")
            if col == 0:
                a.set_ylabel(name.split(":")[0], color=color, weight="bold")
        fig.text(0.5, 0.635 - 0.305 * row, name, ha="center", fontsize=9.5, color=color, weight="bold")
    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_fringe():
    conds = [("A-reference", None, C_SYM, "-")] + [
        (k, p, c, ls) for (k, (p, _)), c, ls in zip(TASK_B_TRAPS.items(),
                                                     [C_TRAP] * 4, ["-", "--", "-.", ":"])]
    dys = np.arange(0, 4.75, 0.25)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, traps, c, ls in conds:
        on, off = [], []
        for dy in dys:
            s = pixel_signals(traps, scale=np.exp(-dy * dy / 8), t_end=20000.0)
            on.append(-s["i_R"].min() * 1e3); off.append(s["i_R"].max() * 1e3)
        on, off = np.array(on), np.array(off)
        axes[0].plot(dys, on / off, color=c, ls=ls, label=named(name))
        if name in ("A-reference", "B-trap-sq0.7", "B-trap-sq0.2"):
            axes[1 if name != "B-trap-sq0.2" else 2].plot(dys, on, color=c, ls=ls, label=f"{name} ON |peak|")
            axes[1 if name != "B-trap-sq0.2" else 2].plot(dys, off, color=c, ls=ls, alpha=0.45,
                                                          label=f"{name} OFF peak")
    axes[0].set_title("ON/OFF peak ratio vs distance from path", loc="left")
    axes[0].set_xlabel("pixel distance from spot path (px)"); axes[0].set_ylabel("|ON| / OFF")
    axes[0].legend(fontsize=7)
    for a, ttl in ((axes[1], "A-reference vs B-trap-sq0.7: peaks near theta"),
                   (axes[2], "B-trap-sq0.2: OFF drops below theta first")):
        a.axhline(THETA * 1e3, color=INK, lw=1, ls=":")
        a.text(0.2, THETA * 1e3 * 1.15, "theta = 4.5 nA", fontsize=7, color=INK)
        a.set_yscale("log"); a.set_xlabel("pixel distance from spot path (px)")
        a.set_ylabel("peak |i_R| (nA)"); a.set_title(ttl, loc="left"); a.legend(fontsize=7)
        a.axvspan(3.5, 4.5, color="#f3e1d6", zorder=0)
    fig.suptitle("Trap asymmetry grows in dimly lit fringe pixels; where one polarity crosses "
                 "theta and the other does not, the tracker gets an ON with no OFF (a trail)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "fig2_fringe_asymmetry.png"), dpi=130)
    plt.close(fig)


def bar_rho(traps, L, e=0.3, v=0.002):
    """|ON|/OFF at one pixel for a soft-edged bar of length L px at v px/us."""
    from validate_single_pixel import run_pixel
    t0 = 3000 + (L / 2) / v
    f = lambda tt: (1 / (1 + np.exp(np.clip(-((tt - t0) * v + L / 2) / e, -700, 700)))
                    - 1 / (1 + np.exp(np.clip(-((tt - t0) * v - L / 2) / e, -700, 700))))
    _, i = run_pixel(f, 2 * t0 + 20000, 2.0, REF, traps)
    return abs(i.min()) / i.max()


def fig_dwell():
    lengths = np.array([0.6, 1, 1.5, 2, 3, 4, 5, 7, 10, 14, 20])
    dwell = lengths / 2.0                      # ms at 2 px/ms
    conds = [("A-reference", None, 1.000, C_SYM, "-")] + [
        (k, p, sq, C_TRAP, ls) for (k, (p, sq)), ls in zip(TASK_B_TRAPS.items(), ["-", "--", "-.", ":"])]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for name, traps, sq, c, ls in conds:
        r = [bar_rho(traps, L) for L in lengths]
        ax.plot(dwell, r, color=c, ls=ls, marker="o", ms=4, label=named(name))
        if traps is not None and sq < 0.9:      # 1.00 / 0.99 ticks would sit on the curves
            ax.annotate(f"rho_sq label {sq:.2f}", xy=(dwell[-1], sq), xytext=(10, 0),
                        textcoords="offset points", fontsize=7, color=c, va="center")
            ax.plot(dwell[-1], sq, marker="_", ms=14, color=c)
    ax.axhline(1, color=INK, lw=0.8, ls=":")
    ax.axvspan(0.3, 1.2, color="#e8eef8", zorder=0)
    ax.text(0.33, 1.38, "sigma = 2 px Gaussian spot\nlights a pixel ~1 ms", fontsize=8, color=C_SYM)
    ax.set_xscale("log")
    ax.set_xlabel("time a pixel stays fully lit (ms)  [bar length / 2 px/ms, edge 0.3 px]")
    ax.set_ylabel("|ON peak| / OFF peak")
    ax.set_title("Device asymmetry (weaker ON, stronger OFF) appears when an object dwells long enough "
                 "for traps to fill\n(sharper edges bring it the rest of the way to the square-pulse "
                 "label; see asymmetry_regimes.txt)", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_dwell_time.png"), dpi=130)
    plt.close(fig)


def regimes_table():
    """Text tables: |ON|/OFF vs Gaussian speed, and vs bar dwell time."""
    from validate_single_pixel import run_pixel, SIGMA
    conds = [("A-reference", None)] + [(k, p) for k, (p, _) in TASK_B_TRAPS.items()]
    lines = ["|ON|/OFF for one Gaussian pass (sigma = 2 px) vs spot speed",
             f"{'model':<14}" + "".join(f"{v:>9}px/ms" for v in (2, 4, 8, 16, 32))]
    for name, traps in conds:
        row = f"{name:<14}"
        for v in (2, 4, 8, 16, 32):
            vv = v / 1000
            t0 = 4 * SIGMA / vv
            _, i = run_pixel(lambda tt: np.exp(-((tt - t0) * vv) ** 2 / (2 * SIGMA ** 2)),
                             2 * t0 + 20000, 1.0 if v >= 32 else 2.0, REF, traps)
            row += f"{abs(i.min()) / i.max():>14.3f}"
        lines.append(row)
    lines += ["", "|ON|/OFF for a soft-edged bar (edge 0.3 px) at 2 px/ms vs time a pixel stays lit",
              f"{'model':<14}" + "".join(f"{L / 2:>9g} ms" for L in (1, 2, 5, 10, 20)) + "   rho_sq"]
    for name, traps in conds:
        sq = 1.0 if traps is None else TASK_B_TRAPS[name][1]
        lines.append(f"{name:<14}" + "".join(f"{bar_rho(traps, L):>12.3f}" for L in (1, 2, 5, 10, 20))
                     + f"{sq:>9.3f}")
    lines += ["", "|ON|/OFF for a 10 ms-dwell bar at 2 px/ms vs edge width (sharper = closer to a square pulse)",
              f"{'model':<14}" + "".join(f"{e:>9g} px" for e in (0.3, 0.1, 0.03)) + "   rho_sq"]
    for name, traps in conds[1:]:
        lines.append(f"{name:<14}" + "".join(f"{bar_rho(traps, 20, e):>12.3f}" for e in (0.3, 0.1, 0.03))
                     + f"{TASK_B_TRAPS[name][1]:>9.3f}")
    text = "\n".join(lines)
    with open(os.path.join(OUT, "asymmetry_regimes.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


def load_summary():
    with open(SUMMARY) as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def color_of(label):
    return C_SYM if label.startswith("A-") else C_TRAP if label.startswith("B-") else C_CTRL


ORDER = ["A-fast", "A-reference", "A-slow", "A-very-slow",
         "B-trap-sq1.0", "B-trap-sq0.7", "B-trap-sq0.5", "B-trap-sq0.2",
         "ctrl-1", "ctrl-0.7", "ctrl-0.5", "ctrl-0.2"]


def fig_metrics(rows):
    r = {x["condition"]: x for x in rows if x["scenario"] == "straight"}
    metrics = [("rho_G", "actual Gaussian ratio rho_G"), ("count_ratio", "ON / OFF event count"),
               ("on_det", "ON detection fraction"), ("iou", "footprint IoU (higher = better)"),
               ("err_mean", "mean trajectory error (px)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5.6), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for a, (key, title) in zip(axes, metrics):
        vals = [num(r[c][key]) for c in ORDER]
        a.barh(y, vals, color=[color_of(c) for c in ORDER], height=0.62)
        for yi, v in zip(y, vals):
            a.text(v, yi, f" {v:.2f}", va="center", fontsize=7.5, color=INK)
        a.set_title(title, fontsize=9, loc="left")
        a.grid(axis="y", visible=False)
        if key in ("rho_G", "count_ratio", "on_det"):
            a.axvline(1, color=INK, lw=0.8, ls=":")
        a.set_xlim(0, max(v for v in vals if not np.isnan(v)) * 1.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([named(c) for c in ORDER], fontsize=8)
    for lab, c in zip(axes[0].get_yticklabels(), ORDER):
        lab.set_color(color_of(c))
    fig.suptitle("Gaussian spot, straight pass, 2 px/ms; same threshold and tracker for every row.   "
                 "BLUE = Task A, no traps   ORANGE = Task B, physical traps (Prof. Cheng's trap model)   "
                 "GREEN = Eq. 6 imposed ON scaling (not physics)", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "fig3_metrics_straight.png"), dpi=130)
    plt.close(fig)


def fig_heatmap(rows):
    scen = [x for x in ["straight", "turn", "fast", "reversal", "disk"]
            if any(r["scenario"] == x for r in rows)]
    M = np.array([[num(next(x["err_mean"] for x in rows if x["scenario"] == s and x["condition"] == c))
                   for s in scen] for c in ORDER])
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    im = ax.imshow(np.log10(M), cmap="Blues", aspect="auto", vmin=np.log10(0.4), vmax=np.log10(15))
    for i in range(len(ORDER)):
        for j in range(len(scen)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if M[i, j] > 4 else INK)
    ax.set_xticks(range(len(scen))); ax.set_xticklabels(scen)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([named(c) for c in ORDER], fontsize=8)
    for lab, c in zip(ax.get_yticklabels(), ORDER):
        lab.set_color(color_of(c))
    ax.grid(False)
    for yline in (3.5, 7.5):
        ax.axhline(yline, color="white", lw=3)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    ticks = [0.5, 1, 2, 5, 10]
    cb.set_ticks(np.log10(ticks)); cb.set_ticklabels([str(t) for t in ticks])
    cb.set_label("mean trajectory error (px), log scale")
    ax.set_title("Mean trajectory error, every model x path\n"
                 "fast = 8 px/ms; disk = flat-top object (our addition)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_error_heatmap.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_pipeline(); print("fig1")
    fig_fringe(); print("fig2")
    rows = load_summary()
    fig_metrics(rows); print("fig3")
    fig_heatmap(rows); print("fig4")
    fig_dwell(); print("fig5")
    regimes_table()


if __name__ == "__main__":
    main()
