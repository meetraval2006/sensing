"""
run_trajectory_study.py

Gaussian-spot event imaging + trajectory tracing on 64 x 64 for:

  Task A  : four electrical pairs, box (A), no traps
  Task B  : zero-trap control (= A-reference) + four active-trap conditions,
            box (S), reference circuit, Gamma = 2400 uA*us
  Control : Eq. (6) output-level ratio control on the A-reference current,
            rho* = 1, 0.7, 0.5, 0.2 (phenomenological, NOT a trap prediction)

on straight / turn / fast / reversal paths, a flat-top disk (our addition,
see scenarios.py), and on repeated passes after
1, 3, 10, 30 ms of darkness. Same theta (4.5 nA) and tracker everywhere.

Outputs (outputs/cpd_trap/):
  summary_basic.csv / .md     metrics table, every condition x basic path.
                              rho_G = one-pass ratio on the straight Gaussian
                              path (the label); rho_this_input = |ON|/OFF at
                              this scenario's first representative pixel
  summary_repeat.csv / .md    per-pass metrics for the repeated-pass runs
  <scenario>/panels_<group>.png
                              u | 1 ms event slice | event-state map | path
  <scenario>/pixel_traces.png representative pixel u, i_R (and T)
  repeat_history.png          second-pass vs first-pass response vs dark gap
  movie_straight.gif          spot movie: A-ref / B-trap-sq0.2 / ctrl 0.2 / A-very-slow
  movie_turn.gif, movie_disk.gif
"""

import csv
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(__file__))
from cpd_trap_model import TASK_A_PAIRS, TASK_B_TRAPS, THETA, named
from simulate import run, metrics, gaussian_ratio
import scenarios as S

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "cpd_trap")
REF = TASK_A_PAIRS["A-reference"]
RHO0 = 1.00397            # trap-free reference rho_G, validate_single_pixel.py
CTRL_TARGETS = (1.0, 0.7, 0.5, 0.2)

GROUPS = {
    "taskA": ["A-fast", "A-reference", "A-slow", "A-very-slow"],
    "taskB": ["A-reference"] + list(TASK_B_TRAPS),
    "control": [f"ctrl-{r:g}" for r in CTRL_TARGETS],
}
GROUP_TITLE = {"taskA": "Task A: no traps, four circuit speeds",
               "taskB": "Task B: symmetric baseline vs physical trap models",
               "control": "Eq. 6 control: baseline current with ON lobe scaled (not physics)"}
RHO_SQ = {"A-reference": 1.000, **{k: v[1] for k, v in TASK_B_TRAPS.items()}}

METRIC_COLS = ["n_on", "n_off", "count_ratio", "on_det", "off_det",
               "on_lat_med", "on_lat_p90", "off_lat_med", "off_lat_p90",
               "iou", "err_mean", "err_p90", "valid_frac", "reversal_delay"]


def ctrl_transform(target):
    k = target / RHO0
    return lambda s: np.where(s < 0, k * s, s)


def run_condition(name, sc):
    """One CPD run; the A-reference run also carries the four Eq. (6)
    controlled-ratio readouts. Returns {readout_label: res-view}."""
    if name in TASK_A_PAIRS:
        circuit, traps = TASK_A_PAIRS[name], None
    else:
        circuit, traps = REF, TASK_B_TRAPS[name][0]
    readouts = [(name, None)]
    if name == "A-reference":
        readouts += [(f"ctrl-{r:g}", ctrl_transform(r)) for r in CTRL_TARGETS]
    return run(sc.u, sc.t_end, S.GRID, circuit, traps, readouts=readouts,
               rep_pixels=sc.rep_pixels, true_center=sc.true_center)


def group_of(label):
    return [g for g, lst in GROUPS.items() if label in lst]


def describe(label, rho_g, rho_here=None):
    """One-line row header: code label | plain name | ratios. rho_here is the
    |ON|/OFF actually measured on this scenario's input; rho_G (straight
    Gaussian pass) and rho_sq (square-pulse calibration) are the reference
    labels the task file asks for."""
    s = named(label)
    parts = []
    if rho_here is not None:
        parts.append(f"|ON|/OFF on this input = {rho_here:.2f}")
    parts.append(f"rho_G = {rho_g:.3f}")
    if label.startswith("B"):
        parts.append(f"rho_sq = {RHO_SQ[label]:.3f}")
    return s + "   --   " + ",  ".join(parts)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def panel_figure(sc, results, labels, rho_g, path, title, rho_here=None):
    """Rows = conditions. Snapshot at the frame nearest the middle of the
    visible interval, plus the full traced path."""
    first = results[labels[0]]
    ft = np.array(first["frames"]["t"])
    vis_t = first["t"][first["visible"]]
    t_snap = vis_t[len(vis_t) // 2] if len(vis_t) else ft[len(ft) // 2]
    fi = int(np.argmin(np.abs(ft - t_snap)))
    t_snap = ft[fi]

    fig, axes = plt.subplots(len(labels), 4, figsize=(15, 3.7 * len(labels)), squeeze=False)
    for k, (row, lab) in enumerate(zip(axes, labels)):
        res = results[lab]
        u = res["frames"]["u"][fi]
        ev = res["events"][lab]
        sl = np.zeros(S.GRID)
        if len(ev):
            w = ev[(ev[:, 0] > t_snap - 1000) & (ev[:, 0] <= t_snap)]
            for t, x, y, p in w:
                sl[int(y), int(x)] = p
        st = res["frames"]["state"][lab][fi]

        row[0].imshow(u, cmap="gray", vmin=0, vmax=1, origin="lower")
        row[0].contour(u, levels=[0.2], colors="y", linewidths=0.8)
        row[0].set_title("input u (yellow: u = 0.2)", fontsize=8)
        row[1].imshow(sl, cmap="bwr", vmin=-1, vmax=1, origin="lower")
        row[1].set_title("events in last 1 ms (red ON, blue OFF)", fontsize=8)
        row[2].imshow(st, cmap="Greys", vmin=0, vmax=1, origin="lower")
        row[2].contour(u, levels=[0.2], colors="orange", linewidths=0.8)
        row[2].set_title("event-state map (black) vs u>=0.2", fontsize=8)
        row[3].plot(res["r_true"][:, 0], res["r_true"][:, 1], "k-", lw=1.2, label="true")
        est = res["r_est"][lab]
        row[3].plot(est[:, 0], est[:, 1], "-", color="tab:red", lw=0.8, label="traced")
        row[3].set_xlim(-2, 66); row[3].set_ylim(-2, 66); row[3].set_aspect("equal")
        row[3].set_title("traced vs true path", fontsize=8)
        row[3].legend(fontsize=6, loc="lower right")
        here = rho_here[lab] if rho_here else None
        row[0].annotate(describe(lab, rho_g[lab], here), xy=(0, 1.17), xycoords="axes fraction",
                        fontsize=9.5, weight="bold", ha="left", va="bottom",
                        color="#2a78d6" if lab.startswith("A") else
                        "#c4501f" if lab.startswith("B") else "#127a55")
        for a in row:
            a.tick_params(labelsize=6)
    fig.suptitle(f"{title}" + "\n" + f"{sc.note};  snapshot at t = {t_snap / 1000:.1f} ms" + "\n"
                 f"theta = {THETA * 1e3:g} nA;  same threshold and tracker (ON sets, OFF resets) "
                 "for every row", fontsize=11, x=0.04, ha="left")
    height = 3.7 * len(labels)
    fig.subplots_adjust(left=0.04, right=0.985, top=1 - 1.45 / height, bottom=0.25 / height,
                        hspace=0.62, wspace=0.18)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pixel_trace_figure(sc, results, labels, rho_g, path):
    n = len(sc.rep_pixels)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 6.5), squeeze=False, sharex=True)
    t_ms = results[labels[0]]["t"] / 1000
    for j, (x, y) in enumerate(sc.rep_pixels):
        axes[0, j].plot(t_ms, results[labels[0]]["traces"]["u"][:, j], "k-", lw=1)
        axes[0, j].set_title(f"pixel ({x},{y}): grayscale input u", fontsize=9)
        for lab in labels:
            res = results[lab]
            axes[1, j].plot(t_ms, res["traces"][lab][:, j] * 1e3, lw=0.9,
                            label=named(lab))
            if res["T_trace"] is not None and lab.startswith("B"):
                axes[0, j].plot(t_ms, res["T_trace"][:, j], lw=0.8, ls="--",
                                label=f"T ({lab})")
        for s in (1, -1):
            axes[1, j].axhline(s * THETA * 1e3, color="k", ls=":", lw=0.7)
        axes[1, j].set_ylabel("i_R (nA)"); axes[1, j].set_xlabel("time (ms)")
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(fontsize=6)
    axes[1, 0].legend(fontsize=6)
    # zoom to the passage of the first representative pixel (+ trap tail)
    u0 = results[labels[0]]["traces"]["u"][:, 0]
    if u0.max() > 0:
        lit = t_ms[u0 > 1e-3]
        axes[1, 0].set_xlim(lit[0] - 1, min(t_ms[-1], lit[-1] + 8))
    fig.suptitle(f"Representative pixels -- {sc.note}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def movie(sc, results, labels, path, every=2):
    first = results[labels[0]]
    ft = first["frames"]["t"]
    idx = list(range(0, len(ft), every))
    fig, axes = plt.subplots(len(labels), 3, figsize=(9, 3 * len(labels)), squeeze=False)
    ims = []
    for row, lab in zip(axes, labels):
        res = results[lab]
        vmax = max(np.abs(f).max() for f in res["frames"]["i_R"][lab]) or 1
        a = row[0].imshow(first["frames"]["u"][0], cmap="gray", vmin=0, vmax=1, origin="lower")
        b = row[1].imshow(res["frames"]["i_R"][lab][0], cmap="bwr_r", vmin=-vmax, vmax=vmax, origin="lower")
        c = row[2].imshow(res["frames"]["state"][lab][0], cmap="Greys", vmin=0, vmax=1, origin="lower")
        row[2].plot(res["r_true"][:, 0], res["r_true"][:, 1], "c-", lw=0.6)
        (dot,) = row[2].plot([], [], "ro", ms=4)
        row[0].set_ylabel(named(lab).replace("  |  ", "\n"), fontsize=7)
        row[1].set_title("i_R (red: negative = rising light = ON)", fontsize=7)
        row[2].set_title("event-state map, red = r_est", fontsize=7)
        for x in row:
            x.set_xticks([]); x.set_yticks([])
        ims.append((a, b, c, dot, res, lab))
    ttl = fig.suptitle("")

    def draw(k):
        fi = idx[k]
        t = ft[fi]
        for a, b, c, dot, res, lab in ims:
            a.set_data(res["frames"]["u"][fi])
            b.set_data(res["frames"]["i_R"][lab][fi])
            c.set_data(res["frames"]["state"][lab][fi])
            ti = min(int(t / res["dt"]) - 1, len(res["t"]) - 1)
            e = res["r_est"][lab][ti]
            dot.set_data([e[0]], [e[1]]) if not np.isnan(e[0]) else dot.set_data([], [])
        ttl.set_text(f"{sc.note}   t = {t / 1000:.1f} ms")
        return []

    anim = FuncAnimation(fig, draw, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"


def write_tables(rows, cols, stem):
    with open(os.path.join(OUT_DIR, stem + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    with open(os.path.join(OUT_DIR, stem + ".md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) if not isinstance(r.get(c), str)
                                      else r.get(c) for c in cols) + " |\n")


# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    conditions = list(TASK_A_PAIRS) + list(TASK_B_TRAPS)
    basic_rows = []
    rho_g_straight = {}

    for sc in S.BASIC:
        sdir = os.path.join(OUT_DIR, sc.name)
        os.makedirs(sdir, exist_ok=True)
        results = {}
        for name in conditions:
            t0 = time.time()
            res = run_condition(name, sc)
            for lab in res["labels"]:
                results[lab] = res
            print(f"  [{sc.name}] {name:<14} {time.time() - t0:5.1f}s", flush=True)
        # actual Gaussian ratio at the first rep pixel (crossed through center
        # on straight/fast/turn); labels use the straight-path value
        rho_g = {lab: gaussian_ratio(results[lab]["traces"][lab][:, 0]) for lab in results}
        if sc.name == "straight":
            rho_g_straight = dict(rho_g)
        labels_rho = rho_g_straight or rho_g
        for lab, res in results.items():
            m = metrics(res, lab, t_reversal=sc.t_reversal, axis=sc.axis)
            basic_rows.append({"scenario": sc.name, "condition": lab,
                               "group": "/".join(group_of(lab)),
                               "rho_sq": fmt(RHO_SQ.get(lab)) if lab.startswith("B") or lab == "A-reference" else "-",
                               "rho_G": fmt(labels_rho[lab]),
                               "rho_this_input": fmt(rho_g[lab]), **m})
        for g, labels in GROUPS.items():
            panel_figure(sc, results, labels, labels_rho,
                         os.path.join(sdir, f"panels_{g}.png"), GROUP_TITLE[g], rho_here=rho_g)
        pixel_trace_figure(sc, results, GROUPS["taskA"], labels_rho,
                           os.path.join(sdir, "pixel_traces_taskA.png"))
        pixel_trace_figure(sc, results, GROUPS["taskB"], labels_rho,
                           os.path.join(sdir, "pixel_traces_taskB.png"))
        pixel_trace_figure(sc, results, ["A-reference"] + GROUPS["control"], labels_rho,
                           os.path.join(sdir, "pixel_traces_control.png"))
        if sc.name == "straight":
            movie(sc, results, ["A-reference", "B-trap-sq0.2", "ctrl-0.2", "A-very-slow"],
                  os.path.join(OUT_DIR, "movie_straight.gif"))
        if sc.name == "disk":
            movie(sc, results, ["A-reference", "B-trap-sq0.5", "B-trap-sq0.2", "ctrl-0.5"],
                  os.path.join(OUT_DIR, "movie_disk.gif"))
        if sc.name == "turn":
            movie(sc, results, ["A-reference", "B-trap-sq0.2", "ctrl-0.2"],
                  os.path.join(OUT_DIR, "movie_turn.gif"), every=3)
        del results

    write_tables(basic_rows, ["scenario", "condition", "group", "rho_sq", "rho_G",
                                "rho_this_input"] + METRIC_COLS,
                 "summary_basic")

    # ---- repeated passes after different dark intervals -------------------
    rep_rows = []
    hist = {}
    for dark in S.REPEAT_DARK_MS:
        sc = S.repeat(dark)
        for name in ["A-reference"] + list(TASK_B_TRAPS):
            t0 = time.time()
            res = run_condition(name, sc)
            print(f"  [{sc.name}] {name:<14} {time.time() - t0:5.1f}s", flush=True)
            for lab in res["labels"]:
                if lab.startswith("ctrl") and lab not in ("ctrl-0.2",):
                    continue
                per = []
                for p in (0, 1):
                    keep = lambda t, x, y, p=p: sc.pass_index(t, x, y) == p
                    m = metrics(res, lab, keep=keep)
                    per.append(m)
                    rep_rows.append({"dark_ms": dark, "condition": lab, "pass": p + 1,
                                     **{k: m[k] for k in ("n_on", "n_off", "on_det", "off_det",
                                                          "on_lat_med", "off_lat_med")}})
                # peak response of the centre pixel on each passage
                tr = res["traces"][lab][:, 0]
                t = res["t"]
                split = sc.arrival(32, 32, sc.delays[1]) - sc.delays[1] / 2
                a, b = tr[t < split], tr[t >= split]
                hist.setdefault(lab, []).append(
                    (dark, abs(b.min()) / abs(a.min()), b.max() / a.max(),
                     per[1]["on_lat_med"] - per[0]["on_lat_med"],
                     per[1]["off_lat_med"] - per[0]["off_lat_med"]))
            if dark == 3 and name == "B-trap-sq0.2":
                pixel_trace_figure(sc, {name: res}, [name], {name: np.nan},
                                   os.path.join(OUT_DIR, "repeat_3ms_pixel_traces_trap0.2.png"))
    write_tables(rep_rows, ["dark_ms", "condition", "pass", "n_on", "n_off", "on_det",
                            "off_det", "on_lat_med", "off_lat_med"], "summary_repeat")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for lab, h in hist.items():
        h = np.array(h)
        axes[0].plot(h[:, 0], h[:, 1], "o-", label=lab)
        axes[1].plot(h[:, 0], h[:, 2], "o-", label=lab)
        axes[2].plot(h[:, 0], h[:, 3], "o-", label=lab)
    axes[0].set_ylabel("pass-2 / pass-1 ON peak |i_R|")
    axes[1].set_ylabel("pass-2 / pass-1 OFF peak")
    axes[2].set_ylabel("pass-2 minus pass-1 median ON latency (us)")
    for a in axes:
        a.set_xscale("log"); a.set_xlabel("dark interval between passes (ms)")
        a.axhline(1 if a is not axes[2] else 0, color="k", lw=0.6, ls=":")
        a.legend(fontsize=7)
    fig.suptitle("Trap history: second passage vs first, centre pixel (32,32), T never reset")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "repeat_history.png"), dpi=120)
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
