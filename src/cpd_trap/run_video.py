"""
run_video.py

CPD event imaging on an ordinary video (traffic footage or any clip).

  video frame -> grayscale u in [0, 1] (no binarization) -> resize
  -> linear interpolation between frames on the 2 us CPD grid
  -> box (A) or box (S) at every pixel -> ON/OFF events (theta = 4.5 nA)
  -> event-state tracker (ON sets, OFF resets)

Time scale. A video frame interval is not a CPD time. --frame-us sets how
much CPD time one video frame represents (default 1000 us, i.e. a 30 fps
clip is replayed ~33x faster than real time). The intermediate optical
trajectory between frames is an ASSUMED linear interpolation, not recovered
sub-frame motion. Why not real time: the readout is a band-pass,
i_R ~ -(tau1 tau2 / RC_s) dI/dt at low frequency (30 us for the reference
pair), so a full 0 -> 1 change spread over a 33 ms frame gives only ~3 nA,
below theta. Pass --frame-us 33333 to see this directly.

Start-up. The CPD starts at T = i_R = z = 0, so the first frame is a step
from darkness. Frame 0 is held for --settle-ms (default 30 ms) and events in
that pre-roll are discarded.

Conditions default to the symmetric reference (A-reference), the physical
trap conditions B-trap-sq0.5 and B-trap-sq0.2, and the Eq. (6) output
control ctrl-0.2. With --truth CSV (frame, x, y) the event-state centroid is
scored against a known single-object trajectory.

Outputs (outputs/cpd_trap/video/<clip>/):
  events_<clip>.gif     frame | per condition: ON/OFF events in that frame
                        interval, and the event-state map
  rates_<clip>.png      ON and OFF event rate vs time per condition
  summary_<clip>.md     event counts, ratios, active pixels left at the last frame,
                        and trajectory error when --truth is given

Usage:
  python make_test_video.py
  python run_video.py ../../outputs/cpd_trap/video/synthetic_traffic.mp4 \
      --truth ../../outputs/cpd_trap/video/synthetic_traffic_truth.csv
  python run_video.py path/to/traffic.mp4 --width 128 --max-frames 150
  python run_video.py ../../videos/ayalon_freeway_480p.webm       --crop 400 190 720 370 --width 160 --max-frames 150 --tag ayalon_freeway
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(__file__))
from cpd_trap_model import CPDArray, TASK_A_PAIRS, TASK_B_TRAPS, THETA, named, ALIAS
from simulate import EventReadout
from run_trajectory_study import ctrl_transform, RHO_SQ

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "cpd_trap", "video")
REF = TASK_A_PAIRS["A-reference"]
# actual one-pass Gaussian ratios (validate_single_pixel.py), for labelling
RHO_G = {"A-fast": 1.000, "A-reference": 1.004, "A-slow": 1.047, "A-very-slow": 1.334,
         "B-trap-sq1.0": 1.124, "B-trap-sq0.7": 1.017, "B-trap-sq0.5": 1.052,
         "B-trap-sq0.2": 1.266, "ctrl-1": 1.0, "ctrl-0.7": 0.7, "ctrl-0.5": 0.5, "ctrl-0.2": 0.2}


def load_frames(path, width, max_frames, start, crop=None, stabilize=False):
    """Grayscale frames in [0, 1]. stabilize=True removes global camera
    translation (shake) by registering every full frame to the first with
    phase correlation BEFORE cropping -- a preprocessing step standing in for
    a fixed sensor, not part of the device model."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, k, ref = [], 0, None
    window = None
    while len(frames) < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        k += 1
        if k <= start:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        if stabilize:
            if ref is None:
                ref = g
                window = cv2.createHanningWindow(g.shape[::-1], cv2.CV_64F)
            else:
                (dx, dy), _ = cv2.phaseCorrelate(ref, g, window)
                M = np.float64([[1, 0, -dx], [0, 1, -dy]])
                g = cv2.warpAffine(g, M, g.shape[::-1], flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
        if crop:
            x0, y0, x1, y1 = crop
            g = g[y0:y1, x0:x1]
        h = int(round(g.shape[0] * width / g.shape[1]))
        frames.append(cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise SystemExit("no frames read")
    return np.clip(np.array(frames), 0, 1), fps, start


def condition_setup(name):
    """(circuit, traps, readout transform) for a condition label."""
    if name.startswith("ctrl-"):
        return REF, None, ctrl_transform(float(name[5:]))
    if name in TASK_A_PAIRS:
        return TASK_A_PAIRS[name], None, None
    return REF, TASK_B_TRAPS[name][0], None


def simulate_video(frames, frame_us, settle_us, name, dt=2.0):
    circuit, traps, transform = condition_setup(name)
    n_f, H, W = frames.shape
    cpd = CPDArray((H, W), circuit, traps)
    ro = EventReadout(name, (H, W), transform)

    def u_fn(t):
        s = (t - settle_us) / frame_us
        if s <= 0:
            return frames[0]
        i = min(int(s), n_f - 1)
        if i >= n_f - 1:
            return frames[-1]
        a = s - i
        return (1 - a) * frames[i] + a * frames[i + 1]

    t_end = settle_us + (n_f - 1) * frame_us
    n = int(round(t_end / dt))
    ev_img = np.zeros((n_f, H, W), np.int8)
    state = np.zeros((n_f, H, W), bool)
    bip = np.zeros((n_f, H, W), bool)
    n_ev_before = 0
    for k in range(n):
        t0 = k * dt
        i_R = cpd.step(u_fn, t0, dt)
        t = t0 + dt
        ro.update(i_R, t)
        if t <= settle_us:
            ro.events.clear()
            ro.state[:] = False
            ro.bipolar[:] = 0
            continue
        fi = min(int((t - settle_us - 1e-9) // frame_us) + 1, n_f - 1)
        new = ro.events[n_ev_before:]
        for _, x, y, p in new:
            ev_img[fi, y, x] = p
        n_ev_before = len(ro.events)
        if abs((t - settle_us) / frame_us - round((t - settle_us) / frame_us)) < dt / (2 * frame_us):
            fs = min(int(round((t - settle_us) / frame_us)), n_f - 1)
            state[fs] = ro.state
            bip[fs] = ro.bipolar != 0
    ev = np.array(ro.events, float).reshape(-1, 4)
    ev[:, 0] -= settle_us
    return ev, ev_img, state, bip


def track_error(maps, truth, obj, xx, yy, W):
    """Centroid-vs-truth error per object id (and 'all'); frames where the
    object centre is outside the view are skipped."""
    out = {}
    for o in [None] + sorted(set(obj[~np.isnan(obj)].astype(int))):
        errs, valid, vis = [], 0, 0
        for f in range(len(maps)):
            if np.isnan(truth[f, 0]) or not (0 <= truth[f, 0] < W):
                continue
            if o is not None and obj[f] != o:
                continue
            vis += 1
            st = maps[f]
            if st.any():
                valid += 1
                errs.append(np.hypot(xx[st].mean() - truth[f, 0], yy[st].mean() - truth[f, 1]))
        out["all" if o is None else o] = (np.array(errs), valid / vis if vis else np.nan)
    return out


def family(name):
    """(short tag, one-line meaning, colour) saying whether a condition is the
    symmetric or an asymmetric response. Colours match the study figures."""
    if name.startswith("A"):
        return "SYMMETRIC", "no traps, ON and OFF equally strong", "#2a78d6"
    if name.startswith("B"):
        return ("ASYMMETRIC (traps)",
                f"Prof. Cheng's trap model, weaker ON / stronger OFF, square-pulse ON/OFF {RHO_SQ[name]:.2f}",
                "#c4501f")
    return ("ASYMMETRIC (imposed)",
            f"symmetric circuit with ON scaled x{name[5:]} afterwards, not trap physics", "#127a55")


def row_heading(name):
    tag, meaning, _ = family(name)
    return f"{name}  =  {tag}:  {meaning}"


def snapshot_sheet(frames, results, truth, clip, frame_us, start, path, n_cols=4):
    """Static 'screenshot' of a run: columns are evenly spaced frames; rows
    are the input frame, then for every condition its ON/OFF events (last 3
    frame intervals) and its spec event-state map (with centroid and, if
    known, the true object centre)."""
    n_f, H, W = frames.shape
    cols = np.linspace(0.12 * n_f, 0.88 * n_f, n_cols).astype(int)
    names = list(results)
    n_rows = 1 + 2 * len(names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 1.95 * n_rows * H / W * 1.6),
                             squeeze=False)
    yy, xx = np.mgrid[0:H, 0:W]
    for j, f in enumerate(cols):
        axes[0, j].imshow(frames[f], cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"frame {f + start}  (t = {f * frame_us / 1000:.0f} ms)", fontsize=8)
        if truth is not None and not np.isnan(truth[f, 0]):
            axes[0, j].plot(truth[f, 0], truth[f, 1], "c+", ms=8)
        for i, name in enumerate(names):
            _, ev_img, state, _ = results[name]
            ev = np.sign(ev_img[max(0, f - 2):f + 1].sum(axis=0))
            a, b = axes[1 + 2 * i, j], axes[2 + 2 * i, j]
            a.imshow(frames[f], cmap="gray", vmin=0, vmax=1, alpha=0.35)
            rgba = np.zeros((H, W, 4))
            rgba[ev > 0] = (0.9, 0.1, 0.1, 1)
            rgba[ev < 0] = (0.1, 0.3, 0.95, 1)
            a.imshow(rgba)
            b.imshow(state[f], cmap="Greys", vmin=0, vmax=1)
            if state[f].any():
                b.plot(xx[state[f]].mean(), yy[state[f]].mean(), "ro", ms=4)
            if truth is not None and not np.isnan(truth[f, 0]):
                b.plot(truth[f, 0], truth[f, 1], "c+", ms=8)
            if j == 0:
                tag, _, fam = family(name)
                tag = tag.replace(" (", "\n(")
                a.set_ylabel(f"{name}\n{tag}\nevents", fontsize=7.5, color=fam, fontweight="bold")
                b.set_ylabel(f"{name}\n{tag}\ntracker map", fontsize=7.5, color=fam, fontweight="bold")
    axes[0, 0].set_ylabel("input frame", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    fig.suptitle(f"{clip}: red = ON, blue = OFF (last 3 frame intervals); black = spec "
                 f"event-state map, red dot = centroid" + (", cyan + = true centre" if truth is not None else ""),
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--start", type=int, default=0, help="skip this many frames")
    ap.add_argument("--frame-us", type=float, default=1000.0)
    ap.add_argument("--settle-ms", type=float, default=30.0)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="crop source frames (source pixels) before resizing")
    ap.add_argument("--stabilize", action="store_true",
                    help="remove global camera shake (phase correlation) before the sensor model")
    ap.add_argument("--truth", default=None,
                    help="CSV frame,x,y[,object] in resized pixels; optional object id "
                         "column gives per-object errors")
    ap.add_argument("--conditions", nargs="+",
                    default=["A-reference", "B-trap-sq0.5", "B-trap-sq0.2", "ctrl-0.2"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--replot", action="store_true",
                    help="redraw figures from the cached arrays of a previous run")
    args = ap.parse_args(argv)

    frames, fps, start = load_frames(args.video, args.width, args.max_frames, args.start,
                                        args.crop, args.stabilize)
    clip = args.tag or os.path.splitext(os.path.basename(args.video))[0]
    if args.frame_us != 1000.0 and args.tag is None:
        clip += f"_{args.frame_us:g}us"
    out = os.path.join(OUT_ROOT, clip)
    os.makedirs(out, exist_ok=True)
    n_f, H, W = frames.shape
    print(f"{clip}: {n_f} frames at {W}x{H}, source {fps:.1f} fps, "
          f"1 frame = {args.frame_us:g} us CPD time")

    truth = None
    if args.truth:
        tr = np.loadtxt(args.truth, delimiter=",", skiprows=1)
        tr = tr[(tr[:, 0] >= start) & (tr[:, 0] < start + n_f)]
        truth = np.full((n_f, 2), np.nan)
        obj = np.full(n_f, np.nan)
        truth[(tr[:, 0] - start).astype(int)] = tr[:, 1:3]
        if tr.shape[1] > 3:
            obj[(tr[:, 0] - start).astype(int)] = tr[:, 3]

    # Simulated arrays are cached next to the outputs (not committed) so the
    # figures can be redrawn with --replot without re-running the sensor model.
    cache = os.path.join(out, f"cache_{clip}.npz")
    results = {}
    if args.replot and os.path.exists(cache):
        z = np.load(cache)
        for name in args.conditions:
            results[name] = tuple(z[f"{name}|{k}"] for k in ("ev", "ev_img", "state", "bip"))
        print(f"  replotting from {cache}")
    else:
        for name in args.conditions:
            t0 = time.time()
            results[name] = simulate_video(frames, args.frame_us, args.settle_ms * 1000, name)
            print(f"  {name:<14} {len(results[name][0]):>7} events  {time.time() - t0:5.1f}s", flush=True)
        np.savez_compressed(cache, **{f"{n}|{k}": a for n, r in results.items()
                                      for k, a in zip(("ev", "ev_img", "state", "bip"), r)})

    # ---- summary -----------------------------------------------------------
    yy, xx = np.mgrid[0:H, 0:W]
    lines = [f"# CPD event imaging on `{clip}`", "",
             f"{n_f} frames, {W}x{H} px, 1 video frame = {args.frame_us:g} us of CPD time, "
             f"theta = {THETA * 1e3:g} nA, frame 0 held {args.settle_ms:g} ms before scoring"
             + (", camera shake removed (--stabilize)." if args.stabilize else "."), ""]
    cols = ["condition", "what it is", "rho_G", "rho_sq", "ON", "OFF", "ON/OFF count", "active px (last frame)",
            "mean active px"]
    if truth is not None:
        cols += ["tracker", "object", "traj err mean (px)", "traj err p90 (px)", "valid frac"]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, (ev, ev_img, state, bip) in results.items():
        n_on = int((ev[:, 3] == 1).sum()) if len(ev) else 0
        n_off = int((ev[:, 3] == -1).sum()) if len(ev) else 0
        row = [name, ALIAS.get(name, ""), f"{RHO_G.get(name, np.nan):.3f}",
               f"{RHO_SQ[name]:.3f}" if name.startswith("B") else "-",
               str(n_on), str(n_off), f"{n_on / n_off:.3f}" if n_off else "-",
               str(int(state[-1].sum())), f"{state.sum(axis=(1, 2)).mean():.1f}"]
        if truth is None:
            lines.append("| " + " | ".join(row) + " |")
            continue
        for tname, maps in (("spec ON-set/OFF-reset", state), ("bipolar", bip)):
            for o, (errs, vf) in track_error(maps, truth, obj, xx, yy, W).items():
                r = row if (tname.startswith("spec") and o == "all") else [""] * len(row)
                lines.append("| " + " | ".join(r + [
                    tname, str(o), f"{errs.mean():.2f}" if len(errs) else "-",
                    f"{np.percentile(errs, 90):.2f}" if len(errs) else "-",
                    f"{vf:.3f}"]) + " |")
    lines += ["", "active px (last frame) = pixels still marked active by the event-state map at "
              "the last frame. Once every object has left the view these are trails "
              "(an ON with no matching OFF: a missed or delayed OFF event).",
              "Labels carry the single-pass Gaussian ratio rho_G and, for physical trap "
              "conditions, the square-pulse calibration ratio rho_sq. ctrl-* rows are the "
              "Eq. (6) phenomenological output control, not a trap prediction.",
              "Trackers: 'spec' is the task's rule (ON sets a pixel, OFF resets it). "
              "'bipolar' (our addition, for video) lets the first event of a passage set the "
              "pixel and the opposite polarity clear it, so dark objects on a brighter "
              "background (which fire OFF first) are tracked too; it is identical to the spec "
              "rule for bright objects on dark."]
    with open(os.path.join(out, f"summary_{clip}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    snapshot_sheet(frames, results, truth, clip, args.frame_us, start,
                   os.path.join(out, f"snapshot_{clip}.png"))

    # ---- event-rate plot ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 3.8))
    bins = np.arange(n_f + 1) * args.frame_us
    for name, (ev, *_) in results.items():
        for p, ls in ((1, "-"), (-1, "--")):
            h, _ = np.histogram(ev[ev[:, 3] == p, 0], bins) if len(ev) else (np.zeros(n_f), None)
            ax.plot(bins[:-1] / 1000, h, ls, lw=1, label=f"{name} {'ON' if p == 1 else 'OFF'}")
    ax.set_xlabel("CPD time (ms)"); ax.set_ylabel("events per frame interval")
    ax.legend(fontsize=6, ncol=2); ax.set_title(f"{clip}: event rates")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"rates_{clip}.png"), dpi=120)
    plt.close(fig)

    # ---- event movie -------------------------------------------------------
    # Each row gets a large coloured heading above it naming the model and
    # whether it is the symmetric or an asymmetric response (a sideways
    # y-label was unreadable at GIF resolution).
    names = list(results)
    fig_h = 0.9 + 3.0 * len(names) * min(1.0, 1.6 * H / W)
    fig, axes = plt.subplots(len(names), 4, figsize=(14, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.9 / fig_h, bottom=0.01,
                        wspace=0.04, hspace=0.75)
    handles = []
    for i, (row, name) in enumerate(zip(axes, names)):
        _, ev_img, state, bip = results[name]
        a = row[0].imshow(frames[0], cmap="gray", vmin=0, vmax=1)
        b = row[1].imshow(ev_img[0], cmap="bwr", vmin=-1, vmax=1)
        c = row[2].imshow(state[0], cmap="Greys", vmin=0, vmax=1)
        d = row[3].imshow(bip[0], cmap="Greys", vmin=0, vmax=1)
        dots = [r.plot([], [], "ro", ms=4)[0] for r in row[2:]]
        for x, t in zip(row, ("input frame", "events: ON red / OFF blue",
                              "spec tracker (ON sets, OFF resets)", "bipolar tracker (our addition)")):
            x.set_title(t, fontsize=9)
            x.set_xticks([]); x.set_yticks([])
        _, _, fam = family(name)
        top = row[0].get_position().y1
        fig.text(0.01, top + 0.34 / fig_h, f"Row {i + 1}:  {row_heading(name)}",
                 fontsize=12, fontweight="bold", color=fam, va="bottom")
        handles.append((a, b, c, d, dots, ev_img, state, bip))
    ttl = fig.text(0.5, 1 - 0.12 / fig_h, "", ha="center", va="top", fontsize=12)

    def draw(f):
        for a, b, c, d, dots, ev_img, state, bip in handles:
            a.set_data(frames[f]); b.set_data(ev_img[f])
            c.set_data(state[f]); d.set_data(bip[f])
            for dot, m in zip(dots, (state[f], bip[f])):
                if truth is not None and m.any():
                    dot.set_data([xx[m].mean()], [yy[m].mean()])
                else:
                    dot.set_data([], [])
        ttl.set_text(f"{clip}  frame {f + start}  (t = {f * args.frame_us / 1000:.1f} ms CPD time)")
        return []

    # every 3rd frame at 70 dpi keeps a 150-frame clip to a few MB for GitHub
    FuncAnimation(fig, draw, frames=range(0, n_f, 3)).save(
        os.path.join(out, f"events_{clip}.gif"), writer=PillowWriter(fps=8), dpi=70)
    plt.close(fig)
    print(f"saved to {out}")
    return out


if __name__ == "__main__":
    main()
