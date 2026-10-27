"""
simulate.py

Streaming 2-D simulation: optical input u(x,y,t) -> CPDArray -> ON/OFF events
-> event-state tracker -> metrics. Nothing of size (n_steps, H, W) is kept in
memory; events, per-step tracker outputs, a few representative pixel traces
and sub-sampled movie frames are recorded as the run streams.

Event rule (spec "Events and trajectory outputs"):
  ON  = first crossing of i_R below -theta in a pixel passage
  OFF = first crossing of i_R above +theta in a pixel passage
A polarity re-arms only once i_R has come back inside +-theta/2, so each lobe
of the (single-signed, non-ringing) band-pass response gives at most one
event. Rising light -> negative i_R -> ON; falling light -> positive -> OFF.

Tracker: ON sets the pixel to 1 in the estimated event-state map, OFF resets
it to 0; r_est is the centroid of active pixels. The binary map is built
AFTER the grayscale CPD response -- u itself is never thresholded except to
define the u >= 0.2 reference footprint used by the metrics.

Readouts: several event detectors can watch the same CPD run, each through a
transform of i_R. This is how the Eq. (6) controlled-ratio comparison is
done: one reference run, extra detectors on (rho*/rho0) * negative lobe.
"""

import numpy as np

from cpd_trap_model import CPDArray, THETA

U_FOOT = 0.2


class EventReadout:
    def __init__(self, label, shape, transform=None, theta=THETA):
        self.label = label
        self.transform = transform
        self.theta = theta
        self.on_armed = np.ones(shape, bool)
        self.off_armed = np.ones(shape, bool)
        self.state = np.zeros(shape, bool)
        self.bipolar = np.zeros(shape, np.int8)
        self.events = []                       # (t_us, x, y, polarity)

    def update(self, i_R, t):
        s = i_R if self.transform is None else self.transform(i_R)
        th = self.theta
        on = self.on_armed & (s < -th)
        off = self.off_armed & (s > th)
        self.on_armed &= ~on
        self.off_armed &= ~off
        self.on_armed |= s > -0.5 * th
        self.off_armed |= s < 0.5 * th
        if on.any():
            ys, xs = np.nonzero(on)
            self.events.extend((t, x, y, 1) for y, x in zip(ys, xs))
        if off.any():
            ys, xs = np.nonzero(off)
            self.events.extend((t, x, y, -1) for y, x in zip(ys, xs))
        self.state |= on
        self.state &= ~off
        # Polarity-agnostic variant (NOT the spec tracker; used for video, where
        # dark objects on a brighter background fire OFF first): the first
        # event of a passage marks the pixel (+1 after ON, -1 after OFF) and
        # the opposite polarity clears it. Identical to `state` for bright
        # objects on a dark background.
        b = self.bipolar
        b[on & (b == 0)] = 2          # temp marker so a same-step OFF can't clear it
        b[on & (b == -1)] = 0
        b[off & (b == 0)] = -1
        b[off & (b == 1)] = 0
        b[b == 2] = 1
        return s


def run(u_fn, t_end, shape, circuit, traps=None, readouts=None, dt=2.0,
        rep_pixels=(), frame_every=500.0, true_center=None):
    """
    u_fn        : t_us -> (H, W) grayscale in [0, 1]
    readouts    : list of (label, transform or None); default one raw readout
    rep_pixels  : (x, y) pixels whose u and per-readout signal are logged
    true_center : optional t_us -> (x, y) of the known object, for metrics
    """
    H, W = shape
    cpd = CPDArray(shape, circuit, traps)
    readouts = [EventReadout(l, shape, f) for l, f in (readouts or [("raw", None)])]
    yy, xx = np.mgrid[0:H, 0:W].astype(float)

    n = int(round(t_end / dt))
    t_axis = np.arange(1, n + 1) * dt
    r_true = np.full((n, 2), np.nan)
    visible = np.zeros(n, bool)
    r_est = {r.label: np.full((n, 2), np.nan) for r in readouts}
    iou = {r.label: np.full(n, np.nan) for r in readouts}
    traces = {"u": np.zeros((n, len(rep_pixels)))}
    for r in readouts:
        traces[r.label] = np.zeros((n, len(rep_pixels)))
    T_trace = np.zeros((n, len(rep_pixels))) if traps is not None else None
    crossings = []                              # (t_us, x, y, +1 entry / -1 exit)
    frames = {"t": [], "u": [], "i_R": {r.label: [] for r in readouts},
              "state": {r.label: [] for r in readouts}}
    next_frame = 0.0
    prev_foot = np.zeros(shape, bool)
    rp = np.array(rep_pixels, int).reshape(-1, 2)

    for k in range(n):
        t0 = k * dt
        i_R = cpd.step(u_fn, t0, dt)
        t = t_axis[k]
        u = u_fn(t)
        foot = u >= U_FOOT
        for sign, m in ((1, foot & ~prev_foot), (-1, ~foot & prev_foot)):
            if m.any():
                ys, xs = np.nonzero(m)
                crossings.extend((t, x, y, sign) for y, x in zip(ys, xs))
        prev_foot = foot
        visible[k] = foot.any()
        if true_center is not None:
            r_true[k] = true_center(t)
        if len(rp):
            traces["u"][k] = u[rp[:, 1], rp[:, 0]]
            if T_trace is not None:
                T_trace[k] = cpd.T[rp[:, 1], rp[:, 0]]

        take_frame = t >= next_frame
        if take_frame:
            frames["t"].append(t)
            frames["u"].append(u.astype(np.float32))
            next_frame += frame_every

        for r in readouts:
            s = r.update(i_R, t)
            st = r.state
            cnt = st.sum()
            if cnt:
                r_est[r.label][k] = (xx[st].mean(), yy[st].mean())
            union = (st | foot).sum()
            if union:
                iou[r.label][k] = (st & foot).sum() / union
            if len(rp):
                traces[r.label][k] = s[rp[:, 1], rp[:, 0]]
            if take_frame:
                frames["i_R"][r.label].append(s.astype(np.float32))
                frames["state"][r.label].append(st.copy())

    return {
        "t": t_axis, "dt": dt, "shape": shape, "visible": visible,
        "r_true": r_true, "r_est": r_est, "iou": iou,
        "events": {r.label: np.array(r.events, float).reshape(-1, 4) for r in readouts},
        "crossings": np.array(crossings, float).reshape(-1, 4),
        "traces": traces, "T_trace": T_trace, "rep_pixels": rep_pixels,
        "frames": frames, "labels": [r.label for r in readouts],
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def match_events(events, crossings, keep=None):
    """
    Pair each u = 0.2 entry crossing with the first ON event, and each exit
    crossing with the first OFF event, inside the same pixel passage:
      ON window for entry j  : (exit_{j-1}, exit_j]
      OFF window for exit j  : (entry_j, entry_{j+1}]
    Only crossings with keep(t, x, y) True are scored (used for per-pass
    metrics on the repeated-pass runs, where passes overlap in time across
    the array and must be separated per pixel). Returns dict of latency arrays (us)
    and detection fractions.
    """
    out = {"on_lat": [], "off_lat": [], "n_entry": 0, "n_exit": 0}
    if len(crossings) == 0:
        return {**out, "on_frac": np.nan, "off_frac": np.nan}
    ev = {}
    for t, x, y, p in events:
        ev.setdefault((int(x), int(y), int(p)), []).append(t)
    cr = {}
    for t, x, y, s in crossings:
        cr.setdefault((int(x), int(y)), []).append((t, int(s)))

    for (x, y), lst in cr.items():
        entries = [t for t, s in lst if s == 1]
        exits = [t for t, s in lst if s == -1]
        on_t = np.array(ev.get((x, y, 1), []))
        off_t = np.array(ev.get((x, y, -1), []))
        for j, te in enumerate(entries):
            if keep is not None and not keep(te, x, y):
                continue
            out["n_entry"] += 1
            lo = exits[j - 1] if j > 0 else -np.inf
            hi = exits[j] if j < len(exits) else np.inf
            c = on_t[(on_t > lo) & (on_t <= hi)]
            if len(c):
                out["on_lat"].append(c.min() - te)
        for j, tx in enumerate(exits):
            if keep is not None and not keep(tx, x, y):
                continue
            out["n_exit"] += 1
            lo = entries[j]
            hi = entries[j + 1] if j + 1 < len(entries) else np.inf
            c = off_t[(off_t > lo) & (off_t <= hi)]
            if len(c):
                out["off_lat"].append(c.min() - tx)
    out["on_lat"] = np.array(out["on_lat"])
    out["off_lat"] = np.array(out["off_lat"])
    out["on_frac"] = len(out["on_lat"]) / out["n_entry"] if out["n_entry"] else np.nan
    out["off_frac"] = len(out["off_lat"]) / out["n_exit"] if out["n_exit"] else np.nan
    return out


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else np.nan


def metrics(res, label, keep=None, t_reversal=None, axis=0):
    """keep(t, x, y) restricts event counts and crossing matching to one
    pass; trajectory/overlap metrics are then skipped (NaN) because the
    centroid of several simultaneous spots is not a single trajectory."""
    t = res["t"]
    ev = res["events"][label]
    if keep is not None and len(ev):
        ev_w = ev[[keep(e[0], int(e[1]), int(e[2])) for e in ev]]
    else:
        ev_w = ev
    n_on = int((ev_w[:, 3] == 1).sum()) if len(ev_w) else 0
    n_off = int((ev_w[:, 3] == -1).sum()) if len(ev_w) else 0
    m = match_events(ev, res["crossings"], keep)

    vis = res["visible"] & (keep is None)
    est = res["r_est"][label]
    valid = vis & ~np.isnan(est[:, 0])
    err = np.linalg.norm(est[valid] - res["r_true"][valid], axis=1) \
        if not np.isnan(res["r_true"]).all() else np.array([])

    out = {
        "n_on": n_on, "n_off": n_off,
        "count_ratio": n_on / n_off if n_off else np.nan,
        "on_det": m["on_frac"], "off_det": m["off_frac"],
        "on_lat_med": _pct(m["on_lat"], 50), "on_lat_p90": _pct(m["on_lat"], 90),
        "off_lat_med": _pct(m["off_lat"], 50), "off_lat_p90": _pct(m["off_lat"], 90),
        "iou": float(np.nanmean(res["iou"][label][vis])) if vis.any() else np.nan,
        "err_mean": float(err.mean()) if len(err) else np.nan,
        "err_p90": _pct(err, 90),
        "valid_frac": valid.sum() / vis.sum() if vis.sum() else np.nan,
    }
    if t_reversal is not None:
        out["reversal_delay"] = reversal_delay(t, est[:, axis], t_reversal)
    return out


def reversal_delay(t, x_est, t_rev, smooth_us=500.0):
    """Time between the true reversal and the extremum of the (lightly
    smoothed) traced coordinate along the motion axis, searched within
    +-10 ms of the true reversal."""
    dt = t[1] - t[0]
    k = max(1, int(round(smooth_us / dt)))
    x = np.where(np.isnan(x_est), np.nan, x_est)
    valid = ~np.isnan(x)
    if valid.sum() < k:
        return np.nan
    x_f = np.interp(t, t[valid], x[valid])
    xs = np.convolve(x_f, np.ones(k) / k, mode="same")
    w = (t > t_rev - 10000) & (t < t_rev + 10000) & valid
    if not w.any():
        return np.nan
    return float(t[w][np.argmax(xs[w])] - t_rev)


def gaussian_ratio(trace):
    """|min| / max of a single pixel's signal trace (rho for that passage)."""
    mx = trace.max()
    return abs(trace.min()) / mx if mx > 0 else np.nan
