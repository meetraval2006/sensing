"""
event_threshold.py

Optional, SECONDARY digital ON/OFF event extraction from the CPD's native
signed analog output. Per review spec section 12: the primary CPD result is
the signed analog map (I_CPD or V_o); a thresholded discrete event stream is
an optional readout stage layered on top, not part of the CPD physics
itself. Kept as a plain function (not a class or part of CPDTIAPixel) so
that choice is visible and easy to skip.

IMPORTANT non-obvious consequence, verified by hand and worth flagging for
the paper -- NOT a bug: spec section 12 defines "ON" as V_o > V_th,+ and
"OFF" as V_o < V_th,-, i.e. directly on the sign of the TIA output voltage.
Because the TIA is INVERTING (V_o = -Rf * I_cpd, spec section 9.1), a
brightness INCREASE produces I_cpd > 0 but V_o < 0, and a brightness
DECREASE produces I_cpd < 0 but V_o > 0. So thresholding V_o per the spec's
own literal definition means "ON" events (as computed here) fire on
brightness DECREASES and "OFF" events fire on brightness INCREASES --
backwards from the usual DVS-literature convention where ON = brighter.
This is a faithful, correct consequence of the specified inverting-TIA
circuit, reproduced exactly as written in the spec, but it will read
backwards against typical event-camera terminology unless called out
explicitly in the paper. If the conventional "ON = brighter" naming is
wanted instead, threshold I_cpd directly (positive = ON) rather than V_o,
or flip the sign convention here.
"""

import numpy as np


def threshold_events(V_o_video, v_th_pos, v_th_neg, dt):
    """
    V_o_video : (n_steps, grid_size, grid_size) TIA voltage (or I_CPD, if you
                want to threshold current directly) video.
    v_th_pos  : ON-event threshold (V_o > v_th_pos fires an ON event)
    v_th_neg  : OFF-event threshold (V_o < v_th_neg fires an OFF event;
                should be negative)
    dt        : timestep (s) -- unused in the computation, kept in the
                signature so callers can convert t_idx to physical time
                without a separate lookup.

    Returns a list of (t_idx, x, y, polarity) tuples, polarity in {+1, -1},
    in the same convention used elsewhere in this repo (compare_models.py,
    animate.py) so existing plotting utilities apply unchanged.

    EDGE-TRIGGERED, per pixel (a Schmitt-trigger comparator, the standard
    circuit-level model for any real threshold-based event pixel): a new ON
    event fires only on the TRANSITION into V_o > v_th_pos, not on every
    subsequent sample that stays above it, and similarly for OFF. A pixel
    re-arms once V_o returns to the dead band between v_th_neg and v_th_pos.

    This replaces an earlier, naive per-sample version that fired an event
    on every single timestep a pixel was above threshold. That produced
    "event counts" in the millions (out of a comparable number of
    pixel-timesteps) for smoothly-decaying analog signals -- i.e. it was
    measuring dwell time above threshold, not sparse transitions, which
    defeats the entire point of an event count as a data-reduction /
    sparsity metric. Kept as a lesson: a level check alone is not an event
    generator without some reset/hysteresis behavior.
    """
    n_steps, gy, gx = V_o_video.shape
    state = np.zeros((gy, gx), dtype=np.int8)  # 0 = dead band, 1 = high (ON fired), -1 = low (OFF fired)
    events = []
    for t_idx in range(n_steps):
        frame = V_o_video[t_idx]
        new_on = (frame > v_th_pos) & (state != 1)
        new_off = (frame < v_th_neg) & (state != -1)
        ys, xs = np.where(new_on)
        events.extend((t_idx, int(x), int(y), 1) for y, x in zip(ys, xs))
        ys, xs = np.where(new_off)
        events.extend((t_idx, int(x), int(y), -1) for y, x in zip(ys, xs))
        state = np.where(frame > v_th_pos, 1, np.where(frame < v_th_neg, -1, 0)).astype(np.int8)
    return events
