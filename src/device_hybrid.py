"""
device_hybrid.py

CUSTOM hybrid device model -- our hypothesis, not from either paper.
This is the thing to actually test/validate at the meeting, not a
established result.

Motivation
----------
Mechanism 1 (dual-branch RC mismatch) gives:
  + exact zero-baseline cancellation under constant light (very clean,
    high temporal precision -- 5 us)
  - linear responsivity -> limited dynamic range, weak signals near the
    noise floor are hard to separate from strong ones

Mechanism 2 (pyro-phototronic) gives:
  + sublinear power-law compression -> very high dynamic range (>110 dB),
    strong low-light sensitivity
  - single-branch design has no built-in cancellation mechanism at
    steady state, relies entirely on the pyroelectric derivative decaying
    to zero, which is more sensitive to thermal drift/noise

Hybrid idea: keep Mechanism 1's two-branch differential structure (for
clean zero-baseline behavior and precise timing), but apply Mechanism 2's
sign-preserving sublinear compression to EACH branch's photocurrent
before differencing:

    I_fast = sign(R*L) * |R*L|^alpha
    I_slow = sign(R*L_filtered) * |R*L_filtered|^alpha   (RC-lagged)
    I_net  = I_fast - I_slow

This should retain exact cancellation at steady state (both branches see
the same compressed steady-state value) while extending dynamic range
via the compression. Whether this actually improves trajectory-tracing
fidelity over either individual mechanism is exactly the open question
to test with compare_models.py -- treat the results as a hypothesis
test, not a foregone conclusion (matching your v6/v7 note about not
assuming an advantage until it's demonstrated).
"""

import numpy as np


class HybridModel:
    def __init__(self, grid_size, dt, tau_slow=5e-6, R=1.0, alpha=0.3,
                 q_thresh=0.02, responsivity_map=None):
        self.grid_size = grid_size
        self.dt = dt
        self.tau_slow = tau_slow
        self.alpha = alpha
        self.q_thresh = q_thresh
        self.R = (responsivity_map if responsivity_map is not None
                  else np.full((grid_size, grid_size), R, dtype=np.float64))

        self.I_slow_raw = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.charge = np.zeros((grid_size, grid_size), dtype=np.float64)

    def _compress(self, x, eps=1e-3):
        # Epsilon-regularized power law -- see device_mechanism2.py for why
        # the eps term is necessary (avoids infinite slope / noise blow-up at x=0).
        return np.sign(x) * (np.power(np.abs(x) + eps, self.alpha) - eps ** self.alpha)

    def step(self, L):
        raw_fast = self.R * L
        self.I_slow_raw += (self.dt / self.tau_slow) * (self.R * L - self.I_slow_raw)

        I_fast = self._compress(raw_fast)
        I_slow = self._compress(self.I_slow_raw)
        I_net = I_fast - I_slow

        self.charge += I_net * self.dt

        events = []
        fired = np.abs(self.charge) >= self.q_thresh
        ys, xs = np.where(fired)
        for y, x in zip(ys, xs):
            while abs(self.charge[y, x]) >= self.q_thresh:
                polarity = 1 if self.charge[y, x] > 0 else -1
                events.append((x, y, polarity))
                self.charge[y, x] -= polarity * self.q_thresh
        return events

    def run(self, L_video):
        all_events = []
        for t_idx in range(L_video.shape[0]):
            step_events = self.step(L_video[t_idx])
            for (x, y, p) in step_events:
                all_events.append((t_idx, x, y, p))
        return all_events
