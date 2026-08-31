"""
device_hybrid.py

CUSTOM hybrid device model -- our hypothesis, not from either paper.
This is the thing to actually test/validate, not an established result.

Motivation
----------
Mechanism 1 (dual-branch RC mismatch) gives:
  + exact zero-baseline cancellation under constant light (very clean,
    high temporal precision -- 5 us)
  + spike amplitude A is EXACTLY linear in programmable responsivity R
    (A = k*R, verified in analysis_responsivity.py, R^2 = 1.00000)
  - linear responsivity -> limited dynamic range, weak signals near the
    noise floor are hard to separate from strong ones

Mechanism 2 (pyro-phototronic) gives:
  + sublinear power-law compression -> very high dynamic range (>110 dB),
    strong low-light sensitivity
  - single-branch design has no built-in cancellation mechanism at
    steady state, relies entirely on the pyroelectric derivative decaying
    to zero, which is more sensitive to thermal drift/noise

Version history (kept here deliberately -- the failure is instructive)
------------------------------------------------------------------------
v1 (superseded): compressed EACH branch separately, then subtracted:
    I_fast = compress(R*L)
    I_slow = compress(R*L_filtered)
    I_net  = I_fast - I_slow
This still cancels exactly at steady state (compress(x) - compress(x) = 0
for any function), but it compresses the wrong quantity. compress(x) =
sign(x)*|x|^alpha has a STEEP slope near x=0 and a SHALLOW slope far from
zero (that is what "sublinear" means). Under a bright baseline, both
branches sit at a large operating point where the compression curve is
nearly flat, so a small transient riding on that baseline gets its
sensitivity ATTENUATED, not boosted -- the opposite of Mechanism 2's actual
effect, which compresses the transient dT/dt directly, not two large
absolute branch currents that happen to get subtracted afterward. This is
why v1 only reached 70.8 dB simulated dynamic range vs Mechanism 2's 113.8
dB using a similar exponent range -- it was compressing the wrong signal.

v2 (current): compute the exact linear branch difference FIRST (identical
to Mechanism 1's I_net), THEN apply the compression to that difference:
    I_diff = R*L - R*L_filtered          (exact, same as Mechanism 1)
    I_net  = compress(I_diff)
Now the compression acts on the transient itself, which sits near zero
where the curve is steepest -- small transients get amplified, matching how
Mechanism 2 actually behaves. Zero-baseline cancellation is still exact and
now trivial (I_diff = 0 => compress(0) = 0), and dynamic range should track
much closer to Mechanism 2's, at some cost to A(R) linearity (compressing
R*(L - L_filtered) is no longer linear in R the way Mechanism 1's raw
difference is) -- see analysis_responsivity.py and RESULTS.md for the
measured tradeoff between v1 and v2.
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

        # v2: difference FIRST (exact, linear, matches Mechanism 1 exactly),
        # THEN compress the resulting transient -- see module docstring for
        # why this order matters.
        I_diff = raw_fast - self.I_slow_raw
        I_net = self._compress(I_diff)

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
