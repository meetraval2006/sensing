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

Version history (kept here deliberately -- every failure is instructive)
--------------------------------------------------------------------------
v1 (superseded): compressed EACH branch separately, then subtracted:
    I_fast = compress(R*L)
    I_slow = compress(R*L_filtered)
    I_net  = I_fast - I_slow
Cancels exactly at steady state, but compresses the wrong quantity --
compress(x) = sign(x)*|x|^alpha has a steep slope near x=0 and a shallow
slope far from zero. Under a bright baseline, both branches sit at a large
operating point where the curve is nearly flat, so a small transient riding
on that baseline gets ATTENUATED, not boosted. Only reached 70.8 dB
simulated dynamic range and traced worse than Mechanism 1.

v2: compute the exact linear branch difference FIRST (identical to
Mechanism 1), THEN compress that difference:
    I_diff = R*L - R*L_filtered
    I_net  = compress(I_diff)
Compression now acts on the transient itself, near zero, where the curve
is steepest. Real improvement over v1: 86.2 dB dynamic range, and
trajectory RMSE within 0.2 px of Mechanism 1 (1.24 px vs 1.03 px at
tau=5ms) on the standard bright-spot circle stimulus.

v3 (current) -- and an important NEGATIVE RESULT along the way:
We then asked "can a hybrid literally beat Mechanism 1's own tracing
accuracy, not just approach it?" The natural next idea: apply compression
ONLY to weak transients (a "linear-core, boosted-tail" law that is EXACTLY
linear, i.e. identical to Mechanism 1, above a knee x0, and only boosts
signal below x0), so strong transients are never distorted and weak ones
(e.g. a moving spot's soft Gaussian tail) get a chance to cross threshold
that Mechanism 1 would miss:
    I_diff = R*L - R*L_filtered                                (exact)
    I_net  = I_diff                          if |I_diff| >= x0
           = sign(I_diff) * x0 * (|I_diff|/x0)^alpha  otherwise

Initial test on the standard circle stimulus looked like a clear win over
Mechanism 1 -- until we controlled for event RATE. This model fires more
events per second than Mechanism 1 at matched q_thresh values, and firing
more events (denser temporal sampling) lowers trajectory RMSE on its own,
independent of the event-generation physics. Once we bisection-tuned both
models' q_thresh to hit IDENTICAL event counts and compared again, the
"win" evaporated: at matched density the two are statistically tied
(sometimes one is 1-2% better, sometimes the other), and at high enough
density Mechanism 1 alone (retuned lower) outright wins. CONCLUSION: on a
uniform-contrast, noise-free stimulus, nothing beats exact linear response
for tracing precision, because there is no real dynamic-range problem for
compression to solve -- every transient is already far above any
reasonable noise floor, so "boosting weak signal" has nothing useful to
boost. This v3 class of model does NOT beat Mechanism 1 on its own turf,
and analysis_hybrid_vs_mechanism1.py keeps that density-matched control
test in the repo rather than deleting the negative result.

Where v3 DOES win, genuinely: fix each model's q_thresh ONCE (tuned on a
normal, well-lit scene) and then dim the scene without retuning. Mechanism
1 goes completely blind (0% coverage) once the signal drops below its
fixed threshold; v3's boosted tail keeps producing usable (if noisier)
events across a much wider brightness range on that SAME fixed setting --
this is literally what "dynamic range" means, and it is demonstrated,
density-effects and all, in analysis_fixed_threshold_range.py. Re-tuning
Mechanism 1's own threshold per-scene closes the gap (it's not that
Mechanism 1 has a worse ceiling -- retuned, it's still better), which
pins down exactly what the hybrid buys you: robustness to NOT knowing to
retune, not higher peak accuracy.
"""

import numpy as np


class HybridModel:
    def __init__(self, grid_size, dt, tau_slow=5e-6, R=1.0, alpha=0.4,
                 x0=0.006, q_thresh=0.02, eps=1e-6, responsivity_map=None):
        """
        x0    : linear/compressed knee. |I_diff| >= x0 passes through
                UNCHANGED (identical to Mechanism 1); below x0 the signal
                is boosted by the sublinear power law. Set x0 well above
                numerical/discretization noise but below the smallest real
                transient you want preserved undistorted.
        alpha : compression exponent applied only below the knee.
        eps   : regularizes the true x=0 point so pure numerical noise
                isn't blown up into spurious events (see device_mechanism2.py).
        """
        self.grid_size = grid_size
        self.dt = dt
        self.tau_slow = tau_slow
        self.alpha = alpha
        self.x0 = x0
        self.q_thresh = q_thresh
        self.eps = eps
        self.R = (responsivity_map if responsivity_map is not None
                  else np.full((grid_size, grid_size), R, dtype=np.float64))

        self.I_slow_raw = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.charge = np.zeros((grid_size, grid_size), dtype=np.float64)

    def _shape(self, x):
        """Linear core (|x| >= x0) + sublinear boosted tail (|x| < x0).
        Continuous at x0 by construction; see module docstring for why
        this ordering (difference first, then shape) matters."""
        ax = np.abs(x)
        boosted = np.sign(x) * self.x0 * np.power(
            np.maximum(ax, self.eps) / self.x0, self.alpha)
        return np.where(ax >= self.x0, x, boosted)

    def step(self, L):
        raw_fast = self.R * L
        self.I_slow_raw += (self.dt / self.tau_slow) * (self.R * L - self.I_slow_raw)

        I_diff = raw_fast - self.I_slow_raw
        I_net = self._shape(I_diff)

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
