"""
device_mechanism1.py

Device model for Mechanism 1: the WSe2 dual-branch event-driven pixel
(Zhou et al., "Computational event-driven vision sensors for in-sensor
spiking neural networks", Nature Electronics 2023).

Physics being modeled
----------------------
Each pixel has two parallel photodiode branches of OPPOSITE polarity:
  - "fast" branch  (no capacitor): current tracks light instantaneously,
        I_fast(t) = R * L(t)
  - "slow" branch  (series capacitor -> RC delay): current lags light,
        tau * dI_slow/dt = R * L(t) - I_slow(t)

Because the branches are opposite-polarity, the net pixel current is
their DIFFERENCE:
        I_net(t) = I_fast(t) - I_slow(t)

Under constant illumination both branches settle to the same steady-state
value R*L, so I_net -> 0 (no output, matches the paper's description of
zero photocurrent under constant light). When light changes, the fast
branch reacts immediately while the slow branch lags, so a transient
appears in I_net -- this transient IS the spike. No downstream threshold
comparator is needed to detect "change"; the RC mismatch does it in the
analog domain.

R (photoresponsivity) is the paper's "programmable, non-volatile" weight:
here it is a per-pixel gain array so you can assign different synaptic
weights across the array if desired.

To turn the continuous I_net(t) into discrete (x, y, t, polarity) events
comparable to a real event-camera output, we run I_net through a simple
charge-integrate-and-fire quantizer: charge accumulates over time, and an
event fires (with reset) each time accumulated charge crosses +-q_thresh.
This is a standard, physically reasonable way to discretize an analog
spiking photocurrent into an event stream, and it preserves the total
"area" (charge) of each transient, so bigger/faster intensity changes
produce more events, matching real event-camera behavior.
"""

import numpy as np


class Mechanism1DualBranch:
    def __init__(self, grid_size, dt, tau_slow=5e-6, R=1.0, q_thresh=0.02,
                 responsivity_map=None):
        """
        tau_slow : RC time constant of the slow branch (s). Paper reports
                   ~5 microseconds temporal resolution for the WSe2 device.
        R        : baseline photoresponsivity (scales current per unit light)
        q_thresh : charge threshold per event (tune this to control event rate)
        responsivity_map : optional (grid_size, grid_size) array of per-pixel
                            programmable responsivity (the "synaptic weight" A).
                            Defaults to uniform R everywhere.
        """
        self.grid_size = grid_size
        self.dt = dt
        self.tau_slow = tau_slow
        self.q_thresh = q_thresh
        self.R = (responsivity_map if responsivity_map is not None
                  else np.full((grid_size, grid_size), R, dtype=np.float64))

        self.I_slow = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.charge = np.zeros((grid_size, grid_size), dtype=np.float64)

    def step(self, L):
        """
        Advance the device model by one timestep given the current
        intensity field L (grid_size x grid_size). Returns a list of
        events: (x, y, polarity) that fired this step (polarity = +1/-1).
        """
        I_fast = self.R * L
        # Explicit Euler update of the RC branch: tau dI/dt = R*L - I
        self.I_slow += (self.dt / self.tau_slow) * (self.R * L - self.I_slow)

        I_net = I_fast - self.I_slow
        self.charge += I_net * self.dt

        events = []
        # A pixel can fire multiple events in one step if the transient is large
        fired = np.abs(self.charge) >= self.q_thresh
        ys, xs = np.where(fired)
        for y, x in zip(ys, xs):
            while abs(self.charge[y, x]) >= self.q_thresh:
                polarity = 1 if self.charge[y, x] > 0 else -1
                events.append((x, y, polarity))
                self.charge[y, x] -= polarity * self.q_thresh
        return events

    def run(self, L_video):
        """
        L_video : (n_steps, grid_size, grid_size) intensity video.
        Returns event list of (t_index, x, y, polarity) tuples.
        """
        all_events = []
        for t_idx in range(L_video.shape[0]):
            step_events = self.step(L_video[t_idx])
            for (x, y, p) in step_events:
                all_events.append((t_idx, x, y, p))
        return all_events
