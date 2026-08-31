"""
memory_decode.py

Implements the "physical observation memory" readout described in your
uploaded notes (v7 review). This is deliberately kept SEPARATE from the
device models above: it is a mechanism-agnostic readout stage that turns
any event stream (from Mechanism 1, Mechanism 2, or the Hybrid model)
into a spatial activity map that can be used to trace a moving object.

Equations (as given in your notes)
-----------------------------------
    dG+/dt = -G+/tau + kappa * E+(t)
    dG-/dt = -G-/tau + kappa * E-(t)
    A_M    = |G+ - G-|

G+/G-  : positive/negative physical memory state per pixel
tau    : fading time -- controls how long a past event's influence lingers
kappa  : event-to-memory gain
E+/E-  : positive/negative event counts (impulses at event arrival times)
A_M    : decoded spatial activity used to localize the object

This is a leaky integrator (same form as a leaky integrate-and-fire
neuron's membrane potential, without the fire/reset): each event adds a
kick of size kappa to the corresponding polarity's memory state, and that
state exponentially decays with time constant tau. A_M is large wherever
recent net (signed) event activity has occurred, and fades out at rate
set by tau, which is exactly the "fading time" your notes highlight as
the thing to test.

Per your notes, this module treats SIGNED cancellation (A_M = |G+ - G-|)
as a hypothesis, not an assumed win -- compare_models.py also reports the
RECTIFIED alternative (A_M = G+ + G-) so the two can be compared, since
your v6 review found them nearly identical at the "starved event"
operating point and v7 wants that re-tested as event supply increases.
"""

import numpy as np


class MemoryDecoder:
    def __init__(self, grid_size, dt, tau=0.02, kappa=1.0):
        """
        tau   : fading time constant (s). Smaller tau = memory forgets
                faster = sharper but more fragile trace. Larger tau =
                smoother trace but more motion blur / lag.
        kappa : event-to-memory gain (how much each single event
                contributes to the memory state)
        """
        self.grid_size = grid_size
        self.dt = dt
        self.tau = tau
        self.kappa = kappa
        self.decay = np.exp(-dt / tau)  # exact exponential decay per step

        self.G_pos = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.G_neg = np.zeros((grid_size, grid_size), dtype=np.float64)

    def step(self, events_this_step):
        """
        events_this_step : list of (x, y, polarity) for events that
                            occurred during this timestep.
        Returns (A_M_signed, A_M_rectified) activity maps for this step.
        """
        # Exponential decay of existing memory
        self.G_pos *= self.decay
        self.G_neg *= self.decay

        # Add new event contributions
        for (x, y, p) in events_this_step:
            if p > 0:
                self.G_pos[y, x] += self.kappa
            else:
                self.G_neg[y, x] += self.kappa

        A_M_signed = np.abs(self.G_pos - self.G_neg)
        A_M_rectified = self.G_pos + self.G_neg
        return A_M_signed, A_M_rectified

    def run(self, events, n_steps, noise_floor=1e-3):
        """
        events   : list of (t_index, x, y, polarity) tuples (from a device model's .run())
        n_steps  : total number of timesteps to decode over
        Returns:
            traced_xy_signed    : (n_steps, 2) array of decoded (x,y) centroid per step
                                   using the signed A_M, NaN where activity is below noise_floor
            traced_xy_rectified : same, using rectified A_M
        """
        # bucket events by timestep for fast lookup
        events_by_step = [[] for _ in range(n_steps)]
        for (t_idx, x, y, p) in events:
            if 0 <= t_idx < n_steps:
                events_by_step[t_idx].append((x, y, p))

        yy, xx = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size), indexing="ij")

        traced_signed = np.full((n_steps, 2), np.nan)
        traced_rect = np.full((n_steps, 2), np.nan)

        for t_idx in range(n_steps):
            A_signed, A_rect = self.step(events_by_step[t_idx])

            for A, out in ((A_signed, traced_signed), (A_rect, traced_rect)):
                total = A.sum()
                if total > noise_floor:
                    cx = (A * xx).sum() / total
                    cy = (A * yy).sum() / total
                    out[t_idx] = [cx, cy]

        return traced_signed, traced_rect
