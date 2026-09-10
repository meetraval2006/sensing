"""
stimuli.py

Optical stimuli for the CPD event-imaging study. Per the review spec section
2: retain the existing Gaussian moving-spot simulator (re-exported here,
unmodified, from the parent src/ package -- not copied) as a benchmark with
an exactly-known trajectory, and add a moving bar/edge stimulus, which gives
a cleaner visualization of CPD-generated ON/OFF event edges.

Both simulators return an intensity/photocurrent video of shape
(n_steps, grid_size, grid_size); per the review spec section 3, this array
is fed directly to CPDTIAPixel as I_L(x,y,t) -- no separate normalization
step is applied here.
"""

import os
import sys

_PARENT_SRC = os.path.join(os.path.dirname(__file__), "..")
if _PARENT_SRC not in sys.path:
    sys.path.insert(0, _PARENT_SRC)

from spot_simulator import SpotSimulator  # noqa: F401  (re-exported, unmodified)

import numpy as np


class MovingBarSimulator:
    """
    A bright rectangular bar sweeping at constant velocity across a dark
    background, with a controllable edge softness (edge_sigma). This is the
    bar/edge analogue of SpotSimulator's `sigma`: it lets the same
    Gamma = v * tau_CPD / sigma scaling law (review spec section 14) be
    computed for edge stimuli, not just Gaussian spots.
    """

    def __init__(self, grid_size=24, n_steps=2000, dt=1e-3,
                 baseline=0.5, amplitude=4.0, bar_width=6.0, edge_sigma=0.75,
                 velocity=10.0, axis="x", start_offset=-4.0):
        """
        grid_size    : pixel grid is grid_size x grid_size
        n_steps      : number of simulation timesteps
        dt           : timestep in seconds
        baseline     : background photocurrent (dark level)
        amplitude    : peak photocurrent added by the bar above baseline
        bar_width    : width of the bright bar (pixels)
        edge_sigma   : edge softness (pixels); 0 -> a perfectly sharp step,
                       larger -> a more gradual transition. This directly
                       sets the "sigma" in Gamma = v*tau_CPD/sigma.
        velocity     : bar sweep speed (pixels/s) along `axis`
        axis         : 'x' (horizontal sweep, bar oriented vertically) or
                       'y' (vertical sweep, bar oriented horizontally)
        start_offset : bar center position (pixels) at t=0, relative to the
                       grid's leading edge (negative = starts off-grid so the
                       leading edge sweeps fully across)
        """
        self.grid_size = grid_size
        self.n_steps = n_steps
        self.dt = dt
        self.baseline = baseline
        self.amplitude = amplitude
        self.bar_width = bar_width
        self.edge_sigma = max(edge_sigma, 1e-6)  # avoid divide-by-zero in the logistic edges
        self.velocity = velocity
        self.axis = axis

        self.t = np.arange(n_steps) * dt
        self.center = start_offset + velocity * self.t  # bar center position vs. time, for ground truth

        yy, xx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        self._coord = xx.astype(np.float64) if axis == "x" else yy.astype(np.float64)

    def intensity_at(self, step_idx):
        """Returns the grid_size x grid_size photocurrent field I_L(x,y) at a given timestep."""
        pos = self.center[step_idx]
        left_edge = pos - self.bar_width / 2.0
        right_edge = pos + self.bar_width / 2.0
        # Smoothed rectangle = difference of two logistic edges, each with
        # softness edge_sigma -- reduces to a perfect step as edge_sigma -> 0.
        rising = 1.0 / (1.0 + np.exp(-(self._coord - left_edge) / self.edge_sigma))
        falling = 1.0 / (1.0 + np.exp(-(self._coord - right_edge) / self.edge_sigma))
        bar = rising - falling
        return self.baseline + self.amplitude * bar

    def generate(self):
        """Returns full photocurrent video, shape (n_steps, grid_size, grid_size)."""
        I_L = np.empty((self.n_steps, self.grid_size, self.grid_size), dtype=np.float64)
        for i in range(self.n_steps):
            I_L[i] = self.intensity_at(i)
        return I_L
