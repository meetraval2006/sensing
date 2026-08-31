"""
spot_simulator.py

Generates a synthetic time-varying light intensity field L(x, y, t):
a Gaussian "spot" of light moving along a known path (default: circular)
across a small pixel grid. This is the ground-truth stimulus fed into
each device model. Because we know the true path, we can score how well
each event-based device model + memory decoder recovers it.
"""

import numpy as np


class SpotSimulator:
    def __init__(self, grid_size=24, n_steps=2000, dt=1e-3,
                 baseline=0.5, amplitude=4.0, sigma=1.8,
                 path="circle", radius=7.0, period=1.0, seed=0):
        """
        grid_size : pixel grid is grid_size x grid_size
        n_steps   : number of simulation timesteps
        dt        : timestep in seconds
        baseline  : background light intensity (arbitrary units, e.g. mW/cm^2)
        amplitude : peak intensity added by the spot above baseline
        sigma     : spot radius (pixels), controls spatial extent of the Gaussian
        path      : "circle" or "line"
        radius    : radius of circular path (pixels) or half-length of line path
        period    : time (s) for one full loop of the circular path
        """
        self.grid_size = grid_size
        self.n_steps = n_steps
        self.dt = dt
        self.baseline = baseline
        self.amplitude = amplitude
        self.sigma = sigma
        self.path = path
        self.radius = radius
        self.period = period

        self.t = np.arange(n_steps) * dt
        self.cx, self.cy = self._ground_truth_path()

        yy, xx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        self._xx = xx.astype(np.float64)
        self._yy = yy.astype(np.float64)

    def _ground_truth_path(self):
        c = self.grid_size / 2.0
        if self.path == "circle":
            omega = 2 * np.pi / self.period
            cx = c + self.radius * np.cos(omega * self.t)
            cy = c + self.radius * np.sin(omega * self.t)
        elif self.path == "line":
            # back-and-forth horizontal sweep
            frac = (self.t % self.period) / self.period
            tri = 1 - np.abs(2 * frac - 1)  # triangle wave 0->1->0
            cx = c - self.radius + 2 * self.radius * tri
            cy = np.full_like(self.t, c)
        else:
            raise ValueError(f"Unknown path type: {self.path}")
        return cx, cy

    def intensity_at(self, step_idx):
        """Returns the grid_size x grid_size intensity field L(x,y) at a given timestep."""
        cx, cy = self.cx[step_idx], self.cy[step_idx]
        d2 = (self._xx - cx) ** 2 + (self._yy - cy) ** 2
        return self.baseline + self.amplitude * np.exp(-d2 / (2 * self.sigma ** 2))

    def generate(self):
        """Returns full intensity video, shape (n_steps, grid_size, grid_size)."""
        L = np.empty((self.n_steps, self.grid_size, self.grid_size), dtype=np.float64)
        for i in range(self.n_steps):
            L[i] = self.intensity_at(i)
        return L
