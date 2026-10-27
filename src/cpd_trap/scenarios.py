"""
scenarios.py

Known-trajectory Gaussian spot stimuli on a 64 x 64 array (spec Eq. 1),
sigma = 2 px. Every path starts and ends >= 4 sigma off the array and is
followed by a 5 ms dark tail so delayed OFF events can still be recorded.

  straight  : y = 32, x -8 -> 72 at 2 px/ms
  turn      : +x at y = 20 to (40, 20), right-angle turn, +y to (40, 72)
  fast      : straight at 8 px/ms (sigma/v = 250 us, comparable to tau1)
  reversal  : +x to (44, 32), reverse, back to (-8, 32); reversal at 26 ms
  disk      : ADDITION, not in the task file -- a flat-topped disk (radius
              5 px, logistic edge 0.3 px) on the straight path. Each pixel
              under it is lit for ~5 ms (> tau_on), which is what lets the
              traps fill and gives the device's |ON| < OFF asymmetry; the
              sigma = 2 Gaussian spot lights a pixel too briefly for that.
  repeat_<d>: two identical spots on the straight path, the second delayed so
              that every pixel on the path is dark (u < 0.2) for d ms between
              passages. T is never reset, so the second passage sees the trap
              occupancy left by the first.

Pixel centers sit on integer coordinates, so y = 32 crosses pixel centers.
"""

import numpy as np

GRID = (64, 64)
SIGMA = 2.0
V_BASE = 0.002            # px/us = 2 px/ms
TAIL = 5000.0             # us of trailing darkness
X0, X1 = -8.0, 72.0
# u >= 0.2 lasts 2 * sigma * sqrt(2 ln 5) / v at a pixel crossed through its center
FOOT_HALF_US = SIGMA * np.sqrt(2 * np.log(5)) / V_BASE

_yy, _xx = np.mgrid[0:GRID[0], 0:GRID[1]].astype(float)


class Scenario:
    def __init__(self, name, waypoints, t_end, rep_pixels, t_reversal=None,
                 delays=(0.0,), axis=0, note="", disk=None):
        """waypoints: list of (t_us, x, y); the path is piecewise linear and
        clamps to the last waypoint. delays: one spot per delay (us)."""
        self.name = name
        self.wp = np.array(waypoints, float)
        self.t_end = t_end
        self.rep_pixels = rep_pixels
        self.t_reversal = t_reversal
        self.delays = delays
        self.axis = axis
        self.note = note
        self.disk = disk            # None = Gaussian spot; (radius, edge) = flat-top disk
        self._cache = {}

    def center(self, t, delay=0.0):
        tt = t - delay
        return (np.interp(tt, self.wp[:, 0], self.wp[:, 1]),
                np.interp(tt, self.wp[:, 0], self.wp[:, 2]))

    def true_center(self, t):
        return self.center(t)

    def u(self, t):
        hit = self._cache.get(t)
        if hit is not None:
            return hit
        out = np.zeros(GRID)
        for d in self.delays:
            if t < d + self.wp[0, 0]:
                continue
            cx, cy = self.center(t, d)
            r2 = (_xx - cx) ** 2 + (_yy - cy) ** 2
            if self.disk is None:
                out += np.exp(-r2 / (2 * SIGMA ** 2))
            else:
                R, e = self.disk
                out += 1.0 / (1.0 + np.exp((np.sqrt(r2) - R) / e))
        np.clip(out, 0.0, 1.0, out=out)
        if len(self._cache) > 4:
            self._cache.clear()
        self._cache[t] = out
        return out

    def arrival(self, x, y, delay=0.0):
        """Time the (straight-path) spot center passes pixel x."""
        return delay + (x - X0) / V_BASE

    def pass_index(self, t, x, y):
        return int(np.argmin([abs(t - self.arrival(x, y, d)) for d in self.delays]))


def straight(v=V_BASE, name="straight"):
    t1 = (X1 - X0) / v
    return Scenario(name, [(0, X0, 32), (t1, X1, 32)], t1 + TAIL,
                    rep_pixels=[(32, 32), (32, 33), (32, 34)],
                    note=f"straight, y=32, {v * 1000:g} px/ms")


def disk(radius=5.0, edge=0.3):
    x0, x1 = -radius - 8, 64 + radius + 8
    t1 = (x1 - x0) / V_BASE
    return Scenario("disk", [(0, x0, 32), (t1, x1, 32)], t1 + TAIL,
                    rep_pixels=[(32, 32), (32, 35), (32, 37)], disk=(radius, edge),
                    note=f"flat-top disk r={radius:g} px, edge {edge:g} px, straight, 2 px/ms "
                         f"(addition: ~{2 * radius / (V_BASE * 1000):g} ms dwell per pixel)")


def turn():
    t_a = (40 - X0) / V_BASE
    t_b = t_a + (X1 - 20) / V_BASE
    return Scenario("turn", [(0, X0, 20), (t_a, 40, 20), (t_b, 40, X1)], t_b + TAIL,
                    rep_pixels=[(20, 20), (40, 20), (40, 40)],
                    note="right-angle turn at (40, 20), 2 px/ms")


def reversal():
    t_a = (44 - X0) / V_BASE
    t_b = 2 * t_a
    return Scenario("reversal", [(0, X0, 32), (t_a, 44, 32), (t_b, X0, 32)], t_b + TAIL,
                    rep_pixels=[(20, 32), (40, 32), (44, 32)], t_reversal=t_a,
                    note="+x to (44, 32), reverse, return; 2 px/ms")


def repeat(dark_ms):
    """Second spot delayed so each path pixel is dark (u < 0.2) for dark_ms."""
    delay = dark_ms * 1000 + 2 * FOOT_HALF_US
    t1 = (X1 - X0) / V_BASE
    return Scenario(f"repeat_{dark_ms:g}ms", [(0, X0, 32), (t1, X1, 32)],
                    t1 + delay + TAIL, rep_pixels=[(32, 32), (16, 32), (48, 32)],
                    delays=(0.0, delay),
                    note=f"two passes, {dark_ms:g} ms of darkness between them at each pixel")


BASIC = [straight(), turn(), straight(v=4 * V_BASE, name="fast"), reversal(), disk()]
REPEAT_DARK_MS = (1, 3, 10, 30)
