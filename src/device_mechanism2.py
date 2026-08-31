"""
device_mechanism2.py

Device model for Mechanism 2: the thin-film multispectral pyro-phototronic
event-based sensor ("Thin-film event-based vision sensors for enhanced
multispectral perception beyond human vision").

Physics being modeled
----------------------
This is a SINGLE two-terminal device (no second branch). The spiking
signal comes from the pyro-phototronic effect: absorbed light heats the
device, and the resulting temperature CHANGE generates a pyroelectric
current. Pyroelectric current is fundamentally a rate-of-change
transducer:
        I_pyro(t) proportional to dT/dt

We model the temperature with its own first-order thermal RC lag (heating
from absorbed light, cooling back down when light is removed):
        tau_th * dT/dt = alpha_heat * L(t) - T(t)

so that I_pyro(t) = kappa * dT/dt.

The paper also reports a SUBLINEAR power-law relationship between spike
current and input power, I ~ P_in^alpha, with different exponents for
positive vs negative spikes (alpha+ = 0.27, alpha- = 0.38 in the paper).
We approximate this by applying a sign-preserving power-law compression
to the raw pyro current:
        I_signed = sign(I_pyro) * |I_pyro|^alpha(sign)

This compression is what gives the device its very high dynamic range
(>110 dB reported) -- weak signals are boosted relative to strong ones,
similar in spirit to the log-compression in conventional DVS pixels, but
here it emerges from the sublinear pyroelectric response itself rather
than an explicit log circuit.

As with Mechanism 1, we discretize the continuous current into events
using a charge integrate-and-fire quantizer.
"""

import numpy as np


class Mechanism2Pyrophototronic:
    def __init__(self, grid_size, dt, tau_th=5e-6, alpha_heat=1.0, kappa=1.0,
                 alpha_pos=0.27, alpha_neg=0.38, q_thresh=0.02):
        """
        tau_th     : thermal RC time constant (s)
        alpha_heat : how efficiently absorbed light raises temperature
        kappa      : pyroelectric conversion gain (current per dT/dt)
        alpha_pos  : sublinear exponent for positive (brightening) spikes
        alpha_neg  : sublinear exponent for negative (darkening) spikes
                     (paper reports these differ, giving asymmetric ON/OFF response)
        q_thresh   : charge threshold per event
        """
        self.grid_size = grid_size
        self.dt = dt
        self.tau_th = tau_th
        self.alpha_heat = alpha_heat
        self.kappa = kappa
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.q_thresh = q_thresh

        self.T = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.charge = np.zeros((grid_size, grid_size), dtype=np.float64)

    def _sublinear(self, I_pyro, eps=1e-3):
        """
        Epsilon-regularized power law: sign(x) * ((|x|+eps)^alpha - eps^alpha).
        A raw power law x^alpha with alpha<1 has INFINITE slope at x=0, so it
        massively amplifies tiny numerical noise near zero (this is what caused
        runaway event counts before this fix). Adding eps keeps the curve
        smooth and bounded near zero while preserving the sublinear
        (dynamic-range-boosting) compression for larger transients, matching
        the qualitative behavior reported in the paper without the numerical
        blow-up.
        """
        out = np.zeros_like(I_pyro)
        pos = I_pyro > 0
        neg = I_pyro < 0
        out[pos] = np.power(I_pyro[pos] + eps, self.alpha_pos) - eps ** self.alpha_pos
        out[neg] = -(np.power(-I_pyro[neg] + eps, self.alpha_neg) - eps ** self.alpha_neg)
        return out

    def step(self, L):
        T_prev = self.T.copy()
        self.T += (self.dt / self.tau_th) * (self.alpha_heat * L - self.T)
        dTdt = (self.T - T_prev) / self.dt

        I_pyro = self.kappa * dTdt
        I_signed = self._sublinear(I_pyro)

        self.charge += I_signed * self.dt

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
