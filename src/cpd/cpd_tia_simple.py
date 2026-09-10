"""
cpd_tia_simple.py

Physical device model for Prof. Cheng's CPD (capacitive photodiode) connected
directly to the virtual-ground input of a transimpedance amplifier (TIA).
Spec: CPD_event_imaging_simulation_review.md, section 9.1.

Physical model (this IS the device model -- the numerical scheme below is
only its discretization, per the spec's section 7.1 warning not to present a
finite difference as the physical mechanism):

    tau_CPD * dI_CPD/dt + I_CPD = eta * tau_CPD * dI_L/dt

    tau_CPD = r * (Cs + Ci)      intrinsic CPD time constant
    eta     = Ci / (Cs + Ci)     edge-current coupling coefficient

I_L(x,y,t) is the single optical input (light-induced photocurrent) -- there
is no separate normalized light variable or amplitude inside this class, per
spec section 3. The class returns both the native device current I_CPD and
the TIA output voltage V_o = -Rf * I_CPD (ideal TIA), or, if a finite TIA
bandwidth is supplied, the first-order-lagged voltage from spec section 5.

Symmetric vs. asymmetric ON/OFF device variant
------------------------------------------------
The boxed equation above is a single LINEAR time-invariant ODE. Used as-is
(one tau_CPD for all transitions), it is the "symmetric" device model.

Prof. Cheng's request for a second model with "slightly asymmetric ON/OFF
peaks" is implemented here as a charge/discharge-dependent time constant:
tau_on applies while I_L is increasing (dI_L/dt >= 0), tau_off applies while
I_L is decreasing (dI_L/dt < 0). This is our own interpretation, agreed with
Meet, of how to turn the spec's single equation into two distinct device
models rather than two measurement conditions of the same device -- IT HAS
NOT BEEN CONFIRMED WITH PROF. CHENG and should be checked before it goes in
the paper. Physical motivation: a real photodiode's differential resistance
often genuinely differs for increasing vs. decreasing photocurrent (carrier
injection vs. recombination dynamics), and both literature devices already
in this repo (Zhou et al.'s WSe2 dual-branch design, and the pyro-phototronic
paper's alpha+=0.27/alpha-=0.38) build in a comparable ON != OFF asymmetry.

IMPORTANT derived subtlety (verified analytically before implementing, kept
here so it isn't rediscovered the hard way): for an IDEALIZED, instantaneous
light step (rise/fall time << tau), the peak of I_CPD right at the edge is
ALWAYS exactly eta*(delta I_L), independent of which tau is active --  tau
only sets how fast the response decays afterward, not the initial peak
height. tau_on/tau_off only produce a genuine PEAK amplitude asymmetry when
the light transition has a FINITE rise/fall time that is not negligible
compared to tau (i.e. tau/transition_time is not close to zero) -- exactly
the same physical regime the spec's Gamma = v*tau_CPD/sigma (section 14)
describes. This is why validate_single_pixel.py tests the asymmetric model
with a finite-rise-time pulse, not only the idealized instantaneous step.
"""

import numpy as np


class CPDTIAPixel:
    def __init__(self, grid_size, dt, tau_cpd=100e-6, eta=0.5, Rf=1.0,
                 tau_cpd_off=None, bw_tia=None):
        """
        grid_size   : pixel grid is grid_size x grid_size
        dt          : internal simulation timestep (s) -- must resolve the
                      smallest tau_cpd/tau_cpd_off and the fastest optical
                      transition of interest (see validate_single_pixel.py)
        tau_cpd     : intrinsic CPD time constant (s) used while I_L is
                      increasing (or the only time constant, if symmetric)
        eta         : Ci / (Cs + Ci), edge-current coupling coefficient
        Rf          : TIA transimpedance gain
        tau_cpd_off : intrinsic CPD time constant (s) used while I_L is
                      decreasing. None (default) => symmetric model,
                      tau_cpd_off = tau_cpd. Set different from tau_cpd for
                      the asymmetric device model.
        bw_tia      : optional finite TIA bandwidth (Hz). None (default) =>
                      ideal TIA, V_o = -Rf * I_CPD instantaneously (spec
                      section 9.1's boxed V_o equation). If given, adds the
                      first-order TIA lag from spec section 5.
        """
        self.grid_size = grid_size
        self.dt = dt
        self.tau_on = tau_cpd
        self.tau_off = tau_cpd_off if tau_cpd_off is not None else tau_cpd
        self.eta = eta
        self.Rf = Rf
        self.tau_tia = 1.0 / (2 * np.pi * bw_tia) if bw_tia else None

        self.I_L_prev = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.I_cpd = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.V_o_state = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._initialized = False

    def step(self, I_L):
        """
        Advance by one timestep given the light-induced photocurrent field
        I_L (grid_size x grid_size). Returns (I_cpd, V_o), each
        grid_size x grid_size, for this step.
        """
        if not self._initialized:
            # First call: assume the field starts at steady state (no prior
            # transient), so dI_L/dt = 0 rather than a spurious jump from 0.
            self.I_L_prev = I_L.copy()
            self._initialized = True

        dI_L_dt = (I_L - self.I_L_prev) / self.dt
        tau = np.where(dI_L_dt >= 0, self.tau_on, self.tau_off)

        # Explicit Euler discretization of the boxed ODE:
        #   tau * dI_cpd/dt + I_cpd = eta * tau * dI_L/dt
        # => I_cpd[n+1] = I_cpd[n] + (dt/tau) * (eta*tau*dI_L_dt - I_cpd[n])
        drive = self.eta * tau * dI_L_dt
        self.I_cpd += (self.dt / tau) * (drive - self.I_cpd)

        self.I_L_prev = I_L.copy()

        if self.tau_tia is None:
            V_o = -self.Rf * self.I_cpd
        else:
            # First-order TIA lag (spec section 5):
            #   tau_TIA * dV_o/dt + V_o = -Rf * I_cpd
            target = -self.Rf * self.I_cpd
            self.V_o_state += (self.dt / self.tau_tia) * (target - self.V_o_state)
            V_o = self.V_o_state.copy()

        return self.I_cpd.copy(), V_o.copy() if self.tau_tia is not None else V_o

    def run(self, I_L_video):
        """
        I_L_video : (n_steps, grid_size, grid_size) light-induced
                    photocurrent video.
        Returns (I_cpd_trace, V_o_trace), each (n_steps, grid_size, grid_size).
        """
        n_steps = I_L_video.shape[0]
        I_cpd_trace = np.empty_like(I_L_video)
        V_o_trace = np.empty_like(I_L_video)
        for t in range(n_steps):
            I_cpd_trace[t], V_o_trace[t] = self.step(I_L_video[t])
        return I_cpd_trace, V_o_trace
