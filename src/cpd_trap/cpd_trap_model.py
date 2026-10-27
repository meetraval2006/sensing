"""
cpd_trap_model.py

Per-pixel CPD readout model with optional trap dynamics, implementing the
boxed systems in CPD_event_imaging_trajectory_simulation_model.md.

Units throughout: time in us, current in uA, Gamma in uA*us.

Task A -- direct photocurrent, no trap state (box A):

    I_P   = I_P0 * u,          0 <= u <= 1
    di_R/dt = z - I_P/RC_s
    dz/dt   = -a z - b i_R + a I_P/RC_s,   a = 1/tau1 + 1/tau2, b = 1/(tau1 tau2)

Task B -- active traps (box S): same circuit, driven by
    I_eff = I_P - Gamma * dT/dt
    dT/dt = u (T_ss - T)/tau_on - (1 - u) T/tau_off

Box A is exactly box S with Gamma = 0 (dT/dt then never enters I_eff), so a
single class covers both; TrapParams=None selects box A and skips T entirely.

The (i_R, z) pair is the exact first-order form of the V5 circuit equation
    i_R'' + a i_R' + b i_R = -(1/RC_s) dI_eff/dt,
i.e. i_R/I_eff = -s / (RC_s (s + 1/tau1)(s + 1/tau2)): a band-pass with two
real poles. A step in light therefore gives a single-signed lobe (negative
for rising light, positive for falling light) with no ringing, which is what
lets the event detector use one crossing per polarity per passage.

u is the CONTINUOUS grayscale illumination (never binarized). It is supplied
as a callable u_fn(t_us) -> (H, W) array so the RK4 integrator can evaluate
it at sub-step times. For piecewise-constant inputs (the square-pulse
calibration) pass u_fn that is constant over each step and align the switch
time to a step boundary.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CircuitParams:
    tau1: float = 150.0     # us
    tau2: float = 20.0      # us
    RC_s: float = 100.0     # us
    I_P0: float = 3.0       # uA

    @property
    def a(self):
        return 1.0 / self.tau1 + 1.0 / self.tau2

    @property
    def b(self):
        return 1.0 / (self.tau1 * self.tau2)


@dataclass(frozen=True)
class TrapParams:
    tau_on: float           # us, capture time under full light
    tau_off: float          # us, release time in darkness
    Gamma: float = 2400.0   # uA*us per unit T
    T_ss: float = 1.0


class CPDArray:
    """Independent CPD pixels on an (H, W) grid, advanced with classical RK4."""

    def __init__(self, shape, circuit=CircuitParams(), traps=None):
        self.shape = shape
        self.c = circuit
        self.traps = traps
        self.i_R = np.zeros(shape)
        self.z = np.zeros(shape)
        self.T = np.zeros(shape) if traps is not None else None

    def _deriv(self, i_R, z, T, u):
        c = self.c
        I_P = c.I_P0 * u
        if T is None:
            I_eff, dT = I_P, None
        else:
            p = self.traps
            dT = u * (p.T_ss - T) / p.tau_on - (1.0 - u) * T / p.tau_off
            I_eff = I_P - p.Gamma * dT
        drive = I_eff / c.RC_s
        di = z - drive
        dz = -c.a * z - c.b * i_R + c.a * drive
        return di, dz, dT

    def step(self, u_fn, t, dt):
        """Advance all states from t to t+dt. Returns i_R after the step."""
        u0 = u_fn(t)
        um = u_fn(t + 0.5 * dt)
        u1 = u_fn(t + dt)
        i, z, T = self.i_R, self.z, self.T
        has_T = T is not None

        k1 = self._deriv(i, z, T, u0)
        k2 = self._deriv(i + 0.5 * dt * k1[0], z + 0.5 * dt * k1[1],
                         T + 0.5 * dt * k1[2] if has_T else None, um)
        k3 = self._deriv(i + 0.5 * dt * k2[0], z + 0.5 * dt * k2[1],
                         T + 0.5 * dt * k2[2] if has_T else None, um)
        k4 = self._deriv(i + dt * k3[0], z + dt * k3[1],
                         T + dt * k3[2] if has_T else None, u1)

        w = dt / 6.0
        self.i_R = i + w * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.z = z + w * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if has_T:
            self.T = T + w * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        return self.i_R


# ---------------------------------------------------------------------------
# Device conditions used in the study (values from the task tables)
# ---------------------------------------------------------------------------

TASK_A_PAIRS = {
    "A-fast":      CircuitParams(tau1=50.0, tau2=10.0),
    "A-reference": CircuitParams(tau1=150.0, tau2=20.0),
    "A-slow":      CircuitParams(tau1=400.0, tau2=40.0),
    "A-very-slow": CircuitParams(tau1=1200.0, tau2=100.0),
}

# label -> (TrapParams, square-pulse calibration ratio from the table)
TASK_B_TRAPS = {
    "B-trap-sq1.0": (TrapParams(tau_on=2899.0, tau_off=3184.0), 0.993),
    "B-trap-sq0.7": (TrapParams(tau_on=1858.0, tau_off=13058.0), 0.695),
    "B-trap-sq0.5": (TrapParams(tau_on=1351.0, tau_off=16967.0), 0.496),
    "B-trap-sq0.2": (TrapParams(tau_on=829.0, tau_off=8786.0), 0.198),
}

THETA = 0.0045  # uA (4.5 nA) event threshold, noiseless demonstration value

# Plain-English names shown next to the code labels in every figure, GIF and
# table (code label -> what it is).
# All of these come from Prof. Cheng's model file; none are literature models.
ALIAS = {
    "A-fast":       "No traps, fast circuit (50/10 us)",
    "A-reference":  "No traps, reference circuit = SYMMETRIC BASELINE",
    "A-slow":       "No traps, slow circuit (400/40 us)",
    "A-very-slow":  "No traps, very slow circuit (1200/100 us)",
    "B-trap-sq1.0": "Traps, square-pulse ON/OFF 0.99 (weakest)",
    "B-trap-sq0.7": "Traps, square-pulse ON/OFF 0.70",
    "B-trap-sq0.5": "Traps, square-pulse ON/OFF 0.50",
    "B-trap-sq0.2": "Traps, square-pulse ON/OFF 0.20 (strongest)",
    "ctrl-1":       "Imposed ON scale 1.0 (= baseline)",
    "ctrl-0.7":     "Imposed ON scale 0.7",
    "ctrl-0.5":     "Imposed ON scale 0.5",
    "ctrl-0.2":     "Imposed ON scale 0.2",
}




def named(label):
    return f"{label}  |  {ALIAS.get(label, label)}"


# Short form for dense figures (video snapshot rows, GIF rows)
SHORT = {
    "A-fast": "no traps, fast", "A-reference": "SYMMETRIC baseline", "A-slow": "no traps, slow",
    "A-very-slow": "no traps, very slow",
    "B-trap-sq1.0": "traps, sq 0.99", "B-trap-sq0.7": "traps, sq 0.70",
    "B-trap-sq0.5": "traps, sq 0.50", "B-trap-sq0.2": "traps, sq 0.20",
    "ctrl-1": "imposed ON x1.0", "ctrl-0.7": "imposed ON x0.7",
    "ctrl-0.5": "imposed ON x0.5", "ctrl-0.2": "imposed ON x0.2",
}
