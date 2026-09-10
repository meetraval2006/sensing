"""
validate_single_pixel.py

Single-pixel validation of CPDTIAPixel against the closed-form square-pulse
response in the review spec, section 8. Per section 11, this MUST pass
before any motion study is trusted.

Two parts:

PART A -- idealized instantaneous step (rise/fall time << tau_CPD), the
exact scenario spec section 8 derives closed form for:
    I_ON,pk  = eta * I0
    I_OFF,pk = -eta * I0 * (1 - exp(-tp/tau_CPD))
Checks all 5 items spec section 11 lists, for the SYMMETRIC model
(tau_on = tau_off), at tau_CPD = 10, 100, 500 us.

PART B -- finite rise/fall time square pulse, comparing the symmetric vs.
asymmetric device models. This is necessary, not optional: we derived
analytically (see cpd_tia_simple.py's docstring) that for an IDEALIZED
instantaneous step, the peak height is ALWAYS exactly eta*deltaI_L
regardless of tau -- tau only sets the decay shape. tau_on != tau_off only
produces a genuine ON/OFF PEAK AMPLITUDE asymmetry when the light transition
has a finite rise/fall time comparable to tau. Part A alone would therefore
make the "asymmetric" model look identical to the symmetric one at its
peaks -- this part demonstrates and checks the actual asymmetric behavior,
using a triangular (linear) ramp for a clean, independently-derived
closed form (see analytical_ramp_peaks() below).
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cpd_tia_simple import CPDTIAPixel

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "cpd")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Stimulus generators
# ---------------------------------------------------------------------------

def idealized_step_pulse(n_steps, dt, t_on, t_off, I0):
    """Single-pixel I_L(t): 0 -> I0 at t_on, back to 0 at t_off, both
    transitions within exactly one timestep (rise/fall time = dt)."""
    t = np.arange(n_steps) * dt
    I_L = np.where((t >= t_on) & (t < t_off), I0, 0.0)
    return t, I_L


def ramp_pulse(n_steps, dt, t_on, t_rise, t_off, t_fall, I0):
    """Single-pixel I_L(t): linear ramp 0->I0 over t_rise starting at t_on,
    holds at I0, then linear ramp I0->0 over t_fall starting at t_off."""
    t = np.arange(n_steps) * dt
    I_L = np.zeros(n_steps)
    rise_mask = (t >= t_on) & (t < t_on + t_rise)
    I_L[rise_mask] = I0 * (t[rise_mask] - t_on) / t_rise
    hold_mask = (t >= t_on + t_rise) & (t < t_off)
    I_L[hold_mask] = I0
    fall_mask = (t >= t_off) & (t < t_off + t_fall)
    I_L[fall_mask] = I0 * (1 - (t[fall_mask] - t_off) / t_fall)
    return t, I_L


# ---------------------------------------------------------------------------
# Analytical formulas
# ---------------------------------------------------------------------------

def analytical_step_peaks(eta, tau, I0, tp):
    """Spec section 8's exact closed form for an idealized instantaneous step."""
    on_peak = eta * I0
    off_peak = -eta * I0 * (1 - np.exp(-tp / tau))
    return on_peak, off_peak


def analytical_ramp_peaks(eta, tau_on, tau_off, I0, t_rise, t_fall, t_plateau):
    """
    Closed form for a finite linear-ramp pulse (derived by hand, documented
    in cpd_tia_simple.py's docstring): rising ramp of duration t_rise uses
    tau_on throughout (dI_L/dt > 0 the whole time), the plateau decays with
    tau_on (dI_L/dt = 0), the falling ramp of duration t_fall uses tau_off
    throughout (dI_L/dt < 0 the whole time).
    """
    def f(tau, t_edge):
        return (tau / t_edge) * (1 - np.exp(-t_edge / tau))

    on_peak = eta * I0 * f(tau_on, t_rise)
    residual_before_off = on_peak * np.exp(-t_plateau / tau_on)
    off_asymptote = -eta * I0 * (tau_off / t_fall)
    off_peak = off_asymptote + (residual_before_off - off_asymptote) * np.exp(-t_fall / tau_off)
    return on_peak, off_peak


# ---------------------------------------------------------------------------
# PART A: idealized step, symmetric model, spec section 11's 5-point checklist
# ---------------------------------------------------------------------------

def part_a():
    print("=" * 78)
    print("PART A: idealized instantaneous step, symmetric CPD model")
    print("(spec section 11's 5-point checklist)")
    print("=" * 78)

    eta = 0.5
    I0 = 1.0
    tau_values_us = [10, 100, 500]
    all_pass = True

    fig, axes = plt.subplots(len(tau_values_us), 3, figsize=(13, 3.6 * len(tau_values_us)))

    for row, tau_us in enumerate(tau_values_us):
        tau = tau_us * 1e-6
        dt = tau / 200.0  # resolve the fastest dynamics with margin
        tp = 20 * tau     # long plateau: tp >> tau, matches spec section 8's "symmetric" limit
        t_on = 5 * tau
        t_off = t_on + tp
        n_steps = int((t_off + 10 * tau) / dt)

        t, I_L = idealized_step_pulse(n_steps, dt, t_on, t_off, I0)
        model = CPDTIAPixel(grid_size=1, dt=dt, tau_cpd=tau, eta=eta, Rf=1.0)
        I_L_grid = I_L.reshape(n_steps, 1, 1)
        I_cpd_trace, V_o_trace = model.run(I_L_grid)
        I_cpd = I_cpd_trace[:, 0, 0]
        V_o = V_o_trace[:, 0, 0]

        on_peak_num = I_cpd[(t >= t_on) & (t < t_on + 5 * dt)].max()
        off_idx = np.where(t >= t_off)[0]
        off_peak_num = I_cpd[off_idx[0]:off_idx[0] + 5].min()

        on_peak_th, off_peak_th = analytical_step_peaks(eta, tau, I0, tp)

        # 5 checks
        check_on_polarity = on_peak_num > 0
        check_off_polarity = off_peak_num < 0
        check_on_amp = abs(on_peak_num - on_peak_th) / abs(on_peak_th) < 0.02
        # decay check: I_cpd should fall to 1/e of its peak after ~tau seconds
        decay_idx = np.searchsorted(t, t_on + tau)
        decayed_val = I_cpd[decay_idx]
        check_decay = abs(decayed_val - on_peak_num / np.e) / (on_peak_num / np.e) < 0.05
        # steady-state check: just before the OFF transition, I_cpd ~ off_peak_th's
        # "settled" companion (should be small since tp >> tau)
        settled_val = I_cpd[off_idx[0] - 1]
        check_steady_state = abs(settled_val) < 0.01 * on_peak_num

        checks = [check_on_polarity, check_off_polarity, check_on_amp,
                  check_decay, check_steady_state]
        row_pass = all(checks)
        all_pass = all_pass and row_pass

        print(f"\ntau_CPD = {tau_us} us:")
        print(f"  ON peak:  numeric={on_peak_num:.6f}  analytical={on_peak_th:.6f}  "
              f"{'PASS' if check_on_amp else 'FAIL'}")
        print(f"  OFF peak: numeric={off_peak_num:.6f}  analytical={off_peak_th:.6f}")
        print(f"  ON polarity positive:  {'PASS' if check_on_polarity else 'FAIL'}")
        print(f"  OFF polarity negative: {'PASS' if check_off_polarity else 'FAIL'}")
        print(f"  Decays to 1/e of peak by t=tau: {'PASS' if check_decay else 'FAIL'}")
        print(f"  Steady-state ~0 before OFF transition: {'PASS' if check_steady_state else 'FAIL'}")
        print(f"  ROW: {'ALL PASS' if row_pass else 'SOME FAILED'}")

        t_us = t * 1e6
        axes[row, 0].plot(t_us, I_L, color="black")
        axes[row, 0].set_title(f"I_L(t), tau_CPD={tau_us}us")
        axes[row, 0].set_xlabel("time (us)")
        axes[row, 1].plot(t_us, I_cpd, color="tab:blue")
        axes[row, 1].axhline(on_peak_th, color="tab:blue", linestyle=":", alpha=0.6)
        axes[row, 1].axhline(off_peak_th, color="tab:blue", linestyle=":", alpha=0.6)
        axes[row, 1].set_title("I_cpd(t) (dotted = analytical peaks)")
        axes[row, 1].set_xlabel("time (us)")
        axes[row, 2].plot(t_us, V_o, color="tab:red")
        axes[row, 2].set_title("V_o(t) = -Rf * I_cpd(t)")
        axes[row, 2].set_xlabel("time (us)")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "validate_part_a_step_response.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nSaved {out_path}")
    print(f"\nPART A OVERALL: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED -- fix before proceeding'}")
    return all_pass


# ---------------------------------------------------------------------------
# PART B: finite-rise-time ramp, symmetric vs. asymmetric model
# ---------------------------------------------------------------------------

def part_b():
    print("\n" + "=" * 78)
    print("PART B: finite-rise-time pulse -- symmetric vs. asymmetric CPD model")
    print("=" * 78)

    eta = 0.5
    I0 = 1.0
    tau_on = 100e-6
    tau_off_asym = 1.5 * tau_on  # "slightly asymmetric": slower recovery/discharge
    t_edge = 80e-6   # rise AND fall time -- comparable to tau, so asymmetry can show up
    t_plateau = 20 * tau_on  # long enough that the ON transient fully settles first
    dt = min(tau_on, t_edge) / 100.0

    t_on = 5 * tau_on
    t_off = t_on + t_edge + t_plateau
    n_steps = int((t_off + t_edge + 10 * tau_off_asym) / dt)

    t, I_L = ramp_pulse(n_steps, dt, t_on, t_edge, t_off, t_edge, I0)
    I_L_grid = I_L.reshape(n_steps, 1, 1)

    results = {}
    for name, tau_off in [("symmetric", tau_on), ("asymmetric", tau_off_asym)]:
        model = CPDTIAPixel(grid_size=1, dt=dt, tau_cpd=tau_on, eta=eta, Rf=1.0,
                             tau_cpd_off=tau_off)
        I_cpd_trace, V_o_trace = model.run(I_L_grid)
        I_cpd = I_cpd_trace[:, 0, 0]

        on_peak_num = I_cpd[(t >= t_on) & (t < t_on + t_edge + 5 * dt)].max()
        off_window = (t >= t_off) & (t < t_off + t_edge + 20 * dt)
        off_peak_num = I_cpd[off_window].min()

        on_peak_th, off_peak_th = analytical_ramp_peaks(
            eta, tau_on, tau_off, I0, t_edge, t_edge, t_plateau)

        asym_ratio_num = abs(off_peak_num) / on_peak_num
        asym_ratio_th = abs(off_peak_th) / on_peak_th

        results[name] = dict(t=t, I_cpd=I_cpd, on_peak_num=on_peak_num,
                              off_peak_num=off_peak_num, on_peak_th=on_peak_th,
                              off_peak_th=off_peak_th, asym_ratio_num=asym_ratio_num,
                              asym_ratio_th=asym_ratio_th)

        check_amp = (abs(on_peak_num - on_peak_th) / on_peak_th < 0.03 and
                     abs(off_peak_num - off_peak_th) / abs(off_peak_th) < 0.05)
        print(f"\n{name} model (tau_on={tau_on*1e6:.0f}us, tau_off={tau_off*1e6:.0f}us):")
        print(f"  ON peak:  numeric={on_peak_num:.6f}  analytical={on_peak_th:.6f}")
        print(f"  OFF peak: numeric={off_peak_num:.6f}  analytical={off_peak_th:.6f}")
        print(f"  |OFF|/|ON| ratio: numeric={asym_ratio_num:.4f}  analytical={asym_ratio_th:.4f}")
        print(f"  Matches closed-form ramp solution: {'PASS' if check_amp else 'FAIL'}")

    sym_ratio = results["symmetric"]["asym_ratio_num"]
    asym_ratio = results["asymmetric"]["asym_ratio_num"]
    print(f"\nSymmetric model |OFF|/|ON| = {sym_ratio:.4f} (should be ~1.0 -- equal peaks)")
    print(f"Asymmetric model |OFF|/|ON| = {asym_ratio:.4f} (should be > 1.0, since "
          f"tau_off > tau_on makes the CPD integrate more of the falling edge -- "
          f"i.e. THIS asymmetric config gives a LARGER OFF peak than ON peak, not "
          f"smaller. If Larry's device shows the opposite direction, set "
          f"tau_cpd_off < tau_cpd instead.)")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(t * 1e6, I_L, "k--", alpha=0.4, label="I_L(t)")
    ax.plot(results["symmetric"]["t"] * 1e6, results["symmetric"]["I_cpd"],
            color="tab:blue", label="I_cpd, symmetric (tau_on=tau_off)")
    ax.plot(results["asymmetric"]["t"] * 1e6, results["asymmetric"]["I_cpd"],
            color="tab:orange", label="I_cpd, asymmetric (tau_off=1.5x tau_on)")
    ax.set_xlabel("time (us)")
    ax.set_ylabel("I_L, I_cpd (a.u.)")
    ax.set_title("Finite-rise-time pulse: symmetric vs. asymmetric CPD ON/OFF peaks")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "validate_part_b_symmetric_vs_asymmetric.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    ok = part_a()
    part_b()
    if not ok:
        sys.exit(1)
