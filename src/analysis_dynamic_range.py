"""
analysis_dynamic_range.py

Reproduces, in spirit, the thin-film paper's dynamic-range characterization
(Li et al., Fig. 3B/C): sweep the input light STEP magnitude (delta P) over
many decades and measure the resulting peak spike-current magnitude for
each device model. The thin-film paper's sub-linear power law
(I ~ P_in^alpha, alpha = 0.27/0.38 < 1) means weak steps are boosted
relative to strong ones -- this compression is what the paper credits for
its >110 dB dynamic range, vs. 60-70 dB for a linear-response (conventional
frame) sensor.

IMPORTANT CAVEAT: our simulation uses arbitrary intensity units, not the
papers' physical mW/cm^2, so the dB numbers below are NOT literally
comparable to the papers' measured 110 dB / 50 dB figures. What IS a valid,
apples-to-apples comparison (because all three models share the same units,
stimulus, and sweep) is the *relative* shape of each model's response curve
and how far down into weak signals each one can still produce a detectable
event before hitting its own tuned noise floor (q_thresh from
compare_models.py) -- i.e. does sub-linear compression actually buy you
detectable range in this simulation, the way the paper's physics predicts.

DR_dB = 20 * log10(deltaP_max / deltaP_min), where deltaP_min is the
smallest step size whose peak response clears the model's detection floor
and deltaP_max is the largest step size tested.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def peak_mechanism1(R, tau_slow, dt, n_steps, step_time, L_base, dP):
    I_slow = R * L_base
    peak = 0.0
    for i in range(n_steps):
        t = i * dt
        L = L_base + dP if t >= step_time else L_base
        I_fast = R * L
        I_slow += (dt / tau_slow) * (R * L - I_slow)
        peak = max(peak, abs(I_fast - I_slow))
    return peak


def peak_mechanism2(tau_th, dt, n_steps, step_time, L_base, dP,
                     alpha_heat=1.0, kappa=1.0, alpha_pos=0.27, eps=1e-3):
    T = alpha_heat * L_base
    peak = 0.0
    for i in range(n_steps):
        t = i * dt
        L = L_base + dP if t >= step_time else L_base
        T_prev = T
        T += (dt / tau_th) * (alpha_heat * L - T)
        dTdt = (T - T_prev) / dt
        I_pyro = kappa * dTdt
        if I_pyro > 0:
            I_signed = np.power(I_pyro + eps, alpha_pos) - eps ** alpha_pos
        else:
            I_signed = 0.0
        peak = max(peak, abs(I_signed))
    return peak


def peak_hybrid(R, alpha, tau_slow, dt, n_steps, step_time, L_base, dP, eps=1e-3):
    def compress(x):
        return np.sign(x) * (np.power(np.abs(x) + eps, alpha) - eps ** alpha)

    I_slow_raw = R * L_base
    peak = 0.0
    for i in range(n_steps):
        t = i * dt
        L = L_base + dP if t >= step_time else L_base
        raw_fast = R * L
        I_slow_raw += (dt / tau_slow) * (R * L - I_slow_raw)
        peak = max(peak, abs(compress(raw_fast) - compress(I_slow_raw)))
    return peak


def dynamic_range_db(dP_values, peaks, floor):
    detectable = dP_values[peaks >= floor]
    if len(detectable) == 0:
        return np.nan, np.nan, np.nan
    dp_min, dp_max = detectable.min(), detectable.max()
    return 20 * np.log10(dp_max / dp_min), dp_min, dp_max


def main():
    dt = 1e-4
    n_steps = 800
    step_time = 0.01
    L_base = 0.5

    # Sweep step size across 6 decades, mirroring the paper's Pin sweep
    # (0.1 uW/cm^2 to 30 mW/cm^2 spans ~5.5 decades).
    dP_values = np.logspace(-5, 1, 40)

    peaks_m1 = np.array([peak_mechanism1(1.0, 5e-3, dt, n_steps, step_time, L_base, dP)
                          for dP in dP_values])
    peaks_m2 = np.array([peak_mechanism2(5e-3, dt, n_steps, step_time, L_base, dP)
                          for dP in dP_values])
    peaks_hy = np.array([peak_hybrid(1.0, 0.4, 5e-3, dt, n_steps, step_time, L_base, dP)
                          for dP in dP_values])

    # Detection floor = each model's tuned q_thresh from compare_models.py,
    # so "detectable" here means "would actually fire an event in our
    # trajectory-tracing comparison run."
    floors = {"Mechanism 1 (linear)": 0.004,
              "Mechanism 2 (sub-linear)": 0.08,
              "Hybrid": 0.0015}
    peaks = {"Mechanism 1 (linear)": peaks_m1,
             "Mechanism 2 (sub-linear)": peaks_m2,
             "Hybrid": peaks_hy}

    fig, ax = plt.subplots(figsize=(7, 5.5))
    results_lines = []
    for name, color in zip(peaks, ["tab:blue", "tab:orange", "tab:green"]):
        ax.loglog(dP_values, peaks[name], "o-", color=color, label=name, markersize=4)
        dr_db, dp_min, dp_max = dynamic_range_db(dP_values, peaks[name], floors[name])
        results_lines.append(
            f"{name:<28} floor={floors[name]:<8} "
            f"detectable dP range=[{dp_min:.2e}, {dp_max:.2e}]  DR = {dr_db:.1f} dB (sim units)")
        ax.axhline(floors[name], color=color, linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Light intensity step size, delta P (a.u., log scale)")
    ax.set_ylabel("Peak transient current amplitude (a.u., log scale)")
    ax.set_title("Dynamic range: peak spike amplitude vs. step size\n"
                 "(cf. thin-film paper Fig. 3B/C: sub-linear response -> higher dynamic range)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "dynamic_range_comparison.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    summary = "\n".join(results_lines) + (
        "\n\nNOTE: dB values use this simulation's arbitrary intensity units "
        "and each model's own hand-tuned q_thresh floor -- they are internally "
        "consistent for comparing the three models to each other, but are NOT "
        "literally comparable to the papers' measured 110 dB / 60-70 dB figures "
        "(those used calibrated mW/cm^2 units and real device noise floors)."
    )
    print(summary)
    with open(os.path.join(OUT_DIR, "dynamic_range_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
