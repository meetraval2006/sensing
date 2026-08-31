"""
analysis_responsivity.py

Reproduces, on our Mechanism 1 model, the WSe2 paper's key programmability
claim (Zhou et al., Fig. 2f, Fig. 3f, and Supplementary Fig. 19): the
pixel's spike amplitude A scales LINEARLY with the electrically-programmed,
non-volatile photoresponsivity R, i.e. A = k * R.

The paper realizes different R values by applying gate-voltage pulses that
non-volatilely shift the WSe2 floating-gate charge state (see "Non-volatile
and programmable WSe2 photodiode" in the paper). In our model, R is exactly
the free parameter that plays that role (device_mechanism1.py's
`responsivity_map` / uniform `R`). We sweep R directly, apply a fixed step
change in light intensity to a single pixel, and record the peak transient
current magnitude BEFORE quantization -- that peak is the model's analogue
of the paper's spike amplitude A.

We do this for:
  - Mechanism 1 (raw dual-branch): I_net = R*L_fast - R*L_slow_filtered is
    exactly linear in R by construction, so this should recover A = k*R with
    R^2 essentially 1.0 -- a direct, checkable confirmation that our
    implementation matches the paper's reported relationship.
  - Hybrid (v3, linear-core + boosted tail, see device_hybrid.py): above a
    knee x0 the transient passes through UNCHANGED, so A(R) is literally
    identical to Mechanism 1's line for large enough R*step. Below the
    knee (small R), the sublinear tail BOOSTS the amplitude above what pure
    linear scaling would give -- so the hybrid's curve should sit on top of
    Mechanism 1's line at low R and merge into it exactly at high R. This
    is the flip side of analysis_fixed_threshold_range.py's finding: the
    same boosted tail that helps a fixed threshold survive a dimmer scene
    also means a fixed R sees more amplitude at weak signal levels here.
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


def peak_amplitude_mechanism1(R, tau_slow, dt, n_steps, step_time, L_before, L_after):
    """Single-pixel step response using the exact update rule in
    device_mechanism1.py's step(), kept continuous (pre-quantization) so we
    can read off its peak -- q_thresh only affects how finely that peak gets
    digitized into events, not the shape of A(R)."""
    I_slow = R * L_before  # start settled at steady state
    peak = 0.0
    for i in range(n_steps):
        t = i * dt
        L = L_after if t >= step_time else L_before
        I_fast = R * L
        I_slow += (dt / tau_slow) * (R * L - I_slow)
        peak = max(peak, abs(I_fast - I_slow))
    return peak


def peak_amplitude_hybrid(R, alpha, x0, tau_slow, dt, n_steps, step_time,
                           L_before, L_after, eps=1e-6):
    """Same step response but through device_hybrid.py's v3 architecture:
    exact linear difference first, then a linear-core/boosted-tail shape --
    passes through UNCHANGED above the knee x0, boosted below it (see
    device_hybrid.py's module docstring)."""
    def shape(x):
        ax = abs(x)
        if ax >= x0:
            return x
        boosted = np.sign(x) * x0 * (max(ax, eps) / x0) ** alpha
        return boosted

    I_slow_raw = R * L_before
    peak = 0.0
    for i in range(n_steps):
        t = i * dt
        L = L_after if t >= step_time else L_before
        raw_fast = R * L
        I_slow_raw += (dt / tau_slow) * (R * L - I_slow_raw)
        I_diff = raw_fast - I_slow_raw
        peak = max(peak, abs(shape(I_diff)))
    return peak


def main():
    dt = 1e-4
    tau_slow = 5e-3
    n_steps = 2000
    step_time = 0.02
    L_before, L_after = 0.5, 2.5  # matches spot_simulator's baseline/peak scale

    # Log-spaced and extended well below the compression knee (R ~ 0.003,
    # see main() below) so the boosted-tail deviation from Mechanism 1's
    # line is actually visible in the plot, not just claimed in text.
    R_values = np.logspace(-4, np.log10(3.0), 40)

    A_mech1 = np.array([
        peak_amplitude_mechanism1(R, tau_slow, dt, n_steps, step_time, L_before, L_after)
        for R in R_values
    ])
    A_hybrid = np.array([
        peak_amplitude_hybrid(R, 0.4, 0.006, tau_slow, dt, n_steps, step_time, L_before, L_after)
        for R in R_values
    ])

    # Linear fit A = k*R for Mechanism 1 (the paper's claimed relationship)
    k_fit, b_fit = np.polyfit(R_values, A_mech1, 1)
    A_fit = k_fit * R_values + b_fit
    ss_res = np.sum((A_mech1 - A_fit) ** 2)
    ss_tot = np.sum((A_mech1 - A_mech1.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    ax = axes[0]
    ax.plot(R_values, A_mech1, "o-", color="tab:blue", markersize=3,
            label=f"Mechanism 1: A = k*R fit, k={k_fit:.3f}, R^2={r2:.5f}")
    ax.plot(R_values, A_hybrid, "s-", color="tab:orange", markersize=3,
            label="Hybrid v3: merges into Mech.1's line at high R")
    ax.set_xlabel("Programmed responsivity, R (a.u.)")
    ax.set_ylabel("Peak transient amplitude, A (a.u.)")
    ax.set_title("Linear scale\n(matches Fig. 2f/3f's typical operating range)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.loglog(R_values, A_mech1, "o-", color="tab:blue", markersize=3, label="Mechanism 1")
    ax2.loglog(R_values, A_hybrid, "s-", color="tab:orange", markersize=3, label="Hybrid v3")
    ax2.set_xlabel("Programmed responsivity, R (a.u., log scale)")
    ax2.set_ylabel("Peak transient amplitude, A (a.u., log scale)")
    ax2.set_title("Log-log scale\n(reveals the boosted tail at low R)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle("Spike-amplitude programmability "
                 "(cf. Zhou et al. Fig. 2f / 3f / Supp. Fig. 19: A = k*R)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, "responsivity_programmability.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    merge_r = 0.006 / (2.5 - 0.5)  # x0 / step-size -- R above this is exactly linear
    summary = (
        f"Mechanism 1 linear fit: A = {k_fit:.4f} * R + {b_fit:.4f}   "
        f"(R^2 = {r2:.5f}; R^2 -> 1.0 confirms exact linearity, matching the "
        f"paper's reported A = k*R relationship)\n"
        f"Hybrid v3 model: for R above ~{merge_r:.3f} (where the transient magnitude "
        f"clears the x0=0.006 knee), amplitude vs. R is IDENTICAL to Mechanism 1's line "
        f"-- no distortion of already-strong signals. Below that, the boosted tail lifts "
        f"amplitude ABOVE the linear line (not below), trading exact proportionality for "
        f"extra sensitivity to weak R -- consistent with the fixed-threshold robustness "
        f"gain measured in analysis_fixed_threshold_range.py."
    )
    print(summary)
    with open(os.path.join(OUT_DIR, "responsivity_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
