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
  - Hybrid: R is scaled first, THEN each branch is sign-preserving
    power-law compressed before differencing, so amplitude vs. R should be
    sub-linear -- directly illustrating the tradeoff documented in
    device_hybrid.py: the hybrid trades away exact linear programmability
    for extended dynamic range (see analysis_dynamic_range.py).
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


def peak_amplitude_hybrid(R, alpha, tau_slow, dt, n_steps, step_time,
                           L_before, L_after, eps=1e-3):
    """Same step response but through device_hybrid.py's compressed branches."""
    def compress(x):
        return np.sign(x) * (np.power(np.abs(x) + eps, alpha) - eps ** alpha)

    I_slow_raw = R * L_before
    peak = 0.0
    for i in range(n_steps):
        t = i * dt
        L = L_after if t >= step_time else L_before
        raw_fast = R * L
        I_slow_raw += (dt / tau_slow) * (R * L - I_slow_raw)
        peak = max(peak, abs(compress(raw_fast) - compress(I_slow_raw)))
    return peak


def main():
    dt = 1e-4
    tau_slow = 5e-3
    n_steps = 2000
    step_time = 0.02
    L_before, L_after = 0.5, 2.5  # matches spot_simulator's baseline/peak scale

    R_values = np.linspace(0.1, 3.0, 25)

    A_mech1 = np.array([
        peak_amplitude_mechanism1(R, tau_slow, dt, n_steps, step_time, L_before, L_after)
        for R in R_values
    ])
    A_hybrid = np.array([
        peak_amplitude_hybrid(R, 0.4, tau_slow, dt, n_steps, step_time, L_before, L_after)
        for R in R_values
    ])

    # Linear fit A = k*R for Mechanism 1 (the paper's claimed relationship)
    k_fit, b_fit = np.polyfit(R_values, A_mech1, 1)
    A_fit = k_fit * R_values + b_fit
    ss_res = np.sum((A_mech1 - A_fit) ** 2)
    ss_tot = np.sum((A_mech1 - A_mech1.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot(R_values, A_mech1, "o-", color="tab:blue",
            label=f"Mechanism 1 (dual-branch): A = k*R fit, k={k_fit:.3f}, R^2={r2:.5f}")
    ax.plot(R_values, A_hybrid, "s-", color="tab:orange",
            label="Hybrid (compressed branches): sub-linear in R")
    ax.set_xlabel("Programmed responsivity, R (a.u.)")
    ax.set_ylabel("Peak transient amplitude, A (a.u.)")
    ax.set_title("Spike-amplitude programmability\n"
                 "(cf. Zhou et al. Fig. 2f / 3f / Supp. Fig. 19: A = k*R)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "responsivity_programmability.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    summary = (
        f"Mechanism 1 linear fit: A = {k_fit:.4f} * R + {b_fit:.4f}   "
        f"(R^2 = {r2:.5f}; R^2 -> 1.0 confirms exact linearity, matching the "
        f"paper's reported A = k*R relationship)\n"
        f"Hybrid model: amplitude vs. R is visibly sub-linear/saturating "
        f"(compression applied after R-scaling breaks exact proportionality) "
        f"-- this is the programmability the hybrid trades away in exchange "
        f"for the dynamic-range gain quantified in analysis_dynamic_range.py."
    )
    print(summary)
    with open(os.path.join(OUT_DIR, "responsivity_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
