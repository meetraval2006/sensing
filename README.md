# Event-Based Vision Device Models — Comparison Repo

**Author:** Meet Raval, Oregon State University ECE, working with Prof. Larry Cheng
**Prepared for:** 8/21 meeting
**Purpose:** Implement, from scratch, two device-physics-grounded event-camera
models (one per project reference paper), a hybrid model combining their
mechanisms, and a shared "physical observation memory" readout to trace a
moving object — then compare all three quantitatively on a controlled
synthetic stimulus, animate it, and validate each model's headline claim
against its source paper.

This addresses the meeting agenda item: *"provide a device model to something
shown below and see what kind of property we can look at ... run both models
separately from both papers and a custom model that takes the best from
both models and implements it."*

**→ For the full quantitative comparison (tables, plots, animation, discussion),
see [`RESULTS.md`](RESULTS.md). This README covers the theory and how to run it.**

Both source PDFs live one directory above this repo
(`../Computational eventdriven vision sensors.pdf` — Zhou et al., *Nature
Electronics* 2023 — and `../Thinfilm eventbased vision sensors for enhanced
multispectral perception beyond human vision.pdf` — Li et al., *InfoMat*
2025). They are **not** included in this repo (or pushed to GitHub) since
they're copyrighted journal articles; see full citations in §7.

---

## 1. What's actually being tested

A synthetic bright spot moves in a known circular path across a small pixel
grid (ground truth is exact, since we generate it). Each device model
"watches" this moving spot and emits its own event stream. A shared
downstream decoder (the "physical observation memory" from your notes)
converts each event stream back into a traced (x, y) path over time. We then
score how close the traced path is to the known ground truth — this is the
"can the model trace a moving spot" test from the meeting notes, and lets us
sweep the memory's fading time constant (`tau`) to see how it trades off
sharpness vs. robustness, per your notes on that being an open question.

Because ground truth is known exactly, this setup isolates **device physics
+ readout quality** from everything else — no need for real hardware or
labeled video to get a first quantitative signal.

---

## 2. Theory behind each model

### 2.1 Mechanism 1 — `src/device_mechanism1.py`
**Source:** Zhou et al., *Nature Electronics* (2023) — WSe2 dual-branch pixel
(your primary reference paper).

Each pixel has two parallel opposite-polarity photodiode branches:
- **Fast branch** (no capacitor): current tracks light instantaneously,
  `I_fast(t) = R * L(t)`
- **Slow branch** (series capacitor → RC delay): current lags light,
  `tau * dI_slow/dt = R*L(t) - I_slow(t)`

Net pixel current is the *difference*: `I_net = I_fast - I_slow`.

- **Constant light** → both branches settle to the same steady-state value →
  they cancel → **zero output**. This matches the paper's description
  exactly: no comparator is needed, the cancellation is a physical property
  of the RC mismatch.
- **Light changes** → fast branch reacts immediately, slow branch lags → for
  that transient window the currents don't cancel → a spike appears,
  positive for brightening, negative for darkening.
- `R` is the paper's non-volatile, electrically **programmable
  responsivity** — modeled here as a per-pixel gain array, so you can assign
  different "synaptic weights" across the array if you want to explore the
  in-sensor-SNN angle later.
- Reported temporal resolution: **5 μs** (`tau_slow` default in the code).

The continuous `I_net(t)` is converted into discrete `(x, y, t, polarity)`
events using a **charge integrate-and-fire quantizer**: charge accumulates,
and an event fires (with reset) whenever it crosses a threshold. This
preserves the "area" of each transient, so bigger/faster changes → more
events, matching real event-camera behavior, without needing a separate
comparator stage that isn't in the paper's actual pixel circuit.

### 2.2 Mechanism 2 — `src/device_mechanism2.py`
**Source:** "Thin-film event-based vision sensors for enhanced multispectral
perception beyond human vision" (your second reference paper).

Fundamentally different physics: a **single two-terminal device** using the
**pyro-phototronic effect**. Absorbed light heats the device; temperature
*change* generates a pyroelectric current:

```
tau_th * dT/dt = alpha_heat * L(t) - T(t)      (thermal RC lag)
I_pyro(t) = kappa * dT/dt                       (pyroelectric law: current ∝ rate of temp change)
```

The paper also reports a **sublinear power-law** relationship between spike
current and input power (`I ~ P_in^alpha`, with different exponents for ON
vs OFF: `alpha+ = 0.27`, `alpha- = 0.38`). This sublinear compression is what
gives the device its very high dynamic range (**>110 dB** reported, vs.
60–70 dB for conventional frame cameras) — weak signals get relatively
boosted, similar in spirit to log-compression but arising from the
pyroelectric response itself.

**Implementation note:** a raw power law with exponent < 1 has *infinite
slope at zero*, which blows up ordinary numerical noise into runaway false
events. We use an epsilon-regularized version,
`sign(x) * ((|x|+eps)^alpha - eps^alpha)`, which preserves the qualitative
sublinear compression while staying numerically well-behaved near zero. This
is documented in code — it's a modeling choice, not something from the
paper, and worth mentioning if asked at the meeting.

### 2.3 Hybrid model — `src/device_hybrid.py`  ⚠️ hypothesis, not established
This is **our own construction**, not from either paper — built to test
whether combining both mechanisms' strengths helps.

- Mechanism 1's strength: exact zero-baseline cancellation, precise 5 μs
  timing, but *linear* responsivity → limited dynamic range.
- Mechanism 2's strength: sublinear compression → huge dynamic range, strong
  low-light sensitivity, but *no* built-in cancellation mechanism at steady
  state (relies entirely on the pyroelectric derivative decaying to zero).

**Hybrid approach:** keep Mechanism 1's two-branch differential structure,
but apply Mechanism 2's sign-preserving sublinear compression to *each
branch* before differencing:

```
I_fast = compress(R * L(t))
I_slow = compress(R * L_filtered(t))      # RC-lagged, same as Mechanism 1
I_net  = I_fast - I_slow
```

Both branches see the same compressed steady-state value, so cancellation
should still hold, while the compression should extend dynamic range. This
is exactly the property the comparison script tests — **treat it as an open
question, not a foregone win**, consistent with your v6/v7 notes about not
assuming an advantage until it's demonstrated (see below).

### 2.4 Physical observation memory (readout) — `src/memory_decode.py`
**Source:** your uploaded notes (v7 review). This is deliberately kept
**separate** from the three device models — it's a mechanism-agnostic
readout that works on any event stream:

```
dG+/dt = -G+/tau + kappa * E+(t)
dG-/dt = -G-/tau + kappa * E-(t)
A_M    = |G+ - G-|
```

This is a leaky integrator: each event nudges the corresponding polarity's
memory state by `kappa`, and that state decays exponentially with time
constant `tau` ("fading time"). `A_M` is large wherever recent net event
activity occurred, and the decoder takes a simple activity-weighted centroid
of `A_M` each step to get a traced (x, y) position — this is the "generate
the fading time to see how well it can trace" test from your notes.

Per your notes, the **signed** cancellation (`A_M = |G+ - G-|`) is a
hypothesis your v6 review found nearly identical to the simpler
**rectified** alternative (`A_M = G+ + G-`) at the "starved event" operating
point. `compare_models.py` reports **both** so you can re-check that
finding as event supply increases (which it does here, since we're not
event-starved in this synthetic setup — worth discussing at the meeting).

---

## 3. Repo structure

```
event_camera_project/
├── README.md                    <- this file (theory + how to run)
├── RESULTS.md                   <- full quantitative comparison (tables, plots, discussion)
├── requirements.txt
├── src/
│   ├── spot_simulator.py            <- ground-truth moving spot generator
│   ├── device_mechanism1.py         <- WSe2 dual-branch model (Zhou et al.)
│   ├── device_mechanism2.py         <- pyro-phototronic model (thin-film paper)
│   ├── device_hybrid.py             <- custom hybrid (hypothesis)
│   ├── memory_decode.py             <- physical observation memory readout
│   ├── compare_models.py            <- trajectory-tracing comparison, makes plots + table
│   ├── animate.py                   <- "light going around" animated GIF, all 3 models side by side
│   ├── analysis_responsivity.py     <- validates A = k*R programmability claim (Mechanism 1)
│   └── analysis_dynamic_range.py    <- validates sub-linear dynamic-range claim (Mechanism 2)
└── outputs/                     <- generated by the scripts above (all regenerable, see below)
    ├── animation.gif
    ├── trajectory_tau_0.005.png / _0.02.png / _0.08.png
    ├── rmse_vs_tau.png
    ├── event_rasters.png
    ├── summary_table.txt
    ├── responsivity_programmability.png / responsivity_summary.txt
    └── dynamic_range_comparison.png / dynamic_range_summary.txt
```

## 4. How to run it

```bash
pip install -r requirements.txt
cd src
python animate.py                    # the "light going around" animation, all 3 models
python compare_models.py             # trajectory-tracing accuracy comparison
python analysis_responsivity.py      # validates Mechanism 1's A = k*R claim
python analysis_dynamic_range.py     # validates Mechanism 2's dynamic-range claim
```

`compare_models.py` will:
1. Generate a 24×24-pixel synthetic video of a spot moving in a circle
   (3000 timesteps, 1 ms each = 3 seconds of simulated motion).
2. Run all three device models on the identical stimulus, producing three
   independent event streams.
3. Decode each event stream into a traced path at three fading times
   (`tau = 0.005s, 0.02s, 0.08s` — fast / medium / slow memory).
4. Score each (model, tau) combination against ground truth using RMSE
   (root-mean-square pixel error) and coverage (fraction of timesteps where
   the decoder had enough activity to produce a position estimate at all).
5. Save comparison plots and a summary table to `outputs/`.

Event-rate thresholds (`q_thresh` in each model) were hand-tuned so all
three models fire a comparable number of events on the same stimulus —
comparing models at wildly different event rates would confound "which
physics traces better" with "which model just fires more often." Re-tune
these if you change the stimulus parameters.

## 5. Current results (this run)

**See [`RESULTS.md`](RESULTS.md) for the full breakdown** — trajectory
accuracy, the `A = k*R` programmability check, the dynamic-range check, and
a cross-axis synthesis table. Headline numbers:

| Model | tau=0.005s RMSE | tau=0.02s RMSE | tau=0.08s RMSE | events generated |
|---|---|---|---|---|
| Mechanism 1 (WSe2 dual-branch) | 1.03 px | 1.39 px | 2.68 px | 3,016 |
| Mechanism 2 (pyro-phototronic) | 1.61 px | 2.04 px | 3.80 px | 6,695 |
| Hybrid (custom) | 1.61 px | 1.72 px | 3.02 px | 2,870 |

- All three models trace the circular path with 99–100% coverage — the
  fundamental mechanism (event-driven change detection → memory decode)
  works for all of them on this simple test.
- Error grows with fading time `tau` in every model — longer memory smooths
  out noise but adds lag/blur.
- Mechanism 1 has the lowest tracing error at every `tau`, consistent with
  its exact steady-state cancellation.
- Separately, `analysis_responsivity.py` confirms Mechanism 1's spike
  amplitude is *exactly* linear in programmed responsivity R² = 1.00000),
  matching the paper's `A = k*R` claim, and `analysis_dynamic_range.py`
  shows Mechanism 2's sub-linear compression gives it a simulated 113.8 dB
  dynamic range — strikingly close to the paper's own reported >110 dB,
  even though that exponent (0.27) wasn't tuned to hit this number. Full
  discussion in `RESULTS.md` §3–§5.

## 6. Known simplifications / good discussion points for the meeting

- Both device models are first-order (single RC time constant) approximations
  of real device physics — real WSe2/pyroelectric devices likely have
  higher-order or nonlinear thermal/electrical dynamics not captured here.
- The stimulus is a single bright Gaussian spot on a flat background — no
  texture, no multiple objects, no real-world lighting variation. Good next
  step: feed in an actual video frame sequence (ties into Stage 1/2 of the
  project plan using v2e-style pipelines).
- `q_thresh` values were manually tuned for comparable event rates rather
  than derived from actual device datasheet specs — once you have real
  measured device parameters, these should be replaced.
- The hybrid model's compression exponent (`alpha=0.4`) was chosen
  arbitrarily as a middle ground between the two papers' reported exponents
  — a parameter sweep would clarify whether the hybrid is a real
  improvement under any conditions.

## 7. Suggested next steps
1. Sweep the hybrid's `alpha` and both models' `tau` values more finely to
   see if there's a regime where the hybrid genuinely outperforms both
   individual mechanisms (dynamic-range-stressed stimuli, e.g. dim spot on
   bright background, are the most likely place to see it).
2. Replace the synthetic Gaussian spot with a real video frame sequence
   (start with something simple, e.g. an OSU hallway walk-through) to bridge
   toward Stage 1 of Prof. Cheng's plan (v2e/IEBCS-based simulation).
3. Once measured device parameters come back from device fabrication/testing
   (or from the corresponding authors, if code/data sharing comes through),
   replace the hand-picked `tau_slow`, `R`, `alpha` values with real
   numbers.

## 8. References

1. Zhou, Y., Fu, J., Chen, Z. et al. Computational event-driven vision
   sensors for in-sensor spiking neural networks. *Nat. Electron.* 6,
   870–878 (2023). https://doi.org/10.1038/s41928-023-01055-2
2. Li, K., Wang, X., Wu, Y. et al. Thin-film event-based vision sensors for
   enhanced multispectral perception beyond human vision. *InfoMat* 7(7),
   e70007 (2025). https://doi.org/10.1002/inf2.70007

Both PDFs are kept locally one directory above this repo for reference and
are intentionally **not committed here** — they're copyrighted journal
articles (article 2 is CC-BY open access and may be redistributed with
attribution, but is still excluded here to keep this repo's scope to code
and generated results only).
