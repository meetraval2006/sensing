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

### 2.3 Hybrid model — `src/device_hybrid.py`  ⚠️ our construction, empirically tested (including a kept negative result)
This is **our own construction**, not from either paper — built to test
whether combining both mechanisms' strengths helps. It went through three
designs; all are documented (see the module's docstring for the full
reasoning) because the failures are as informative as the fix:

- Mechanism 1's strength: exact zero-baseline cancellation, precise 5 μs
  timing, but *linear* responsivity → limited dynamic range.
- Mechanism 2's strength: sublinear compression → huge dynamic range, strong
  low-light sensitivity, but *no* built-in cancellation mechanism at steady
  state (relies entirely on the pyroelectric derivative decaying to zero).

**v1 (superseded):** compressed *each branch separately*, then subtracted.
Compresses the wrong quantity — small transients riding on a bright
baseline get *attenuated*, not boosted. Modest dynamic-range gain (70.8 dB
vs. Mechanism 1's 67.7 dB), worse tracing than Mechanism 1.

**v2 (superseded):** compute the exact linear branch difference *first*,
*then* compress that difference, so compression acts where its curve is
steepest. Real gains: 86.2 dB dynamic range, tracing within 0.2 px of
Mechanism 1.

**v3 (current) — we then tried to literally beat Mechanism 1's tracing
accuracy, not just approach it, and the honest result was mixed:**
make the response **exactly linear** (byte-for-byte identical to Mechanism 1)
above a compression knee `x0`, and only boost signal below it — so a moving
spot's weak Gaussian tail can contribute events without ever distorting the
strong-signal region:

```
I_diff = R * L(t) - R * L_filtered(t)               # exact, same as Mechanism 1
I_net  = I_diff                        if |I_diff| >= x0
       = sign(I_diff) * x0 * (|I_diff|/x0)^alpha     otherwise
```

The first comparison looked like a clear win — until we controlled for
event rate (v3 was simply firing more events at those thresholds, and
denser sampling lowers RMSE on its own). A proper density-matched test
(`analysis_hybrid_vs_mechanism1.py`) shows the apparent win **evaporates**:
at matched event count, Mechanism 1 wins or ties every time, by small
(0.3–4.1%) margins. **The hybrid does not beat Mechanism 1's raw tracing
accuracy on this stimulus, full stop** — kept in the repo as a documented
negative result rather than deleted.

What v3 *does* win, genuinely: `analysis_fixed_threshold_range.py` shows
that with each model's threshold tuned **once** on a normal scene and never
retuned, Mechanism 1 goes completely blind (0 events) once the scene dims
below its operating range, while v3's boosted tail still produces some
usable signal one brightness step further down. That's a real, structural,
but narrower advantage than "beats Mechanism 1" — it's "survives conditions
Mechanism 1 wasn't tuned for," which is the actual, correctly-scoped
analogue of a dynamic-range claim. Full story, numbers, and the
counterfactual that pins this down (retuning Mechanism 1 per-scene closes
the gap) in `RESULTS.md` §2.5–§2.7.

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
│   ├── compare_models.py                    <- trajectory-tracing comparison, makes plots + table
│   ├── animate.py                           <- "light going around" animated GIF, all 3 models side by side
│   ├── analysis_responsivity.py             <- validates A = k*R programmability claim (Mechanism 1)
│   ├── analysis_dynamic_range.py            <- validates sub-linear dynamic-range claim (Mechanism 2)
│   ├── analysis_hybrid_vs_mechanism1.py     <- density-matched control test (the "did the hybrid really win?" check)
│   └── analysis_fixed_threshold_range.py    <- the hybrid's real, narrower advantage: fixed-threshold robustness
└── outputs/                     <- generated by the scripts above (all regenerable, see below)
    ├── animation.gif
    ├── trajectory_tau_0.005.png / _0.02.png / _0.08.png
    ├── rmse_vs_tau.png
    ├── event_rasters.png
    ├── summary_table.txt
    ├── responsivity_programmability.png / responsivity_summary.txt
    ├── dynamic_range_comparison.png / dynamic_range_summary.txt
    ├── hybrid_vs_mechanism1_density_matched.png / hybrid_vs_mechanism1_summary.txt
    └── fixed_threshold_dynamic_range.png / fixed_threshold_range_summary.txt
```

## 4. How to run it

```bash
pip install -r requirements.txt
cd src
python animate.py                          # the "light going around" animation, all 3 models
python compare_models.py                   # trajectory-tracing accuracy comparison
python analysis_responsivity.py            # validates Mechanism 1's A = k*R claim
python analysis_dynamic_range.py           # validates Mechanism 2's dynamic-range claim
python analysis_hybrid_vs_mechanism1.py    # controls for event-rate, checks if hybrid really beats Mech. 1
python analysis_fixed_threshold_range.py   # tests the hybrid's real advantage: fixed-threshold robustness
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
| Hybrid (custom, v3) | 1.06 px | 1.40 px | 2.69 px | 2,912 |

- All three models trace the circular path with 99–100% coverage — the
  fundamental mechanism (event-driven change detection → memory decode)
  works for all of them on this simple test.
- Error grows with fading time `tau` in every model — longer memory smooths
  out noise but adds lag/blur.
- At matched event count, the Hybrid (v3) is now statistically
  indistinguishable from Mechanism 1 (1.06 px vs. 1.03 px) — **we tried to
  make it actually beat Mechanism 1 and, after controlling for a real
  event-rate confound, confirmed it does not** (`RESULTS.md` §2.6). It does
  have one genuine, narrower advantage: with a threshold tuned once and
  never retuned, it keeps producing usable signal on scenes dim enough to
  leave Mechanism 1 completely blind (`RESULTS.md` §2.7).
- Separately, `analysis_responsivity.py` confirms Mechanism 1's spike
  amplitude is *exactly* linear in programmed responsivity (R² = 1.00000),
  matching the paper's `A = k*R` claim, and `analysis_dynamic_range.py`
  shows Mechanism 2's sub-linear compression gives it a simulated 113.8 dB
  dynamic range — strikingly close to the paper's own reported >110 dB,
  even though that exponent (0.27) wasn't tuned to hit this number. Full
  story, including the negative result, in `RESULTS.md` §2.5–§5.

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
- The hybrid v3's knee (`x0=0.006`) and exponent (`alpha=0.4`) are still
  hand-picked, not swept. We did test whether the hybrid genuinely
  outperforms Mechanism 1 (that was the whole point of v3) and found,
  after controlling for event-rate, that it does not on raw tracing
  accuracy (`RESULTS.md` §2.6) — its real, narrower win is fixed-threshold
  robustness to scene dimming (`RESULTS.md` §2.7), and how far a parameter
  sweep could push that specific advantage is still open.

## 7. Suggested next steps
1. Sweep the hybrid v3's `x0`/`alpha` and both models' `tau` values more
   finely specifically on the fixed-threshold-robustness axis (`RESULTS.md`
   §2.7), since that's the one place a real, demonstrated advantage exists
   — a stimulus with genuine sensor noise (not just a clean synthetic spot)
   is also worth testing, since §2.6's negative result may be specific to
   the noise-free stimulus used here.
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
