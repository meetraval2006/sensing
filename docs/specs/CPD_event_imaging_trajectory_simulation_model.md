# CPD model for Gaussian-spot event imaging and trajectory tracing

## What to simulate

Move a **Gaussian light spot** across a 64 × 64 pixel array. At every pixel, simulate the CPD readout current, convert threshold crossings into ON and OFF events, and reconstruct the spot's path from the events. Compare electrical timing without traps (Task A) with trap-assisted response (Task B). These are illustrative parameters, not fitted device values.

The spot center $\mathbf r_{\rm true}(t)=(x_c(t),y_c(t))$ is the known trajectory. Let its spatial standard deviation be $\sigma=2$ pixels. **$u(x,y,t)$ is normalized grayscale illumination, continuously ranging from 0 (dark) to 1 (peak brightness); it is not thresholded into a binary pixel input.** The illumination and direct photocurrent at pixel $(x,y)$ are

$$
u(x,y,t)=\exp\!\left[-\frac{(x-x_c(t))^2+(y-y_c(t))^2}{2\sigma^2}\right],\qquad
I_P(x,y,t)=I_{P0}u(x,y,t). \tag{1}
$$

Start the spot several $\sigma$ outside the array, move it along a straight path at 2 pixels/ms, and then simulate a separate path with a right-angle turn. Run a faster pass and a reversal after the basic runs. At a stationary pixel crossed through its center, the temporal width is $\sigma/v=1$ ms for $\sigma=2$ pixels and speed $v=2$ pixels/ms.

## Boxed equations for the simulator

**For Task B, implement this system independently at every pixel.** All three states, $T$, $i_R$, and $z$, persist between time steps and through changes in illumination.

$$
\boxed{\begin{gathered}
I_P=I_{P0}u,\qquad 0\leq u\leq1,\\[5pt]
\frac{dT}{dt}=u\frac{T_{\mathrm{ss}}-T}{\tau_{\mathrm{on}}}
-(1-u)\frac{T}{\tau_{\mathrm{off}}},\\[7pt]
I_{\mathrm{eff}}=I_P-\Gamma\frac{dT}{dt},\\[7pt]
a=\frac1{\tau_1}+\frac1{\tau_2},\qquad b=\frac1{\tau_1\tau_2},\\[5pt]
\frac{di_R}{dt}=z-\frac{I_{\mathrm{eff}}}{RC_s},\qquad
\frac{dz}{dt}=-az-bi_R+\frac{aI_{\mathrm{eff}}}{RC_s}.
\end{gathered}} \tag{S}
$$

Here $T$ is normalized occupied-trap density, $i_R$ is measured readout current, $I_{\rm eff}$ is current driving the circuit, and $z$ is an auxiliary electrical state. $\Gamma$ converts a unit change in $T$ to coupled charge. The pair $\tau_1,\tau_2$ sets the two circuit relaxation times; $\tau_{\rm on}$ and $\tau_{\rm off}$ set trap filling under full illumination and release in darkness. $RC_s$ is the fixed forcing scale. Use time in **µs**, current in **µA**, and $\Gamma$ in **µA·µs**.

At each time step: compute the **actual grayscale value** $u$ at every pixel from the spot position (or read its normalized value from a video frame); evaluate $dT/dt$ and $I_{\rm eff}$ from the current $T$; advance $T$, $i_R$, and $z$; then check $i_R$ for an event threshold crossing. Initialize $T=i_R=z=0$ while the spot is outside the array. Use $\Delta t=2$ µs initially and check a representative run at 1 µs. The equations accept smoothly varying grayscale input directly; **no ideal-edge jump or optical binarization is applied during trajectory simulation**.

### Task A: direct photocurrent model (no trap variables)

For Task A, implement the following equations directly at each pixel. The optical input is the Gaussian grayscale value from Eq. (1), and only the electrical states $i_R$ and $z$ need to be integrated. There is no trap state or effective-current variable in this implementation.

$$
\boxed{\begin{gathered}
I_P=I_{P0}u,\qquad 0\leq u\leq1,\\[4pt]
a=\frac1{\tau_1}+\frac1{\tau_2},\qquad b=\frac1{\tau_1\tau_2},\\[5pt]
\frac{di_R}{dt}=z-\frac{I_P}{RC_s},\qquad
\frac{dz}{dt}=-az-bi_R+\frac{aI_P}{RC_s}.
\end{gathered}} \tag{A}
$$

Initialize $i_R=z=0$ before illumination. This is the zero-trap reference used again in Task B. Use box (S) only for Task B's active-trap conditions.

## Values to enter

| Quantity | Task A: no traps | Task B: with traps |
| --- | ---: | ---: |
| Peak photocurrent I_P0 | 3 µA | 3 µA |
| Normalized T_ss | Not used | 1 |
| Coupling Γ | Not used | 2400 µA·µs = 2.4 nC per unit T |
| RC_s | 100 µs | 100 µs |
| Reference τ₁, τ₂ | 150 µs, 20 µs | 150 µs, 20 µs |
| Event threshold θ | 0.0045 µA (4.5 nA) | 0.0045 µA (4.5 nA) |

Enter the peak photocurrent directly; optical power is unnecessary for these comparisons. If a measured responsivity $\mathcal R$ is available, its corresponding peak optical power is $P_0=I_{P0}/\mathcal R$. The illustrative 3 µA photocurrent matches the stated microamp scale. Γ and I_P0 set the *relative* capture strength: at full light and T = 0, the capture current is ΓT_ss/τ_on. With τ_on ≥ 800 µs, this does not exceed 3 µA. The example threshold of 4.5 nA is for a **noiseless demonstration**, not a recommended hardware setting: the weakest trap-free Gaussian response peaks near 9 nA, only twice this threshold. For measured data, choose the threshold from the readout noise at the operating bandwidth and confirm that the weakest useful signal crosses it. Changing only the threshold changes event counts and trajectory results.

## Task A: electrical timing without traps

Run box (A) with the following four electrical time-constant pairs. Keep the spot, I_P0, RC_s, threshold, and tracker fixed.

| Case | τ₁ | τ₂ | Expected difference |
| --- | ---: | ---: | --- |
| Fast | 50 µs | 10 µs | Narrow events near the light change |
| Reference | 150 µs | 20 µs | Intermediate timing; also used in Task B |
| Slow | 400 µs | 40 µs | Broader and later events |
| Very slow | 1200 µs | 100 µs | Strongest overlap and lag on fast motion |

**Meaning of “symmetric.”** There is no trap-induced ON/OFF asymmetry in Task A. For an *isolated 2.5 ms ideal square pulse* with the reference τ₁ = 150 µs and τ₂ = 20 µs, the ON/OFF peak ratio ρ is essentially **1**, so this exact reference is the no-trap row for the square-pulse calibration in Task B. A moving Gaussian spot is a different optical input. Its ON/OFF peaks need not be exactly equal because the electrical modes are causal and can overlap. With σ = 2 pixels and v = 2 pixels/ms, illustrative no-trap Gaussian peak ratios for the four pairs are approximately **1.000, 1.004, 1.047, and 1.334**. Report these as electrical-only ratios, not as trap-induced asymmetry.

## Task B: trap-induced asymmetry, including the zero-trap control

First run a **zero-trap control** using box (A), with no trap state. This is Task A's reference electrical pair. Then keep τ₁ = 150 µs, τ₂ = 20 µs, I_P0 = 3 µA, T_ss = 1, Γ = 2400 µA·µs, and RC_s = 100 µs fixed for the four **active-trap** runs. Use the same Gaussian spot, threshold, and tracker throughout. The trap equation continuously interpolates between its full-light and dark limits:

$$
u=1:\quad \frac{dT}{dt}=\frac{T_{\mathrm{ss}}-T}{\tau_{\mathrm{on}}},\quad
I_{\mathrm{eff}}=I_{P0}-\Gamma\frac{T_{\mathrm{ss}}-T}{\tau_{\mathrm{on}}}; \tag{2a}
$$

$$
u=0:\quad \frac{dT}{dt}=-\frac{T}{\tau_{\mathrm{off}}},\quad
I_{\mathrm{eff}}=\Gamma\frac{T}{\tau_{\mathrm{off}}}. \tag{2b}
$$

Equations (2a)–(2b) are **limiting cases only**. During the Gaussian passage, most pixels have $0<u<1$; use the continuous trap equation in box (S), not a switch between these two limiting equations. Thus $\tau_{\rm on}$ controls the light-driven capture rate and $\tau_{\rm off}$ controls release, including release while illumination is partial. **Do not reset $T$ when the spot leaves a pixel**; its remaining value produces history on a later pass.

### Where the four ratios came from

The four τ_on–τ_off pairs below were calibrated with an **isolated ideal square pulse**, not with the Gaussian trajectory input. At one pixel, set I_P = 3 µA for 2.5 ms and zero afterward; initialize T = i_R = z = 0. Integrate box (S) through the ON interval and at least 12 ms of darkness. Keep the states continuous when I_P switches. Define

$$
\rho_{\mathrm{square}}=
\frac{|\min_{0<t<2.5\,\mathrm{ms}} i_R(t)|}
{\max_{t>2.5\,\mathrm{ms}} i_R(t)}. \tag{3}
$$

Sweep $\tau_{\rm on}\geq800$ µs and $\tau_{\rm off}>\tau_{\rm on}$, choose pairs closest to the four targets, and recompute the current peaks after rounding the pair. The numbers below used 18 ms of post-pulse darkness. They can be reproduced from box (S) by replacing the Gaussian grayscale $u$ with a 0/1 square pulse for **calibration only**. The Gaussian event-imaging runs always retain continuous grayscale $u$.

| Condition | Γ (µA·µs) | τ_on | τ_off | Square ON peak | Square OFF peak | ρ_square | ρ_G for one Gaussian pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Zero traps** | **0** | — | — | −0.44007 µA | +0.44007 µA | **1.000** | **1.004** |
| Active traps, target 1.0 | 2400 | 2.899 ms | 3.184 ms | −0.32007 µA | +0.32241 µA | 0.993 | 1.12 |
| Active traps, target 0.7 | 2400 | 1.858 ms | 13.058 ms | −0.25413 µA | +0.36591 µA | 0.695 | 1.02 |
| Active traps, target 0.5 | 2400 | 1.351 ms | 16.967 ms | −0.18636 µA | +0.37584 µA | 0.496 | 1.05 |
| Active traps, target 0.2 | 2400 | 0.829 ms | 8.786 ms | −0.07458 µA | +0.37605 µA | 0.198 | 1.27 |

**Important for interpretation:** the zero-trap row and the active-trap target-1.0 row both have nearly equal square-pulse peak magnitudes, but they have different waveforms and trap histories. The table labels active-trap conditions by their *square-pulse calibration ratio*. Applying those conditions to a moving Gaussian spot does **not** guarantee Gaussian-waveform ratios of 1, 0.7, 0.5, and 0.2. Report both ρ_square and the actual Gaussian ratio ρ_G; the latter directly describes the trajectory input. If the goal changes to *prescribing those four ratios in the Gaussian application itself*, the parameter search must be repeated using the Gaussian input, and fixed Γ and T_ss may not reach all targets.

### Optional controlled-ratio Gaussian comparison

To answer the separate application question **“What would ratios 1, 0.7, 0.5, and 0.2 do to tracking a Gaussian spot?”**, run a clearly labeled *output-level control* in addition to the physical trap runs. Let $i_0(t)$ be the trap-free Gaussian current with the reference electrical pair, and let $\rho_0$ be its measured ON/OFF peak ratio (approximately 1.004 here). For target ratio $\rho_\ast$, define

$$
i_{\rho_\ast}(t)=\begin{cases}
(\rho_\ast/\rho_0)\,i_0(t),&i_0(t)<0,\\
i_0(t),&i_0(t)\geq0.
\end{cases} \tag{6}
$$

Use the same threshold and tracker on these four controlled currents. This directly isolates the impact of **ON/OFF magnitude ratio** on Gaussian event imaging and trajectory tracing. It is a **phenomenological output control**, not a current predicted by the trap equations, and must be reported separately from the τ_on–τ_off study. The physical runs determine whether trap kinetics actually produce a comparable effect for this optical input.

## Events and trajectory outputs

Record an **ON event** at the first negative crossing of −θ and an **OFF event** at the first positive crossing of +θ for each pixel passage. Use θ = 0.0045 µA (4.5 nA) and at most one event of each polarity per passage. Preserve the physical sign convention: rising light tends to produce negative i_R, falling light positive i_R. Store (x, y, time, polarity), and show ON/OFF events in different colors.

For a simple tracer, let an ON event set its pixel to 1 in an **estimated event-state map** and an OFF event reset it to 0. This binary map is created *after* simulating the grayscale optical response; it does not replace the grayscale input $u$. The centroid of active pixels is $\mathbf r_{\rm est}(t)$. Compare it to the known Gaussian center $\mathbf r_{\rm true}(t)$. For the image-overlap metric only, define a true spot footprint by $u(x,y,t)\geq0.2$; do not apply that threshold to the CPD input. A missing ON event makes a hole and a delayed OFF event makes a trail.

| Measure | How to calculate it |
| --- | --- |
| ON/OFF event count | Count each polarity; also report their count ratio separately from ρ_square and ρ_G |
| ON/OFF detection fraction | Fraction of true u = 0.2 footprint entry/exit crossings with a matching event |
| ON/OFF latency | Median and 90th-percentile event time minus the corresponding u = 0.2 crossing time |
| Image overlap | Intersection over union of estimated active pixels and true u ≥ 0.2 footprint |
| Trajectory error | Mean and 90th-percentile distance between r_est and r_true, in pixels |
| Valid-trace fraction | Fraction of visible time with at least one estimated active pixel |
| Reversal delay | Time between the true reversal and the traced-path reversal |

Save the spot movie, representative pixel input and current, polarity event images, estimated footprint, and traced versus true path. Evaluate straight motion, a turn, a faster pass, reversal, and a repeated pass after different dark intervals. Apply the **same threshold and tracker** to Task A, the physical Task B, and the optional controlled-ratio comparison. Label each result by its actual Gaussian ratio, as well as its square-pulse calibration ratio when applicable.

## Connection to original V5 physics

The physical trap equation is

$$
\frac{dT}{dt}=\gamma n_p(t)(T_0-T)-\beta T. \tag{7}
$$

At full constant illumination, $\tau_{\mathrm{on}}^{-1}=\gamma n_p+\beta$, $T_{\mathrm{ss}}=\gamma n_pT_0/(\gamma n_p+\beta)$; in darkness, $\tau_{\mathrm{off}}^{-1}=\beta$. Taking carrier supply proportional to the Gaussian illumination u yields the trap update in box (S). For positive capture and release rates, use τ_on < τ_off; all four table pairs satisfy this. The V5 circuit equation is

$$
\frac{d^2i_R}{dt^2}+\left(\frac1{\tau_1}+\frac1{\tau_2}\right)\frac{di_R}{dt}
+\frac{i_R}{\tau_1\tau_2}=-\frac1{RC_s}\frac{dI_{\mathrm{eff}}}{dt}. \tag{8}
$$

The auxiliary state z in box (S) is an exactly equivalent first-order form that avoids numerically differentiating optical changes. For an ideal square-pulse edge, i_R, z, and T stay continuous while $\Delta(di_R/dt)=-\Delta I_{\mathrm{eff}}/(RC_s)$. For Gaussian trajectory input, solve box (A) for Task A and box (S) for active-trap Task B continuously.
