# CPD-Based Event Dynamic Imaging Simulation

## 1. Objective

The current repository is a validated event-imaging framework built around literature-reported event-sensor models. The next step is to retain the existing scene generation, event representation, visualization, and motion-analysis infrastructure while replacing the literature-derived sensor front end with our physically derived capacitive photodiode (CPD) model.

The immediate goal is:

```text
optical scene
    ↓
photocurrent I_L(x,y,t)
    ↓
physical CPD model
    ↓
i_CPD(x,y,t)
    ↓
TIA voltage V_o(x,y,t)
    ↓
signed analog CPD event image / event video
    ↓
optional ON/OFF threshold for discrete event stream
```

The first priority is to demonstrate that the CPD produces useful event images for moving objects. Do not introduce LIF, fading-memory integration, or machine learning until this CPD event-imaging stage has been validated.

---

## 2. What to retain from the current repository

Keep the existing infrastructure for:

- synthetic moving targets;
- per-pixel time-dependent optical input;
- ON/OFF event representation `(t, x, y, polarity)`;
- event-image and event-video visualization;
- trajectory ground truth and reconstruction metrics;
- literature models as reference baselines.

Replace only the existing literature-derived event-sensor front end with the CPD model described below.

The current Gaussian moving-spot simulation should remain as a convenient benchmark because its trajectory is known exactly. A moving edge or moving rectangular bar should also be added because it provides a cleaner visualization of CPD-generated ON/OFF event edges.

---

## 3. Signal definitions and model interface

For the CPD simulation, use **one input quantity only** for the optical excitation:

```text
I_L(x,y,t)        light-induced photocurrent at each pixel
i_CPD(x,y,t)      native CPD transient output current
V_o(x,y,t)        experimentally measurable TIA output voltage
E(x,y,t,p)        optional digital ON/OFF events
```

Here $I_L(x,y,t)$ is the **single optical-input variable used by the simulation**: the light-induced photocurrent corresponding to the instantaneous illumination at that pixel. It is the electrical representation of the incident light at the CPD input. Therefore the CPD model should **not simultaneously use a normalized light variable $L$ and a separate amplitude $I_0$**. Any conversion from physical irradiance to photocurrent can be calibrated outside the CPD dynamic model when experimental responsivity data are available.

The CPD code should return

```text
i_CPD(t)
V_o(t)
```

so that the device output current and the experimentally measurable TIA voltage remain the quantities emphasized in the simulation.

---

# 4. Latest physical model: CPD connected directly to a TIA

The primary simulation should now use the **CPD connected directly to the virtual-ground input of a transimpedance amplifier (TIA)**. This is different from the earlier passive-load configuration in which an external resistor $R$ participates directly in the CPD dynamics.

The physical CPD still contains:

- semiconductor resistance $r$;
- semiconductor capacitance $C_s$;
- insulating capacitance $C_i$;
- light-induced photocurrent source $I_L(t)$.
![alt text](image.png)
However, the TIA holds the external readout electrode close to virtual ground. Under the ideal-TIA approximation, the external load-resistor mode is removed from the CPD itself.

Let $v(t)$ be the internal CPD node voltage. Current balance gives

$$
I_L(t)=\frac{v(t)}{r}+(C_s+C_i)\frac{dv}{dt}.
$$

The current delivered to the TIA input is the current through $C_i$:

$$
i_{\mathrm{CPD}}(t)=C_i\frac{dv}{dt}.
$$

Define the intrinsic CPD time constant

$$
\tau_{\mathrm{CPD}}=r(C_s+C_i).
$$

Eliminating $v(t)$ gives the first-order high-pass relation

$$
\tau_{\mathrm{CPD}}\frac{di_{\mathrm{CPD}}}{dt}
+i_{\mathrm{CPD}}
=rC_i\frac{dI_L}{dt}.
$$

The transfer function is

$$
\frac{I_{\mathrm{CPD}}(s)}{I_L(s)}
=
\frac{rC_i s}{1+r(C_s+C_i)s}
=
\frac{C_i}{C_s+C_i}
\frac{\tau_{\mathrm{CPD}}s}{1+\tau_{\mathrm{CPD}}s}.
$$

Therefore the CPD behaves as an intrinsic first-order temporal high-pass / differentiating photosensor.

It is useful to define

$$
\eta=\frac{C_i}{C_s+C_i},
$$

so that

$$
\frac{I_{\mathrm{CPD}}(s)}{I_L(s)}
=\eta\frac{\tau_{\mathrm{CPD}}s}{1+\tau_{\mathrm{CPD}}s}.
$$

The two most important physical quantities are therefore:

```text
intrinsic response time    tau_CPD = r (Cs + Ci)
edge-current coupling      eta     = Ci / (Cs + Ci)
```

This representation is more appropriate for the next simulation than the earlier passive-$R$ second-order model because the intended experimental readout is now CPD $\rightarrow$ TIA.

The earlier passive-load model should not be deleted. It should remain as an **alternate readout architecture** for comparison.

---

# 5. Role of the TIA in the latest model

For an ideal TIA with transimpedance gain $R_f$,

$$
V_o(t)=-R_f i_{\mathrm{CPD}}(t).
$$

Therefore

$$
\frac{V_o(s)}{I_L(s)}
=-R_f\eta\frac{\tau_{\mathrm{CPD}}s}{1+\tau_{\mathrm{CPD}}s}.
$$

The crucial consequence is:

> **$R_f$ controls output-voltage gain but does not set the intrinsic CPD decay time in the ideal-TIA limit.**

This is fundamentally different from the earlier passive-load case, where an external resistance directly participated in the device time constants.

For a real TIA, finite bandwidth should be added as a separate readout dynamics. A useful first approximation is

$$
H_{\mathrm{TIA}}(s)
=
\frac{-R_f}{1+s\tau_{\mathrm{TIA}}},
$$

with

$$
\tau_{\mathrm{TIA}}\approx\frac{1}{2\pi f_{\mathrm{BW,TIA}}}.
$$

Then

$$
\frac{V_{\mathrm{out}}(s)}{I_L(s)}
=
-R_f\eta
\frac{\tau_{\mathrm{CPD}}s}{1+\tau_{\mathrm{CPD}}s}
\frac{1}{1+s\tau_{\mathrm{TIA}}}.
$$

This gives a clean separation between:

```text
CPD physics       tau_CPD = r(Cs + Ci)
TIA gain          Rf
TIA distortion    tau_TIA or bandwidth
```

That separation should be preserved throughout the simulation.

---

# 6. Revised primary device model

Create a new primary model such as

```text
src/cpd/cpd_tia_simple.py
```

with a class such as

```text
CPDTIAPixel
```

The fundamental physical parameters should be

```text
r_sl       semiconductor-layer resistance r
Cs         semiconductor capacitance Cs
Ci         insulating capacitance Ci
Rf         TIA transimpedance gain
BW_tia     TIA bandwidth; infinity for ideal TIA
dt         internal simulation timestep
```

The optical scene should supply $I_L(x,y,t)$ directly as the light-induced photocurrent input. No separate $L$, $I_0$, or optical-responsivity parameter is required inside the reduced CPD dynamic model.

The code should return both

```text
i_CPD(t)
V_o(t)
```

so that sensor physics and electronic readout can be analyzed independently.

---

# 7. Physical model versus numerical implementation

The **physical CPD model used for the imaging study is the continuous input-output equation in terms of the light-induced photocurrent $I_L(t)$ and the measurable CPD current $I_{\mathrm{CPD}}(t)$**:

$$
\boxed{
\tau_{\mathrm{CPD}}\frac{dI_{\mathrm{CPD}}(t)}{dt}
+I_{\mathrm{CPD}}(t)
=\eta\tau_{\mathrm{CPD}}\frac{dI_L(t)}{dt}
}
$$

This is the equation that should be identified throughout the project as the **CPD simulation model**. It expresses the physical capacitive temporal differentiation directly. The CPD operates continuously; it does not require a shutter, frame-to-frame storage, or a digital subtraction operation in the physical device.

The internal node voltage $v(t)$ introduced in the derivation in Section 4 is only an intermediate variable used to derive this measurable input-output relation. It is not an experimentally measured output and should not be emphasized in the imaging results or model interface.

## 7.1 Numerical implementation

A computer must represent the continuous optical trajectory on a finite time grid. This numerical sampling is an implementation of the continuous model, **not part of the physical CPD mechanism**.

For synthetic moving objects, first define a continuous or finely interpolated $I_L(x,y,t)$ trajectory and numerically solve the continuous CPD equation at each pixel. The numerical solver may internally evaluate the change of $I_L(t)$ or use an equivalent state-space/filter realization, but the project should not present a discrete difference such as $I_L[n]-I_L[n-1]$ as the physical CPD model.

The numerical timestep $\Delta t$ should resolve both the fastest CPD dynamics and the fastest optical variation of interest. For waveform-level validation, use several samples across the shortest relevant time constant and verify convergence by reducing $\Delta t$ until $I_{\mathrm{CPD}}(t)$ and the event images no longer change materially.

The simulation interface should therefore be presented as

```text
continuous/fine-time I_L(x,y,t)
        ↓
continuous CPD differential equation
        ↓
I_CPD(x,y,t)
        ↓
V_o(x,y,t) = -Rf I_CPD(x,y,t)
```

For a nonideal first-order TIA, the optional readout equation is

$$
\tau_{\mathrm{TIA}}\frac{dV_o}{dt}+V_o
=-R_f I_{\mathrm{CPD}}.
$$

For the initial ideal-TIA study, use simply

$$
\boxed{V_o(t)=-R_f I_{\mathrm{CPD}}(t).}
$$

---

# 8. Analytical square-pulse response of the latest model

Use the square pulse

$$
I_L(t)=
\begin{cases}
0, & t<0,\\
I_0, & 0\le t<t_p,\\
0, & t\ge t_p.
\end{cases}
$$

For light turn-on,

$$
i_{\mathrm{CPD}}(t)
=
\eta I_0 e^{-t/\tau_{\mathrm{CPD}}},
\qquad 0\le t<t_p.
$$

At light turn-off,

$$
i_{\mathrm{CPD}}(t)
=
-\eta I_0
\left(1-e^{-t_p/\tau_{\mathrm{CPD}}}\right)
 e^{-(t-t_p)/\tau_{\mathrm{CPD}}},
\qquad t\ge t_p.
$$

Thus the ON peak is

$$
I_{\mathrm{ON,pk}}=\eta I_0,
$$

and the OFF peak magnitude is

$$
|I_{\mathrm{OFF,pk}}|
=
\eta I_0\left(1-e^{-t_p/\tau_{\mathrm{CPD}}}\right).
$$

For

$$
t_p\gg\tau_{\mathrm{CPD}},
$$

ON and OFF peaks approach equal magnitude.

For

$$
t_p\ll\tau_{\mathrm{CPD}},
$$

the OFF response is smaller because the internal CPD state has not fully charged before the light is removed.

This analytical result should be the first numerical validation target.

---

## 9. Simplification for the immediate event-imaging study

### 9.1 Single physical CPD equation used in the imaging simulation

Use **one continuous-time physical equation** for every pixel:

$$
\boxed{
\tau_{\mathrm{CPD}}\frac{dI_{\mathrm{CPD}}(x,y,t)}{dt}
+I_{\mathrm{CPD}}(x,y,t)
=\eta\tau_{\mathrm{CPD}}\frac{dI_L(x,y,t)}{dt}
}
$$

where

$$
\tau_{\mathrm{CPD}}=r(C_s+C_i),
\qquad
\eta=\frac{C_i}{C_s+C_i}.
$$

This equation is the **device model**. $I_L(x,y,t)$ is the continuously varying light-induced photocurrent incident on pixel $(x,y)$, and $I_{\mathrm{CPD}}(x,y,t)$ is the native measurable CPD output current.

The equation directly contains the key physical feature of the device: the CPD is driven by the temporal derivative $dI_L/dt$. The differentiation arises from the CPD capacitance itself. It should therefore not be described as a digital subtraction between two sampled light values.

The ideal-TIA output is

$$
\boxed{
V_o(x,y,t)=-R_f I_{\mathrm{CPD}}(x,y,t).
}
$$

The light-ON and light-OFF exponential equations in Section 8 are **special analytical solutions used only to validate this continuous model**. They are not separate equations used by the imaging simulation.

For a step change of magnitude $\Delta I_L$, the continuous model gives the immediate peak

$$
I_{\mathrm{CPD,pk}}=\eta\,\Delta I_L.
$$

For the special validation step $0\rightarrow I_0$, this becomes

$$
I_{\mathrm{CPD,pk}}=\eta I_0.
$$

Thus $I_0$ appears only as the amplitude of a particular validation step. It is not an additional light-input variable in the general imaging simulation.

For the first system-level study, assume the TIA is sufficiently fast that its own rise time can be neglected. The main independent CPD parameter is then

$$
\boxed{
\tau_{\mathrm{CPD}}=10\text{--}500~\mu\mathrm{s}.
}
$$

The numerical code should solve the boxed continuous CPD equation on a sufficiently fine time grid and report $I_{\mathrm{CPD}}$ and $V_o$. Any finite-difference, interpolation, ODE-solver, or equivalent filter realization belongs only to the numerical-method implementation and should not replace the continuous equation as the stated physical model.

---

## 10. Required code revision

Create a CPD pixel model, for example

```text
src/cpd/cpd_tia_simple.py
```

with a class such as

```text
CPDTIAPixel
```

and parameters

```text
tau_cpd
eta
Rf
dt
```

For the reduced event-imaging model, it is acceptable to parameterize the device directly by `tau_cpd` and `eta` rather than sweeping `r`, `Cs`, and `Ci`.

The physical relations

$$
\tau_{\mathrm{CPD}}=r(C_s+C_i)
$$

and

$$
\eta=\frac{C_i}{C_s+C_i}
$$

should remain documented, but mapping individual physical component values is not required in the immediate simulation.

Use the continuous CPD input-output differential equation in Section 9.1 as the primary physical model. Implement it numerically with a sufficiently resolved time grid while keeping the code interface expressed in terms of the physical input $I_L$ and measurable outputs $I_{\mathrm{CPD}}$ and $V_o$.

---

## 11. Single-pixel validation

Before running a 2-D scene, verify the CPD model with a square optical pulse.

For representative values

$$
\tau_{\mathrm{CPD}}
=
10,\ 100,\ 500~\mu\mathrm{s},
$$

plot:

```text
light-induced photocurrent I_L(t)
CPD current i_CPD(t)
TIA voltage V_o(t)
```

Verify:

1. the ON transient has the expected polarity;
2. the OFF transient has the opposite polarity;
3. the ON peak satisfies

$$
I_{\mathrm{ON,pk}}=\eta\,\Delta I_L;
$$

4. the transient decays with the specified

$$
\tau_{\mathrm{CPD}};
$$

5. the steady-state CPD current approaches zero under constant illumination.

This validation should be completed before any motion study.

---

## 12. CPD event representation

The primary event-imaging result should first use the **native signed CPD response without imposing a threshold**. At each time $t$, display either

$$
I_{\mathrm{CPD}}(x,y,t)
$$

or the directly measurable TIA voltage

$$
V_o(x,y,t)=-R_f I_{\mathrm{CPD}}(x,y,t).
$$

Positive and negative values represent the two temporal-change polarities, while the magnitude retains the strength of the CPD response. This signed analog map is the most direct representation of what the CPD physically produces.

A digital ON/OFF event stream should be treated as an **optional readout stage**, not as part of the CPD physics. If conventional event-camera output is needed, generate an ON event when

$$
V_o>V_{\mathrm{th,+}}
$$

and an OFF event when

$$
V_o<V_{\mathrm{th,-}}.
$$

These optional discrete events can be stored as

```text
(t, x, y, polarity)
```

When thresholded results are compared across different values of $\tau_{\mathrm{CPD}}$, keep the threshold definition consistent so that apparent changes in event morphology are not caused only by threshold selection.

The recommended order is therefore

```text
I_L(x,y,t)
    ↓
physical CPD model
    ↓
I_CPD(x,y,t)
    ↓
V_o(x,y,t)
    ↓
signed analog event image          ← primary
    ↓
optional ON/OFF threshold
    ↓
discrete event stream (t,x,y,p)    ← secondary
```

---

## 13. Main event-imaging simulation

After the single-pixel model is validated, replace the previous literature sensor with the CPD model in the existing moving-object framework.

### 13.1 Moving bar or moving edge

Start with a bright rectangular bar moving across a dark background.

The expected response is

```text
leading edge   → one CPD polarity
bar interior   → response decays toward zero
trailing edge  → opposite CPD polarity
```

For each tested

$$
\tau_{\mathrm{CPD}},
$$

show the actual spatial event image, not only scalar metrics.

Required visualization:

```text
optical frame | signed CPD current map | TIA voltage map | optional thresholded ON/OFF event image
```

Also generate an event-video animation for representative cases.

### 13.2 Moving Gaussian spot

Then repeat the existing moving Gaussian-spot trajectory with the CPD front end.

Show:

- input object;
- CPD event image;
- reconstructed trajectory;
- ground-truth trajectory.

This provides a direct comparison with the previous literature-based simulations.

---

## 14. Effect of CPD decay time on event imaging

Sweep

$$
\tau_{\mathrm{CPD}}
=
10,\ 20,\ 50,\ 100,\ 200,\ 300,\ 500~\mu\mathrm{s}
$$

over several motion velocities.

A key physical scaling is

$$
\boxed{
\ell_{\mathrm{event}}
=
v\tau_{\mathrm{CPD}}
}
$$

where

$$
v
$$

is the target velocity.

This is the distance traveled during one CPD decay time and provides an estimate of temporal-to-spatial event smearing.

For an object or edge with characteristic width

$$
\sigma,
$$

define

$$
\boxed{
\Gamma
=
\frac{v\tau_{\mathrm{CPD}}}{\sigma}
}.
$$

Interpretation:

```text
Gamma << 1   CPD response is fast relative to target motion;
             event localization should be sharp.

Gamma ~ 1    CPD persistence and target motion are comparable;
             visible event trails and overlap are expected.

Gamma >> 1   event responses strongly overlap;
             localization and reconstruction may degrade.
```

The corresponding design condition to test is

$$
\boxed{
\tau_{\mathrm{CPD}}
\ll
\frac{\sigma}{v}
}.
$$

---

## 15. Required event-imaging outputs

The student should produce the following outputs before moving to a memory or ML model:

1. analytical versus numerical single-pixel CPD response;
2. CPD current and TIA voltage for ON/OFF optical transitions;
3. moving-bar ON/OFF event images for several values of $\tau_{\mathrm{CPD}}$;
4. moving-bar event-video animation;
5. moving-Gaussian event images and reconstructed trajectory;
6. event-trail width versus $v\tau_{\mathrm{CPD}}$;
7. event localization or trajectory error versus $\tau_{\mathrm{CPD}}$ and velocity;
8. event count or density versus $\tau_{\mathrm{CPD}}$;
9. representative comparison with the previous literature sensor model.

The central question is:

> How does the experimentally relevant CPD decay time affect the spatial and temporal quality of the resulting event image?

---

## 16. Ordinary-video extension

Only after the controlled synthetic simulations are working should ordinary video be introduced.

The processing chain is

```text
video
  ↓
L(x,y,t)
  ↓
I_L(x,y,t)
  ↓
CPD
  ↓
i_CPD(x,y,t)
  ↓
TIA
  ↓
V_o(x,y,t)
  ↓
ON/OFF threshold
  ↓
event video
```

For ordinary frame-based video, clearly distinguish the video frame interval from the much shorter CPD internal timestep. Temporal interpolation between frames may be used for the first implementation, but it should be documented as an assumed intermediate optical trajectory rather than true recovered sub-frame motion.

---

## 17. Next stage after CPD event imaging: reservoir computing

Do not add this stage until the CPD event images have been validated.

After the CPD front end is demonstrated, **reservoir computing (RC) should be investigated as the primary temporal-computing approach before introducing a conventional deep-learning model**. RC is still a machine-learning method, but it is attractive here because the internal dynamical system can remain fixed and only a simple output layer needs to be trained.

This is particularly compatible with the intended device concept. The CPD already provides fast temporal differentiation,

$$
I_L(t) \rightarrow I_{\mathrm{CPD}}(t),
$$

while a fading-memory element or an LIF-like dynamical element can provide temporal integration and memory. The combination can therefore serve as a physical reservoir rather than merely generating an event tensor that is sent to a large software neural network.

A simple fading reservoir state can be represented by

$$
\tau_k \frac{dx_k(t)}{dt}
=
-x_k(t)+\kappa_k I_{\mathrm{CPD}}(t),
$$

where $x_k(t)$ is reservoir state $k$, $\tau_k$ is its fading-memory time constant, and $\kappa_k$ is its coupling coefficient to the CPD output.

Multiple reservoir states with different time constants can encode different portions of the recent temporal history:

$$
\tau_1,\; \tau_2,\; \tau_3,\ldots
$$

The number of distinct time constants should initially be kept small so that the model remains compatible with practical hardware implementation. A two- or three-time-constant reservoir should therefore be tested before considering a large continuously distributed reservoir.

For a sensor array, the reservoir can be driven by the CPD output at each pixel:

$$
\tau_k \frac{\partial x_k(x,y,t)}{\partial t}
=
-x_k(x,y,t)
+\kappa_k I_{\mathrm{CPD}}(x,y,t).
$$

The reservoir states can then be collected into a feature vector,

$$
\mathbf{x}(t)
=
[x_1(t),x_2(t),\ldots,x_N(t)]^T,
$$

or, for image data, into spatial reservoir-state maps $x_k(x,y,t)$.

The trained output can then be kept simple:

$$
\mathbf{y}(t)=W_{\mathrm{out}}\mathbf{x}(t),
$$

where only $W_{\mathrm{out}}$ is trained, for example by linear regression or ridge regression.

The preferred later-stage pipeline is therefore

```text
optical scene
    ↓
CPD temporal differentiation
    ↓
I_CPD(x,y,t) / V_o(x,y,t)
    ↓
fading-memory or LIF-like reservoir states
    ↓
linear trained readout
    ↓
classification / recognition
```

This approach should be evaluated before a conventional CNN, RNN, transformer, or other deep-learning classifier because it tests a more important hardware question: **how much useful temporal recognition can be performed by the intrinsic sensor and memory dynamics with only a simple trained readout?**

LIF remains a possible implementation of a reservoir node, but it should not automatically be treated as a separate mandatory stage between the CPD and machine learning. In particular, the CPD output may directly drive fading-memory or LIF reservoir states. If spikes are desired, an LIF threshold can generate them; if continuous reservoir states are sufficient, the membrane/fading states can be read directly.

A useful later comparison is therefore:

```text
A. CPD only → simple readout
B. CPD → fading/LIF reservoir → linear readout
C. CPD → conventional ML/deep-learning model
```

The purpose of comparison C is primarily to provide an accuracy benchmark. The main hardware-oriented direction should first test whether approach B can obtain strong dynamic-recognition performance with substantially simpler training and a more direct physical implementation.

---

## 18. Immediate implementation sequence

```text
1. Replace literature sensor front end with CPD model.
2. Validate one CPD pixel analytically and numerically.
3. Generate signed analog CPD/TIA event images; add ON/OFF thresholding only as an optional discrete-event readout.
4. Show moving-bar event images.
5. Sweep tau_CPD and motion speed.
6. Show moving-Gaussian event images and trajectory.
7. Compare representative CPD result with previous sensor model.
8. Only after these are complete, add a fading-memory/LIF-like reservoir.
9. Train a simple linear/ridge-regression readout from the reservoir states.
10. Compare CPD-only, CPD + reservoir computing, and CPD + conventional ML as later benchmarks.
```

The immediate deliverable is therefore a **working CPD-based event imager**. Reservoir computing is the preferred next-stage temporal-computing study; conventional deep learning should be added later as a benchmark rather than assumed to be the primary processing route.
