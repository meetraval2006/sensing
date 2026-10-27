# CPD event imaging on `synthetic_traffic`

160 frames, 128x72 px, 1 video frame = 1000 us of CPD time, theta = 4.5 nA, frame 0 held 30 ms before scoring.

| condition | what it is | rho_G | rho_sq | ON | OFF | ON/OFF count | active px (last frame) | mean active px | tracker | object | traj err mean (px) | traj err p90 (px) | valid frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-reference | No traps, reference circuit = SYMMETRIC BASELINE | 1.004 | - | 3223 | 3224 | 1.000 | 1772 | 490.8 | spec ON-set/OFF-reset | all | 19.25 | 56.09 | 1.000 |
|  |  |  |  |  |  |  |  |  | spec ON-set/OFF-reset | 0 | 2.10 | 3.90 | 1.000 |
|  |  |  |  |  |  |  |  |  | spec ON-set/OFF-reset | 1 | 36.40 | 63.62 | 1.000 |
|  |  |  |  |  |  |  |  |  | bipolar | all | 7.33 | 16.97 | 1.000 |
|  |  |  |  |  |  |  |  |  | bipolar | 0 | 4.02 | 8.61 | 1.000 |
|  |  |  |  |  |  |  |  |  | bipolar | 1 | 10.64 | 19.02 | 1.000 |
| B-trap-sq0.5 | Traps, square-pulse ON/OFF 0.50 | 1.052 | 0.496 | 3162 | 3182 | 0.994 | 1726 | 474.7 | spec ON-set/OFF-reset | all | 18.89 | 56.33 | 1.000 |
|  |  |  |  |  |  |  |  |  | spec ON-set/OFF-reset | 0 | 1.38 | 2.16 | 1.000 |
|  |  |  |  |  |  |  |  |  | spec ON-set/OFF-reset | 1 | 36.41 | 62.84 | 1.000 |
|  |  |  |  |  |  |  |  |  | bipolar | all | 6.20 | 15.96 | 1.000 |
|  |  |  |  |  |  |  |  |  | bipolar | 0 | 2.61 | 5.02 | 1.000 |
|  |  |  |  |  |  |  |  |  | bipolar | 1 | 9.78 | 19.35 | 1.000 |
| B-trap-sq0.2 | Traps, square-pulse ON/OFF 0.20 (strongest) | 1.266 | 0.198 | 3123 | 3133 | 0.997 | 1692 | 463.2 | spec ON-set/OFF-reset | all | 19.05 | 56.54 | 1.000 |
|  |  |  |  |  |  |  |  |  | spec ON-set/OFF-reset | 0 | 0.98 | 1.55 | 1.000 |
|  |  |  |  |  |  |  |  |  | spec ON-set/OFF-reset | 1 | 37.12 | 62.92 | 1.000 |
|  |  |  |  |  |  |  |  |  | bipolar | all | 5.45 | 17.99 | 1.000 |
|  |  |  |  |  |  |  |  |  | bipolar | 0 | 1.11 | 1.61 | 1.000 |
|  |  |  |  |  |  |  |  |  | bipolar | 1 | 9.80 | 21.36 | 1.000 |
| ctrl-0.2 | Imposed ON scale 0.2 | 0.200 | - | 699 | 3224 | 0.217 | 14 | 35.5 | spec ON-set/OFF-reset | all | 10.51 | 40.03 | 0.742 |
|  |  |  |  |  |  |  |  |  | spec ON-set/OFF-reset | 0 | 0.90 | 1.67 | 1.000 |
|  |  |  |  |  |  |  |  |  | spec ON-set/OFF-reset | 1 | 30.32 | 52.20 | 0.485 |
|  |  |  |  |  |  |  |  |  | bipolar | all | 31.65 | 56.29 | 1.000 |
|  |  |  |  |  |  |  |  |  | bipolar | 0 | 29.24 | 53.30 | 1.000 |
|  |  |  |  |  |  |  |  |  | bipolar | 1 | 34.05 | 57.84 | 1.000 |

active px (last frame) = pixels still marked active by the event-state map at the last frame. Once every object has left the view these are trails (an ON with no matching OFF: a missed or delayed OFF event).
Labels carry the single-pass Gaussian ratio rho_G and, for physical trap conditions, the square-pulse calibration ratio rho_sq. ctrl-* rows are the Eq. (6) phenomenological output control, not a trap prediction.
Trackers: 'spec' is the task's rule (ON sets a pixel, OFF resets it). 'bipolar' (our addition, for video) lets the first event of a passage set the pixel and the opposite polarity clear it, so dark objects on a brighter background (which fire OFF first) are tracked too; it is identical to the spec rule for bright objects on dark.
