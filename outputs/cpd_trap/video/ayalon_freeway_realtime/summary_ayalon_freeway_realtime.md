# CPD event imaging on `ayalon_freeway_realtime`

30 frames, 80x45 px, 1 video frame = 33333 us of CPD time, theta = 4.5 nA, frame 0 held 30 ms before scoring.

| condition | what it is | rho_G | rho_sq | ON | OFF | ON/OFF count | active px (last frame) | mean active px |
|---|---|---|---|---|---|---|---|---|
| A-reference | No traps, reference circuit = SYMMETRIC BASELINE | 1.004 | - | 0 | 0 | - | 0 | 0.0 |

active px (last frame) = pixels still marked active by the event-state map at the last frame. Once every object has left the view these are trails (an ON with no matching OFF: a missed or delayed OFF event).
Labels carry the single-pass Gaussian ratio rho_G and, for physical trap conditions, the square-pulse calibration ratio rho_sq. ctrl-* rows are the Eq. (6) phenomenological output control, not a trap prediction.
Trackers: 'spec' is the task's rule (ON sets a pixel, OFF resets it). 'bipolar' (our addition, for video) lets the first event of a passage set the pixel and the opposite polarity clear it, so dark objects on a brighter background (which fire OFF first) are tracked too; it is identical to the spec rule for bright objects on dark.
