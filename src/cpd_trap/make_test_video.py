"""
make_test_video.py

Writes a synthetic, traffic-like grayscale clip with a KNOWN object
trajectory, so the video pipeline (run_video.py) can be scored the same way
as the Gaussian-spot study before it is pointed at real footage.

Scene: textured grey road with dashed lane markings; one car at a time
(a bright car, then a dark car, so both "brighter-than-background" and
"darker-than-background" objects are exercised), each with soft edges.
The second car changes lanes partway through.

Outputs (default videos/, the drop-in input folder):
  synthetic_traffic.mp4        the clip (8-bit grayscale stored as BGR)
  synthetic_traffic_truth.csv  frame, x, y of the active car centre (px),
                               object (0 = bright car, 1 = dark car)
"""

import argparse
import os
import numpy as np
import cv2

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "videos")


def render(W=128, H=72, n_frames=160, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    road = 0.42 + 0.04 * rng.standard_normal((H, W))
    road = cv2.GaussianBlur(road, (0, 0), 1.0)
    road[(yy > H / 2 - 1) & (yy < H / 2 + 1) & ((xx // 8) % 2 == 0)] = 0.85
    road[(yy < 6) | (yy > H - 7)] = 0.25

    def car(cx, cy, L=16, Wd=9, level=0.95):
        m = (1 / (1 + np.exp(-(Wd / 2 - np.abs(yy - cy)) / 0.7))) * \
            (1 / (1 + np.exp(-(L / 2 - np.abs(xx - cx)) / 0.7)))
        return m, level

    frames, truth = [], []
    half = n_frames // 2
    for f in range(n_frames):
        img = road.copy()
        if f < half:                      # bright car, lane 1, left -> right
            cx = -12 + (W + 24) * f / (half - 1)
            cy = H * 0.32
            m, lvl = car(cx, cy)
        else:                             # dark car, right -> left, lane change
            g = (f - half) / (n_frames - half - 1)
            cx = W + 12 - (W + 24) * g
            cy = H * 0.68 - (H * 0.36) * np.clip((g - 0.4) / 0.25, 0, 1)
            m, lvl = car(cx, cy, level=0.08)
        img = img * (1 - m) + lvl * m
        frames.append(np.clip(img, 0, 1))
        truth.append((f, cx, cy, 0 if f < half else 1))
    return frames, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--frames", type=int, default=160)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frames, truth = render(n_frames=args.frames)
    H, W = frames[0].shape
    path = os.path.join(args.out, "synthetic_traffic.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for fr in frames:
        g = (fr * 255).astype(np.uint8)
        vw.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    vw.release()
    np.savetxt(os.path.join(args.out, "synthetic_traffic_truth.csv"), np.array(truth),
               delimiter=",", header="frame,x,y,object", comments="", fmt=["%d", "%.3f", "%.3f", "%d"])
    print(f"wrote {path} ({len(frames)} frames, {W}x{H})")


if __name__ == "__main__":
    main()
