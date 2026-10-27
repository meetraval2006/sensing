"""
run_video_folder.py

Drop-in video workflow: put clips in event_camera_project/videos/ and run

    python run_video_folder.py            # every clip in videos/
    python run_video_folder.py traffic    # only clips whose name contains "traffic"
    python run_video_folder.py --replot   # redraw figures from cached arrays only

Each clip is run through run_video.py with the symmetric reference and the
asymmetric conditions (B-trap-sq0.5, B-trap-sq0.2, ctrl-0.2 by default).
Results land in outputs/cpd_trap/video/<tag>/: summary_<tag>.md,
snapshot_<tag>.png (static screenshot of the output), events_<tag>.gif and
rates_<tag>.png.

Optional sidecar config videos/<clip-name>.json, keys = run_video.py flags
with underscores, e.g.

    {"crop": [400, 190, 720, 370], "width": 160, "max_frames": 150, "stabilize": true}

or several runs of the same clip:

    {"runs": [{"tag": "freeway"},
              {"tag": "freeway_realtime", "frame_us": 33333, "max_frames": 30,
               "conditions": ["A-reference"]}]}

A "truth" path is resolved relative to videos/. Without a sidecar the clip
runs at width 128, up to 160 frames, 1 frame = 1000 us of CPD time.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import run_video

VIDEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "videos"))
EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".ogv", ".m4v")


def argv_for(path, cfg):
    argv = [path]
    for k, v in cfg.items():
        if k == "truth":
            v = os.path.join(VIDEO_DIR, v)
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            argv += [flag] if v else []
        elif isinstance(v, list):
            argv += [flag] + [str(x) for x in v]
        else:
            argv += [flag, str(v)]
    return argv


def main():
    args = sys.argv[1:]
    replot = "--replot" in args
    args = [a for a in args if a != "--replot"]
    pattern = args[0] if args else ""
    clips = sorted(f for f in os.listdir(VIDEO_DIR)
                   if f.lower().endswith(EXTS) and pattern in f)
    if not clips:
        raise SystemExit(f"no videos in {VIDEO_DIR} (extensions {', '.join(EXTS)})")
    for clip in clips:
        stem = os.path.splitext(clip)[0]
        cfg_path = os.path.join(VIDEO_DIR, stem + ".json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        runs = cfg.get("runs", [cfg])
        for run_cfg in runs:
            run_cfg = {"tag": stem, **run_cfg}
            print(f"\n=== {clip}  ->  {run_cfg['tag']} ===", flush=True)
            run_video.main(argv_for(os.path.join(VIDEO_DIR, clip), run_cfg)
                           + (["--replot"] if replot else []))


if __name__ == "__main__":
    main()
