# videos/ — drop-in input folder for the CPD video demo

Put any clip here (`.mp4 .avi .mov .mkv .webm .ogv .m4v`) and run

```bash
cd src/cpd_trap
python run_video_folder.py          # all clips
python run_video_folder.py traffic  # only names containing "traffic"
```

Outputs go to `outputs/cpd_trap/video/<tag>/`: `snapshot_<tag>.png` (static
screenshot), `events_<tag>.gif` (event video), `rates_<tag>.png` and
`summary_<tag>.md`. Every clip is run with the symmetric baseline
(`A-reference`), two physical trap models (`B-trap-sq0.5`, `B-trap-sq0.2`) and
the imposed 0.2 control (`ctrl-0.2`).

An optional `<clip-name>.json` next to a clip sets crop / resolution / frame
time / truth file / `"stabilize": true` (removes global camera motion before
the sensor model, recommended for real footage); see the docstring of
`src/cpd_trap/run_video_folder.py`. Add `--replot` to `run_video.py` to redraw a
clip's figures from its cached arrays without re-simulating.

Included:

| file | what |
|---|---|
| `synthetic_traffic.mp4` + `_truth.csv` | our own synthetic clip (bright car, then dark car with a lane change) with known car centre per frame, so tracking error can be scored |
| `ayalon_freeway_480p.webm` | real freeway footage, **not committed** (third-party, CC BY-SA 3.0, see `ayalon_freeway_480p_SOURCE.txt`). Download: `curl -L -o ayalon_freeway_480p.webm "https://upload.wikimedia.org/wikipedia/commons/transcoded/7/74/Ayalon_Freeway_Azriely_Towers_View_P1150162.ogv/Ayalon_Freeway_Azriely_Towers_View_P1150162.ogv.480p.vp9.webm"` |
