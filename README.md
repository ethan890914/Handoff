# Handoff

Gesture-controlled computer interface.

## Development setup

The project requires Python 3.11 or newer. Create and activate a virtual environment,
then install the package with development tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Run the checks with:

```bash
pytest
ruff check .
mypy src
```

The project uses a `src/` layout. See [docs/architecture.md](docs/architecture.md) for
the layer boundaries and adapter pipeline. See [gesture-mapping.md](docs/gesture-mapping.md)
for the planned gesture/action reference. The project includes a MediaPipe Hand
Landmarker model under `models/hand_landmarker.task`. To manually check basic gesture
recognition, run:

```bash
python -m handoff --check-gestures
```

The command opens a mirrored live window with the stable gesture and confidence overlaid,
prints gesture transitions, and does not trigger keyboard or mouse actions. Its
default checkpoint profile activates only index-finger cursor tracking,
release-based left click, and two-finger vertical scrolling. The other
recognizers remain implemented but are inactive until selected through a
different `--gesture-profile` JSON file.
Transient one-frame labels during gesture changes are filtered by the preview
controller.
Press `q` or `Esc` to stop. Use `--model` to override the bundled model path.
Hold two fingers pointed up or down to scroll; releasing or changing the pose
stops it. `--scroll-repeat-interval` controls the scrolling rate, while
`--scroll-stable-frames` and `--scroll-direction-ratio` make scroll recognition
more selective. A click is a tight, still thumb–index pinch released after a short hold; tune it with
`--pinch-distance`, `--click-hold-frames`, and `--click-max-motion`. In a
profile that enables zoom, hold thumb and index extended with the other fingers
curled and a clearly separated gap. Spread the thumb and index apart to zoom in
or bring them together
to zoom out. Zoom waits for five stable frames and repeats
every 0.6 seconds by default; `--zoom-stable-frames`,
`--zoom-repeat-interval`, and `--zoom-distance-threshold` tune that behavior.
`--landmark-smoothing` filters hand-landmark jitter before gesture recognition.
The default mappings are in `config/gesture_mappings.json`.

## Checkpoint-one mouse control

After confirming recognition with the safe preview, enable cursor tracking, release-based
left click, and two-finger scrolling with:

```bash
python -m handoff --control
```

Add `--preview` to keep the mirrored camera window and its debounced gesture
status visible while actions are dispatched:

```bash
python -m handoff --control --preview
```

On macOS, grant the terminal application running Handoff permission in **System Settings
→ Privacy & Security → Accessibility** before using control mode. `--control` has no live
preview window unless `--preview` is provided; press `q` or `Esc` in that window, or `Ctrl-C`
in the launching terminal, to stop it. Tune pointer movement
with `--cursor-pixels-per-unit` and wheel intensity with `--scroll-steps-per-unit`. The
control pipeline activates only the labels in `config/checkpoint_one_gestures.json`, and its
action mappings remain editable in `config/gesture_mappings.json`.

## Cursor tracking calibration

Relative cursor tracking is the default. To use clutch-safe absolute tracking,
select it explicitly; the comfortable central camera region then maps across
the display, while starting an index-pointing pose anchors your current pointer
position so it does not jump. First run the safe preview and note the
`Camera: (x, y)` values while holding your pointing hand at its comfortable
left, top, right, and bottom limits. Provide those normalized camera bounds as
follows:

```bash
python -m handoff --check-gestures
```

```bash
python -m handoff --control --preview --cursor-mode absolute \
  --cursor-active-region 0.15 0.15 0.85 0.85
```

The values are `LEFT TOP RIGHT BOTTOM`, each between `0` and `1`; the default
central region above is a starting point. `--gesture-mappings` can still select
a custom external mapping and overrides `--cursor-mode`.
