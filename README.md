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
stops it. `--scroll-repeat-interval` controls the scrolling rate. A click is a
tight, still thumb–index pinch released after a short hold; tune it with
`--pinch-distance`, `--click-hold-frames`, and `--click-max-motion`. In a
profile that enables zoom, hold thumb and index extended with the other fingers
curled and a clearly separated gap. Spread the thumb and index apart to zoom in
or bring them together
to zoom out. Zoom waits for five stable frames and repeats
every 0.6 seconds by default; `--zoom-stable-frames`,
`--zoom-repeat-interval`, and `--zoom-distance-threshold` tune that behavior.
`--landmark-smoothing` filters hand-landmark jitter before gesture recognition.
The default mappings are in `config/gesture_mappings.json`.
