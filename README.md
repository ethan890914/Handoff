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
prints gesture transitions, and does not trigger keyboard or mouse actions. It
recognizes palms, individual fingers, thumb pinches for left/right click, a
closed-fist reset, two-finger vertical scrolling, and thumb-pointing navigation.
Transient one-frame labels during gesture changes are filtered by the preview
controller.
Press `q` or `Esc` to stop. Use `--model` to override the bundled model path.
The default mappings are in `config/gesture_mappings.json`.
