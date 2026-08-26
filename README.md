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
the layer boundaries and adapter pipeline. The project includes a MediaPipe Hand
Landmarker model under `models/hand_landmarker.task`. To manually check basic gesture
recognition, run:

```bash
python -m handoff --check-gestures
```

The command opens a live window with the current gesture and confidence overlaid,
prints transitions between `unknown`, `palm`, and individual finger labels, and
does not trigger keyboard or mouse actions. Press `q` or `Esc` to stop. Use
`--model` to override the bundled model path.
