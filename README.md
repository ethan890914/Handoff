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

Run the scaffold checks with:

```bash
pytest
ruff check .
mypy src
```

The project uses a `src/` layout. See [docs/architecture.md](docs/architecture.md) for
the layer boundaries and planned adapter pipeline. Runtime behavior is not implemented
yet; `python -m handoff` reports the scaffold status.
