PYTHON ?= python3

.PHONY: setup test lint typecheck run

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e '.[dev]'

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check .

typecheck:
	.venv/bin/python -m mypy src

run:
	.venv/bin/python -m handoff
