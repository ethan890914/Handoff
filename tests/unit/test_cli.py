"""Tests for config-backed cursor-mode selection."""

from handoff.cli import _default_gesture_mappings


def test_relative_mode_uses_the_default_mapping() -> None:
    assert _default_gesture_mappings("relative").name == "gesture_mappings.json"


def test_absolute_mode_uses_the_absolute_mapping() -> None:
    assert _default_gesture_mappings("absolute").name == "gesture_mappings_absolute.json"
