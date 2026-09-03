"""Tests for externally selected active gesture checkpoints."""

from handoff.adapters.gesture_profile import JsonGestureProfile
from handoff.domain.models import GestureLabel


def test_checkpoint_one_keeps_only_cursor_click_and_scroll_active() -> None:
    profile = JsonGestureProfile("config/checkpoint_one_gestures.json")

    assert profile.enabled_labels == frozenset(
        (
            GestureLabel.INDEX,
            GestureLabel.LEFT_CLICK,
            GestureLabel.SCROLL_UP,
            GestureLabel.SCROLL_DOWN,
        )
    )
