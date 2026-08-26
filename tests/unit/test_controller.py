"""Tests for gesture debounce and reset safety behavior."""

from handoff.application.controller import DebouncedGestureController
from handoff.domain.models import GestureLabel, GesturePrediction


def test_emits_only_after_stable_frames() -> None:
    controller = DebouncedGestureController(stable_frames=2, clock=lambda: 0.0)
    palm = GesturePrediction(GestureLabel.PALM, 0.9)

    assert controller.update(palm) is None
    event = controller.update(palm)

    assert event is not None
    assert event.label is GestureLabel.PALM
    assert controller.update(palm) is None


def test_reset_pauses_tracking_for_one_second() -> None:
    now = [10.0]
    controller = DebouncedGestureController(
        stable_frames=1,
        reset_pause_seconds=1.0,
        clock=lambda: now[0],
    )

    event = controller.update(GesturePrediction(GestureLabel.RESET, 0.9))
    assert event is not None
    assert event.label is GestureLabel.RESET
    assert controller.tracking_paused

    now[0] = 10.5
    assert controller.update(GesturePrediction(GestureLabel.PALM, 0.9)) is None
    assert controller.tracking_paused

    now[0] = 11.1
    event = controller.update(GesturePrediction(GestureLabel.PALM, 0.9))
    assert event is not None
    assert event.label is GestureLabel.PALM
    assert not controller.tracking_paused
