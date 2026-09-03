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


def test_scroll_pose_repeats_at_a_controlled_rate() -> None:
    now = [0.0]
    controller = DebouncedGestureController(
        stable_frames=2,
        repeat_interval_seconds=0.2,
        clock=lambda: now[0],
    )
    scroll = GesturePrediction(GestureLabel.SCROLL_UP, 0.9)

    assert controller.update(scroll) is None
    event = controller.update(scroll)

    assert event is not None
    assert event.label is GestureLabel.SCROLL_UP

    now[0] = 0.19
    assert controller.update(scroll) is None

    now[0] = 0.2
    repeated_event = controller.update(scroll)

    assert repeated_event is not None
    assert repeated_event.label is GestureLabel.SCROLL_UP


def test_released_click_is_dispatched_without_another_debounce_delay() -> None:
    controller = DebouncedGestureController(stable_frames=3, clock=lambda: 0.0)
    click = GesturePrediction(GestureLabel.LEFT_CLICK, 0.9)

    event = controller.update(click)

    assert event is not None
    assert event.label is GestureLabel.LEFT_CLICK


def test_zoom_requires_a_longer_hold_and_repeats_more_slowly() -> None:
    now = [0.0]
    controller = DebouncedGestureController(
        stable_frames=2,
        repeat_interval_seconds=0.2,
        zoom_stable_frames=4,
        zoom_repeat_interval_seconds=0.6,
        clock=lambda: now[0],
    )
    zoom = GesturePrediction(GestureLabel.ZOOM_IN, 0.9)

    assert controller.update(zoom) is None
    assert controller.update(zoom) is None
    assert controller.update(zoom) is None
    event = controller.update(zoom)

    assert event is not None
    assert event.label is GestureLabel.ZOOM_IN

    now[0] = 0.59
    assert controller.update(zoom) is None

    now[0] = 0.6
    repeated_event = controller.update(zoom)

    assert repeated_event is not None
    assert repeated_event.label is GestureLabel.ZOOM_IN


def test_index_tracking_emits_relative_cursor_motion_and_supports_clutching() -> None:
    controller = DebouncedGestureController(stable_frames=1, cursor_dead_zone=0.01)
    index = lambda position: GesturePrediction(GestureLabel.INDEX, 0.9, position)

    assert controller.update(index((0.5, 0.5))) is not None
    event = controller.update(index((0.54, 0.52)))

    assert event is not None
    assert event.cursor_delta is not None
    assert abs(event.cursor_delta[0] - 0.04) < 0.000001
    assert abs(event.cursor_delta[1] - 0.02) < 0.000001

    controller.update(GesturePrediction(GestureLabel.UNKNOWN, 0.0))
    assert controller.update(index((0.1, 0.1))) is not None
    assert controller.update(index((0.1, 0.1))) is None


def test_disabled_gestures_do_not_emit_events() -> None:
    controller = DebouncedGestureController(
        stable_frames=1,
        enabled_labels=frozenset((GestureLabel.INDEX,)),
    )

    assert controller.update(GesturePrediction(GestureLabel.ZOOM_IN, 0.9)) is None
