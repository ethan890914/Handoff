"""Tests for the isolated checkpoint-one mouse dispatcher."""

import pytest

from handoff.adapters.dispatcher import PynputInputDispatcher
from handoff.domain.models import Action, ActionType


class _MouseDriver:
    def __init__(self) -> None:
        self.moves: list[tuple[int, int]] = []
        self.absolute_moves: list[tuple[int, int]] = []
        self.clicks = 0
        self.scrolls: list[int] = []
        self.position = (500, 300)

    def move_by(self, horizontal: int, vertical: int) -> None:
        self.moves.append((horizontal, vertical))

    def move_to(self, horizontal: int, vertical: int) -> None:
        self.absolute_moves.append((horizontal, vertical))
        self.position = (horizontal, vertical)

    def current_position(self) -> tuple[int, int]:
        return self.position

    def screen_size(self) -> tuple[int, int]:
        return 1_000, 600

    def click_left(self) -> None:
        self.clicks += 1

    def scroll_vertical(self, steps: int) -> None:
        self.scrolls.append(steps)


def test_dispatches_cursor_delta_as_relative_pixel_movement() -> None:
    driver = _MouseDriver()
    dispatcher = PynputInputDispatcher(cursor_pixels_per_unit=500, driver=driver)

    dispatcher.dispatch(Action(ActionType.MOVE_MOUSE_BY, (0.02, -0.01)))

    assert driver.moves == [(10, -5)]


def test_dispatches_left_click() -> None:
    driver = _MouseDriver()
    dispatcher = PynputInputDispatcher(driver=driver)

    dispatcher.dispatch(Action(ActionType.CLICK, "left"))

    assert driver.clicks == 1


def test_dispatches_signed_scroll() -> None:
    driver = _MouseDriver()
    dispatcher = PynputInputDispatcher(scroll_steps_per_unit=2, driver=driver)

    dispatcher.dispatch(Action(ActionType.SCROLL, 1.0))
    dispatcher.dispatch(Action(ActionType.SCROLL, -1.0))

    assert driver.scrolls == [2, -2]


def test_absolute_tracking_anchors_without_a_cursor_jump_then_maps_camera_position() -> None:
    driver = _MouseDriver()
    dispatcher = PynputInputDispatcher(cursor_active_region=(0.0, 0.0, 1.0, 1.0), driver=driver)

    dispatcher.dispatch(Action(ActionType.BEGIN_MOUSE_TRACKING, (0.2, 0.3)))
    dispatcher.dispatch(Action(ActionType.MOVE_MOUSE, (0.2, 0.3)))
    dispatcher.dispatch(Action(ActionType.MOVE_MOUSE, (0.3, 0.3)))

    assert driver.absolute_moves == [(500, 300), (600, 300)]


def test_rejects_actions_outside_the_checkpoint_scope() -> None:
    dispatcher = PynputInputDispatcher(driver=_MouseDriver())

    with pytest.raises(ValueError, match="unsupported checkpoint action"):
        dispatcher.dispatch(Action(ActionType.PRESS_KEY, "space"))
