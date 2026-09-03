"""Pynput-backed mouse dispatcher for the checkpoint-one actions.

The rest of the application deals only in :class:`handoff.domain.models.Action`.
Pynput is deliberately confined to this module so it can be replaced without
changing gesture recognition, control, or mapping code.
"""

from __future__ import annotations

import sys
from typing import Protocol

from handoff.domain.models import Action, ActionType


class MouseDriver(Protocol):
    """Small mouse-output boundary that makes dispatcher tests side-effect free."""

    def move_by(self, horizontal: int, vertical: int) -> None:
        """Move the pointer by a relative number of screen pixels."""
        ...

    def move_to(self, horizontal: int, vertical: int) -> None:
        """Move the pointer to an absolute screen position in pixels."""
        ...

    def current_position(self) -> tuple[int, int]:
        """Return the current absolute pointer position in screen pixels."""
        ...

    def screen_size(self) -> tuple[int, int]:
        """Return the main display dimensions in screen pixels."""
        ...

    def click_left(self) -> None:
        """Click the primary mouse button once."""
        ...

    def scroll_vertical(self, steps: int) -> None:
        """Scroll by signed vertical wheel steps."""
        ...


class PynputMouseDriver:
    """Translate the small mouse-output boundary to pynput calls."""

    def __init__(self) -> None:
        _require_macos_accessibility_permission()
        try:
            from pynput import mouse  # type: ignore[import-untyped]
            from Quartz import CGDisplayBounds, CGMainDisplayID  # type: ignore[import-untyped]
        except ImportError as error:  # pragma: no cover - installation failure path
            raise RuntimeError(
                "Mouse control requires the optional 'pynput' dependency. "
                "Install Handoff with its runtime dependencies."
            ) from error
        self._controller = mouse.Controller()
        self._left_button = mouse.Button.left
        self._display_bounds = CGDisplayBounds(CGMainDisplayID())

    def move_by(self, horizontal: int, vertical: int) -> None:
        self._controller.move(horizontal, vertical)

    def move_to(self, horizontal: int, vertical: int) -> None:
        self._controller.position = (horizontal, vertical)

    def current_position(self) -> tuple[int, int]:
        horizontal, vertical = self._controller.position
        return int(horizontal), int(vertical)

    def screen_size(self) -> tuple[int, int]:
        return round(self._display_bounds.size.width), round(self._display_bounds.size.height)

    def click_left(self) -> None:
        self._controller.click(self._left_button)

    def scroll_vertical(self, steps: int) -> None:
        self._controller.scroll(0, steps)


class PynputInputDispatcher:
    """Dispatch checkpoint-one abstract actions through a mouse driver.

    Camera positions map through a calibrated active region to the main display;
    the dispatcher anchors that mapping on each new index-tracking session.
    Relative cursor deltas remain available through the fallback mapping, with
    their scale controlled by ``cursor_pixels_per_unit``. Neither detail leaks
    into the controller or JSON mapping.
    """

    def __init__(
        self,
        *,
        cursor_pixels_per_unit: float = 1_250.0,
        scroll_steps_per_unit: float = 1.0,
        cursor_active_region: tuple[float, float, float, float] = (0.15, 0.15, 0.85, 0.85),
        driver: MouseDriver | None = None,
    ) -> None:
        if cursor_pixels_per_unit <= 0:
            raise ValueError("cursor_pixels_per_unit must be positive")
        if scroll_steps_per_unit <= 0:
            raise ValueError("scroll_steps_per_unit must be positive")
        _validate_active_region(cursor_active_region)
        self._cursor_pixels_per_unit = cursor_pixels_per_unit
        self._scroll_steps_per_unit = scroll_steps_per_unit
        self._cursor_active_region = cursor_active_region
        self._driver = PynputMouseDriver() if driver is None else driver
        self._cursor_anchor: tuple[int, int] | None = None

    def dispatch(self, action: Action) -> None:
        """Perform one supported action or reject an invalid checkpoint mapping."""

        if action.type is ActionType.MOVE_MOUSE_BY:
            horizontal, vertical = _cursor_delta(action.value)
            pixels_x = round(horizontal * self._cursor_pixels_per_unit)
            pixels_y = round(vertical * self._cursor_pixels_per_unit)
            if pixels_x or pixels_y:
                self._driver.move_by(pixels_x, pixels_y)
            return

        if action.type is ActionType.BEGIN_MOUSE_TRACKING:
            target_x, target_y = self._camera_position_to_screen(_cursor_position(action.value))
            current_x, current_y = self._driver.current_position()
            self._cursor_anchor = (current_x - target_x, current_y - target_y)
            return

        if action.type is ActionType.MOVE_MOUSE:
            if action.value == "center":
                width, height = self._driver.screen_size()
                self._driver.move_to(width // 2, height // 2)
                return
            target_x, target_y = self._camera_position_to_screen(_cursor_position(action.value))
            if self._cursor_anchor is None:
                current_x, current_y = self._driver.current_position()
                self._cursor_anchor = (current_x - target_x, current_y - target_y)
            anchor_x, anchor_y = self._cursor_anchor
            width, height = self._driver.screen_size()
            self._driver.move_to(
                _clamp(target_x + anchor_x, 0, width - 1),
                _clamp(target_y + anchor_y, 0, height - 1),
            )
            return

        if action.type is ActionType.CLICK:
            if action.value != "left":
                raise ValueError("checkpoint dispatcher supports only a left click")
            self._driver.click_left()
            return

        if action.type is ActionType.SCROLL:
            if isinstance(action.value, bool) or not isinstance(action.value, (int, float)):
                raise ValueError("scroll action value must be a number")
            steps = round(action.value * self._scroll_steps_per_unit)
            if steps:
                self._driver.scroll_vertical(steps)
            return

        raise ValueError(f"unsupported checkpoint action: {action.type.value}")

    def _camera_position_to_screen(self, position: tuple[float, float]) -> tuple[int, int]:
        left, top, right, bottom = self._cursor_active_region
        normalized_x = _clamp_float((position[0] - left) / (right - left), 0.0, 1.0)
        normalized_y = _clamp_float((position[1] - top) / (bottom - top), 0.0, 1.0)
        width, height = self._driver.screen_size()
        return round(normalized_x * (width - 1)), round(normalized_y * (height - 1))


def _cursor_delta(value: str | float | tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("move_mouse_by action value must be a two-item cursor delta")
    horizontal, vertical = value
    if (
        isinstance(horizontal, bool)
        or isinstance(vertical, bool)
        or not isinstance(horizontal, (int, float))
        or not isinstance(vertical, (int, float))
    ):
        raise ValueError("cursor delta coordinates must be numbers")
    return float(horizontal), float(vertical)


def _cursor_position(value: str | float | tuple[float, float]) -> tuple[float, float]:
    return _cursor_delta(value)


def _validate_active_region(region: tuple[float, float, float, float]) -> None:
    left, top, right, bottom = region
    if not all(0 <= coordinate <= 1 for coordinate in region):
        raise ValueError("cursor_active_region coordinates must be between 0 and 1")
    if left >= right or top >= bottom:
        raise ValueError("cursor_active_region must have left < right and top < bottom")


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _require_macos_accessibility_permission() -> None:
    """Fail before opening the camera when macOS would reject mouse output."""

    if sys.platform != "darwin":
        return
    from ApplicationServices import AXIsProcessTrusted  # type: ignore[import-untyped]

    if not AXIsProcessTrusted():
        raise RuntimeError(
            "macOS Accessibility permission is required for mouse control. "
            "Enable it for the terminal running Handoff in System Settings > "
            "Privacy & Security > Accessibility, then run --control again."
        )
