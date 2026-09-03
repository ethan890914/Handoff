"""Gesture debounce and safety state management."""

from __future__ import annotations

import time
from collections.abc import Callable

from handoff.domain.models import GestureEvent, GestureLabel, GesturePrediction


class DebouncedGestureController:
    """Emit stable gesture transitions and pause after the reset gesture."""

    def __init__(
        self,
        *,
        stable_frames: int = 2,
        reset_pause_seconds: float = 1.0,
        repeating_labels: frozenset[GestureLabel] | None = None,
        instant_labels: frozenset[GestureLabel] | None = None,
        repeat_interval_seconds: float = 0.2,
        zoom_repeat_interval_seconds: float = 0.6,
        right_click_stable_frames: int = 3,
        zoom_stable_frames: int = 5,
        enabled_labels: frozenset[GestureLabel] | None = None,
        cursor_dead_zone: float = 0.003,
        cursor_gain: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        if reset_pause_seconds < 0:
            raise ValueError("reset_pause_seconds cannot be negative")
        if repeat_interval_seconds <= 0:
            raise ValueError("repeat_interval_seconds must be positive")
        if zoom_repeat_interval_seconds <= 0:
            raise ValueError("zoom_repeat_interval_seconds must be positive")
        if right_click_stable_frames < 1:
            raise ValueError("right_click_stable_frames must be at least 1")
        if zoom_stable_frames < 1:
            raise ValueError("zoom_stable_frames must be at least 1")
        if cursor_dead_zone < 0:
            raise ValueError("cursor_dead_zone cannot be negative")
        if cursor_gain <= 0:
            raise ValueError("cursor_gain must be positive")
        self._stable_frames = stable_frames
        self._reset_pause_seconds = reset_pause_seconds
        self._repeating_labels = (
            frozenset(
                (
                    GestureLabel.SCROLL_UP,
                    GestureLabel.SCROLL_DOWN,
                    GestureLabel.ZOOM_IN,
                    GestureLabel.ZOOM_OUT,
                )
            )
            if repeating_labels is None
            else repeating_labels
        )
        self._instant_labels = (
            frozenset((GestureLabel.LEFT_CLICK,))
            if instant_labels is None
            else instant_labels
        )
        self._repeat_interval_seconds = repeat_interval_seconds
        self._zoom_repeat_interval_seconds = zoom_repeat_interval_seconds
        self._right_click_stable_frames = right_click_stable_frames
        self._zoom_stable_frames = zoom_stable_frames
        self._enabled_labels = enabled_labels
        self._cursor_dead_zone = cursor_dead_zone
        self._cursor_gain = cursor_gain
        self._clock = clock
        self._candidate: GestureLabel | None = None
        self._candidate_frames = 0
        self._active: GestureLabel | None = None
        self._paused_until = 0.0
        self._last_repeat_at: float | None = None
        self._last_cursor_position: tuple[float, float] | None = None

    @property
    def tracking_paused(self) -> bool:
        """Whether cursor tracking should currently be suppressed."""

        return self._clock() < self._paused_until

    def update(self, prediction: GesturePrediction) -> GestureEvent | None:
        """Return a debounced transition, if one is ready."""

        now = self._clock()
        if now < self._paused_until:
            return None

        if self._enabled_labels is not None and prediction.label not in self._enabled_labels:
            self._clear_active_state()
            return None

        if prediction.label in self._instant_labels:
            self._clear_active_state()
            return GestureEvent(prediction.label)

        if prediction.label is not self._candidate:
            self._candidate = prediction.label
            self._candidate_frames = 1
        else:
            self._candidate_frames += 1

        if self._candidate_frames < self._required_stable_frames(prediction.label):
            return None
        if prediction.label is self._active:
            if prediction.label is GestureLabel.INDEX:
                return self._cursor_event(prediction.cursor_position)
            if (
                prediction.label in self._repeating_labels
                and self._last_repeat_at is not None
                and now - self._last_repeat_at >= self._repeat_interval_for(prediction.label)
            ):
                self._last_repeat_at = now
                return GestureEvent(prediction.label)
            return None

        self._active = prediction.label
        self._last_repeat_at = now if prediction.label in self._repeating_labels else None
        if prediction.label is GestureLabel.INDEX:
            self._last_cursor_position = prediction.cursor_position
        if prediction.label is GestureLabel.RESET:
            self._paused_until = now + self._reset_pause_seconds
        return GestureEvent(prediction.label)

    def _cursor_event(self, position: tuple[float, float] | None) -> GestureEvent | None:
        if position is None:
            self._last_cursor_position = None
            return None
        if self._last_cursor_position is None:
            self._last_cursor_position = position
            return None
        previous_x, previous_y = self._last_cursor_position
        self._last_cursor_position = position
        delta_x = (position[0] - previous_x) * self._cursor_gain
        delta_y = (position[1] - previous_y) * self._cursor_gain
        if delta_x * delta_x + delta_y * delta_y <= self._cursor_dead_zone**2:
            return None
        return GestureEvent(GestureLabel.INDEX, cursor_delta=(delta_x, delta_y))

    def _clear_active_state(self) -> None:
        self._candidate = None
        self._candidate_frames = 0
        self._active = None
        self._last_repeat_at = None
        self._last_cursor_position = None

    def _required_stable_frames(self, label: GestureLabel) -> int:
        if label is GestureLabel.RIGHT_CLICK:
            return max(self._stable_frames, self._right_click_stable_frames)
        if label in (GestureLabel.ZOOM_IN, GestureLabel.ZOOM_OUT):
            return max(self._stable_frames, self._zoom_stable_frames)
        return self._stable_frames

    def _repeat_interval_for(self, label: GestureLabel) -> float:
        if label in (GestureLabel.ZOOM_IN, GestureLabel.ZOOM_OUT):
            return self._zoom_repeat_interval_seconds
        return self._repeat_interval_seconds
