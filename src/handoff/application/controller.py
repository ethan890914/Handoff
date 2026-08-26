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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        if reset_pause_seconds < 0:
            raise ValueError("reset_pause_seconds cannot be negative")
        self._stable_frames = stable_frames
        self._reset_pause_seconds = reset_pause_seconds
        self._clock = clock
        self._candidate: GestureLabel | None = None
        self._candidate_frames = 0
        self._active: GestureLabel | None = None
        self._paused_until = 0.0

    @property
    def tracking_paused(self) -> bool:
        """Whether cursor tracking should currently be suppressed."""

        return self._clock() < self._paused_until

    def update(self, prediction: GesturePrediction) -> GestureEvent | None:
        """Return a debounced transition, if one is ready."""

        now = self._clock()
        if now < self._paused_until:
            return None

        if prediction.label is not self._candidate:
            self._candidate = prediction.label
            self._candidate_frames = 1
        else:
            self._candidate_frames += 1

        if self._candidate_frames < self._stable_frames:
            return None
        if prediction.label is self._active:
            return None

        self._active = prediction.label
        if prediction.label is GestureLabel.RESET:
            self._paused_until = now + self._reset_pause_seconds
        return GestureEvent(prediction.label)
