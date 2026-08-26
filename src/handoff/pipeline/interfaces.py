"""Fixed interfaces between capture, perception, control, and dispatch."""

from collections.abc import Iterator, Sequence
from typing import Protocol

from handoff.domain.models import (
    Action,
    GestureEvent,
    GesturePrediction,
    HandObservation,
)


class CameraSource(Protocol):
    """Provide raw camera frames."""

    def frames(self) -> Iterator[object]:
        """Yield frames until stopped."""
        ...


class PerceptionBackend(Protocol):
    """Convert a raw frame into normalized landmarks."""

    def detect(self, frame: object) -> HandObservation | None:
        """Return a hand observation, if one is present."""
        ...


class GestureClassifier(Protocol):
    """Classify a hand observation using rules or a trained model."""

    def classify(self, observation: HandObservation) -> GesturePrediction:
        """Return a backend-neutral prediction."""
        ...


class GestureController(Protocol):
    """Debounce predictions and emit discrete gesture events."""

    def update(self, prediction: GesturePrediction) -> GestureEvent | None:
        """Return an event only when the state machine accepts a transition."""
        ...


class ActionMapper(Protocol):
    """Map domain events using external configuration."""

    def map(self, event: GestureEvent) -> Sequence[Action]:
        """Return zero or more abstract actions."""
        ...


class InputDispatcher(Protocol):
    """Dispatch abstract actions through the host operating system."""

    def dispatch(self, action: Action) -> None:
        """Perform one action."""
        ...
