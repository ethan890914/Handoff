"""Stable data contracts shared by pipeline stages."""

from dataclasses import dataclass
from enum import StrEnum


class GestureLabel(StrEnum):
    """Labels emitted by a gesture classifier."""

    UNKNOWN = "unknown"
    PALM = "palm"
    FINGER = "finger"


@dataclass(frozen=True, slots=True)
class Landmark:
    """One normalized hand landmark."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class HandObservation:
    """Backend-neutral hand data from one frame."""

    landmarks: tuple[Landmark, ...]
    handedness: str
    confidence: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class GesturePrediction:
    """Classifier output for one frame or short frame window."""

    label: GestureLabel
    confidence: float


@dataclass(frozen=True, slots=True)
class GestureEvent:
    """A debounced gesture transition suitable for action mapping."""

    label: GestureLabel


class ActionType(StrEnum):
    """OS-independent actions supported by the dispatcher contract."""

    PRESS_KEY = "press_key"
    MOVE_MOUSE = "move_mouse"
    SCROLL = "scroll"


@dataclass(frozen=True, slots=True)
class Action:
    """An abstract action; OS adapters decide how to execute it."""

    type: ActionType
    value: str | float
