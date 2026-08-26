"""Stable data contracts shared by pipeline stages."""

from dataclasses import dataclass
from enum import StrEnum


class GestureLabel(StrEnum):
    """Labels emitted by a gesture classifier."""

    UNKNOWN = "unknown"
    PALM = "palm"
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    NAVIGATE_LEFT = "navigate_left"
    NAVIGATE_RIGHT = "navigate_right"
    RESET = "reset"
    # Kept for compatibility with clients that consumed the original label.
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
    # Original normalized image coordinates retained for motion/cursor tracking.
    wrist_position: tuple[float, float] | None = None
    # Original midpoint of the index and middle fingertips for two-finger motion.
    motion_position: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class GesturePrediction:
    """Classifier output for one frame or short frame window.

    Confidence is a calibrated estimate from 0 to 1 for the selected label;
    it is not a raw MediaPipe gesture probability.
    """

    label: GestureLabel
    confidence: float


@dataclass(frozen=True, slots=True)
class GestureEvent:
    """A debounced gesture transition suitable for action mapping."""

    label: GestureLabel


class ActionType(StrEnum):
    """OS-independent actions supported by the dispatcher contract."""

    PRESS_KEY = "press_key"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    CLICK = "click"
    MOVE_MOUSE = "move_mouse"
    SCROLL = "scroll"
    PAUSE_TRACKING = "pause_tracking"


@dataclass(frozen=True, slots=True)
class Action:
    """An abstract action; OS adapters decide how to execute it."""

    type: ActionType
    value: str | float
