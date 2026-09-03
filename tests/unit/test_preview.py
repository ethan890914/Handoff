"""Tests for the side-effect-free manual gesture preview loop."""

from io import StringIO

from handoff.application.controller import DebouncedGestureController
from handoff.application.preview import run_gesture_preview
from handoff.domain.models import GestureLabel, GesturePrediction


class _Camera:
    def frames(self):
        yield from ("no-hand", "finger", "finger", "palm")


class _Perception:
    def detect(self, frame: object) -> object | None:
        return None if frame == "no-hand" else frame


class _Classifier:
    def classify(self, observation: object) -> GesturePrediction:
        label = GestureLabel.FINGER if observation == "finger" else GestureLabel.PALM
        return GesturePrediction(label, 0.8)


class _Display:
    def __init__(self) -> None:
        self.overlays: list[str] = []

    def show(self, frame: object, overlay: str) -> bool:
        self.overlays.append(overlay)
        return frame != "finger"


class _RecordingDisplay(_Display):
    def show(self, frame: object, overlay: str) -> bool:
        self.overlays.append(overlay)
        return True


class _TransitionCamera:
    def frames(self):
        yield from ("finger", "palm", "palm")


class _CursorClassifier:
    def classify(self, observation: object) -> GesturePrediction:
        return GesturePrediction(GestureLabel.INDEX, 0.8, cursor_position=(0.25, 0.75))


def test_prints_only_gesture_transitions() -> None:
    output = StringIO()

    run_gesture_preview(_Camera(), _Perception(), _Classifier(), output=output)

    assert output.getvalue().splitlines() == [
        "gesture: unknown confidence: 0.00",
        "gesture: finger confidence: 0.80",
        "gesture: palm confidence: 0.80",
    ]


def test_display_can_stop_the_preview() -> None:
    display = _Display()

    run_gesture_preview(_Camera(), _Perception(), _Classifier(), display=display)

    assert display.overlays == [
        "Gesture: unknown (0.00)",
        "Gesture: finger (0.80)",
    ]


def test_preview_filters_single_frame_transition_labels() -> None:
    display = _RecordingDisplay()

    run_gesture_preview(
        _TransitionCamera(),
        _Perception(),
        _Classifier(),
        display=display,
        controller=DebouncedGestureController(stable_frames=2, clock=lambda: 0.0),
    )

    assert display.overlays == [
        "Gesture: unknown (0.00)",
        "Gesture: unknown (0.00)",
        "Gesture: palm (0.80)",
    ]


def test_preview_shows_camera_coordinates_for_cursor_calibration() -> None:
    display = _RecordingDisplay()

    run_gesture_preview(
        _TransitionCamera(),
        _Perception(),
        _CursorClassifier(),
        display=display,
        max_frames=1,
    )

    assert display.overlays == ["Gesture: index (0.80)  Camera: (0.25, 0.75)"]
