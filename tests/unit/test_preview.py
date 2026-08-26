"""Tests for the side-effect-free manual gesture preview loop."""

from io import StringIO

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
