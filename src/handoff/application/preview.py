"""Safe manual gesture-recognition preview."""

from __future__ import annotations

import sys
from typing import Protocol, TextIO

from handoff.domain.models import GestureLabel, GesturePrediction
from handoff.pipeline.interfaces import CameraSource, GestureClassifier, PerceptionBackend


class FrameDisplay(Protocol):
    """Display a frame and return whether the preview should continue."""

    def show(self, frame: object, overlay: str) -> bool:
        """Render one frame and return ``False`` when the user requests exit."""
        ...


def run_gesture_preview(
    camera: CameraSource,
    perception: PerceptionBackend,
    classifier: GestureClassifier,
    *,
    output: TextIO = sys.stdout,
    max_frames: int | None = None,
    display: FrameDisplay | None = None,
) -> None:
    """Print and optionally display gesture changes without dispatching actions."""

    previous: GestureLabel | None = None
    for frame_number, frame in enumerate(camera.frames(), start=1):
        prediction = _classify_frame(frame, perception, classifier)
        overlay = f"Gesture: {prediction.label.value} ({prediction.confidence:.2f})"
        if display is not None and not display.show(frame, overlay):
            break
        if prediction.label is not previous:
            print(
                f"gesture: {prediction.label.value} "
                f"confidence: {prediction.confidence:.2f}",
                file=output,
                flush=True,
            )
            previous = prediction.label

        if max_frames is not None and frame_number >= max_frames:
            break


def _classify_frame(
    frame: object,
    perception: PerceptionBackend,
    classifier: GestureClassifier,
) -> GesturePrediction:
    observation = perception.detect(frame)
    if observation is None:
        return GesturePrediction(GestureLabel.UNKNOWN, 0.0)
    return classifier.classify(observation)
