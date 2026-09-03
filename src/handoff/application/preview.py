"""Safe manual gesture-recognition preview."""

from __future__ import annotations

import sys
from typing import TextIO

from handoff.domain.models import GestureLabel, GesturePrediction
from handoff.pipeline.interfaces import (
    CameraSource,
    FrameDisplay,
    GestureClassifier,
    GestureController,
    PerceptionBackend,
)


def run_gesture_preview(
    camera: CameraSource,
    perception: PerceptionBackend,
    classifier: GestureClassifier,
    *,
    output: TextIO = sys.stdout,
    max_frames: int | None = None,
    display: FrameDisplay | None = None,
    controller: GestureController | None = None,
) -> None:
    """Print and optionally display stable gesture changes without dispatching actions."""

    previous: GestureLabel | None = None
    stable_prediction = GesturePrediction(GestureLabel.UNKNOWN, 0.0)
    for frame_number, frame in enumerate(camera.frames(), start=1):
        frame_prediction = _classify_frame(frame, perception, classifier)
        prediction = frame_prediction
        if controller is not None:
            event = controller.update(prediction)
            if event is not None:
                stable_prediction = prediction
            prediction = stable_prediction

        overlay = _overlay(prediction, cursor_position=frame_prediction.cursor_position)
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


def _overlay(
    prediction: GesturePrediction,
    *,
    cursor_position: tuple[float, float] | None,
) -> str:
    overlay = f"Gesture: {prediction.label.value} ({prediction.confidence:.2f})"
    if cursor_position is not None:
        overlay += f"  Camera: ({cursor_position[0]:.2f}, {cursor_position[1]:.2f})"
    return overlay
