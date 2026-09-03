"""Application-level orchestration for live gesture control."""

from __future__ import annotations

from handoff.domain.models import GestureLabel, GesturePrediction
from handoff.pipeline.interfaces import (
    ActionMapper,
    CameraSource,
    FrameDisplay,
    GestureClassifier,
    GestureController,
    InputDispatcher,
    PerceptionBackend,
)


class HandoffApplication:
    """Run the dependency-injected pipeline without knowing adapter details."""

    def __init__(
        self,
        *,
        camera: CameraSource,
        perception: PerceptionBackend,
        classifier: GestureClassifier,
        controller: GestureController,
        action_mapper: ActionMapper,
        dispatcher: InputDispatcher,
        display: FrameDisplay | None = None,
    ) -> None:
        self._camera = camera
        self._perception = perception
        self._classifier = classifier
        self._controller = controller
        self._action_mapper = action_mapper
        self._dispatcher = dispatcher
        self._display = display

    def run(self, *, max_frames: int | None = None) -> None:
        """Process frames until the source ends or the optional limit is reached."""

        displayed_prediction = GesturePrediction(GestureLabel.UNKNOWN, 0.0)
        for frame_number, frame in enumerate(self._camera.frames(), start=1):
            observation = self._perception.detect(frame)
            prediction = (
                GesturePrediction(GestureLabel.UNKNOWN, 0.0)
                if observation is None
                else self._classifier.classify(observation)
            )
            event = self._controller.update(prediction)
            if event is not None:
                displayed_prediction = prediction
            if self._display is not None:
                overlay = (
                    f"Gesture: {displayed_prediction.label.value} "
                    f"({displayed_prediction.confidence:.2f})"
                )
                if prediction.cursor_position is not None:
                    overlay += (
                        f"  Camera: ({prediction.cursor_position[0]:.2f}, "
                        f"{prediction.cursor_position[1]:.2f})"
                    )
                if not self._display.show(frame, overlay):
                    break
            if event is not None:
                for action in self._action_mapper.map(event):
                    self._dispatcher.dispatch(action)
            if max_frames is not None and frame_number >= max_frames:
                break
