"""MediaPipe hand-landmark adapter.

MediaPipe types are intentionally confined to this module.  The rest of the
application receives only :class:`~handoff.domain.models.HandObservation`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Any, cast

from handoff.domain.models import HandObservation, Landmark

_LANDMARK_COUNT = 21


class MediaPipePerception:
    """Convert MediaPipe Hand Landmarker results to domain observations.

    ``frame`` must be an RGB image accepted by ``mediapipe.Image`` (normally a
    NumPy array).  A landmarker can be injected for tests or for application
    composition.  When it is not injected, ``model_path`` must point to a
    compatible MediaPipe hand-landmarker task model.
    """

    def __init__(
        self,
        model_path: str | PathLike[str] | None = None,
        *,
        landmarker: Any | None = None,
        mediapipe_module: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mp: Any = mediapipe_module
        self._clock = clock
        self._last_timestamp_ms = -1

        if landmarker is None:
            if model_path is None:
                raise ValueError("model_path is required when landmarker is not injected")
            self._mp = self._mp or self._load_mediapipe()
            landmarker = self._create_landmarker(Path(model_path))
        elif self._mp is None:
            self._mp = self._load_mediapipe()

        self._landmarker = landmarker

    @staticmethod
    def _load_mediapipe() -> Any:
        try:
            mp = import_module("mediapipe")
        except ImportError as error:
            raise RuntimeError(
                "MediaPipe is required for MediaPipePerception; install the runtime dependency"
            ) from error
        return cast(Any, mp)

    def _create_landmarker(self, model_path: Path) -> Any:
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe hand-landmarker model not found: {model_path}")

        base_options = self._mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = self._mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self._mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
        )
        return self._mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: object) -> HandObservation | None:
        """Detect the first hand in an RGB frame, if one is present."""

        timestamp = self._next_timestamp()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms=timestamp)
        return self._to_observation(result, timestamp_ms=timestamp)

    def close(self) -> None:
        """Release MediaPipe resources."""

        self._landmarker.close()

    def _next_timestamp(self) -> int:
        timestamp_ms = int(self._clock() * 1000)
        self._last_timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    @staticmethod
    def _to_observation(result: Any, *, timestamp_ms: int) -> HandObservation | None:
        hands = getattr(result, "hand_landmarks", ())
        if not hands:
            return None

        raw_landmarks = tuple(hands[0])
        if len(raw_landmarks) != _LANDMARK_COUNT:
            raise ValueError(
                f"MediaPipe returned {len(raw_landmarks)} landmarks; "
                f"expected {_LANDMARK_COUNT}"
            )

        landmarks = MediaPipePerception._normalize_landmarks(raw_landmarks)
        handedness, confidence = MediaPipePerception._read_handedness(result)
        return HandObservation(
            landmarks=landmarks,
            handedness=handedness,
            confidence=confidence,
            timestamp=timestamp_ms / 1000,
        )

    @staticmethod
    def _normalize_landmarks(raw_landmarks: tuple[Any, ...]) -> tuple[Landmark, ...]:
        wrist = raw_landmarks[0]
        wrist_x = float(wrist.x)
        wrist_y = float(wrist.y)
        wrist_z = float(wrist.z)
        translated = tuple(
            (
                float(point.x) - wrist_x,
                float(point.y) - wrist_y,
                float(point.z) - wrist_z,
            )
            for point in raw_landmarks
        )
        scale = max(math.sqrt(x * x + y * y + z * z) for x, y, z in translated)
        if math.isclose(scale, 0.0):
            raise ValueError("MediaPipe returned a degenerate hand with no measurable scale")

        return tuple(Landmark(x / scale, y / scale, z / scale) for x, y, z in translated)

    @staticmethod
    def _read_handedness(result: Any) -> tuple[str, float]:
        handedness = getattr(result, "handedness", ())
        if not handedness or not handedness[0]:
            return "unknown", 0.0

        category = handedness[0][0]
        name = getattr(category, "category_name", None) or getattr(
            category, "display_name", None
        )
        score = float(getattr(category, "score", 0.0))
        return str(name or "unknown").lower(), max(0.0, min(1.0, score))
