"""Tests for translation at the MediaPipe adapter boundary."""

from types import SimpleNamespace

import pytest

from handoff.adapters.perception import MediaPipePerception


class _FakeImage:
    def __init__(self, *, image_format: object, data: object) -> None:
        self.image_format = image_format
        self.data = data


class _FakeMediaPipe:
    Image = _FakeImage
    ImageFormat = SimpleNamespace(SRGB="srgb")


class _FakeLandmarker:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[object, int]] = []

    def detect_for_video(self, image: _FakeImage, *, timestamp_ms: int) -> object:
        self.calls.append((image, timestamp_ms))
        return self.result

    def close(self) -> None:
        pass


def test_translates_a_hand_to_wrist_relative_normalized_landmarks() -> None:
    raw = [SimpleNamespace(x=0.5, y=0.5, z=0.2) for _ in range(21)]
    raw[8] = SimpleNamespace(x=0.5, y=1.5, z=0.2)
    result = SimpleNamespace(
        hand_landmarks=[raw],
        handedness=[[SimpleNamespace(category_name="Left", score=0.87)]],
    )
    landmarker = _FakeLandmarker(result)
    perception = MediaPipePerception(
        landmarker=landmarker,
        mediapipe_module=_FakeMediaPipe,
        clock=iter((10.0,)).__next__,
    )

    observation = perception.detect(object())

    assert observation is not None
    assert len(observation.landmarks) == 21
    assert observation.landmarks[0].x == 0.0
    assert observation.landmarks[0].y == 0.0
    assert observation.landmarks[8].y == 1.0
    assert observation.handedness == "left"
    assert observation.confidence == 0.87
    assert observation.timestamp == 10.0
    assert observation.wrist_position == (0.5, 0.5)
    assert observation.motion_position == (0.5, 1.0)
    assert landmarker.calls[0][1] == 10_000


def test_returns_none_when_mediapipe_detects_no_hand() -> None:
    landmarker = _FakeLandmarker(SimpleNamespace(hand_landmarks=[], handedness=[]))
    perception = MediaPipePerception(
        landmarker=landmarker,
        mediapipe_module=_FakeMediaPipe,
        clock=lambda: 10.0,
    )

    assert perception.detect(object()) is None


def test_video_timestamps_are_strictly_increasing() -> None:
    result = SimpleNamespace(hand_landmarks=[], handedness=[])
    landmarker = _FakeLandmarker(result)
    perception = MediaPipePerception(
        landmarker=landmarker,
        mediapipe_module=_FakeMediaPipe,
        clock=iter((10.0, 10.0)).__next__,
    )

    perception.detect(object())
    perception.detect(object())

    assert [timestamp for _, timestamp in landmarker.calls] == [10_000, 10_001]


def test_requires_a_model_when_constructing_a_real_landmarker() -> None:
    with pytest.raises(ValueError, match="model_path is required"):
        MediaPipePerception(mediapipe_module=_FakeMediaPipe)
