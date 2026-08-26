"""Deterministic tests for the OpenCV camera adapter."""

import pytest

from handoff.adapters.camera import OpenCVCamera


class _FakeCapture:
    def __init__(self, frames: list[object], *, opened: bool = True) -> None:
        self.frames = frames
        self.opened = opened
        self.read_fails = False
        self.index = 0
        self.released = False
        self.settings: list[tuple[object, object]] = []

    def isOpened(self) -> bool:
        return self.opened and not self.released and self.index < len(self.frames)

    def read(self) -> tuple[bool, object]:
        if self.read_fails:
            return False, None
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def set(self, property_id: object, value: object) -> bool:
        self.settings.append((property_id, value))
        return True

    def release(self) -> None:
        self.released = True


class _FakeCV2:
    CAP_PROP_FRAME_WIDTH = "width"
    CAP_PROP_FRAME_HEIGHT = "height"
    CAP_PROP_FPS = "fps"
    COLOR_BGR2RGB = "bgr2rgb"
    COLOR_RGB2BGR = "rgb2bgr"
    FONT_HERSHEY_SIMPLEX = "font"
    LINE_AA = "line"

    def __init__(self, capture: _FakeCapture) -> None:
        self.capture = capture
        self.overlays: list[tuple[object, ...]] = []
        self.windows: list[tuple[object, ...]] = []
        self.keys = [0]

    def VideoCapture(self, device_index: int) -> _FakeCapture:
        assert device_index == 2
        return self.capture

    @staticmethod
    def cvtColor(frame: object, conversion: object) -> tuple[str, object, object]:
        return ("rgb", frame, conversion)

    def putText(self, *args: object) -> None:
        self.overlays.append(args)

    def imshow(self, *args: object) -> None:
        self.windows.append(args)

    def waitKey(self, delay: int) -> int:
        assert delay == 1
        return self.keys.pop(0)

    def destroyWindow(self, name: object) -> None:
        self.windows.append(("destroy", name))


def test_yields_rgb_frames_and_releases_capture() -> None:
    capture = _FakeCapture(["frame-1", "frame-2"])
    camera = OpenCVCamera(
        device_index=2,
        width=800,
        height=600,
        fps=24,
        cv2_module=_FakeCV2(capture),
    )

    frames = list(camera.frames())

    assert frames == [("rgb", "frame-1", "bgr2rgb"), ("rgb", "frame-2", "bgr2rgb")]
    assert capture.settings == [("width", 800), ("height", 600), ("fps", 24)]
    assert capture.released


def test_raises_when_camera_cannot_be_opened() -> None:
    capture = _FakeCapture([], opened=False)

    with pytest.raises(RuntimeError, match="Could not open camera device 2"):
        OpenCVCamera(device_index=2, cv2_module=_FakeCV2(capture))

    assert capture.released


def test_raises_when_frame_read_fails() -> None:
    capture = _FakeCapture(["frame"])
    camera = OpenCVCamera(device_index=2, cv2_module=_FakeCV2(capture))
    capture.read_fails = True

    with pytest.raises(RuntimeError, match="Could not read a frame"):
        next(camera.frames())

    assert capture.released


def test_shows_rgb_frame_and_stops_on_q() -> None:
    capture = _FakeCapture(["frame"])
    cv2 = _FakeCV2(capture)
    cv2.keys = [ord("q")]
    camera = OpenCVCamera(device_index=2, cv2_module=cv2)

    should_continue = camera.show("rgb-frame", "Gesture: palm (0.90)")
    camera.close()

    assert not should_continue
    assert cv2.overlays[0][1] == "Gesture: palm (0.90)"
    assert cv2.windows[0][0] == "Handoff Gesture Preview"
    assert cv2.windows[-1] == ("destroy", "Handoff Gesture Preview")
