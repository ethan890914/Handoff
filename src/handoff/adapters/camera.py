"""OpenCV camera adapter.

OpenCV-specific capture and color conversion stay in this module.  Downstream
perception code receives RGB image frames as opaque objects.
"""

from __future__ import annotations

from collections.abc import Iterator
from importlib import import_module
from typing import Any, cast


class OpenCVCamera:
    """Capture RGB frames from one webcam using OpenCV."""

    _WINDOW_NAME = "Handoff Gesture Preview"

    def __init__(
        self,
        device_index: int = 0,
        *,
        width: int | None = 640,
        height: int | None = 480,
        fps: int | None = 30,
        mirror: bool = True,
        cv2_module: Any | None = None,
    ) -> None:
        self._cv2: Any = cv2_module or self._load_opencv()
        self._mirror = mirror
        self._capture = self._cv2.VideoCapture(device_index)
        self._closed = False
        self._window_open = False

        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open camera device {device_index}")

        if width is not None:
            self._capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self._capture.set(self._cv2.CAP_PROP_FPS, fps)

    @staticmethod
    def _load_opencv() -> Any:
        try:
            cv2 = import_module("cv2")
        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required for OpenCVCamera; install the runtime dependencies"
            ) from error
        return cast(Any, cv2)

    def frames(self) -> Iterator[object]:
        """Yield RGB frames until the camera closes or a read fails."""

        if self._closed:
            raise RuntimeError("camera is closed")

        try:
            while self._capture.isOpened():
                ok, bgr_frame = self._capture.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the camera")
                rgb_frame = self._cv2.cvtColor(bgr_frame, self._cv2.COLOR_BGR2RGB)
                if self._mirror:
                    rgb_frame = self._cv2.flip(rgb_frame, 1)
                yield rgb_frame
        finally:
            self.close()

    def show(self, frame: object, overlay: str) -> bool:
        """Show an RGB frame with an overlay; return ``False`` on quit input."""

        if self._closed:
            raise RuntimeError("camera is closed")

        bgr_frame = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
        self._cv2.putText(
            bgr_frame,
            overlay,
            (10, 30),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            self._cv2.LINE_AA,
        )
        self._cv2.imshow(self._WINDOW_NAME, bgr_frame)
        self._window_open = True
        key = self._cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))

    def close(self) -> None:
        """Release the webcam if it is still open."""

        if not self._closed:
            self._capture.release()
            self._closed = True
        if self._window_open:
            self._cv2.destroyWindow(self._WINDOW_NAME)
            self._window_open = False

    def __enter__(self) -> OpenCVCamera:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
