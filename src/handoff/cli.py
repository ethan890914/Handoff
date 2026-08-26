"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from handoff import __version__

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"


def main(argv: Sequence[str] | None = None) -> None:
    """Run a safe manual gesture preview when requested."""

    parser = argparse.ArgumentParser(prog="handoff")
    parser.add_argument(
        "--check-gestures",
        action="store_true",
        help="read the webcam and print basic gesture recognition results",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_MODEL_PATH,
        help="path to a MediaPipe Hand Landmarker .task model",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    if not args.check_gestures:
        print(
            f"Handoff {__version__}: use --check-gestures --model /path/to/hand_landmarker.task "
            "to preview basic gestures"
        )
        return

    if not args.model.is_file():
        parser.error(f"MediaPipe model not found: {args.model}")

    from handoff.adapters.camera import OpenCVCamera
    from handoff.adapters.classifier import RuleBasedGestureClassifier
    from handoff.adapters.perception import MediaPipePerception
    from handoff.application.controller import DebouncedGestureController
    from handoff.application.preview import run_gesture_preview

    print("Starting gesture preview. Press Ctrl-C to stop.")
    with OpenCVCamera(
        device_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    ) as camera:
        perception = MediaPipePerception(model_path=args.model)
        try:
            run_gesture_preview(
                camera,
                perception,
                RuleBasedGestureClassifier(),
                display=camera,
                controller=DebouncedGestureController(stable_frames=3),
            )
        except KeyboardInterrupt:
            print("\nGesture preview stopped.")
        finally:
            perception.close()
