"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from handoff import __version__

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
_DEFAULT_GESTURE_PROFILE = (
    Path(__file__).resolve().parents[2] / "config" / "checkpoint_one_gestures.json"
)


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
    parser.add_argument(
        "--gesture-profile",
        type=Path,
        default=_DEFAULT_GESTURE_PROFILE,
        help="JSON allowlist of gestures active in this run",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--scroll-repeat-interval",
        type=float,
        default=0.2,
        help="seconds between scroll actions while a two-finger direction pose is held",
    )
    parser.add_argument(
        "--pinch-distance",
        type=float,
        default=0.16,
        help="maximum normalized thumb-to-fingertip distance accepted as a click pinch",
    )
    parser.add_argument(
        "--click-hold-frames",
        type=int,
        default=3,
        help="consecutive tight-pinch frames required before a release emits a click",
    )
    parser.add_argument(
        "--click-max-motion",
        type=float,
        default=0.025,
        help="maximum wrist movement allowed while holding a click pinch",
    )
    parser.add_argument(
        "--zoom-repeat-interval",
        type=float,
        default=0.6,
        help="seconds between zoom actions while a zoom pose is held",
    )
    parser.add_argument(
        "--zoom-stable-frames",
        type=int,
        default=5,
        help="consecutive zoom frames required before the first zoom action",
    )
    parser.add_argument(
        "--zoom-ready-distance",
        type=float,
        default=0.30,
        help="minimum separated thumb-index gap required before zoom can arm",
    )
    parser.add_argument(
        "--zoom-distance-threshold",
        type=float,
        default=0.08,
        help="thumb-index separation change required to enter zoom mode",
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.5,
        help="minimum hand-detection confidence accepted for gesture output",
    )
    parser.add_argument(
        "--landmark-smoothing",
        type=float,
        default=0.55,
        help="landmark filter weight from 0 (smoothest) to 1 (most responsive)",
    )
    args = parser.parse_args(argv)

    if not args.check_gestures:
        print(
            f"Handoff {__version__}: use --check-gestures --model /path/to/hand_landmarker.task "
            "to preview basic gestures"
        )
        return

    if not args.model.is_file():
        parser.error(f"MediaPipe model not found: {args.model}")
    if not args.gesture_profile.is_file():
        parser.error(f"gesture profile not found: {args.gesture_profile}")
    if not 0 < args.landmark_smoothing <= 1:
        parser.error("landmark_smoothing must be between 0 and 1")

    from handoff.adapters.camera import OpenCVCamera
    from handoff.adapters.classifier import RuleBasedGestureClassifier
    from handoff.adapters.gesture_profile import JsonGestureProfile
    from handoff.adapters.perception import MediaPipePerception
    from handoff.application.controller import DebouncedGestureController
    from handoff.application.preview import run_gesture_preview

    try:
        gesture_profile = JsonGestureProfile(args.gesture_profile)
        classifier = RuleBasedGestureClassifier(
            pinch_distance=args.pinch_distance,
            click_hold_frames=args.click_hold_frames,
            click_max_motion=args.click_max_motion,
            zoom_ready_distance=args.zoom_ready_distance,
            zoom_distance_threshold=args.zoom_distance_threshold,
            minimum_observation_confidence=args.minimum_confidence,
        )
    except ValueError as error:
        parser.error(str(error))

    try:
        controller = DebouncedGestureController(
            stable_frames=3,
            repeat_interval_seconds=args.scroll_repeat_interval,
            zoom_repeat_interval_seconds=args.zoom_repeat_interval,
            zoom_stable_frames=args.zoom_stable_frames,
            enabled_labels=gesture_profile.enabled_labels,
        )
    except ValueError as error:
        parser.error(str(error))

    print("Starting gesture preview. Press Ctrl-C to stop.")
    with OpenCVCamera(
        device_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    ) as camera:
        perception = MediaPipePerception(
            model_path=args.model,
            landmark_smoothing=args.landmark_smoothing,
        )
        try:
            run_gesture_preview(
                camera,
                perception,
                classifier,
                display=camera,
                controller=controller,
            )
        except KeyboardInterrupt:
            print("\nGesture preview stopped.")
        finally:
            perception.close()
