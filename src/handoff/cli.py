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
_DEFAULT_GESTURE_MAPPINGS = (
    Path(__file__).resolve().parents[2] / "config" / "gesture_mappings.json"
)


def main(argv: Sequence[str] | None = None) -> None:
    """Run a safe manual gesture preview when requested."""

    parser = argparse.ArgumentParser(prog="handoff")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-gestures",
        action="store_true",
        help="read the webcam and print basic gesture recognition results",
    )
    mode.add_argument(
        "--control",
        action="store_true",
        help="enable checkpoint-one mouse control; requires macOS Accessibility permission",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show the mirrored live gesture view while using --control",
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
    parser.add_argument(
        "--gesture-mappings",
        type=Path,
        default=_DEFAULT_GESTURE_MAPPINGS,
        help="JSON gesture-to-action mapping used by --control",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--scroll-repeat-interval",
        type=float,
        default=0.16,
        help="seconds between scroll actions while a two-finger direction pose is held",
    )
    parser.add_argument(
        "--scroll-stable-frames",
        type=int,
        default=4,
        help="consecutive two-finger scroll frames required before scrolling starts",
    )
    parser.add_argument(
        "--scroll-direction-ratio",
        type=float,
        default=1.7,
        help="minimum vertical-to-horizontal ratio required for a scroll pose",
    )
    parser.add_argument(
        "--pinch-distance",
        type=float,
        default=0.14,
        help="maximum normalized thumb-to-fingertip distance accepted as a click pinch",
    )
    parser.add_argument(
        "--click-hold-frames",
        type=int,
        default=4,
        help="consecutive tight-pinch frames required before a release emits a click",
    )
    parser.add_argument(
        "--click-max-motion",
        type=float,
        default=0.018,
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
    parser.add_argument(
        "--cursor-pixels-per-unit",
        type=float,
        default=1_250.0,
        help="screen pixels moved for one normalized cursor-delta unit with relative mappings",
    )
    parser.add_argument(
        "--cursor-active-region",
        type=float,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=(0.15, 0.15, 0.85, 0.85),
        help="camera region mapped to the screen in absolute mode, as normalized coordinates",
    )
    parser.add_argument(
        "--scroll-steps-per-unit",
        type=float,
        default=1.0,
        help="mouse-wheel steps for one configured scroll unit in --control mode",
    )
    args = parser.parse_args(argv)

    if not args.check_gestures and not args.control:
        print(
            f"Handoff {__version__}: use --check-gestures to preview gestures or "
            "--control to enable checkpoint-one mouse control"
        )
        return
    if args.preview and not args.control:
        parser.error("--preview can only be used with --control")

    if not args.model.is_file():
        parser.error(f"MediaPipe model not found: {args.model}")
    if not args.gesture_profile.is_file():
        parser.error(f"gesture profile not found: {args.gesture_profile}")
    if args.control and not args.gesture_mappings.is_file():
        parser.error(f"gesture mappings not found: {args.gesture_mappings}")
    if not 0 < args.landmark_smoothing <= 1:
        parser.error("landmark_smoothing must be between 0 and 1")
    if args.cursor_pixels_per_unit <= 0:
        parser.error("cursor_pixels_per_unit must be positive")
    if args.scroll_steps_per_unit <= 0:
        parser.error("scroll_steps_per_unit must be positive")
    cursor_active_region = (
        args.cursor_active_region[0],
        args.cursor_active_region[1],
        args.cursor_active_region[2],
        args.cursor_active_region[3],
    )

    from handoff.adapters.camera import OpenCVCamera
    from handoff.adapters.classifier import RuleBasedGestureClassifier
    from handoff.adapters.gesture_profile import JsonGestureProfile
    from handoff.adapters.perception import MediaPipePerception
    from handoff.application.controller import DebouncedGestureController

    try:
        gesture_profile = JsonGestureProfile(args.gesture_profile)
        classifier = RuleBasedGestureClassifier(
            pinch_distance=args.pinch_distance,
            click_hold_frames=args.click_hold_frames,
            click_max_motion=args.click_max_motion,
            scroll_direction_ratio=args.scroll_direction_ratio,
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
            scroll_stable_frames=args.scroll_stable_frames,
            zoom_repeat_interval_seconds=args.zoom_repeat_interval,
            zoom_stable_frames=args.zoom_stable_frames,
            enabled_labels=gesture_profile.enabled_labels,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.check_gestures:
        from handoff.application.preview import run_gesture_preview

        print("Starting gesture preview. Press Ctrl-C to stop.")
    else:
        from handoff.adapters.action_mapper import JsonActionMapper
        from handoff.adapters.dispatcher import PynputInputDispatcher
        from handoff.application.service import HandoffApplication

        try:
            action_mapper = JsonActionMapper(args.gesture_mappings)
            dispatcher = PynputInputDispatcher(
                cursor_pixels_per_unit=args.cursor_pixels_per_unit,
                scroll_steps_per_unit=args.scroll_steps_per_unit,
                cursor_active_region=cursor_active_region,
            )
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))

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
            if args.check_gestures:
                run_gesture_preview(
                    camera,
                    perception,
                    classifier,
                    display=camera,
                    controller=controller,
                )
            else:
                status = "with live preview" if args.preview else "without preview"
                print(
                    "Starting checkpoint-one mouse control "
                    f"{status}. Press Ctrl-C in this terminal to stop."
                )
                application = HandoffApplication(
                    camera=camera,
                    perception=perception,
                    classifier=classifier,
                    controller=controller,
                    action_mapper=action_mapper,
                    dispatcher=dispatcher,
                    display=camera if args.preview else None,
                )
                application.run()
        except KeyboardInterrupt:
            print("\nHandoff stopped.")
        finally:
            perception.close()
