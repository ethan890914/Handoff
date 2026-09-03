# Architecture

Handoff is organized as a ports-and-adapters pipeline:

```text
Camera → Perception → Classifier → Controller → Action mapper → Dispatcher → OS
```

## Layers

- `handoff.domain` contains stable, dependency-free contracts: landmarks, gesture
  predictions, debounced events, and abstract actions.
- `handoff.pipeline.interfaces` defines protocols for every stage. This allows a
  rule-based classifier to be replaced by a trained model without changing control or
  dispatch code.
- `handoff.application` owns pipeline orchestration and composition.
- `handoff.adapters` contains camera, MediaPipe/vision, classifier, and OS input
  integrations. Backend-specific types must stop at this boundary.
- `handoff.cli` is the process entry point for the safe manual gesture preview.

The intended flow is raw frames to normalized 21-point landmarks, then predictions,
debounced gesture events, configured actions, and finally OS-level input. Gesture-to-
action mappings belong in external configuration rather than hard-coded conditionals.

## Initial gesture recognition

`handoff.adapters.perception.MediaPipePerception` is the MediaPipe Tasks adapter.
It uses `HandLandmarker` in video mode, translates the first detected hand to
wrist-relative, scale-normalized landmarks, and preserves handedness, confidence,
timestamp, and image-space tracking positions in the domain contract.

`handoff.adapters.classifier.RuleBasedGestureClassifier` currently recognizes:

- `palm` — all five fingers extended;
- `thumb`, `index`, `middle`, `ring`, or `pinky` — one raised finger;
- `left_click` / `right_click` — thumb-index or thumb-middle pinch;
- `scroll_up` / `scroll_down` — two straight fingers held pointing up or down;
- `zoom_in` / `zoom_out` — a thumb–index span change while thumb and index are
  held extended;
- `navigate_left` / `navigate_right` — thumb pointing left or right;
- `reset` — closed fist;
- `unknown` — anything else.

These classifier labels are translated into abstract actions by the JSON action
mapper. In checkpoint-one control mode, `PynputInputDispatcher` is the only
module that imports `pynput`; it translates abstract calibrated mouse movement,
left clicks, and signed scrolling to macOS input events. Pynput and macOS APIs
do not appear in the classifier, controller, mapper, or application pipeline.
The manual preview mirrors the camera frame and displays the debounced
classification result, then stops before action mapping or dispatch; platform
permissions are checked only when the explicit `--control` mode is selected.

## Checkpoint one

`config/checkpoint_one_gestures.json` is the active gesture allowlist for the
preview: `index`, `left_click`, `scroll_up`, and `scroll_down`. Other labels and
their mappings remain available but are deliberately gated off. Index tracking
emits backend-neutral camera positions alongside cursor deltas; the default
mapping uses relative tracking, while `--cursor-mode absolute` selects a
calibrated mapping anchored on index-pose entry. Leaving the index pose resets
the anchor, providing a clutch for hand repositioning.
