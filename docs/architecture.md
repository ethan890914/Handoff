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
and timestamp in the domain contract.

`handoff.adapters.classifier.RuleBasedGestureClassifier` currently recognizes:

- `palm` — all five fingers extended;
- `thumb`, `index`, `middle`, `ring`, or `pinky` — one raised finger;
- `unknown` — anything else.

These are classifier labels only. No gesture-to-action mappings are included yet.
The manual preview displays the frame and classification result, then stops before
action mapping or dispatch; platform permissions, safety behavior, and desktop
control remain separate future stages.
