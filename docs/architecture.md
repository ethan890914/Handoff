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
- `handoff.cli` is the future process entry point.

The intended flow is raw frames to normalized 21-point landmarks, then predictions,
debounced gesture events, configured actions, and finally OS-level input. Gesture-to-
action mappings belong in external configuration rather than hard-coded conditionals.
The current entry point is deliberately a placeholder; platform permissions, safety
behavior, and concrete backends must be decided before enabling desktop control.
