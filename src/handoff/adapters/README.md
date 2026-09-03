# Adapter boundary

Adapter modules:

- `camera.py` — OpenCV webcam capture, RGB conversion, and lifecycle management.
- `perception.py` — MediaPipe Tasks hand landmarks translated into the domain
  `HandObservation` contract.
- `classifier.py` — deterministic rule-based recognition of `palm` and
  individual raised fingers; a trained classifier can implement the same
  interface later.
- `dispatcher.py` — the checkpoint-one Pynput mouse dispatcher. It supports
  relative pointer movement, left clicks, and vertical scrolling, and checks
  macOS Accessibility permission before enabling real control.

Adapters translate third-party data and errors into the contracts in
`handoff.pipeline.interfaces`. Do not expose backend-specific types upstream.

`MediaPipePerception` expects an RGB frame and a compatible Hand Landmarker task
model file. It uses MediaPipe's video-running mode and supplies monotonic
millisecond timestamps. The default model is committed at
`models/hand_landmarker.task`; runtime configuration can override its path.

`OpenCVCamera` yields RGB frames and releases the webcam when its frame iterator
ends or the camera context exits. The manual preview command uses this source
to open a labeled live preview window without dispatching any actions.
