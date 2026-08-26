# Adapter boundary

Planned adapter modules:

- `camera.py` — webcam capture and lifecycle management.
- `perception.py` — MediaPipe or another landmark backend.
- `classifier.py` — rule-based classifier first; trained model later.
- `dispatcher.py` — keyboard/mouse/accessibility integration.

Adapters translate third-party data and errors into the contracts in
`handoff.pipeline.interfaces`. Do not expose backend-specific types upstream.
