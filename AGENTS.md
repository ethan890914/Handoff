# Project overview

A camera-based gesture control app: a webcam feed is processed to detect hand landmarks, landmarks are classified into gestures, and recognized gestures are dispatched as OS-level keyboard/mouse actions (zoom, app switching, mouse movement, scroll). Runs as a background process so it can control whatever application currently has focus.

Milestones: **Phase 1** — MediaPipe + rule-based gesture classification. Phase 2 (trained/finetuned gesture classifier) is planned but not yet implemented; the architecture below exists specifically to make that transition a swap, not a rewrite.

## Architecture — read this before changing anything

The app is a strict pipeline with fixed interfaces between stages. Each stage only knows about the interface of its neighbors, never their implementation:

```
Camera capture → Perception adapter → Gesture classifier → Debounce/state machine → Action mapping → Input dispatcher → OS
```

**The rule that matters most: never let a backend-specific detail leak past its own layer.** Concretely:
- MediaPipe-specific types/APIs must not appear outside the perception adapter.
- Whether the gesture classifier is rule-based or a trained model must not matter to the debounce/state machine, action mapping, or dispatcher.
- The specific input-simulation library (pyautogui/pynput/etc.) must not appear outside the input dispatcher.

If a change requires touching more than one layer to add a gesture, swap a model, or change an action mapping, that's a sign the interface boundary has been violated somewhere — fix the boundary, don't just patch around it.

### Interface contracts

- **Perception → Gesture:** a standardized landmark object — 21 hand landmarks, normalized (wrist-relative, scale-normalized), handedness, per-detection confidence, timestamp. This is the one schema every perception backend must produce.
- **Gesture → Control:** gesture label + confidence, computed from one frame (static gestures) or a short window of frames (dynamic gestures). Both the rule-based classifier and any future trained classifier implement this same interface.
- **Control → Action:** discrete gesture *events* (post-debounce), not raw per-frame labels.
- **Action → Dispatcher:** abstract action descriptions (press key, move mouse to, scroll by) — never calls into the input library directly from upstream code.

Gesture-to-action mappings live in external config (JSON/YAML), not in code.

## Repo conventions

- One module per pipeline stage; no stage imports another stage's internal implementation, only its public interface.
- New gesture types are added via config + a classifier update, not by touching the debounce, action-mapping, or dispatcher layers.
- Any new perception or classifier backend is added as a new implementation of the existing interface, registered via config — not by branching existing code with conditionals.
- Keep the rule-based classifier intact and selectable even after a trained classifier exists (config-selectable fallback), don't delete it when Phase 2 lands.

## Commit and PR guide

- PR title format: feature-name-with-hyphen
- Use concise imperative subjects (for example, `Add gesture event model`) and keep each commit focused. Pull requests should explain the behavior change, identify validation performed, link any issue, and include screenshots or a short recording for user-interface changes. Call out platform-specific requirements and permission changes explicitly.

## Dev environment tips
No build system, dependency manifest, or runnable development command is defined yet. Do not add undocumented setup assumptions. Once Python tooling is introduced, record the canonical environment setup and commands here (for example, `python -m venv .venv`, `pip install -r requirements.txt`, and `python -m handoff`).

- New behavior should include focused tests under `tests/`, named `test_<behavior>.py`, and should be runnable with the project’s documented test command. Keep gesture-recognition logic deterministic and isolate camera, input, and desktop side effects behind mocks or adapters.
- If you need to add new packages, ask the user for approval.


## What NOT to do

- Don't hardcode gesture→action mappings in code — they belong in config.
- Don't add a new perception or classifier backend by branching existing functions with if/else on backend type — implement the shared interface instead.
- Don't call the input-simulation library from outside the dispatcher module.
- Don't remove the rule-based classifier path when adding a trained model.