# Gesture mapping reference

This document is the implementation order and reference vocabulary for
gesture-to-action mappings. Mappings belong in
[`config/gesture_mappings.json`](../config/gesture_mappings.json), not in
classifier or dispatcher code.

## Functions ordered by simplicity

| Order | Function | Suggested gesture | Implementation notes |
| ---: | --- | --- | --- |
| 1 | Press a single key | One raised finger held briefly | Static label plus debounced transition. |
| 2 | Media controls | Dedicated raised-finger label | Uses a platform-specific media key in the dispatcher. |
| 3 | Modifier key | Hold a configured finger gesture | Needs key-down/key-up lifecycle handling. |
| 4 | Left click | Thumb–index pinch | Emits one click on a debounced pinch transition. |
| 5 | Right click | Thumb–middle-finger pinch | Same pinch detection with a different target finger. |
| 6 | Vertical scroll | Two fingers moving up/down | Requires dynamic movement tracking. |
| 7 | Horizontal navigation | Thumb pointing left/right | Suitable for browser or workspace navigation. |
| 8 | Zoom | Pinch distance changing | Requires continuous distance tracking. |
| 9 | Cursor movement | Index fingertip position | Uses continuous position, not just a gesture label. |
| 10 | Drag and drop | Held thumb–index pinch while moving | Combines pinch state with cursor movement. |
| 11 | App switching | Swipe or modifier plus gesture | Emulates Alt-Tab or Command-Tab. |
| 12 | Keyboard typing | Gesture sequence | Possible, but slower and less reliable than a keyboard. |

## Reset gesture

A closed fist is classified as `reset`. After a debounced reset event, the
configured actions move the cursor to the screen center and tracking is paused
for 1 second. During that pause the controller ignores predictions, preventing
the reset gesture from immediately moving the cursor again.

The current dynamic gesture thresholds use the original image-space midpoint of
the index and middle fingertips, smoothed across frames. A two-finger pose
(index and middle extended, ring and pinky curled) followed by dominant vertical
movement emits `scroll_up` or `scroll_down`. Horizontal navigation uses a static
thumb-pointing-left or thumb-pointing-right pose, avoiding lateral hand movement
being mistaken for navigation.

`palm` remains a neutral gesture by default and should not be mapped to an
action that can be triggered accidentally while the hand is relaxed.
