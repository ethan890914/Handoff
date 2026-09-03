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
| 4 | Left click | Tight, still thumb–index pinch then release | Emits one click on release after a short held candidate. |
| 5 | Right click | Thumb–middle-finger pinch | Same pinch detection with a different target finger. |
| 6 | Vertical scroll | Two fingers moving up/down | Requires dynamic movement tracking. |
| 7 | Horizontal navigation | Thumb pointing left/right | Suitable for browser or workspace navigation. |
| 8 | Zoom | Thumb and index extended; change thumb–index distance | Enters a separate mode from click and scroll; repeats slowly while held. |
| 9 | Cursor movement | Index fingertip position | Uses continuous position, not just a gesture label. |
| 10 | Drag and drop | Held thumb–index pinch while moving | Combines pinch state with cursor movement. |
| 11 | App switching | Swipe or modifier plus gesture | Emulates Alt-Tab or Command-Tab. |
| 12 | Keyboard typing | Gesture sequence | Possible, but slower and less reliable than a keyboard. |

## Reset gesture

A closed fist is classified as `reset`. After a debounced reset event, the
configured actions move the cursor to the screen center and tracking is paused
for 1 second. During that pause the controller ignores predictions, preventing
the reset gesture from immediately moving the cursor again.

The current scroll gesture is static: hold the index and middle fingers straight
and pointing up for `scroll_up`, or down for `scroll_down`. The controller emits
the configured scroll action at a bounded repeat interval while that stable pose
is held, and stops as soon as the pose changes. Horizontal navigation uses a
static thumb-pointing-left or thumb-pointing-right pose, avoiding lateral hand
movement being mistaken for navigation.

## Zoom gesture

Hold the thumb and index straight with the other fingers curled. Begin with a
clearly separated thumb–index gap, then spread them apart to enter `zoom_in`,
or bring them together to enter `zoom_out`. A tight thumb–index contact is
reserved for click, which is emitted only when a short, still pinch is released.
Once zoom activates, click recognition stays locked until a neutral release, so
releasing a zoom gesture cannot create an accidental click. The classifier
re-arms when the span returns near its starting size. The controller deliberately
uses a longer stable hold and a longer repeat interval for zoom than for
scrolling, avoiding rapid page-scale changes from a brief or noisy pose. The
default action mappings use
`ctrl+plus` and `ctrl+minus`; platform-specific deployments can replace those
strings in the JSON mapping without changing gesture logic.

`palm` remains a neutral gesture by default and should not be mapped to an
action that can be triggered accidentally while the hand is relaxed.
