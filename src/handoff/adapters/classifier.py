"""Backend-neutral rule-based gesture classifier."""

from __future__ import annotations

import math

from handoff.domain.models import GestureLabel, GesturePrediction, HandObservation, Landmark

_WRIST = 0
_FINGER_CHAINS = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
_NON_THUMB_FINGERS = ("index", "middle", "ring", "pinky")
_TWO_FINGER_FINGERS = ("index", "middle")


class RuleBasedGestureClassifier:
    """Recognize an open palm, single raised fingers, pinches, and reset.

    Extension is determined from the ratio between a fingertip's distance from
    the wrist and its PIP/IP joint's distance from the wrist.  This works on
    the adapter's wrist-relative, scale-normalized landmarks and does not
    depend on camera resolution or absolute hand position.  For non-thumb
    gestures, the thumb is allowed to be either curled or extended because
    index-pointing commonly includes an extended thumb.
    """

    def __init__(
        self,
        *,
        extended_ratio: float = 1.10,
        curled_ratio: float = 0.95,
        pinch_distance: float = 0.25,
        movement_threshold: float = 0.06,
        thumb_extension_distance: float = 0.25,
        motion_smoothing: float = 0.35,
    ) -> None:
        if curled_ratio >= extended_ratio:
            raise ValueError("curled_ratio must be smaller than extended_ratio")
        if pinch_distance <= 0:
            raise ValueError("pinch_distance must be positive")
        if movement_threshold <= 0:
            raise ValueError("movement_threshold must be positive")
        if thumb_extension_distance <= 0:
            raise ValueError("thumb_extension_distance must be positive")
        if not 0 < motion_smoothing <= 1:
            raise ValueError("motion_smoothing must be between 0 and 1")
        self._extended_ratio = extended_ratio
        self._curled_ratio = curled_ratio
        self._pinch_distance = pinch_distance
        self._movement_threshold = movement_threshold
        self._thumb_extension_distance = thumb_extension_distance
        self._motion_smoothing = motion_smoothing
        self._motion_origin: tuple[float, float] | None = None
        self._motion_position: tuple[float, float] | None = None
        self._motion_triggered = False
        self._motion_label: GestureLabel | None = None
        self._motion_confidence_value = 0.0
        self._motion_pose_lost = False

    def classify(self, observation: HandObservation) -> GesturePrediction:
        """Return a prediction for one hand observation."""

        if len(observation.landmarks) != 21:
            raise ValueError("gesture classification requires exactly 21 hand landmarks")

        states = {
            finger: self._state(observation.landmarks, chain)
            for finger, chain in _FINGER_CHAINS.items()
        }
        if states["thumb"][0] == "extended" and not self._thumb_is_extended(
            observation.landmarks
        ):
            states["thumb"] = ("curled", 0.0)
        if all(states[finger][0] == "extended" for finger in _FINGER_CHAINS):
            return GesturePrediction(
                GestureLabel.PALM,
                self._pattern_confidence(observation, states, tuple(_FINGER_CHAINS)),
            )

        pinch_candidates = self._pinch_candidates(observation.landmarks, states)
        if not any(state == "extended" for state, _ in states.values()) and not pinch_candidates:
            return GesturePrediction(
                GestureLabel.RESET,
                self._pattern_confidence(observation, states, tuple(_FINGER_CHAINS)),
            )

        if pinch_candidates:
            pinch_finger = min(
                pinch_candidates,
                key=lambda candidate: pinch_candidates[candidate],
            )
            pinch_label = {
                "index": GestureLabel.LEFT_CLICK,
                "middle": GestureLabel.RIGHT_CLICK,
            }[pinch_finger]
            return GesturePrediction(
                pinch_label,
                self._pinch_confidence(observation, pinch_finger),
            )

        motion_prediction = self._classify_motion(observation, states)
        if motion_prediction is not None:
            return motion_prediction

        non_thumb_extended = tuple(
            finger for finger in _NON_THUMB_FINGERS if states[finger][0] == "extended"
        )
        finger: str | None = None
        if len(non_thumb_extended) == 1:
            # An extended thumb is allowed alongside a pointing finger.
            finger = non_thumb_extended[0]
        elif not non_thumb_extended and states["thumb"][0] == "extended":
            direction = self._thumb_direction(observation.landmarks)
            if direction is not None:
                return GesturePrediction(
                    direction,
                    self._pattern_confidence(observation, states, ("thumb",)),
                )
            finger = "thumb"

        if finger is not None:
            required_curled = (
                _NON_THUMB_FINGERS
                if finger == "thumb"
                else tuple(other for other in _NON_THUMB_FINGERS if other != finger)
            )
            if all(states[other][0] == "curled" for other in required_curled):
                label = {
                    "thumb": GestureLabel.THUMB,
                    "index": GestureLabel.INDEX,
                    "middle": GestureLabel.MIDDLE,
                    "ring": GestureLabel.RING,
                    "pinky": GestureLabel.PINKY,
                }[finger]
                evidence_fingers = (finger, *required_curled)
                return GesturePrediction(
                    label,
                    self._pattern_confidence(observation, states, evidence_fingers),
                )

        return GesturePrediction(
            GestureLabel.UNKNOWN,
            self._unknown_confidence(observation, states),
        )

    def _pinch_candidates(
        self,
        landmarks: tuple[Landmark, ...],
        states: dict[str, tuple[str, float]],
    ) -> dict[str, float]:
        candidates = {}
        for finger in _TWO_FINGER_FINGERS:
            if (
                self._thumb_is_extended(landmarks)
                and states[finger][0] in ("curled", "unknown")
                and all(
                    states[other][0] in ("curled", "extended")
                    for other in self._pinch_allowed_other_fingers(finger)
                )
                and all(
                    states[other][0] == "curled"
                    for other in self._pinch_required_curled_fingers(finger)
                )
            ):
                distance = self._pinch_distance_between(landmarks, finger)
                if distance <= self._pinch_distance:
                    candidates[finger] = distance
        return candidates

    def _pinch_distance_between(self, landmarks: tuple[Landmark, ...], finger: str) -> float:
        tip = 8 if finger == "index" else 12
        return _distance(landmarks[4], landmarks[tip])

    def _pinch_confidence(self, observation: HandObservation, finger: str) -> float:
        distance = self._pinch_distance_between(observation.landmarks, finger)
        geometry = max(0.0, min(1.0, 1.0 - distance / self._pinch_distance))
        return _combined_confidence(geometry, observation.confidence)

    def _classify_motion(
        self,
        observation: HandObservation,
        states: dict[str, tuple[str, float]],
    ) -> GesturePrediction | None:
        two_finger_pose = (
            all(
                self._motion_finger_is_extended(observation.landmarks, finger)
                for finger in _TWO_FINGER_FINGERS
            )
            and all(states[finger][0] != "extended" for finger in ("ring", "pinky"))
        )
        position = observation.motion_position or observation.wrist_position
        if position is None:
            self._clear_motion_state()
            return None

        if not two_finger_pose:
            if self._motion_origin is None or self._motion_pose_lost:
                self._clear_motion_state()
                return None
            # Keep one frame of trajectory state when a moving fingertip is
            # briefly occluded or its extension ratio becomes uncertain.
            self._motion_pose_lost = True
        else:
            self._motion_pose_lost = False

        if self._motion_origin is None:
            self._motion_origin = position
            self._motion_position = position
            self._motion_triggered = False
            self._motion_label = None
            return None
        if self._motion_triggered and self._motion_label is not None:
            return GesturePrediction(self._motion_label, self._motion_confidence_value)

        previous_x, previous_y = self._motion_position or self._motion_origin
        current_x, current_y = position
        smoothing = self._motion_smoothing
        self._motion_position = (
            previous_x + smoothing * (current_x - previous_x),
            previous_y + smoothing * (current_y - previous_y),
        )
        origin_x, origin_y = self._motion_origin
        delta_x = self._motion_position[0] - origin_x
        delta_y = self._motion_position[1] - origin_y
        if abs(delta_y) >= self._movement_threshold and abs(delta_y) >= abs(delta_x) * 1.5:
            self._motion_triggered = True
            label = GestureLabel.SCROLL_DOWN if delta_y > 0 else GestureLabel.SCROLL_UP
            confidence = self._motion_confidence(observation, abs(delta_y))
            self._motion_label = label
            self._motion_confidence_value = confidence
            return GesturePrediction(label, confidence)
        return None

    @staticmethod
    def _pinch_allowed_other_fingers(finger: str) -> tuple[str, ...]:
        if finger == "middle":
            # A thumb-middle pinch commonly leaves the index finger raised.
            return ("index",)
        return ()

    @staticmethod
    def _pinch_required_curled_fingers(finger: str) -> tuple[str, ...]:
        allowed = RuleBasedGestureClassifier._pinch_allowed_other_fingers(finger)
        return tuple(
            other
            for other in _NON_THUMB_FINGERS
            if other != finger and other not in allowed
        )

    def _thumb_is_extended(self, landmarks: tuple[Landmark, ...]) -> bool:
        first_segment = _distance(landmarks[2], landmarks[3])
        second_segment = _distance(landmarks[3], landmarks[4])
        chain_length = first_segment + second_segment
        if chain_length == 0:
            return False
        direct_span = _distance(landmarks[2], landmarks[4])
        straightness = direct_span / chain_length
        return (
            direct_span >= self._thumb_extension_distance
            and straightness >= 0.45
        )

    def _thumb_direction(self, landmarks: tuple[Landmark, ...]) -> GestureLabel | None:
        if not self._thumb_is_extended(landmarks):
            return None
        delta_x = landmarks[4].x - landmarks[2].x
        delta_y = landmarks[4].y - landmarks[2].y
        if abs(delta_x) < abs(delta_y) * 1.2:
            return None
        return GestureLabel.NAVIGATE_RIGHT if delta_x > 0 else GestureLabel.NAVIGATE_LEFT

    def _clear_motion_state(self) -> None:
        self._motion_origin = None
        self._motion_position = None
        self._motion_triggered = False
        self._motion_label = None
        self._motion_confidence_value = 0.0
        self._motion_pose_lost = False

    @staticmethod
    def _motion_finger_is_extended(
        landmarks: tuple[Landmark, ...], finger: str
    ) -> bool:
        _, pip_or_ip, tip = _FINGER_CHAINS[finger]
        pip_distance = _distance(landmarks[_WRIST], landmarks[pip_or_ip])
        if pip_distance == 0:
            return False
        # Motion gestures use a slightly more permissive gate than static
        # labels because a moving fingertip is often foreshortened.
        return _distance(landmarks[_WRIST], landmarks[tip]) / pip_distance >= 0.9

    def _motion_confidence(self, observation: HandObservation, distance: float) -> float:
        geometry = min(1.0, 0.5 + (distance - self._movement_threshold) / self._movement_threshold)
        return _combined_confidence(geometry, observation.confidence)

    def _state(
        self, landmarks: tuple[Landmark, ...], chain: tuple[int, int, int]
    ) -> tuple[str, float]:
        _, pip_or_ip, tip = chain
        wrist = landmarks[_WRIST]
        pip_distance = _distance(wrist, landmarks[pip_or_ip])
        tip_distance = _distance(wrist, landmarks[tip])
        if pip_distance == 0:
            return "unknown", 0.0

        ratio = tip_distance / pip_distance
        if ratio >= self._extended_ratio:
            return "extended", _margin_confidence(ratio - self._extended_ratio)
        if ratio <= self._curled_ratio:
            return "curled", _margin_confidence(self._curled_ratio - ratio)
        return "unknown", 0.0

    @staticmethod
    def _pattern_confidence(
        observation: HandObservation,
        states: dict[str, tuple[str, float]],
        fingers: tuple[str, ...],
    ) -> float:
        geometric_confidence = min(states[finger][1] for finger in fingers)
        return _combined_confidence(geometric_confidence, observation.confidence)

    @staticmethod
    def _unknown_confidence(
        observation: HandObservation,
        states: dict[str, tuple[str, float]],
    ) -> float:
        recognized = sum(state != "unknown" for state, _ in states.values())
        geometric_confidence = recognized / len(states)
        return _combined_confidence(geometric_confidence, observation.confidence)


def _distance(first: Landmark, second: Landmark) -> float:
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )


def _margin_confidence(margin: float) -> float:
    """Map a threshold margin to a stable confidence in the [0, 1] range."""

    return min(1.0, 0.5 + max(0.0, margin))


def _combined_confidence(geometry: float, observation: float) -> float:
    """Blend geometric evidence with the upstream hand observation score."""

    return round(0.75 * geometry + 0.25 * max(0.0, min(1.0, observation)), 3)
