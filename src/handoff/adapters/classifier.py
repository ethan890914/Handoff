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
    """Recognize static hand poses, including held two-finger scrolling.

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
        pinch_distance: float = 0.14,
        click_hold_frames: int = 4,
        click_max_motion: float = 0.018,
        scroll_direction_ratio: float = 1.7,
        zoom_ready_distance: float = 0.30,
        zoom_distance_threshold: float = 0.08,
        zoom_rearm_distance: float | None = None,
        zoom_smoothing: float = 0.35,
        thumb_extension_distance: float = 0.25,
        minimum_observation_confidence: float = 0.5,
    ) -> None:
        if curled_ratio >= extended_ratio:
            raise ValueError("curled_ratio must be smaller than extended_ratio")
        if pinch_distance <= 0:
            raise ValueError("pinch_distance must be positive")
        if click_hold_frames < 1:
            raise ValueError("click_hold_frames must be at least 1")
        if click_max_motion < 0:
            raise ValueError("click_max_motion cannot be negative")
        if scroll_direction_ratio <= 1:
            raise ValueError("scroll_direction_ratio must be greater than 1")
        if zoom_ready_distance <= pinch_distance:
            raise ValueError("zoom_ready_distance must be larger than pinch_distance")
        if zoom_distance_threshold <= 0:
            raise ValueError("zoom_distance_threshold must be positive")
        if zoom_rearm_distance is not None and zoom_rearm_distance < 0:
            raise ValueError("zoom_rearm_distance cannot be negative")
        if not 0 < zoom_smoothing <= 1:
            raise ValueError("zoom_smoothing must be between 0 and 1")
        if thumb_extension_distance <= 0:
            raise ValueError("thumb_extension_distance must be positive")
        if not 0 <= minimum_observation_confidence <= 1:
            raise ValueError("minimum_observation_confidence must be between 0 and 1")
        self._extended_ratio = extended_ratio
        self._curled_ratio = curled_ratio
        self._pinch_distance = pinch_distance
        self._click_hold_frames = click_hold_frames
        self._click_max_motion = click_max_motion
        self._scroll_direction_ratio = scroll_direction_ratio
        self._zoom_ready_distance = zoom_ready_distance
        self._zoom_distance_threshold = zoom_distance_threshold
        self._zoom_rearm_distance = (
            zoom_distance_threshold * 0.35
            if zoom_rearm_distance is None
            else zoom_rearm_distance
        )
        self._zoom_smoothing = zoom_smoothing
        self._thumb_extension_distance = thumb_extension_distance
        self._minimum_observation_confidence = minimum_observation_confidence
        self._zoom_origin_distance: float | None = None
        self._zoom_distance: float | None = None
        self._zoom_label: GestureLabel | None = None
        self._zoom_click_lockout = False
        self._click_candidate_frames = 0
        self._click_candidate_position: tuple[float, float] | None = None
        self._click_candidate_confidence = 0.0

    def classify(self, observation: HandObservation) -> GesturePrediction:
        """Return a prediction for one hand observation."""

        if len(observation.landmarks) != 21:
            raise ValueError("gesture classification requires exactly 21 hand landmarks")
        if observation.confidence < self._minimum_observation_confidence:
            self._clear_zoom_state()
            self._clear_click_state()
            self._zoom_click_lockout = False
            return GesturePrediction(GestureLabel.UNKNOWN, observation.confidence)

        states = {
            finger: self._state(observation.landmarks, chain)
            for finger, chain in _FINGER_CHAINS.items()
        }
        if states["thumb"][0] == "extended" and not self._thumb_is_extended(
            observation.landmarks
        ):
            states["thumb"] = ("curled", 0.0)

        pinch_candidates = self._pinch_candidates(observation.landmarks, states)
        if "middle" in pinch_candidates:
            self._clear_zoom_state()
            self._clear_click_state()
            return GesturePrediction(
                GestureLabel.RIGHT_CLICK,
                self._pinch_confidence(observation, "middle"),
            )

        zoom_prediction = self._classify_zoom(observation, states)
        if zoom_prediction is not None:
            return zoom_prediction

        click_prediction = self._classify_left_click(
            observation,
            left_pinch="index" in pinch_candidates,
            other_pinch="middle" in pinch_candidates,
        )
        if click_prediction is not None:
            return click_prediction

        if all(states[finger][0] == "extended" for finger in _FINGER_CHAINS):
            return GesturePrediction(
                GestureLabel.PALM,
                self._pattern_confidence(observation, states, tuple(_FINGER_CHAINS)),
            )

        if not any(state == "extended" for state, _ in states.values()) and not pinch_candidates:
            return GesturePrediction(
                GestureLabel.RESET,
                self._pattern_confidence(observation, states, tuple(_FINGER_CHAINS)),
            )

        if pinch_candidates:
            # The index pinch is handled above as a release-triggered left
            # click. A middle pinch has already returned directly.
            return GesturePrediction(GestureLabel.UNKNOWN, observation.confidence)

        zoom_prediction = self._classify_zoom(observation, states)
        if zoom_prediction is not None:
            return zoom_prediction

        scroll_prediction = self._classify_static_scroll(observation, states)
        if scroll_prediction is not None:
            return scroll_prediction

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
                confidence = self._pattern_confidence(observation, states, evidence_fingers)
                if label is GestureLabel.INDEX:
                    return GesturePrediction(
                        label,
                        confidence,
                        cursor_position=observation.cursor_position,
                    )
                return GesturePrediction(label, confidence)

        return GesturePrediction(
            GestureLabel.UNKNOWN,
            self._unknown_confidence(observation, states),
        )

    def _classify_zoom(
        self,
        observation: HandObservation,
        states: dict[str, tuple[str, float]],
    ) -> GesturePrediction | None:
        zoom_pose = (
            self._thumb_is_extended(observation.landmarks)
            and self._finger_is_straight(observation.landmarks, "index")
            and _distance(observation.landmarks[4], observation.landmarks[8])
            >= self._zoom_ready_distance
            and all(
                states[finger][0] != "extended" for finger in ("middle", "ring", "pinky")
            )
        )
        if not zoom_pose:
            self._clear_zoom_state()
            return None

        # A zoom pose is deliberately separate from a touch/click pose.  It
        # neutralizes any pending click before measuring the zoom span.
        self._clear_click_state()

        distance = _distance(observation.landmarks[4], observation.landmarks[8])
        if self._zoom_origin_distance is None:
            self._zoom_origin_distance = distance
            self._zoom_distance = distance
            return None

        previous_distance = self._zoom_distance or self._zoom_origin_distance
        smoothed_distance = previous_distance + self._zoom_smoothing * (
            distance - previous_distance
        )
        self._zoom_distance = smoothed_distance
        signed_distance = smoothed_distance - self._zoom_origin_distance

        if self._zoom_label is None:
            if abs(signed_distance) < self._zoom_distance_threshold:
                return None
            self._zoom_label = (
                GestureLabel.ZOOM_IN if signed_distance > 0 else GestureLabel.ZOOM_OUT
            )
            self._zoom_click_lockout = True
            return GesturePrediction(
                self._zoom_label,
                self._zoom_confidence(observation, abs(signed_distance)),
            )

        direction = 1.0 if self._zoom_label is GestureLabel.ZOOM_IN else -1.0
        if direction * signed_distance <= self._zoom_rearm_distance:
            self._zoom_origin_distance = smoothed_distance
            self._zoom_label = None
            return None

        return GesturePrediction(
            self._zoom_label,
            self._zoom_confidence(observation, abs(signed_distance)),
        )

    def _classify_left_click(
        self,
        observation: HandObservation,
        *,
        left_pinch: bool,
        other_pinch: bool,
    ) -> GesturePrediction | None:
        if self._zoom_click_lockout:
            if not left_pinch and not other_pinch:
                # One neutral frame releases the lockout; it cannot also start
                # a new click candidate, preventing a zoom release click.
                self._zoom_click_lockout = False
            return None

        if left_pinch:
            position = observation.wrist_position
            if self._click_candidate_frames == 0:
                self._click_candidate_position = position
                self._click_candidate_confidence = self._pinch_confidence(observation, "index")
            elif (
                position is not None
                and self._click_candidate_position is not None
                and _position_distance(position, self._click_candidate_position)
                > self._click_max_motion
            ):
                self._clear_click_state()
                return GesturePrediction(GestureLabel.UNKNOWN, observation.confidence)
            self._click_candidate_frames += 1
            return GesturePrediction(GestureLabel.UNKNOWN, observation.confidence)

        if other_pinch:
            self._clear_click_state()
            return None

        if self._click_candidate_frames >= self._click_hold_frames:
            confidence = self._click_candidate_confidence
            self._clear_click_state()
            return GesturePrediction(GestureLabel.LEFT_CLICK, confidence)

        self._clear_click_state()
        return None

    def _clear_zoom_state(self) -> None:
        self._zoom_origin_distance = None
        self._zoom_distance = None
        self._zoom_label = None

    def _clear_click_state(self) -> None:
        self._click_candidate_frames = 0
        self._click_candidate_position = None
        self._click_candidate_confidence = 0.0

    def _zoom_confidence(self, observation: HandObservation, distance: float) -> float:
        geometry = min(
            1.0,
            0.5 + (distance - self._zoom_distance_threshold) / self._zoom_distance_threshold,
        )
        return _combined_confidence(geometry, observation.confidence)

    def _pinch_candidates(
        self,
        landmarks: tuple[Landmark, ...],
        states: dict[str, tuple[str, float]],
    ) -> dict[str, float]:
        candidates = {}
        for finger in _TWO_FINGER_FINGERS:
            if (
                self._thumb_can_pinch(landmarks)
                and self._finger_can_pinch(states, finger)
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

    @staticmethod
    def _finger_can_pinch(
        states: dict[str, tuple[str, float]], finger: str
    ) -> bool:
        if finger == "index":
            return states[finger][0] in ("curled", "unknown", "extended")
        return states[finger][0] in ("curled", "unknown")

    def _pinch_distance_between(self, landmarks: tuple[Landmark, ...], finger: str) -> float:
        tip = 8 if finger == "index" else 12
        return _distance(landmarks[4], landmarks[tip])

    @staticmethod
    def _thumb_can_pinch(landmarks: tuple[Landmark, ...]) -> bool:
        """Accept a bent-but-reached thumb while rejecting a thumb folded in a fist."""

        return _distance(landmarks[2], landmarks[4]) >= 0.15

    def _pinch_confidence(self, observation: HandObservation, finger: str) -> float:
        distance = self._pinch_distance_between(observation.landmarks, finger)
        geometry = max(0.0, min(1.0, 1.0 - distance / self._pinch_distance))
        return _combined_confidence(geometry, observation.confidence)

    def _classify_static_scroll(
        self,
        observation: HandObservation,
        states: dict[str, tuple[str, float]],
    ) -> GesturePrediction | None:
        two_finger_pose = (
            all(
                self._finger_is_straight(observation.landmarks, finger)
                for finger in _TWO_FINGER_FINGERS
            )
            and all(states[finger][0] != "extended" for finger in ("ring", "pinky"))
        )
        if not two_finger_pose:
            return None

        vectors = tuple(
            self._finger_direction(observation.landmarks, finger)
            for finger in _TWO_FINGER_FINGERS
        )
        horizontal = sum(vector[0] for vector in vectors) / len(vectors)
        vertical = sum(vector[1] for vector in vectors) / len(vectors)
        if abs(vertical) < abs(horizontal) * self._scroll_direction_ratio:
            return None

        label = GestureLabel.SCROLL_DOWN if vertical > 0 else GestureLabel.SCROLL_UP
        verticality = abs(vertical) / math.sqrt(horizontal * horizontal + vertical * vertical)
        return GesturePrediction(label, _combined_confidence(verticality, observation.confidence))

    @staticmethod
    def _finger_direction(
        landmarks: tuple[Landmark, ...], finger: str
    ) -> tuple[float, float]:
        mcp, _, tip = _FINGER_CHAINS[finger]
        return landmarks[tip].x - landmarks[mcp].x, landmarks[tip].y - landmarks[mcp].y

    @staticmethod
    def _finger_is_straight(landmarks: tuple[Landmark, ...], finger: str) -> bool:
        mcp, pip_or_ip, tip = _FINGER_CHAINS[finger]
        first_segment = _distance(landmarks[mcp], landmarks[pip_or_ip])
        second_segment = _distance(landmarks[pip_or_ip], landmarks[tip])
        chain_length = first_segment + second_segment
        if chain_length == 0:
            return False
        return _distance(landmarks[mcp], landmarks[tip]) / chain_length >= 0.8

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


def _position_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _margin_confidence(margin: float) -> float:
    """Map a threshold margin to a stable confidence in the [0, 1] range."""

    return min(1.0, 0.5 + max(0.0, margin))


def _combined_confidence(geometry: float, observation: float) -> float:
    """Blend geometric evidence with the upstream hand observation score."""

    return round(0.75 * geometry + 0.25 * max(0.0, min(1.0, observation)), 3)
