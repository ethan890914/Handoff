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


class RuleBasedGestureClassifier:
    """Recognize an open palm and single raised fingers.

    Extension is determined from the ratio between a fingertip's distance from
    the wrist and its PIP/IP joint's distance from the wrist.  This works on
    the adapter's wrist-relative, scale-normalized landmarks and does not
    depend on camera resolution or absolute hand position.  For non-thumb
    gestures, the thumb is allowed to be either curled or extended because
    index-pointing commonly includes an extended thumb.
    """

    def __init__(self, *, extended_ratio: float = 1.10, curled_ratio: float = 0.95) -> None:
        if curled_ratio >= extended_ratio:
            raise ValueError("curled_ratio must be smaller than extended_ratio")
        self._extended_ratio = extended_ratio
        self._curled_ratio = curled_ratio

    def classify(self, observation: HandObservation) -> GesturePrediction:
        """Return a prediction for one hand observation."""

        if len(observation.landmarks) != 21:
            raise ValueError("gesture classification requires exactly 21 hand landmarks")

        states = {
            finger: self._state(observation.landmarks, chain)
            for finger, chain in _FINGER_CHAINS.items()
        }
        if all(states[finger][0] == "extended" for finger in _FINGER_CHAINS):
            return GesturePrediction(
                GestureLabel.PALM,
                self._pattern_confidence(observation, states, tuple(_FINGER_CHAINS)),
            )

        non_thumb_extended = tuple(
            finger for finger in _NON_THUMB_FINGERS if states[finger][0] == "extended"
        )
        finger: str | None = None
        if len(non_thumb_extended) == 1:
            # An extended thumb is allowed alongside a pointing finger.
            finger = non_thumb_extended[0]
        elif not non_thumb_extended and states["thumb"][0] == "extended":
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
