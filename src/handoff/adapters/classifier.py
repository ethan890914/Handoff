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
    """Recognize an open palm and a single pointing finger.

    Extension is determined from the ratio between a fingertip's distance from
    the wrist and its PIP/IP joint's distance from the wrist.  This works on
    the adapter's wrist-relative, scale-normalized landmarks and does not
    depend on camera resolution or absolute hand position.
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
            confidence = self._pattern_confidence(states, tuple(_FINGER_CHAINS))
            return GesturePrediction(GestureLabel.PALM, confidence)

        if (
            states["index"][0] == "extended"
            and all(states[finger][0] == "curled" for finger in _NON_THUMB_FINGERS[1:])
        ):
            confidence = self._pattern_confidence(states, _NON_THUMB_FINGERS)
            return GesturePrediction(GestureLabel.FINGER, confidence)

        return GesturePrediction(GestureLabel.UNKNOWN, self._unknown_confidence(states))

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
            return "extended", min(1.0, (ratio - self._extended_ratio) / self._extended_ratio)
        if ratio <= self._curled_ratio:
            return "curled", min(1.0, (self._curled_ratio - ratio) / self._curled_ratio)
        return "unknown", 0.0

    @staticmethod
    def _pattern_confidence(
        states: dict[str, tuple[str, float]],
        fingers: tuple[str, ...],
    ) -> float:
        return round(min(states[finger][1] for finger in fingers), 3)

    @staticmethod
    def _unknown_confidence(states: dict[str, tuple[str, float]]) -> float:
        recognized = sum(state != "unknown" for state, _ in states.values())
        return round(recognized / len(states), 3)


def _distance(first: Landmark, second: Landmark) -> float:
    return math.sqrt(
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )
