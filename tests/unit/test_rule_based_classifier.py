"""Deterministic tests for the first basic gesture rules."""

from handoff.adapters.classifier import RuleBasedGestureClassifier
from handoff.domain.models import GestureLabel, HandObservation, Landmark


def test_open_hand_is_classified_as_palm() -> None:
    prediction = RuleBasedGestureClassifier().classify(_observation(extended=range(5)))

    assert prediction.label is GestureLabel.PALM
    assert prediction.confidence > 0.5


def test_single_extended_index_is_classified_as_finger() -> None:
    prediction = RuleBasedGestureClassifier().classify(_observation(extended=(1,)))

    assert prediction.label is GestureLabel.FINGER
    assert prediction.confidence > 0.0


def test_closed_hand_is_unknown() -> None:
    prediction = RuleBasedGestureClassifier().classify(_observation(extended=()))

    assert prediction.label is GestureLabel.UNKNOWN


def _observation(*, extended: range | tuple[int, ...]) -> HandObservation:
    points = [Landmark(0.0, 0.0, 0.0) for _ in range(21)]
    chains = ((2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    directions = ((1.0, 0.0), (0.0, -1.0), (0.2, -1.0), (-0.2, -1.0), (-0.4, -1.0))

    for finger, (mcp, pip, tip) in enumerate(chains):
        direction_x, direction_y = directions[finger]
        if finger in extended:
            points[mcp] = Landmark(direction_x * 0.2, direction_y * 0.2, 0.0)
            points[pip] = Landmark(direction_x * 0.5, direction_y * 0.5, 0.0)
            points[tip] = Landmark(direction_x * 1.0, direction_y * 1.0, 0.0)
        else:
            points[mcp] = Landmark(direction_x * 0.2, direction_y * 0.2, 0.0)
            points[pip] = Landmark(direction_x * 0.8, direction_y * 0.8, 0.0)
            points[tip] = Landmark(direction_x * 0.4, direction_y * 0.4, 0.0)

    return HandObservation(tuple(points), "left", 0.9, 1.0)
