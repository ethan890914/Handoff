"""Deterministic tests for the first basic gesture rules."""

from handoff.adapters.classifier import RuleBasedGestureClassifier
from handoff.domain.models import GestureLabel, HandObservation, Landmark


def test_open_hand_is_classified_as_palm() -> None:
    prediction = RuleBasedGestureClassifier().classify(_observation(extended=range(5)))

    assert prediction.label is GestureLabel.PALM
    assert prediction.confidence > 0.5


def test_single_extended_index_is_classified_specifically() -> None:
    prediction = RuleBasedGestureClassifier().classify(_observation(extended=(1,)))

    assert prediction.label is GestureLabel.INDEX
    assert prediction.confidence > 0.75


def test_each_single_extended_nonthumb_finger_gets_a_specific_label() -> None:
    expected = (
        GestureLabel.INDEX,
        GestureLabel.MIDDLE,
        GestureLabel.RING,
        GestureLabel.PINKY,
    )

    for finger, label in enumerate(expected, start=1):
        prediction = RuleBasedGestureClassifier().classify(_observation(extended=(finger,)))
        assert prediction.label is label
        assert prediction.confidence > 0.75


def test_thumb_pointing_right_is_navigate_right() -> None:
    prediction = RuleBasedGestureClassifier().classify(_observation(extended=(0,)))

    assert prediction.label is GestureLabel.NAVIGATE_RIGHT


def test_thumb_pointing_left_is_navigate_left() -> None:
    points = list(_observation(extended=(0,)).landmarks)
    points[2] = Landmark(-0.20, 0.0, 0.0)
    points[3] = Landmark(-0.50, 0.0, 0.0)
    points[4] = Landmark(-1.00, 0.0, 0.0)

    prediction = RuleBasedGestureClassifier().classify(
        HandObservation(tuple(points), "left", 0.9, 1.0)
    )

    assert prediction.label is GestureLabel.NAVIGATE_LEFT


def test_closed_hand_is_reset() -> None:
    prediction = RuleBasedGestureClassifier().classify(_observation(extended=()))

    assert prediction.label is GestureLabel.RESET


def test_low_confidence_observation_is_neutral() -> None:
    classifier = RuleBasedGestureClassifier(
        minimum_observation_confidence=0.5,
    )
    classifier.classify(_observation(extended=(1, 2), wrist_position=(0.5, 0.5)))

    low_confidence = classifier.classify(
        _observation(
            extended=(1, 2),
            wrist_position=(0.5, 0.3),
            confidence=0.2,
        )
    )
    assert low_confidence.label is GestureLabel.UNKNOWN


def test_thumb_index_pinch_clicks_on_release_after_a_stable_hold() -> None:
    classifier = RuleBasedGestureClassifier(click_hold_frames=3)
    points = list(_observation(extended=()).landmarks)
    _make_thumb_extended(points)
    points[4] = Landmark(0.10, 0.0, 0.0)
    points[8] = Landmark(0.12, 0.0, 0.0)

    pinch = HandObservation(tuple(points), "left", 0.9, 1.0)
    assert classifier.classify(pinch).label is GestureLabel.UNKNOWN
    assert classifier.classify(pinch).label is GestureLabel.UNKNOWN
    assert classifier.classify(pinch).label is GestureLabel.UNKNOWN
    prediction = classifier.classify(_observation(extended=()))

    assert prediction.label is GestureLabel.LEFT_CLICK
    assert prediction.confidence > 0.75


def test_straight_index_pinch_is_reserved_for_click_not_zoom() -> None:
    classifier = RuleBasedGestureClassifier(click_hold_frames=1)
    points = list(_observation(extended=(0, 1)).landmarks)
    points[4] = Landmark(0.02, -0.98, 0.0)
    pinch = HandObservation(tuple(points), "left", 0.9, 1.0)

    assert classifier.classify(pinch).label is GestureLabel.UNKNOWN
    prediction = classifier.classify(_observation(extended=()))

    assert prediction.label is GestureLabel.LEFT_CLICK


def test_moving_a_click_pinch_cancels_the_pending_click() -> None:
    classifier = RuleBasedGestureClassifier(click_hold_frames=2, click_max_motion=0.02)
    points = list(_observation(extended=()).landmarks)
    _make_thumb_extended(points)
    points[4] = Landmark(0.10, 0.0, 0.0)
    points[8] = Landmark(0.12, 0.0, 0.0)
    first_pinch = HandObservation(
        tuple(points), "left", 0.9, 1.0, wrist_position=(0.5, 0.5)
    )
    moved_pinch = HandObservation(
        tuple(points), "left", 0.9, 1.0, wrist_position=(0.6, 0.5)
    )

    assert classifier.classify(first_pinch).label is GestureLabel.UNKNOWN
    assert classifier.classify(moved_pinch).label is GestureLabel.UNKNOWN
    assert classifier.classify(_observation(extended=())).label is not GestureLabel.LEFT_CLICK


def test_nearby_thumb_index_without_a_full_pinch_is_not_a_click() -> None:
    points = list(_observation(extended=()).landmarks)
    _make_thumb_extended(points)
    points[4] = Landmark(0.10, 0.0, 0.0)
    points[8] = Landmark(0.29, 0.0, 0.0)

    prediction = RuleBasedGestureClassifier().classify(
        HandObservation(tuple(points), "left", 0.9, 1.0)
    )

    assert prediction.label is not GestureLabel.LEFT_CLICK


def test_thumb_middle_pinch_is_right_click() -> None:
    # The index finger may naturally remain raised during a thumb-middle pinch.
    points = list(_observation(extended=(1,)).landmarks)
    _make_thumb_extended(points)
    points[4] = Landmark(0.10, 0.0, 0.0)
    points[12] = Landmark(0.12, 0.0, 0.0)

    prediction = RuleBasedGestureClassifier().classify(
        HandObservation(tuple(points), "left", 0.9, 1.0)
    )

    assert prediction.label is GestureLabel.RIGHT_CLICK
    assert prediction.confidence > 0.75


def test_pinch_with_uncertain_target_finger_is_still_a_click() -> None:
    classifier = RuleBasedGestureClassifier(click_hold_frames=1)
    points = list(_observation(extended=()).landmarks)
    _make_thumb_extended(points)
    points[4] = Landmark(0.10, 0.0, 0.0)
    points[6] = Landmark(0.20, 0.0, 0.0)
    points[8] = Landmark(0.20, 0.0, 0.0)

    pinch = HandObservation(tuple(points), "left", 0.9, 1.0)
    classifier.classify(pinch)
    prediction = classifier.classify(_observation(extended=()))

    assert prediction.label is GestureLabel.LEFT_CLICK


def test_closed_hand_thumb_wrap_is_not_left_click() -> None:
    points = list(_observation(extended=()).landmarks)
    points[4] = Landmark(0.10, 0.0, 0.0)
    points[8] = Landmark(0.12, 0.0, 0.0)

    prediction = RuleBasedGestureClassifier().classify(
        HandObservation(tuple(points), "left", 0.9, 1.0)
    )

    assert prediction.label is GestureLabel.RESET


def test_closed_hand_with_angled_thumb_is_reset() -> None:
    points = list(_observation(extended=()).landmarks)
    # The thumb tip is far enough from the wrist to look extended by the
    # normal ratio, but remains close to its MCP joint as it would in a fist.
    points[2] = Landmark(0.20, 0.0, 0.0)
    points[3] = Landmark(0.35, 0.0, 0.0)
    points[4] = Landmark(0.42, 0.0, 0.0)

    prediction = RuleBasedGestureClassifier().classify(
        HandObservation(tuple(points), "left", 0.9, 1.0)
    )

    assert prediction.label is GestureLabel.RESET


def test_two_finger_pointing_down_is_scroll_down() -> None:
    prediction = RuleBasedGestureClassifier().classify(
        _observation(extended=(1, 2), point_down=(1, 2))
    )

    assert prediction.label is GestureLabel.SCROLL_DOWN
    assert prediction.confidence > 0.75


def test_two_finger_pointing_up_is_scroll_up() -> None:
    prediction = RuleBasedGestureClassifier().classify(_observation(extended=(1, 2)))

    assert prediction.label is GestureLabel.SCROLL_UP


def test_two_finger_scroll_allows_a_naturally_extended_thumb() -> None:
    prediction = RuleBasedGestureClassifier().classify(_observation(extended=(0, 1, 2)))

    assert prediction.label is GestureLabel.SCROLL_UP


def test_thumb_index_spread_is_zoom_in() -> None:
    classifier = RuleBasedGestureClassifier(
        zoom_distance_threshold=0.1,
        zoom_smoothing=1.0,
    )
    baseline = _observation(extended=(0, 1))
    spread_points = list(baseline.landmarks)
    spread_points[4] = Landmark(1.2, 0.0, 0.0)

    classifier.classify(baseline)
    prediction = classifier.classify(
        HandObservation(tuple(spread_points), "left", 0.9, 1.0)
    )

    assert prediction.label is GestureLabel.ZOOM_IN


def test_thumb_index_closing_is_zoom_out() -> None:
    classifier = RuleBasedGestureClassifier(
        zoom_distance_threshold=0.1,
        zoom_smoothing=1.0,
    )
    baseline = _observation(extended=(0, 1))
    closing_points = list(baseline.landmarks)
    closing_points[4] = Landmark(0.8, 0.0, 0.0)

    classifier.classify(baseline)
    prediction = classifier.classify(
        HandObservation(tuple(closing_points), "left", 0.9, 1.0)
    )

    assert prediction.label is GestureLabel.ZOOM_OUT


def test_zoom_mode_locks_out_click_until_a_neutral_release() -> None:
    classifier = RuleBasedGestureClassifier(
        click_hold_frames=1,
        zoom_distance_threshold=0.1,
        zoom_smoothing=1.0,
    )
    zoom_ready = _observation(extended=(0, 1))
    spread_points = list(zoom_ready.landmarks)
    spread_points[4] = Landmark(1.2, 0.0, 0.0)
    click_points = list(_observation(extended=()).landmarks)
    _make_thumb_extended(click_points)
    click_points[4] = Landmark(0.10, 0.0, 0.0)
    click_points[8] = Landmark(0.12, 0.0, 0.0)

    classifier.classify(zoom_ready)
    assert classifier.classify(
        HandObservation(tuple(spread_points), "left", 0.9, 1.0)
    ).label is GestureLabel.ZOOM_IN
    assert classifier.classify(
        HandObservation(tuple(click_points), "left", 0.9, 1.0)
    ).label is GestureLabel.UNKNOWN
    assert classifier.classify(_observation(extended=())).label is not GestureLabel.LEFT_CLICK


def test_two_finger_pointing_sideways_does_not_scroll() -> None:
    prediction = RuleBasedGestureClassifier().classify(
        _observation(extended=(1, 2), point_sideways=(1, 2))
    )

    assert prediction.label is GestureLabel.UNKNOWN


def test_two_finger_horizontal_motion_is_not_navigation() -> None:
    classifier = RuleBasedGestureClassifier()
    prediction = classifier.classify(_observation(extended=(1, 2), point_sideways=(1, 2)))

    assert prediction.label is GestureLabel.UNKNOWN


def _make_thumb_extended(points: list[Landmark]) -> None:
    points[2] = Landmark(-0.50, 0.0, 0.0)
    points[3] = Landmark(-0.10, 0.0, 0.0)


def _observation(
    *,
    extended: range | tuple[int, ...],
    wrist_position: tuple[float, float] | None = None,
    confidence: float = 0.9,
    point_down: tuple[int, ...] = (),
    point_sideways: tuple[int, ...] = (),
) -> HandObservation:
    points = [Landmark(0.0, 0.0, 0.0) for _ in range(21)]
    chains = ((2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    directions = ((1.0, 0.0), (0.0, -1.0), (0.2, -1.0), (-0.2, -1.0), (-0.4, -1.0))

    for finger, (mcp, pip, tip) in enumerate(chains):
        direction_x, direction_y = directions[finger]
        if finger in point_down:
            direction_y *= -1
        if finger in point_sideways:
            direction_x, direction_y = 1.0, 0.0
        if finger in extended:
            points[mcp] = Landmark(direction_x * 0.2, direction_y * 0.2, 0.0)
            points[pip] = Landmark(direction_x * 0.5, direction_y * 0.5, 0.0)
            points[tip] = Landmark(direction_x * 1.0, direction_y * 1.0, 0.0)
        else:
            points[mcp] = Landmark(direction_x * 0.2, direction_y * 0.2, 0.0)
            points[pip] = Landmark(direction_x * 0.8, direction_y * 0.8, 0.0)
            points[tip] = Landmark(direction_x * 0.4, direction_y * 0.4, 0.0)

    return HandObservation(tuple(points), "left", confidence, 1.0, wrist_position)
