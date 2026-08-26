"""Tests for external gesture-to-action mappings."""

from handoff.adapters.action_mapper import JsonActionMapper
from handoff.domain.models import ActionType, GestureEvent, GestureLabel


def test_loads_actions_from_reference_config() -> None:
    mapper = JsonActionMapper("config/gesture_mappings.json")

    actions = mapper.map(GestureEvent(GestureLabel.LEFT_CLICK))

    assert actions[0].type is ActionType.CLICK
    assert actions[0].value == "left"


def test_reset_maps_to_center_and_pause() -> None:
    mapper = JsonActionMapper("config/gesture_mappings.json")

    actions = mapper.map(GestureEvent(GestureLabel.RESET))

    assert [(action.type, action.value) for action in actions] == [
        (ActionType.MOVE_MOUSE, "center"),
        (ActionType.PAUSE_TRACKING, 1.0),
    ]
