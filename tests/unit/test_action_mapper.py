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


def test_zoom_mapping_is_loaded_from_external_configuration() -> None:
    mapper = JsonActionMapper("config/gesture_mappings.json")

    actions = mapper.map(GestureEvent(GestureLabel.ZOOM_IN))

    assert [(action.type, action.value) for action in actions] == [
        (ActionType.PRESS_KEY, "ctrl+plus"),
    ]


def test_absolute_cursor_mapping_anchors_then_uses_camera_position() -> None:
    mapper = JsonActionMapper("config/gesture_mappings.json")

    start_actions = mapper.map(
        GestureEvent(
            GestureLabel.INDEX,
            cursor_position=(0.2, 0.3),
            cursor_tracking_started=True,
        )
    )
    move_actions = mapper.map(
        GestureEvent(
            GestureLabel.INDEX,
            cursor_delta=(0.1, -0.2),
            cursor_position=(0.3, 0.3),
        )
    )

    assert [(action.type, action.value) for action in start_actions] == [
        (ActionType.BEGIN_MOUSE_TRACKING, (0.2, 0.3)),
        (ActionType.MOVE_MOUSE, (0.2, 0.3)),
    ]
    assert [(action.type, action.value) for action in move_actions] == [
        (ActionType.MOVE_MOUSE, (0.3, 0.3)),
    ]


def test_relative_cursor_mapping_remains_selectable() -> None:
    mapper = JsonActionMapper("config/gesture_mappings_relative.json")

    actions = mapper.map(GestureEvent(GestureLabel.INDEX, cursor_delta=(0.1, -0.2)))

    assert [(action.type, action.value) for action in actions] == [
        (ActionType.MOVE_MOUSE_BY, (0.1, -0.2)),
    ]
