"""JSON-backed gesture-to-action mapping adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from handoff.domain.models import Action, ActionType, GestureEvent


class JsonActionMapper:
    """Map gesture events to abstract actions loaded from JSON."""

    def __init__(self, config_path: str | Path) -> None:
        path = Path(config_path)
        with path.open(encoding="utf-8") as config_file:
            raw_config: Any = json.load(config_file)
        if not isinstance(raw_config, dict):
            raise ValueError("gesture mapping config must be a JSON object")
        self._mapping = raw_config

    def map(self, event: GestureEvent) -> tuple[Action, ...]:
        """Return configured abstract actions for one gesture event."""

        entries = self._mapping.get(event.label.value, [])
        if not isinstance(entries, list):
            raise ValueError(f"mapping for {event.label.value!r} must be a list")

        actions = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("each gesture action must be an object")
            try:
                action_type = ActionType(entry["type"])
                value = entry["value"]
            except (KeyError, ValueError) as error:
                raise ValueError(f"invalid action mapping for {event.label.value!r}") from error
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError("action value must be a string or number")
            if action_type is ActionType.MOVE_MOUSE_BY and value == "cursor_delta":
                if event.cursor_delta is None:
                    continue
                actions.append(Action(action_type, event.cursor_delta))
            else:
                actions.append(Action(action_type, value))
        return tuple(actions)
