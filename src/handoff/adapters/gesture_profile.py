"""External configuration for enabling a safe subset of gesture labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from handoff.domain.models import GestureLabel


class JsonGestureProfile:
    """Load an allowlist of active gesture labels without changing classifiers."""

    def __init__(self, config_path: str | Path) -> None:
        with Path(config_path).open(encoding="utf-8") as config_file:
            raw_config: Any = json.load(config_file)
        if not isinstance(raw_config, dict) or not isinstance(
            raw_config.get("enabled_gestures"), list
        ):
            raise ValueError("gesture profile must contain an enabled_gestures list")
        try:
            self.enabled_labels = frozenset(
                GestureLabel(label) for label in raw_config["enabled_gestures"]
            )
        except (TypeError, ValueError) as error:
            raise ValueError("gesture profile contains an invalid gesture label") from error
