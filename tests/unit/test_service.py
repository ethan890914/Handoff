"""Tests for live-pipeline orchestration without camera or desktop side effects."""

from handoff.application.service import HandoffApplication
from handoff.domain.models import Action, ActionType, GestureEvent, GestureLabel, GesturePrediction


class _Camera:
    def __init__(self, *frames: object) -> None:
        self._frames = frames

    def frames(self):
        yield from self._frames


class _Perception:
    def detect(self, frame: object) -> object | None:
        return None if frame == "no-hand" else frame


class _Classifier:
    def __init__(self) -> None:
        self.observations: list[object] = []

    def classify(self, observation: object) -> GesturePrediction:
        self.observations.append(observation)
        return GesturePrediction(GestureLabel.LEFT_CLICK, 0.9)


class _Controller:
    def __init__(self) -> None:
        self.predictions: list[GesturePrediction] = []

    def update(self, prediction: GesturePrediction) -> GestureEvent | None:
        self.predictions.append(prediction)
        if prediction.label is GestureLabel.LEFT_CLICK:
            return GestureEvent(prediction.label)
        return None


class _Mapper:
    def __init__(self) -> None:
        self.events: list[GestureEvent] = []

    def map(self, event: GestureEvent) -> tuple[Action, ...]:
        self.events.append(event)
        return (
            Action(ActionType.CLICK, "left"),
            Action(ActionType.SCROLL, -1.0),
        )


class _Dispatcher:
    def __init__(self) -> None:
        self.actions: list[Action] = []

    def dispatch(self, action: Action) -> None:
        self.actions.append(action)


class _Display:
    def __init__(self, *, continue_running: bool = True) -> None:
        self.continue_running = continue_running
        self.overlays: list[str] = []

    def show(self, frame: object, overlay: str) -> bool:
        self.overlays.append(overlay)
        return self.continue_running


def test_runs_events_through_mapping_and_dispatch_in_order() -> None:
    classifier = _Classifier()
    controller = _Controller()
    mapper = _Mapper()
    dispatcher = _Dispatcher()
    application = HandoffApplication(
        camera=_Camera("hand"),
        perception=_Perception(),
        classifier=classifier,
        controller=controller,
        action_mapper=mapper,
        dispatcher=dispatcher,
    )

    application.run()

    assert classifier.observations == ["hand"]
    assert [event.label for event in mapper.events] == [GestureLabel.LEFT_CLICK]
    assert dispatcher.actions == [
        Action(ActionType.CLICK, "left"),
        Action(ActionType.SCROLL, -1.0),
    ]


def test_no_hand_resets_control_without_classifying_or_dispatching() -> None:
    classifier = _Classifier()
    controller = _Controller()
    mapper = _Mapper()
    dispatcher = _Dispatcher()
    application = HandoffApplication(
        camera=_Camera("no-hand"),
        perception=_Perception(),
        classifier=classifier,
        controller=controller,
        action_mapper=mapper,
        dispatcher=dispatcher,
    )

    application.run()

    assert classifier.observations == []
    assert controller.predictions == [GesturePrediction(GestureLabel.UNKNOWN, 0.0)]
    assert mapper.events == []
    assert dispatcher.actions == []


def test_optional_preview_shows_debounced_status_without_dispatching_after_quit() -> None:
    classifier = _Classifier()
    controller = _Controller()
    mapper = _Mapper()
    dispatcher = _Dispatcher()
    display = _Display(continue_running=False)
    application = HandoffApplication(
        camera=_Camera("hand"),
        perception=_Perception(),
        classifier=classifier,
        controller=controller,
        action_mapper=mapper,
        dispatcher=dispatcher,
        display=display,
    )

    application.run()

    assert display.overlays == ["Gesture: left_click (0.90)"]
    assert mapper.events == []
    assert dispatcher.actions == []
