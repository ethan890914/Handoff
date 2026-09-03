"""Smoke tests for the initial project scaffold."""

from handoff import __version__
from handoff.domain.models import GestureLabel, GesturePrediction


def test_package_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_domain_contract_can_be_constructed() -> None:
    prediction = GesturePrediction(label=GestureLabel.UNKNOWN, confidence=0.0)
    assert prediction.label is GestureLabel.UNKNOWN
