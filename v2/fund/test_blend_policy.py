"""BlendPolicy Kelly-field validation tests — method scoping and round-trips."""

import pytest
from pydantic import ValidationError

from v2.fund.spec import BlendPolicy


def test_kelly_policy_accepts_buffer():
    policy = BlendPolicy(method="kelly", payoff_ratio=1.5, buffer=0.25)
    assert policy.buffer == 0.25
    assert policy.payoff_ratio == 1.5


def test_kelly_defaults():
    policy = BlendPolicy(method="kelly")
    assert policy.buffer == 0.2  # the standard 20% safety margin
    assert policy.payoff_ratio == 1.0
    assert policy.market_neutral is False


def test_kelly_rejects_market_neutral():
    with pytest.raises(ValidationError, match="market_neutral"):
        BlendPolicy(method="kelly", market_neutral=True)


def test_conviction_rejects_custom_kelly_values():
    with pytest.raises(ValidationError, match="kelly"):
        BlendPolicy(method="conviction_weighted", buffer=0.3)
    with pytest.raises(ValidationError, match="kelly"):
        BlendPolicy(method="conviction_weighted", payoff_ratio=2.0)


def test_defaults_survive_dump_round_trip():
    """Spec dumps and CycleRecord round-trips carry every default field."""
    conviction = BlendPolicy(method="conviction_weighted")
    assert BlendPolicy(**conviction.model_dump()) == conviction
    kelly = BlendPolicy(method="kelly")
    assert BlendPolicy(**kelly.model_dump()) == kelly


def test_buffer_bounds():
    with pytest.raises(ValidationError):
        BlendPolicy(method="kelly", buffer=1.0)
    with pytest.raises(ValidationError):
        BlendPolicy(method="kelly", buffer=-0.1)
