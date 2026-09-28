"""run_cycle Kelly dispatch tests — CLM-style probabilities through the whole pipeline."""

import pytest

from v2.brokers.sim import SimBroker
from v2.data.models import Price
from v2.fund.spec import Fund, FundSpec
from v2.models import Signal
from v2.pipeline.run_cycle import run_cycle


class FakeDataClient:
    """Canned closes; a ticker absent from `closes` has no bars."""

    def __init__(self, closes):
        self._closes = closes

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        close = self._closes.get(ticker)
        if close is None:
            return []
        return [Price(open=close, close=close, high=close, low=close,
                      volume=1000, time=f"{end_date}T00:00:00Z")]


class FakeProbAnalyst:
    """Fixed probabilities per ticker — the metadata shape CLMAnalyst emits."""

    def __init__(self, name="clm", probs=None):
        self._name = name
        self._probs = probs or {}

    @property
    def name(self):
        return self._name

    def predict(self, ticker, date, data_client):
        p = self._probs.get(
            ticker, {"appreciate": 1 / 3, "flat": 1 / 3, "depreciate": 1 / 3}
        )
        return Signal(
            model_name=self._name,
            ticker=ticker,
            date=date,
            value=p["appreciate"] - p["depreciate"],
            metadata={"probabilities": p, "abstained": False},
        )


def _spec(universe, max_position_pct=1.0, blend=None):
    return FundSpec(
        name="kelly-fund",
        universe=universe,
        strategies=[{"name": "s", "models": [{"name": "clm"}],
                     "blend": blend or {"method": "kelly", "buffer": 0.2}}],
        risk={"max_position_pct": max_position_pct, "max_gross_exposure": 1.0},
        capital=100_000.0,
    )


def test_kelly_blend_dispatches_with_buffer():
    probs = {"appreciate": 0.7, "flat": 0.1, "depreciate": 0.2}
    spec = _spec(["AAPL"])
    fund = Fund(spec, models={"s": [FakeProbAnalyst(probs={"AAPL": probs})]})

    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient({"AAPL": 200.0}))

    expected = ((0.7 - 0.2) / 0.9) * 0.8  # f* x (1 - buffer)
    assert record.strategies[0].weights["AAPL"] == pytest.approx(expected)
    assert record.final_weights["AAPL"] == pytest.approx(expected)
    assert record.positions["AAPL"] == int(expected * 100_000 / 200)  # floor sizing


def test_kelly_weight_is_still_risk_clamped():
    probs = {"appreciate": 0.9, "flat": 0.05, "depreciate": 0.05}
    spec = _spec(["AAPL"], max_position_pct=0.25)
    fund = Fund(spec, models={"s": [FakeProbAnalyst(probs={"AAPL": probs})]})

    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient({"AAPL": 200.0}))

    f = ((0.9 - 0.05) / 0.95) * 0.8  # ≈ 0.7158, well above the 25% cap
    assert record.target_weights["AAPL"] == pytest.approx(f)
    assert record.final_weights["AAPL"] == pytest.approx(0.25)
    assert [c.limit for c in record.clamps] == ["max_position_pct"]


def test_kelly_vs_conviction_diverge_on_same_signals():
    """Same probabilities: conviction spreads full gross; Kelly sizes to edge."""
    probs = {"AAPL": {"appreciate": 0.7, "flat": 0.1, "depreciate": 0.2},
             "MSFT": {"appreciate": 0.4, "flat": 0.1, "depreciate": 0.5}}

    def run(blend):
        fund = Fund(_spec(["AAPL", "MSFT"], blend=blend),
                    models={"s": [FakeProbAnalyst(probs=probs)]})
        return run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                         FakeDataClient({"AAPL": 200.0, "MSFT": 400.0}))

    kelly = run({"method": "kelly", "buffer": 0.2})
    conviction = run({"method": "conviction_weighted", "gross_target": 1.0})

    # Conviction book deploys the full gross target; Kelly leaves the buffer + no-edge names in cash.
    assert sum(abs(w) for w in conviction.final_weights.values()) == pytest.approx(1.0)
    assert sum(abs(w) for w in kelly.final_weights.values()) < 1.0
    # Both agree on direction.
    for rec in (kelly, conviction):
        assert rec.final_weights["AAPL"] > 0 > rec.final_weights["MSFT"]
