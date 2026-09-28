"""CLMAnalyst tests — fake CLM client + fake data client, no network."""

import pytest

from v2.clm import CLMError
from v2.clm.cache import CLMCache
from v2.data.client import FDClientError
from v2.data.models import FinancialMetrics
from v2.models import Signal
from v2.signals import CLMAnalyst


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeCLMClient:
    """Canned probability distribution; counts calls; can raise instead."""

    model = "clm-latest"

    def __init__(self, probabilities=None, error=None):
        self._probabilities = probabilities or {
            "appreciate": 0.5, "flat": 0.2, "depreciate": 0.3,
        }
        self._error = error
        self.calls = 0

    def system_one(self, state, questions):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return {"model": self.model, "answers": {"direction": {
            "type": "choice",
            "choice": "appreciate",
            "probabilities": self._probabilities,
        }}}


class MockDataClient:
    def __init__(self, metrics=None, error=None):
        self._metrics = metrics or []
        self._error = error

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        if self._error is not None:
            raise self._error
        return self._metrics

    def get_company_facts(self, ticker):
        return None


def _history(n=8):
    quarters = ["2024-12-31", "2024-09-30", "2024-06-30", "2024-03-31",
                "2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31"]
    return [
        FinancialMetrics(
            ticker="TEST", report_period=q, period="ttm", filing_date=q,
            return_on_equity=0.2, gross_margin=0.4, book_value_per_share=10.0,
            market_cap=1e9,
        )
        for q in quarters[:n]
    ]


def _analyst(tmp_path, client):
    return CLMAnalyst(client=client, cache=CLMCache(tmp_path / "clm"))


# ---------------------------------------------------------------------------
# Signal folding
# ---------------------------------------------------------------------------

def test_value_is_up_minus_down(tmp_path):
    client = FakeCLMClient(probabilities={
        "appreciate": 0.7, "flat": 0.1, "depreciate": 0.2,
    })
    sig = _analyst(tmp_path, client).predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))

    assert isinstance(sig, Signal)
    assert sig.model_name == "clm"
    assert sig.value == pytest.approx(0.5)  # 0.7 - 0.2
    assert sig.metadata["abstained"] is False
    assert sig.metadata["probabilities"] == {
        "appreciate": 0.7, "flat": 0.1, "depreciate": 0.2,
    }
    assert "70%" in sig.reasoning


def test_negative_conviction(tmp_path):
    client = FakeCLMClient(probabilities={
        "appreciate": 0.2, "flat": 0.1, "depreciate": 0.7,
    })
    sig = _analyst(tmp_path, client).predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))
    assert sig.value == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_unchanged_snapshot_never_pays_twice(tmp_path):
    client = FakeCLMClient()
    analyst = _analyst(tmp_path, client)
    data = MockDataClient(metrics=_history())

    first = analyst.predict("TEST", "2025-01-15", data)
    second = analyst.predict("TEST", "2025-02-15", data)  # same snapshot, later date

    assert client.calls == 1
    assert first.metadata["cached"] is False
    assert second.metadata["cached"] is True
    assert second.value == pytest.approx(first.value)
    assert second.metadata["prompt_key"] == first.metadata["prompt_key"]


# ---------------------------------------------------------------------------
# Failure contract
# ---------------------------------------------------------------------------

def test_clm_failure_abstains(tmp_path):
    client = FakeCLMClient(error=CLMError("service down"))
    sig = _analyst(tmp_path, client).predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))
    assert sig.metadata["abstained"] is True
    assert "clm call failed" in sig.metadata["abstain_reason"]
    assert sig.value == 0.0


def test_contract_violation_abstains(tmp_path):
    client = FakeCLMClient(probabilities={"appreciate": 0.6, "flat": 0.4})  # missing 'depreciate'
    sig = _analyst(tmp_path, client).predict("TEST", "2025-01-15", MockDataClient(metrics=_history()))
    assert sig.metadata["abstained"] is True
    assert "missing options" in sig.metadata["abstain_reason"]


def test_insufficient_data_abstains(tmp_path):
    sig = _analyst(tmp_path, FakeCLMClient()).predict(
        "TEST", "2025-01-15", MockDataClient(metrics=[]),
    )
    assert sig.metadata["abstained"] is True
    assert "insufficient data" in sig.metadata["abstain_reason"]


def test_data_error_propagates(tmp_path):
    """Fail loud: a broken data layer must not become a quiet neutral view."""
    data = MockDataClient(error=FDClientError("API down"))
    with pytest.raises(FDClientError):
        _analyst(tmp_path, FakeCLMClient()).predict("TEST", "2025-01-15", data)
