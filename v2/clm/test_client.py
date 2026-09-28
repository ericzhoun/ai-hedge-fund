"""CLMClient + CLMCache tests — stubbed HTTP session, no network."""

import json as _json

import pytest
import requests

from v2.clm import CLMClient, CLMError, choice_probabilities
from v2.clm.cache import CLMCache, clm_key


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        # Mirror requests: raises ValueError on a non-JSON body.
        if self._payload is not None:
            return self._payload
        return _json.loads(self.text)


@pytest.fixture
def client():
    return CLMClient(base_url="http://127.0.0.1:8700", model="clm-latest")


def _stub_post(client, responses):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    client._session.post = fake_post
    return calls


def _answers_payload():
    return {"model": "clm-latest", "answers": {"direction": {
        "type": "choice", "choice": "appreciate",
        "probabilities": {"appreciate": 0.5, "flat": 0.2, "depreciate": 0.3},
    }}, "usage": {"input_tokens": 10}}


# ---------------------------------------------------------------------------
# Scoring calls
# ---------------------------------------------------------------------------

def test_system_one_happy_path(client):
    calls = _stub_post(client, [_FakeResponse(200, _answers_payload())])
    payload = client.system_one("state text", {"direction": {"type": "choice"}})

    assert payload["answers"]["direction"]["type"] == "choice"
    assert len(calls) == 1
    assert calls[0]["url"] == "http://127.0.0.1:8700/v1/systemone"
    assert calls[0]["json"]["state"] == "state text"
    assert calls[0]["json"]["model"] == "clm-latest"


def test_http_error_raises(client):
    _stub_post(client, [_FakeResponse(500, text="internal error")])
    with pytest.raises(CLMError, match="500"):
        client.system_one("s", {})


def test_connection_error_raises(client):
    _stub_post(client, [requests.ConnectionError("boom")])
    with pytest.raises(CLMError, match="unreachable"):
        client.system_one("s", {})


def test_non_json_raises(client):
    _stub_post(client, [_FakeResponse(200, text="<html>")])
    with pytest.raises(CLMError, match="non-JSON"):
        client.system_one("s", {})


def test_missing_answers_raises(client):
    _stub_post(client, [_FakeResponse(200, {"model": "clm-latest"})])
    with pytest.raises(CLMError, match="missing 'answers'"):
        client.system_one("s", {})


def test_health(client):
    def fake_get(url, timeout=None):
        return _FakeResponse(200, {"ok": True})

    client._session.get = fake_get
    assert client.health() is True

    client._session.get = lambda url, timeout=None: _FakeResponse(503)
    assert client.health() is False


# ---------------------------------------------------------------------------
# choice_probabilities
# ---------------------------------------------------------------------------

def test_choice_probabilities_parse():
    probs = choice_probabilities({
        "type": "choice", "choice": "a",
        "probabilities": {"a": 0.6, "b": 0.4},
    })
    assert probs == {"a": 0.6, "b": 0.4}


def test_choice_probabilities_wrong_type_raises():
    with pytest.raises(CLMError, match="choice"):
        choice_probabilities({"type": "noul", "noul": 0.6})


def test_choice_probabilities_empty_raises():
    with pytest.raises(CLMError, match="no probabilities"):
        choice_probabilities({"type": "choice", "probabilities": {}})


def test_choice_probabilities_zero_sum_raises():
    with pytest.raises(CLMError, match="zero"):
        choice_probabilities({
            "type": "choice", "probabilities": {"a": 0.0, "b": 0.0},
        })


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_round_trip(tmp_path):
    cache = CLMCache(tmp_path / "clm")
    key = clm_key("clm-latest", "state", "{}")
    assert cache.get(key) is None

    cache.put(key, {"probabilities": {"appreciate": 0.5, "flat": 0.2, "depreciate": 0.3}})
    hit = cache.get(key)
    assert hit["probabilities"]["appreciate"] == 0.5
    assert "created_at" in hit


def test_cache_key_stable_and_scoped():
    assert clm_key("m", "s", "q") == clm_key("m", "s", "q")
    assert clm_key("m", "s", "q") != clm_key("m", "s", "q2")
    assert clm_key("m", "s", "q") != clm_key("m2", "s", "q")


def test_cache_corrupt_entry_is_a_miss(tmp_path):
    cache = CLMCache(tmp_path / "clm")
    key = clm_key("m", "s", "q")
    (tmp_path / "clm").mkdir()
    (tmp_path / "clm" / f"{key}.json").write_text("{ not json")
    assert cache.get(key) is None
