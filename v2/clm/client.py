"""CLM client — the local CLM-8B System One scoring service over HTTP.

CLM (Contrastive-LM's CLM-v0.1-8B) scores candidate actions against a state
instead of generating text: it returns a probability distribution over the
closed option set you hand it. This client talks to a local `clm-serve`
(default http://127.0.0.1:8700) over its TypeSafe-compatible API:

    POST /v1/systemone  {"state", "model", "questions"}  ->  {"answers", ...}

The service itself is installed outside this repo (see ~/clm); only the
HTTP contract is pinned here, so any compatible server — local or remote,
via CLM_BASE_URL — works unchanged.

Failure semantics mirror the LLM providers (v2/llm/client.py): this client
RAISES on transport/HTTP/parse failure — the analyst layer above decides
whether that becomes an abstain, not the client.
"""

from __future__ import annotations

import os

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8700"
DEFAULT_MODEL = "clm-latest"


class CLMError(RuntimeError):
    """CLM scoring failed: unreachable service, HTTP error, or unusable
    response. Distinct from "the model abstained" — that is a Signal fact."""


class CLMClient:
    """Thin HTTP client for one clm-serve endpoint.

    Usage::

        client = CLMClient()
        payload = client.system_one(state, {"direction": {...}})
        payload["answers"]["direction"]["probabilities"]
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("CLM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.environ.get("CLM_MODEL") or DEFAULT_MODEL
        self._timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def system_one(self, state: str, questions: dict) -> dict:
        """One scoring call: every question answered against one state.

        Returns the parsed response body ({"model", "answers", "usage"}).
        Raises CLMError on any failure — never returns a partial answer.
        """
        body = {"state": state, "model": self.model, "questions": questions}
        try:
            response = self._session.post(
                f"{self.base_url}/v1/systemone", json=body, timeout=self._timeout
            )
        except requests.RequestException as exc:
            raise CLMError(f"CLM unreachable at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            raise CLMError(
                f"CLM error {response.status_code}: {response.text[:300]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise CLMError(f"CLM returned non-JSON: {response.text[:200]!r}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
            raise CLMError(f"CLM response missing 'answers': {str(payload)[:200]}")
        return payload

    def health(self) -> bool:
        """True if the service responds healthy on /health."""
        try:
            response = self._session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200 and bool(response.json().get("ok"))
        except (requests.RequestException, ValueError):
            return False


def choice_probabilities(answer: dict) -> dict[str, float]:
    """Extract the distribution from a choice answer.

    The wire shape is
    ``{"type": "choice", "choice": ..., "probabilities": {option: p}}``;
    anything else is a contract violation and raises CLMError.
    """
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise CLMError(f"expected a choice answer, got {str(answer)[:200]}")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        raise CLMError(f"choice answer has no probabilities: {str(answer)[:200]}")
    out = {str(k): float(v) for k, v in probabilities.items()}
    if sum(out.values()) <= 0:
        raise CLMError(f"choice probabilities sum to zero: {out}")
    return out
