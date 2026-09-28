"""CLM analyst — the local CLM-8B System One model as an alpha model.

CLM scores candidate actions against a state instead of generating text
(CLM-v0.1-8B: a frozen Qwen3-8B encoder + trained projection heads). The
state here is the same point-in-time fundamentals snapshot the LLM
personas reason over; the candidate set is a closed, typed question about
the next quarter's price path:

    appreciate | flat | depreciate

The softmax over CLM's scores IS the answer distribution. Folded into the
Signal contract:

    value = p(appreciate) - p(depreciate),   in [-1, +1]

and the raw probabilities ride along in `metadata.probabilities`, where the
Kelly blend policy (v2/portfolio/kelly.py) consumes them for sizing.

Failure contract (mirrors LLMAgent — locked decisions):
- Data-layer errors PROPAGATE (a broken snapshot must never become a quiet
  neutral view).
- CLM call/parse failures ABSTAIN (Signal(value=0.0, abstained=True)): the
  scoring service is infrastructure like an LLM provider, and one
  unreachable local server must not crash an entire backtest. The record
  keeps the reason.
- Every decision persists its exact state + question + answer distribution
  (CLMCache); an unchanged snapshot never pays a second call.
"""

from __future__ import annotations

import json
import logging

from v2.clm import CLMCache, CLMClient, CLMError, choice_probabilities, clm_key
from v2.data.protocol import DataClient
from v2.features.snapshot import FundamentalsSnapshot, InsufficientData, build_snapshot
from v2.models import Signal
from v2.signals.base import AlphaModel

logger = logging.getLogger(__name__)

DIRECTION_QUESTION_ID = "direction"

# The closed option set the direction question offers; answers missing any
# of these are a contract violation (config drift), not a smaller answer.
_EXPECTED_OPTIONS = {"appreciate", "flat", "depreciate"}


def direction_question() -> dict:
    """The typed question CLM answers for every ticker: next-quarter path."""
    return {
        "type": "choice",
        "instructions": (
            "Over the next quarter (about three months), which price path is "
            "most likely for this stock, given only the point-in-time "
            "fundamentals above?"
        ),
        "criteria": {
            "appreciate": "The stock is more likely to appreciate (end the quarter net up)",
            "flat": "The stock drifts sideways; neither direction clearly dominates",
            "depreciate": "The stock is more likely to depreciate (end the quarter net down)",
        },
    }


class CLMAnalyst(AlphaModel):
    """Point-in-time fundamentals -> CLM direction probabilities -> Signal."""

    def __init__(
        self,
        client: CLMClient | None = None,
        cache: CLMCache | None = None,
    ) -> None:
        self._client = client if client is not None else CLMClient()
        self._cache = cache if cache is not None else CLMCache()

    @property
    def name(self) -> str:
        return "clm"

    # ------------------------------------------------------------------
    # AlphaModel interface
    # ------------------------------------------------------------------

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        try:
            snapshot = self.build_snapshot(ticker, date, data_client)
        except InsufficientData as exc:
            return self._abstain(ticker, date, f"insufficient data: {exc}")
        # Any other data-layer exception (e.g. FDClientError) propagates.

        state = snapshot.render()
        questions = {DIRECTION_QUESTION_ID: direction_question()}
        key = clm_key(self._client.model, state, json.dumps(questions, sort_keys=True))

        cached = self._cache.get(key)
        if cached is not None and "probabilities" in cached:
            return self._to_signal(
                ticker, date, cached["probabilities"], key, snapshot, cached=True
            )

        try:
            payload = self._client.system_one(state, questions)
            probabilities = choice_probabilities(payload["answers"][DIRECTION_QUESTION_ID])
            _require_direction_options(probabilities)
        except (CLMError, KeyError) as exc:
            logger.warning("clm call failed for %s@%s: %s", ticker, date, exc)
            return self._abstain(ticker, date, f"clm call failed: {exc}")

        self._cache.put(key, {
            "agent": self.name,
            "model": self._client.model,
            "ticker": ticker,
            "as_of": date,
            "snapshot_hash": snapshot.content_hash,
            "state": state,
            "questions": questions,
            "probabilities": probabilities,
        })
        return self._to_signal(ticker, date, probabilities, key, snapshot, cached=False)

    # ------------------------------------------------------------------
    # Subclass surface (same hook as LLMAgent, for symmetry)
    # ------------------------------------------------------------------

    def build_snapshot(self, ticker: str, date: str, data_client: DataClient) -> FundamentalsSnapshot:
        """What the scorer is allowed to see: the shared point-in-time
        fundamentals snapshot."""
        return build_snapshot(ticker, date, data_client)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_signal(
        self,
        ticker: str,
        date: str,
        probabilities: dict[str, float],
        key: str,
        snapshot: FundamentalsSnapshot,
        cached: bool,
    ) -> Signal:
        p_up = probabilities.get("appreciate", 0.0)
        p_flat = probabilities.get("flat", 0.0)
        p_down = probabilities.get("depreciate", 0.0)
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=p_up - p_down,
            reasoning=(
                f"CLM direction: appreciate {p_up:.0%}, flat {p_flat:.0%}, "
                f"depreciate {p_down:.0%} over the next quarter"
            ),
            metadata={
                "probabilities": {
                    "appreciate": p_up,
                    "flat": p_flat,
                    "depreciate": p_down,
                },
                "model": self._client.model,
                "prompt_key": key,
                "snapshot_hash": snapshot.content_hash,
                "cached": cached,
                "abstained": False,
            },
        )

    def _abstain(self, ticker: str, date: str, reason: str) -> Signal:
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=0.0,
            reasoning=f"abstained: {reason}",
            metadata={"abstained": True, "abstain_reason": reason, "cached": False},
        )


def _require_direction_options(probabilities: dict[str, float]) -> None:
    """The direction answer must offer exactly the expected options."""
    missing = _EXPECTED_OPTIONS - set(probabilities)
    if missing:
        raise CLMError(f"direction answer missing options: {sorted(missing)}")
