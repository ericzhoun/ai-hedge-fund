"""CLM scoring cache — one JSON file per CLM scoring decision.

Mirrors the LLM prompt-cache contract (v2/llm/cache.py), for the same three
reasons:

1. a cache: re-running the analyst over an unchanged snapshot costs one
   local forward pass, not a second scoring round-trip;
2. the persistence record: the exact state + question + answer
   distribution behind every Signal, for replay and audit;
3. the debug trail: the raw response stays on disk even when it fails to
   parse.

Files live under .v2_cache/clm/ (gitignored), keyed by a hash of
(model, state, question set). The as-of date is deliberately not part of
the key: the snapshot it derives from already excludes dates between
filings, and identical data must share one entry, not pay twice.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CACHE_DIR = Path(".v2_cache/clm")


def clm_key(model: str, state: str, questions_json: str) -> str:
    """Cache key for one (model, state, question set) combination."""
    payload = f"clm|{model}|{state}|{questions_json}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class CLMCache:
    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> None:
        self._dir = Path(cache_dir)

    def get(self, key: str) -> dict | None:
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None  # corrupt cache entry -> treat as miss, will be rewritten

    def put(self, key: str, record: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        record = {**record, "created_at": datetime.now(timezone.utc).isoformat()}
        path = self._dir / f"{key}.json"
        path.write_text(json.dumps(record, indent=2))
