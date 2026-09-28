"""v2 CLM layer — local CLM-8B System One scoring over HTTP + disk cache.

CLM scores candidate actions against a state and returns probabilities
(no text generation). The analyst that consumes this layer lives in
v2/signals/clm.py; the Kelly blend that consumes its probabilities lives
in v2/portfolio/kelly.py.
"""

from v2.clm.cache import CLMCache, clm_key
from v2.clm.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    CLMClient,
    CLMError,
    choice_probabilities,
)

__all__ = [
    "CLMCache",
    "CLMClient",
    "CLMError",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "choice_probabilities",
    "clm_key",
]
