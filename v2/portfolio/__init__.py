"""v2 portfolio construction — blend analyst views into target weights.

Two blend policies, one contract (BlendResult):

- conviction_weighted — capital flows proportionally to blended conviction;
- kelly               — per-name fractional Kelly off win probabilities,
                          with an explicit safety buffer (default 20%).

Later: mean-variance optimization, Black-Litterman, risk parity.
"""

from v2.portfolio.construction import BlendResult, blend_signals
from v2.portfolio.kelly import blend_kelly, kelly_fraction

__all__ = ["BlendResult", "blend_signals", "blend_kelly", "kelly_fraction"]
