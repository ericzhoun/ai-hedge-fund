"""Kelly sizing — probability-scaled weights with an explicit safety buffer.

Where `blend_signals` normalizes convictions into a sleeve ("capital flows
proportionally to blended conviction"), the Kelly blend sizes each name
from its *probability of winning* — the way a bettor with an edge sizes a
book:

    f* = p - (1 - p) / b          (classical Kelly fraction)

    p = win probability, b = payoff ratio (win per unit risked).
    No edge (f* <= 0) means no position: Kelly refuses coin-flips.

For a directional stock bet the outcome space is three-way (up / flat /
down), and flat maps to a push — stake returned, no P&L. This is exact,
not an approximation: a push contributes log(1) = 0 to expected log
wealth, so it drops out of the maximization and p, q are legitimately
taken CONDITIONAL on the quarter resolving:

    p = p_up / (p_up + p_down),   q = p_down / (p_up + p_down)

With b = 1 that reads `f* = (p_up - p_down) / (p_up + p_down)`.

Where do probabilities come from? A Signal's `metadata.probabilities`
(the CLM analyst emits exactly this shape). Signals without explicit
probabilities fall back to conviction-as-probability, `p = (1 + |c|) / 2`,
which is coarser but keeps the blend usable for any model in the registry.

The **buffer** reserves a fraction of the budget: `weight = f* x (1 - buffer)`
whenever the Kelly stakes fit inside `gross_cap`, and Kelly *proportions*
scaled to `gross_cap x (1 - buffer)` when they do not. Either way the book
never deploys more than 80% of what full Kelly would (default buffer 0.2),
and the remainder sits in cash. Full Kelly is famously too hot for noisy
edge estimates; the reserve is the explicit safety margin, and it bites in
both regimes by construction.

Formally, with f_i the per-name Kelly fractions and C the gross cap:

    weights_i = sign_i x f_i / max(sum(f), C) x C x (1 - buffer)

which is exactly `f_i x (1 - buffer)` when `sum(f) <= C` (the usual case:
a few names, modest edges) and preserves proportions at a buffered budget
when the edge is large enough to want the whole book.

The fund's risk limits remain the master gate: conviction requests, risk
disposes; Kelly requests, risk disposes.
"""

from __future__ import annotations

from v2.models import Signal
from v2.portfolio.construction import BlendResult, blend_signals

# Below this, (p_up + p_down) is numerically all-flat noise -> no bet to size.
_ALL_FLAT_EPS = 1e-9


def kelly_fraction(p_win: float, payoff_ratio: float = 1.0) -> float:
    """The classical Kelly fraction f* = p - (1 - p) / b.

    Returns the fraction of bankroll the bet deserves; <= 0 means no bet.
    """
    return p_win - (1.0 - p_win) / payoff_ratio


def blend_kelly(
    signals: list[Signal],
    model_weights: dict[str, float],
    payoff_ratio: float = 1.0,
    buffer: float = 0.2,
    gross_cap: float | None = None,
) -> BlendResult:
    """Probability-weighted (Kelly) target weights for one strategy sleeve.

    Convictions come from the standard blend (so the audit trail is
    identical to `blend_signals`); the WEIGHTS come from buffered Kelly
    sizing instead of conviction normalization:

        direction_t = sign(blended conviction)
        p_t, q_t    = win / loss probabilities in that direction
        f_t         = max(kelly_fraction(p_t, b), 0)     (no edge, no bet)
        weight_t    = direction_t * f_t * (1 - buffer)

    A ticker whose signals carry `metadata.probabilities` (CLM) uses
    those, aggregated over the models that published them; otherwise the
    fallback `p = (1 + |conviction|) / 2` applies. Abstained signals are
    excluded from both the blend and the probability aggregation.

    Args:
        signals:       Every model's Signal for every ticker this cycle.
        model_weights: model_name -> blend weight from the StrategySpec.
        payoff_ratio:  Win/loss payoff ratio b (win per unit risked).
        buffer:        Budget fraction reserved as a safety margin: the book
                       deploys at most (1 - buffer) of what full Kelly would.
        gross_cap:     Cap on deployment (None = uncapped). The buffered book
                       never exceeds gross_cap, and when Kelly would want the
                       whole cap it deploys only gross_cap x (1 - buffer).
    """
    base = blend_signals(signals, model_weights, gross_target=1.0)
    convictions = base.convictions

    # Contributing (non-abstained) signals per ticker, with their blend weights.
    by_ticker: dict[str, list[tuple[float, Signal]]] = {}
    for signal in signals:
        if signal.metadata.get("abstained") is True:
            continue
        by_ticker.setdefault(signal.ticker, []).append(
            (model_weights[signal.model_name], signal)
        )

    weights: dict[str, float] = {t: 0.0 for t in convictions}
    f_star: dict[str, float] = {}
    for ticker, conviction in convictions.items():
        if conviction == 0.0:
            f_star[ticker] = 0.0
            continue
        long_side = conviction > 0

        pairs = by_ticker.get(ticker, [])
        prob_pairs = [(w, s) for w, s in pairs if "probabilities" in s.metadata]
        if prob_pairs:
            w_total = sum(w for w, _ in prob_pairs)
            p_up = sum(
                w * s.metadata["probabilities"].get("appreciate", 0.0)
                for w, s in prob_pairs
            ) / w_total
            p_down = sum(
                w * s.metadata["probabilities"].get("depreciate", 0.0)
                for w, s in prob_pairs
            ) / w_total
            resolved = p_up + p_down
            if resolved <= _ALL_FLAT_EPS:
                f_star[ticker] = 0.0  # all-flat distribution: no bet to size
                continue
            p_win = (p_up if long_side else p_down) / resolved
        else:
            # Conviction-as-probability fallback: |c| = 1 -> p = 1, c = 0 -> 0.5.
            p_win = (1.0 + abs(conviction)) / 2.0

        f_star[ticker] = max(kelly_fraction(p_win, payoff_ratio), 0.0)

    gross_f = sum(f_star.values())
    if gross_f > _ALL_FLAT_EPS:
        # The buffered budget: what Kelly wants, capped, then reserved by the
        # buffer. Under the cap this is exactly the per-bet fractional-Kelly
        # stakes; at the cap it holds Kelly proportions on (1 - buffer) x cap.
        budget = min(gross_f, gross_cap if gross_cap is not None else gross_f)
        deploy = budget * (1.0 - buffer)
        for ticker, f in f_star.items():
            if f <= 0.0:
                continue
            stake = f / gross_f * deploy
            long_side = convictions[ticker] > 0
            weights[ticker] = 0.0 if stake == 0.0 else stake * (1 if long_side else -1)

    return BlendResult(convictions=convictions, weights=weights)
