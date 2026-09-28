"""kelly_fraction + blend_kelly tests — pure math, hand-built signals."""

import pytest

from v2.models import Signal
from v2.portfolio.kelly import blend_kelly, kelly_fraction


def _sig(model, ticker, value, probs=None, abstained=False):
    metadata = {"abstained": True} if abstained else {}
    if probs is not None:
        metadata["probabilities"] = probs
    return Signal(model_name=model, ticker=ticker, date="2024-06-03",
                  value=value, metadata=metadata)


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------

def test_kelly_fraction_classic():
    assert kelly_fraction(0.6) == pytest.approx(0.2)   # 0.6 - 0.4
    assert kelly_fraction(0.5) == pytest.approx(0.0)   # fair bet -> no stake
    assert kelly_fraction(0.4) == pytest.approx(-0.2)  # worse than fair -> no bet


@pytest.mark.parametrize("p,b,expected", [
    (0.5, 2.0, 0.25),     # 0.5 - 0.5/2
    (0.5, 3.0, 1 / 3),    # 0.5 - 0.5/3
    (0.6, 2.0, 0.4),      # 0.6 - 0.4/2
])
def test_kelly_fraction_payoff_ratio(p, b, expected):
    assert kelly_fraction(p, payoff_ratio=b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# The blend: probabilities -> buffered Kelly weights
# ---------------------------------------------------------------------------

def test_kelly_blend_sizes_off_probabilities():
    """p_up=0.7, p_down=0.2 -> p=7/9, f=5/9, weight=5/9 x 0.8 (buffer)."""
    sig = _sig("clm", "AAPL", 0.5,
               probs={"appreciate": 0.7, "flat": 0.1, "depreciate": 0.2})
    result = blend_kelly([sig], {"clm": 1.0})
    assert result.weights["AAPL"] == pytest.approx((0.7 - 0.2) / 0.9 * 0.8)


def test_kelly_blend_negative_side():
    sig = _sig("clm", "NVDA", -0.5,
               probs={"appreciate": 0.2, "flat": 0.1, "depreciate": 0.7})
    result = blend_kelly([sig], {"clm": 1.0})
    assert result.weights["NVDA"] == pytest.approx(-(0.7 - 0.2) / 0.9 * 0.8)


def test_default_buffer_is_20_percent():
    sig = _sig("clm", "AAPL", 0.5,
               probs={"appreciate": 0.7, "flat": 0.1, "depreciate": 0.2})
    default = blend_kelly([sig], {"clm": 1.0})
    explicit = blend_kelly([sig], {"clm": 1.0}, buffer=0.2)
    full = blend_kelly([sig], {"clm": 1.0}, buffer=0.0)
    assert default.weights["AAPL"] == pytest.approx(explicit.weights["AAPL"])
    assert default.weights["AAPL"] == pytest.approx(full.weights["AAPL"] * 0.8)


def test_payoff_ratio_scales_stake():
    """b=2: p=7/9 -> f = 7/9 - (2/9)/2 = 2/3."""
    sig = _sig("clm", "AAPL", 0.5,
               probs={"appreciate": 0.7, "flat": 0.1, "depreciate": 0.2})
    result = blend_kelly([sig], {"clm": 1.0}, payoff_ratio=2.0, buffer=0.0)
    assert result.weights["AAPL"] == pytest.approx(2 / 3)


def test_no_edge_no_bet():
    """Positive blended conviction but negative measured edge -> Kelly refuses."""
    s1 = _sig("a", "AAPL", 1.0)  # no probabilities -> fallback path
    s2 = _sig("b", "AAPL", -0.9,
              probs={"appreciate": 0.1, "flat": 0.1, "depreciate": 0.8})
    result = blend_kelly([s1, s2], {"a": 1.0, "b": 1.0})
    assert result.convictions["AAPL"] > 0
    assert result.weights["AAPL"] == 0.0


def test_all_flat_distribution_no_bet():
    s1 = _sig("a", "AAPL", 1.0)  # no probabilities
    s2 = _sig("b", "AAPL", 0.0,
              probs={"appreciate": 0.0, "flat": 1.0, "depreciate": 0.0})
    result = blend_kelly([s1, s2], {"a": 1.0, "b": 1.0})
    assert result.weights["AAPL"] == 0.0


def test_abstained_signals_do_not_size():
    sig = _sig("clm", "AAPL", 0.5,
               probs={"appreciate": 0.7, "flat": 0.1, "depreciate": 0.2},
               abstained=True)
    result = blend_kelly([sig], {"clm": 1.0})
    assert result.convictions["AAPL"] == 0.0
    assert result.weights["AAPL"] == 0.0


def test_fallback_conviction_as_probability():
    """No probabilities in metadata: p = (1 + |c|) / 2."""
    sig = _sig("a", "AAPL", 0.6)
    result = blend_kelly([sig], {"a": 1.0})
    assert result.weights["AAPL"] == pytest.approx(0.6 * 0.8)  # f=0.6, x0.8


def test_probabilities_aggregate_weighted_over_models():
    a = _sig("a", "AAPL", 0.6,
             probs={"appreciate": 0.8, "flat": 0.0, "depreciate": 0.2})
    b = _sig("b", "AAPL", 0.2,
             probs={"appreciate": 0.4, "flat": 0.0, "depreciate": 0.6})
    result = blend_kelly([a, b], {"a": 3.0, "b": 1.0}, buffer=0.0)
    p_up = (3 * 0.8 + 1 * 0.4) / 4  # 0.7
    p_down = (3 * 0.2 + 1 * 0.6) / 4  # 0.3
    assert result.weights["AAPL"] == pytest.approx((p_up - p_down) / (p_up + p_down))


# ---------------------------------------------------------------------------
# The gross cap: buffered reserve semantics
# ---------------------------------------------------------------------------

def test_gross_cap_reserves_buffer_when_kelly_wants_more():
    """Kelly stakes exceed the cap -> Kelly proportions on (1-buffer) x cap."""
    sigs = [
        _sig("clm", t, 0.5,
             probs={"appreciate": 0.7, "flat": 0.1, "depreciate": 0.2})
        for t in ("AAPL", "MSFT")
    ]
    result = blend_kelly(sigs, {"clm": 1.0}, gross_cap=0.5)
    # f* = 5/9 each; gross_f = 10/9 > 0.5 -> deploy 0.5 x 0.8 = 0.4
    assert sum(abs(w) for w in result.weights.values()) == pytest.approx(0.4)
    assert result.weights["AAPL"] == pytest.approx(result.weights["MSFT"])


def test_gross_cap_never_levers_up():
    sig = _sig("clm", "AAPL", 0.5,
               probs={"appreciate": 0.7, "flat": 0.1, "depreciate": 0.2})
    result = blend_kelly([sig], {"clm": 1.0}, gross_cap=5.0)
    assert result.weights["AAPL"] == pytest.approx((0.7 - 0.2) / 0.9 * 0.8)


def test_buffer_always_reduces_deployment():
    """The buffered book is exactly (1-buffer) x the unbuffered book, in both regimes."""
    probs = {"appreciate": 0.7, "flat": 0.1, "depreciate": 0.2}
    sigs = [_sig("clm", t, 0.5, probs=probs) for t in ("AAPL", "MSFT", "NVDA")]

    for cap in (None, 1.0, 0.5):
        full = blend_kelly(sigs, {"clm": 1.0}, buffer=0.0, gross_cap=cap)
        buffered = blend_kelly(sigs, {"clm": 1.0}, buffer=0.2, gross_cap=cap)
        gross_full = sum(abs(w) for w in full.weights.values())
        gross_buf = sum(abs(w) for w in buffered.weights.values())
        assert gross_buf == pytest.approx(gross_full * 0.8), f"cap={cap}"
        # Proportions are preserved (Kelly decides relative sizes either way)
        for t in full.weights:
            share_full = full.weights[t] / gross_full
            share_buf = buffered.weights[t] / gross_buf
            assert share_buf == pytest.approx(share_full), f"cap={cap}, {t}"
