import sys, os, json
sys.path.insert(0, '/Users/ericz/Documents/GitHub/ai-hedge-fund')
from src.main import run_hedge_fund
portfolio = {
    "cash": 100000.0,
    "margin_requirement": 0.0,
    "margin_used": 0.0,
    "positions": {"TSLA": {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0}},
    "realized_gains": {"TSLA": {"long": 0.0, "short": 0.0}},
}
result = run_hedge_fund(
    tickers=["TSLA"],
    start_date="2026-02-01",
    end_date="2026-07-01",
    portfolio=portfolio,
    show_reasoning=False,
    selected_analysts=["day_swing_trader"],
    model_name="claude-opus-4-8",
    model_provider="Anthropic"
)
print(json.dumps(result["analyst_signals"]["day_swing_trader_agent"], indent=2))
