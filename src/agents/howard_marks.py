"""Howard Marks analyst agent.

Implements Howard Marks' investment philosophy as described in his Oaktree Capital
memos and books ("The Most Important Thing", "Mastering the Market Cycle").

Core tenets encoded here:
  1. Risk control, not risk avoidance — asymmetric outcomes are the goal.
  2. Second-level thinking — disagree with consensus AND be right.
  3. Market cycle awareness — where are we in the cycle?
  4. Buy below intrinsic value — price matters more than quality alone.
  5. Contrarian positioning — be fearful when others are greedy.
  6. Defensive investing — the best returns come from limiting losses.
  7. Patient opportunism — wait for fat pitches.
  8. Knowing what you don't know — intellectual humility.
"""

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, get_prices, prices_to_df, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
import numpy as np
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.utils.api_key import get_api_key_from_state


class HowardMarksSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def howard_marks_agent(state: AgentState, agent_id: str = "howard_marks_agent"):
    """Analyzes stocks through Howard Marks' lens of risk control, cycles, and second-level thinking."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    interval = data.get("interval", "day")
    interval_multiplier = data.get("interval_multiplier", 1)

    marks_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "net_income",
                "operating_margin",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "shareholders_equity",
                "outstanding_shares",
                "debt_to_equity",
                "capital_expenditure",
                "depreciation_and_amortization",
                "dividends_and_other_cash_distributions",
                "gross_profit",
            ],
            end_date,
            period="annual",
            limit=10,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Fetching price data for cycle analysis")
        prices = get_prices(ticker, end_date[:4] + "-01-01", end_date, interval=interval, interval_multiplier=interval_multiplier, api_key=api_key)

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing risk-return profile")
        risk_analysis = analyze_risk_return(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Assessing market cycle position")
        cycle_analysis = analyze_market_cycle(prices, metrics)

        progress.update_status(agent_id, ticker, "Evaluating intrinsic value & margin of safety")
        value_analysis = analyze_value_vs_price(financial_line_items, market_cap)

        progress.update_status(agent_id, ticker, "Measuring contrarian indicators")
        contrarian_analysis = analyze_contrarian_indicators(prices, metrics)

        progress.update_status(agent_id, ticker, "Assessing defensive quality")
        defensive_analysis = analyze_defensive_quality(metrics, financial_line_items)

        # Aggregate analysis
        total_score = (
            risk_analysis["score"]
            + cycle_analysis["score"]
            + value_analysis["score"]
            + contrarian_analysis["score"]
            + defensive_analysis["score"]
        )
        max_score = 25  # 5 from each of 5 analyses

        analysis_bundle = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_score,
            "risk_analysis": risk_analysis,
            "cycle_analysis": cycle_analysis,
            "value_analysis": value_analysis,
            "contrarian_analysis": contrarian_analysis,
            "defensive_analysis": defensive_analysis,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating Howard Marks analysis")
        output = generate_marks_output(
            ticker=ticker,
            analysis_data=analysis_bundle,
            state=state,
            agent_id=agent_id,
        )

        marks_analysis[ticker] = {
            "signal": output.signal,
            "confidence": output.confidence,
            "reasoning": output.reasoning,
        }
        progress.update_status(agent_id, ticker, "Done", analysis=output.reasoning)

    message = HumanMessage(content=json.dumps(marks_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(marks_analysis, "Howard Marks Agent")

    state["data"]["analyst_signals"][agent_id] = marks_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


# ---------------------------------------------------------------------------
# Sub-analyses
# ---------------------------------------------------------------------------

def analyze_risk_return(metrics: list, financial_line_items: list) -> dict:
    """Marks says: 'Risk control is the most important thing.'

    Evaluates earnings stability, leverage prudence, and downside protection.
    """
    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data for risk-return analysis"}

    score = 0
    reasoning = []

    # 1. Earnings volatility — Marks wants consistency, not fireworks
    net_incomes = [item.net_income for item in financial_line_items if item.net_income is not None]
    if len(net_incomes) >= 4:
        avg = sum(net_incomes) / len(net_incomes)
        if avg > 0:
            variance = sum((x - avg) ** 2 for x in net_incomes) / len(net_incomes)
            cv = (variance ** 0.5) / abs(avg)  # coefficient of variation
            if cv < 0.3:
                score += 2
                reasoning.append(f"Low earnings volatility (CV={cv:.2f}) — Marks values predictable cash flows")
            elif cv < 0.6:
                score += 1
                reasoning.append(f"Moderate earnings volatility (CV={cv:.2f})")
            else:
                reasoning.append(f"High earnings volatility (CV={cv:.2f}) — risky for Marks' framework")
        else:
            reasoning.append("Average net income is negative — high risk")
    else:
        reasoning.append("Insufficient earnings history for volatility analysis")

    # 2. Leverage — Marks is cautious about debt
    debt_ratios = [item.debt_to_equity for item in financial_line_items if item.debt_to_equity is not None]
    if debt_ratios:
        avg_de = sum(debt_ratios) / len(debt_ratios)
        if avg_de < 0.5:
            score += 2
            reasoning.append(f"Conservative leverage (avg D/E={avg_de:.2f}) — aligns with defensive investing")
        elif avg_de < 1.0:
            score += 1
            reasoning.append(f"Moderate leverage (avg D/E={avg_de:.2f})")
        else:
            reasoning.append(f"High leverage (avg D/E={avg_de:.2f}) — Marks would flag fragility risk")
    else:
        # Fallback to liabilities/assets
        ratios = []
        for item in financial_line_items:
            if item.total_liabilities and item.total_assets and item.total_assets > 0:
                ratios.append(item.total_liabilities / item.total_assets)
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            if avg_ratio < 0.5:
                score += 1
                reasoning.append(f"Moderate liabilities-to-assets ratio ({avg_ratio:.2f})")
            else:
                reasoning.append(f"High liabilities-to-assets ({avg_ratio:.2f})")
        else:
            reasoning.append("No leverage data available")

    # 3. Free cash flow positivity — asymmetric upside
    fcf_values = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    if fcf_values:
        positive_pct = sum(1 for f in fcf_values if f > 0) / len(fcf_values)
        if positive_pct >= 0.8:
            score += 1
            reasoning.append(f"FCF positive {positive_pct:.0%} of periods — strong cash generation")
        else:
            reasoning.append(f"FCF positive only {positive_pct:.0%} of periods")
    else:
        reasoning.append("No FCF data")

    return {"score": min(score, 5), "details": "; ".join(reasoning)}


def analyze_market_cycle(prices: list, metrics: list) -> dict:
    """Marks' cycle framework: 'We may never know where we're going, but we'd better know where we are.'

    Uses price momentum, valuation spread, and mean-reversion signals.
    """
    score = 0
    reasoning = []

    if not prices:
        return {"score": 0, "details": "No price data for cycle analysis"}

    import pandas as pd
    df = prices_to_df(prices)
    if df.empty or len(df) < 20:
        return {"score": 0, "details": "Insufficient price history for cycle analysis"}

    closes = df["close"].to_numpy()
    current = float(closes[-1])

    # 1. Distance from 52-week (or available) moving average
    lookback = min(252, len(closes))
    ma = float(np.mean(closes[-lookback:]))
    deviation = (current - ma) / ma

    if deviation > 0.30:
        reasoning.append(f"Price is {deviation:.0%} above long-term avg — late cycle / euphoria territory")
        # Late cycle = bearish from Marks' perspective (others are greedy)
    elif deviation > 0.10:
        score += 1
        reasoning.append(f"Price {deviation:.0%} above avg — mid-cycle, moderate optimism")
    elif deviation > -0.10:
        score += 2
        reasoning.append(f"Price near long-term avg ({deviation:+.0%}) — equilibrium zone")
    elif deviation > -0.25:
        score += 3
        reasoning.append(f"Price {deviation:.0%} below avg — potential value zone (fear)")
    else:
        score += 4
        reasoning.append(f"Price {deviation:.0%} below avg — deep value / panic territory (Marks would be very interested)")

    # 2. Recent volatility as a cycle indicator
    if len(closes) >= 30:
        returns = np.diff(np.log(closes[-30:]))
        vol_30d = float(np.std(returns) * np.sqrt(252))
        if vol_30d > 0.40:
            score += 1
            reasoning.append(f"High 30-day annualized vol ({vol_30d:.0%}) — market stress may create opportunities")
        elif vol_30d < 0.15:
            reasoning.append(f"Low vol ({vol_30d:.0%}) — possible complacency")
        else:
            reasoning.append(f"Normal vol ({vol_30d:.0%})")

    # 3. P/E context from metrics (if available)
    if metrics:
        pe = getattr(metrics[0], 'price_to_earnings_ratio', None)
        if pe is not None:
            if pe < 12:
                score += 1
                reasoning.append(f"Low P/E ({pe:.1f}) suggests market pessimism — contrarian opportunity")
            elif pe > 30:
                reasoning.append(f"High P/E ({pe:.1f}) — Marks would caution about excessive optimism")
            else:
                reasoning.append(f"P/E of {pe:.1f} is in normal range")

    return {"score": min(score, 5), "details": "; ".join(reasoning)}


def analyze_value_vs_price(financial_line_items: list, market_cap: float) -> dict:
    """Marks: 'The relationship between price and value holds the key to success in investing.'

    Conservative DCF + margin of safety analysis.
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data for value analysis", "margin_of_safety": None}

    score = 0
    details = []

    # Use free cash flow for DCF
    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0

    if fcf <= 0:
        return {"score": 0, "details": f"No positive FCF (={fcf}); cannot compute intrinsic value", "margin_of_safety": None}

    # Conservative Marks-style DCF: low growth, high discount rate
    growth_rate = 0.04  # Conservative 4% — Marks doesn't chase growth
    discount_rate = 0.12  # Higher discount rate — margin of safety built in
    terminal_multiple = 10  # Conservative exit multiple
    projection_years = 5

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value

    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    if margin_of_safety > 0.40:
        score += 5
        details.append(f"Deep margin of safety ({margin_of_safety:.0%}) — Marks would call this 'a fat pitch'")
    elif margin_of_safety > 0.20:
        score += 3
        details.append(f"Good margin of safety ({margin_of_safety:.0%}) — price below conservative IV")
    elif margin_of_safety > 0:
        score += 1
        details.append(f"Slim margin of safety ({margin_of_safety:.0%}) — some cushion")
    elif margin_of_safety > -0.20:
        details.append(f"Near fair value (MoS={margin_of_safety:.0%}) — no compelling discount")
    else:
        details.append(f"Trading above intrinsic value (MoS={margin_of_safety:.0%}) — Marks would wait for a better price")

    details.append(f"Conservative IV: ${intrinsic_value:,.0f} vs Market Cap: ${market_cap:,.0f}")

    return {"score": min(score, 5), "details": "; ".join(details), "margin_of_safety": margin_of_safety}


def analyze_contrarian_indicators(prices: list, metrics: list) -> dict:
    """Marks: 'To achieve superior results, you have to hold non-consensus views that turn out to be right.'

    Looks for signs of excessive pessimism or optimism.
    """
    score = 0
    reasoning = []

    if not prices:
        return {"score": 0, "details": "No data for contrarian analysis"}

    import pandas as pd
    df = prices_to_df(prices)
    if df.empty or len(df) < 20:
        return {"score": 0, "details": "Insufficient data for contrarian analysis"}

    closes = df["close"].to_numpy()
    current = float(closes[-1])

    # 1. Drawdown from recent high — magnitude of pessimism
    if len(closes) >= 50:
        recent_high = float(np.max(closes[-252:] if len(closes) >= 252 else closes))
        drawdown = (current - recent_high) / recent_high

        if drawdown < -0.30:
            score += 3
            reasoning.append(f"Stock down {drawdown:.0%} from recent high — deep pessimism, contrarian opportunity")
        elif drawdown < -0.15:
            score += 2
            reasoning.append(f"Stock down {drawdown:.0%} from high — moderate sell-off")
        elif drawdown < -0.05:
            score += 1
            reasoning.append(f"Minor pullback ({drawdown:.0%})")
        elif drawdown > -0.02:
            reasoning.append(f"Near highs ({drawdown:.0%}) — consensus is bullish, less room for contrarian edge")

    # 2. Recent momentum reversal (potential bottom formation)
    if len(closes) >= 30:
        last_5 = closes[-5:]
        prev_25 = closes[-30:-5]
        if float(np.mean(last_5)) > float(np.mean(prev_25)) and float(np.min(prev_25)) < float(np.mean(closes)):
            score += 1
            reasoning.append("Recent price stabilization after weakness — possible cycle turn")

    # 3. Valuation compression check
    if metrics and len(metrics) >= 2:
        current_pe = getattr(metrics[0], 'price_to_earnings_ratio', None)
        historical_pes = [getattr(m, 'price_to_earnings_ratio', None) for m in metrics if getattr(m, 'price_to_earnings_ratio', None) is not None]
        if current_pe and len(historical_pes) >= 3:
            avg_pe = sum(historical_pes) / len(historical_pes)
            if current_pe < avg_pe * 0.75:
                score += 1
                reasoning.append(f"P/E ({current_pe:.1f}) well below historical avg ({avg_pe:.1f}) — valuation compression")
            elif current_pe > avg_pe * 1.25:
                reasoning.append(f"P/E ({current_pe:.1f}) above historical avg ({avg_pe:.1f}) — expensive by own standards")

    return {"score": min(score, 5), "details": "; ".join(reasoning) if reasoning else "No strong contrarian signals detected"}


def analyze_defensive_quality(metrics: list, financial_line_items: list) -> dict:
    """Marks: 'If we avoid the losers, the winners will take care of themselves.'

    Checks for characteristics that limit downside: margin stability,
    cash generation, conservative capital allocation.
    """
    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data for defensive analysis"}

    score = 0
    reasoning = []

    # 1. Operating margin stability
    margins = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
    if len(margins) >= 3:
        avg = sum(margins) / len(margins)
        variance = sum((m - avg) ** 2 for m in margins) / len(margins)
        stability = 1 - ((variance ** 0.5) / abs(avg)) if avg > 0 else 0

        if stability > 0.80 and avg > 0.15:
            score += 2
            reasoning.append(f"Stable high margins (avg {avg:.0%}, stability {stability:.0%}) — defensive moat")
        elif stability > 0.60:
            score += 1
            reasoning.append(f"Reasonably stable margins (stability {stability:.0%})")
        else:
            reasoning.append(f"Volatile margins (stability {stability:.0%}) — less defensive")
    else:
        reasoning.append("Insufficient margin history")

    # 2. Asset quality — equity growing, not shrinking
    equities = [item.shareholders_equity for item in financial_line_items
                if item.shareholders_equity is not None]
    if len(equities) >= 3:
        growing = sum(1 for i in range(len(equities) - 1) if equities[i] > equities[i + 1])
        growth_pct = growing / (len(equities) - 1)
        if growth_pct >= 0.7:
            score += 1
            reasoning.append(f"Equity growing in {growth_pct:.0%} of periods — value accumulation")
        else:
            reasoning.append(f"Equity growing in only {growth_pct:.0%} of periods")

    # 3. Conservative capital allocation
    capex_values = [item.capital_expenditure for item in financial_line_items if item.capital_expenditure is not None]
    fcf_values = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    if capex_values and fcf_values:
        avg_capex = abs(sum(capex_values) / len(capex_values))
        avg_fcf = sum(fcf_values) / len(fcf_values) if fcf_values else 0
        if avg_fcf > 0 and avg_capex > 0:
            capex_intensity = avg_capex / (avg_fcf + avg_capex)
            if capex_intensity < 0.4:
                score += 1
                reasoning.append(f"Low capex intensity ({capex_intensity:.0%}) — capital-light model")
            elif capex_intensity < 0.6:
                reasoning.append(f"Moderate capex intensity ({capex_intensity:.0%})")
            else:
                reasoning.append(f"High capex intensity ({capex_intensity:.0%}) — capital-heavy")

    # 4. Dividend track record
    divs = [item.dividends_and_other_cash_distributions for item in financial_line_items
            if item.dividends_and_other_cash_distributions is not None]
    if divs:
        paying = sum(1 for d in divs if d < 0)  # negative = cash outflow
        if paying >= len(divs) * 0.7:
            score += 1
            reasoning.append("Consistent dividend payer — defensive income stream")

    return {"score": min(score, 5), "details": "; ".join(reasoning) if reasoning else "Limited defensive analysis available"}


# ---------------------------------------------------------------------------
# LLM judgment
# ---------------------------------------------------------------------------

def generate_marks_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> HowardMarksSignal:
    """Produce a Howard Marks-style investment signal via LLM."""

    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "risk_return": analysis_data.get("risk_analysis", {}).get("details"),
        "cycle_position": analysis_data.get("cycle_analysis", {}).get("details"),
        "value_vs_price": analysis_data.get("value_analysis", {}).get("details"),
        "margin_of_safety": analysis_data.get("value_analysis", {}).get("margin_of_safety"),
        "contrarian_signals": analysis_data.get("contrarian_analysis", {}).get("details"),
        "defensive_quality": analysis_data.get("defensive_analysis", {}).get("details"),
        "market_cap": analysis_data.get("market_cap"),
    }

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Howard Marks, co-chairman of Oaktree Capital Management. You make investment
decisions based on these core principles from your memos and books:

1. RISK CONTROL IS THE MOST IMPORTANT THING
   - Focus on limiting downside, not maximizing upside
   - "If we avoid the losers, the winners will take care of themselves"
   - Asymmetric returns: participate in upside while limiting downside

2. SECOND-LEVEL THINKING
   - First-level: "This is a good company, let's buy."
   - Second-level: "This is a good company, but everyone thinks it's great and the stock is overpriced — sell."
   - You must disagree with consensus AND be right

3. MARKET CYCLES — "Where Are We?"
   - Markets swing between euphoria and panic like a pendulum
   - Late-cycle euphoria = time to be cautious / defensive
   - Early-cycle pessimism = time to be aggressive / buy
   - The key question: "Where are we in the cycle?"

4. PRICE VS VALUE
   - A great company at a bad price is a bad investment
   - A mediocre company at a great price can be a good investment
   - Margin of safety is everything

5. CONTRARIAN POSITIONING
   - "To buy when others are despondently selling and to sell when others are euphorically buying"
   - "The error of optimism" is the greatest risk in late cycles
   - Be skeptical of consensus views

6. DEFENSIVE INVESTING
   - Prefer consistency over home runs
   - Companies with stable margins, low leverage, and strong cash flow survive cycles
   - "Never forget the six-foot-tall man who drowned crossing the stream that was five feet deep on average"

7. KNOWING WHAT YOU DON'T KNOW
   - Intellectual humility: acknowledge uncertainty
   - Never 100% confident — real world is probabilistic

Signal rules:
- Bullish: Margin of safety > 0, defensive quality is strong, contrarian opportunity exists (pessimism priced in)
- Bearish: Overvalued (negative margin of safety), late-cycle euphoria, poor defensive characteristics
- Neutral: Fair value, mixed signals, or insufficient conviction for a strong view

Confidence scale:
- 80-95%: Deep value with multiple defensive characteristics aligning — rare
- 60-79%: Good setup with some uncertainty — most common for actionable views
- 40-59%: Mixed signals, would wait for more clarity
- 20-39%: Leaning one direction but low conviction
- Never exceed 95% — intellectual humility is paramount

Keep reasoning under 200 words. Cite specific data points. Return JSON only."""
            ),
            (
                "human",
                "Ticker: {ticker}\nFacts:\n{facts}\n\n"
                "Return exactly:\n"
                '{{\n  "signal": "bullish" | "bearish" | "neutral",\n'
                '  "confidence": float,\n'
                '  "reasoning": "string"\n}}'
            ),
        ]
    )

    prompt = template.invoke({
        "ticker": ticker,
        "facts": json.dumps(facts, indent=2, default=str),
    })

    def create_default():
        return HowardMarksSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=HowardMarksSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default,
    )
