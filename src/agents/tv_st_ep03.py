from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
import numpy as np
import pandas as pd
from typing_extensions import Literal

from src.tools.api import get_prices, prices_to_df
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state

class TVSTSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")

def geo_vol(high, low, close, volume):
    """Geometric volume estimation (Buy vs Sell) from Pine Script."""
    r = high - low
    if pd.isna(volume):
        return 0.0, 0.0
    if r == 0:
        return volume * 0.5, volume * 0.5
    buy_vol = volume * ((close - low) / r)
    sell_vol = volume * ((high - close) / r)
    return buy_vol, sell_vol

def analyze_blocks(df: pd.DataFrame, window: int = 100, num_groups: int = 20) -> dict:
    """Group recent bars into blocks and analyze composite metrics."""
    if len(df) < window:
        window = len(df)
    
    # We need at least enough bars to form groups
    if window < num_groups:
        num_groups = max(1, window)
        
    group_size = max(1, window // num_groups)
    recent_df = df.iloc[-window:].copy()
    
    blocks = []
    
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, window)
        block_df = recent_df.iloc[start_idx:end_idx]
        
        if block_df.empty:
            continue
            
        b_open = block_df['open'].iloc[0]
        b_close = block_df['close'].iloc[-1]
        b_high = block_df['high'].max()
        b_low = block_df['low'].min()
        b_vol = block_df['volume'].sum()
        
        # Aggregate Buy/Sell Volume
        b_buy_vol = 0
        b_sell_vol = 0
        for _, row in block_df.iterrows():
            b, s = geo_vol(row['high'], row['low'], row['close'], row['volume'])
            b_buy_vol += b
            b_sell_vol += s
            
        blocks.append({
            'open': b_open, 'close': b_close, 'high': b_high, 'low': b_low,
            'volume': b_vol, 'buy_vol': b_buy_vol, 'sell_vol': b_sell_vol,
            'delta': b_buy_vol - b_sell_vol
        })
        
    return blocks

def detect_trend(blocks: list) -> dict:
    """Simple trend detection based on composite block positioning."""
    if len(blocks) < 2:
        return {"trend": "RANGE", "strength": "WEAK"}
        
    recent = blocks[-1]
    prev = blocks[-2]
    
    mid_recent = (recent['high'] + recent['low']) / 2
    mid_prev = (prev['high'] + prev['low']) / 2
    
    if mid_recent > mid_prev:
        trend = "UPTREND"
    elif mid_recent < mid_prev:
        trend = "DOWNTREND"
    else:
        trend = "RANGE"
        
    return {"trend": trend, "strength": "MODERATE"}

def tv_st_ep03_agent(state: AgentState, agent_id: str = "tv_st_ep03_agent"):
    """
    Smart Trader EP03 Agent based on Ata Sabancı's TradingView script.
    Focuses on block-based composite candles, volume delta, and trend channels.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    interval = data.get("interval", "day")
    interval_multiplier = data.get("interval_multiplier", 1)
    
    # We need enough historical data for the window (e.g. 100 bars)
    import datetime
    start_date = (datetime.datetime.fromisoformat(end_date) - datetime.timedelta(days=150)).date().isoformat()
    
    tv_analysis = {}
    
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching price data")
        prices = get_prices(ticker, start_date, end_date, interval=interval, interval_multiplier=interval_multiplier, api_key=api_key)
        df = prices_to_df(prices) if prices else pd.DataFrame()
        
        if df.empty or len(df) < 10:
            tv_analysis[ticker] = {"signal": "neutral", "confidence": 0, "reasoning": "Insufficient price data"}
            continue
            
        progress.update_status(agent_id, ticker, "Analyzing blocks & volume")
        blocks = analyze_blocks(df, window=100, num_groups=20)
        trend_info = detect_trend(blocks)
        
        # Analyze volume momentum over recent blocks
        recent_blocks = blocks[-5:] if len(blocks) >= 5 else blocks
        total_buy = sum(b['buy_vol'] for b in recent_blocks)
        total_sell = sum(b['sell_vol'] for b in recent_blocks)
        net_delta = total_buy - total_sell
        
        pressure = "BUYING" if net_delta > 0 else ("SELLING" if net_delta < 0 else "BALANCED")
        
        analysis_data = {
            "trend": trend_info['trend'],
            "trend_strength": trend_info['strength'],
            "recent_pressure": pressure,
            "net_delta": net_delta,
            "blocks_analyzed": len(blocks)
        }
        
        progress.update_status(agent_id, ticker, "Generating TV-ST-EP03 analysis")
        
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a TradingView indicator analyst named TV-ST-EP03 based on Ata Sabancı's script.\n"
                    "You analyze price by grouping candles into composite blocks and using a geometric volume engine.\n"
                    "Rules:\n"
                    "- Bullish: Uptrend and Buying pressure.\n"
                    "- Bearish: Downtrend and Selling pressure.\n"
                    "- Neutral: Conflicting signals (e.g. Uptrend but Selling pressure) or Range.\n"
                    "Return JSON with signal, confidence (0-100), and concise reasoning."
                ),
                (
                    "human",
                    "Ticker: {ticker}\n"
                    "Facts:\n{facts}\n\n"
                    "Return exactly:\n"
                    "{{\n"
                    '  "signal": "bullish" | "bearish" | "neutral",\n'
                    '  "confidence": int,\n'
                    '  "reasoning": "short justification"\n'
                    "}}"
                )
            ]
        )
        
        prompt = template.invoke({
            "facts": json.dumps(analysis_data, separators=(",", ":"), ensure_ascii=False),
            "ticker": ticker
        })
        
        def create_default():
            return TVSTSignal(signal="neutral", confidence=50, reasoning="Insufficient data")
            
        output = call_llm(
            prompt=prompt,
            pydantic_model=TVSTSignal,
            agent_name=agent_id,
            state=state,
            default_factory=create_default,
        )
        
        tv_analysis[ticker] = {
            "signal": output.signal,
            "confidence": output.confidence,
            "reasoning": output.reasoning
        }
        progress.update_status(agent_id, ticker, "Done", analysis=output.reasoning)
        
    message = HumanMessage(content=json.dumps(tv_analysis), name=agent_id)
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(tv_analysis, agent_id)
        
    state["data"]["analyst_signals"][agent_id] = tv_analysis
    progress.update_status(agent_id, None, "Done")
    
    return {"messages": [message], "data": state["data"]}
