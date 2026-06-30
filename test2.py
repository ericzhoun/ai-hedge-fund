import sys, os
sys.path.insert(0, '/Users/ericz/Documents/GitHub/ai-hedge-fund')
from src.tools.api import get_prices, prices_to_df
from src.agents.day_swing_trader import build_features
prices = get_prices("TSLA", "2026-02-01", "2026-05-01", "day", 1)
print(f"Number of prices: {len(prices)}")
df = prices_to_df(prices)
print(f"DF shape: {df.shape}")
features = build_features(df)
import json
print(json.dumps(features, indent=2))
