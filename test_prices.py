from src.tools.api import get_prices
from dotenv import load_dotenv

load_dotenv()

prices = get_prices("AAPL", "2025-06-01", "2026-04-23", "day", 1)
print(f"Prices fetched: {len(prices)}")
if prices:
    print(f"Last price time: {prices[-1].time}")
    print(f"Last price value: {prices[-1].close}")
else:
    print("Failed to fetch prices. Check API key and response.")
