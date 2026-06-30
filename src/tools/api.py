import datetime
import logging
import os
import pandas as pd
import requests
import time

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.
    
    Args:
        url: The URL to request
        headers: Headers to include in the request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests
        max_retries: Maximum number of retries (default: 3)
    
    Returns:
        requests.Response: The response object
    
    Raises:
        Exception: If the request fails with a non-429 error
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)
        
        if response.status_code == 429 and attempt < max_retries:
            # Linear backoff: 60s, 90s, 120s, 150s...
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue
        
        # Return the response (whether success, other errors, or final 429)
        return response


def get_prices(ticker: str, start_date: str, end_date: str, interval: str = "day", interval_multiplier: int = 1, api_key: str = None, include_snapshot: bool = True) -> list[Price]:
    """Fetch price data from cache or API, optionally appending the latest snapshot.

    When ``include_snapshot`` is True (the default for live analysis), the most
    recent intraday snapshot is fetched from ``/prices/snapshot/`` and merged
    into the historical price list.  The snapshot uses proper OHLCV fields and
    is marked with ``is_snapshot=True`` so downstream agents can distinguish it
    from historical bars.

    If the snapshot's timestamp matches the last historical bar (same date for
    daily data), the snapshot *replaces* the historical bar to provide the
    freshest data.  Otherwise it is appended as a new bar.

    Set ``include_snapshot=False`` during backtesting to avoid contaminating
    historical data with live prices.
    """
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    prices = []
    # Cache by ticker and interval format to share data among all analysts
    cache_key = f"{ticker}_{interval}_{interval_multiplier}"
    
    # Always fetch a 5-year superset to satisfy any analyst's range requirements
    fetch_start = (datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")
    
    cached_data = _cache.get_prices(cache_key)
    if not cached_data:
        # If not in cache, fetch the superset from API
        url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval={interval}&interval_multiplier={interval_multiplier}&start_date={fetch_start}&end_date={end_date}"
        response = _make_api_request(url, headers)
        if response.status_code == 200:
            try:
                price_response = PriceResponse(**response.json())
                prices = price_response.prices
                if prices:
                    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
                    cached_data = _cache.get_prices(cache_key)
            except Exception as e:
                logger.warning("Failed to parse price response for %s: %s", ticker, e)
                
    if cached_data:
        # Filter the cached superset to exactly what the analyst requested
        prices = [Price(**price) for price in cached_data if start_date <= price["time"][:10] <= end_date]

    # Optionally fetch the live snapshot and merge it
    if include_snapshot:
        # Use an in-memory cache attached to the _cache object for the lifetime of this run
        if not hasattr(_cache, "snapshot_cache"):
            _cache.snapshot_cache = {}
            
        snapshot_data = _cache.snapshot_cache.get(ticker)
        
        if not snapshot_data:
            snapshot_url = f"https://api.financialdatasets.ai/prices/snapshot/?ticker={ticker}"
            snapshot_response = _make_api_request(snapshot_url, headers)
            if snapshot_response.status_code == 200:
                try:
                    snapshot_data = snapshot_response.json().get("snapshot", {})
                    if snapshot_data:
                        _cache.snapshot_cache[ticker] = snapshot_data
                except Exception as e:
                    logger.warning("Failed to parse snapshot response for %s: %s", ticker, e)
                    
        if snapshot_data:
            snapshot_price = Price(
                open=snapshot_data.get("open", snapshot_data.get("price", 0)),
                close=snapshot_data.get("close", snapshot_data.get("price", 0)),
                high=snapshot_data.get("high", snapshot_data.get("price", 0)),
                low=snapshot_data.get("low", snapshot_data.get("price", 0)),
                volume=snapshot_data.get("volume", 0),
                time=snapshot_data.get("time", ""),
                is_snapshot=True,
            )
            # Deduplicate: if the snapshot covers the same period as the
            # last historical bar, replace it (snapshot is fresher).
            if prices and snapshot_price.time:
                last_hist_date = prices[-1].time[:10]  # YYYY-MM-DD
                snap_date = snapshot_price.time[:10]
                if last_hist_date == snap_date:
                    prices[-1] = snapshot_price
                else:
                    prices.append(snapshot_price)
            else:
                prices.append(snapshot_price)

    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Use a broader cache key to group calls (ticker + period)
    cache_key = f"{ticker}_{period}"
    
    cached_data = _cache.get_financial_metrics(cache_key)
    if not cached_data:
        # If not in cache, fetch a large superset from API
        headers = {}
        financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        if financial_api_key:
            headers["X-API-KEY"] = financial_api_key

        url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit=20&period={period}"
        response = _make_api_request(url, headers)
        if response.status_code == 200:
            try:
                metrics_response = FinancialMetricsResponse(**response.json())
                financial_metrics = metrics_response.financial_metrics
                if financial_metrics:
                    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
                    cached_data = _cache.get_financial_metrics(cache_key)
            except Exception as e:
                logger.warning("Failed to parse financial metrics response for %s: %s", ticker, e)

    if cached_data:
        # Sort by date descending and filter to requested limit
        sorted_metrics = sorted(cached_data, key=lambda x: x["report_period"], reverse=True)
        return [FinancialMetrics(**metric) for metric in sorted_metrics[:limit]]
        
    return []


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from API."""
    # Use a broader cache key (ticker + period)
    cache_key = f"{ticker}_{period}"
    
    cached_data = _cache.get_line_items(cache_key)
    if not cached_data:
        headers = {}
        financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        if financial_api_key:
            headers["X-API-KEY"] = financial_api_key

        url = "https://api.financialdatasets.ai/financials/search/line-items"

        # Fetch a superset of common line items used by all analysts
        all_line_items = [
            "capital_expenditure", "cash_and_equivalents", "current_assets", "current_liabilities",
            "debt_to_equity", "depreciation_and_amortization", "dividends_and_other_cash_distributions",
            "earnings_per_share", "ebit", "ebitda", "free_cash_flow", "goodwill_and_intangible_assets",
            "gross_margin", "gross_profit", "interest_expense", "issuance_or_purchase_of_equity_shares",
            "net_income", "operating_expense", "operating_income", "operating_margin",
            "outstanding_shares", "research_and_development", "return_on_invested_capital",
            "revenue", "shareholders_equity", "total_assets", "total_debt", "total_liabilities",
            "working_capital", "book_value_per_share"
        ]

        body = {
            "tickers": [ticker],
            "line_items": all_line_items,
            "end_date": end_date,
            "period": period,
            "limit": 20,
        }
        response = _make_api_request(url, headers, method="POST", json_data=body)
        if response.status_code == 200:
            try:
                data = response.json()
                response_model = LineItemResponse(**data)
                search_results = response_model.search_results
                if search_results:
                    _cache.set_line_items(cache_key, [m.model_dump() for m in search_results])
                    cached_data = _cache.get_line_items(cache_key)
            except Exception as e:
                logger.warning("Failed to parse line items response for %s: %s", ticker, e)

    if cached_data:
        sorted_items = sorted(cached_data, key=lambda x: x["report_period"], reverse=True)
        return [LineItem(**item) for item in sorted_items[:limit]]
        
    return []


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_trades = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = InsiderTradeResponse(**data)
            insider_trades = response_model.insider_trades
        except Exception as e:
            logger.warning("Failed to parse insider trades response for %s: %s", ticker, e)
            break

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break

        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_news = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = CompanyNewsResponse(**data)
            company_news = response_model.news
        except Exception as e:
            logger.warning("Failed to parse company news response for %s: %s", ticker, e)
            break

        if not company_news:
            break

        all_news.extend(company_news)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break

        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from the API."""
    # Check if end_date is today
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        # Get the market cap from company facts API
        headers = {}
        financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        if financial_api_key:
            headers["X-API-KEY"] = financial_api_key

        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        response = _make_api_request(url, headers)
        if response.status_code != 200:
            print(f"Error fetching company facts: {ticker} - {response.status_code}")
            return None

        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap

    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if not financial_metrics:
        return None

    market_cap = financial_metrics[0].market_cap

    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame.

    The resulting DataFrame includes an ``is_snapshot`` boolean column so that
    downstream agents can distinguish the live intraday snapshot bar from
    historical bars (e.g. to weight it differently in indicator calculations).
    """
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"], format="mixed", utc=True).dt.tz_localize(None)
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Ensure is_snapshot column exists (older cached data may lack it)
    if "is_snapshot" not in df.columns:
        df["is_snapshot"] = False
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str, interval: str = "day", interval_multiplier: int = 1, api_key: str = None, include_snapshot: bool = True) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, interval=interval, interval_multiplier=interval_multiplier, api_key=api_key, include_snapshot=include_snapshot)
    return prices_to_df(prices)
