import json
import os
import logging

logger = logging.getLogger(__name__)

class Cache:
    """Persistent cache for API responses."""

    def __init__(self, cache_dir: str = "."):
        self.cache_file = os.path.join(cache_dir, "market_data_cache.json")
        self._prices_cache: dict[str, list[dict[str, any]]] = {}
        self._financial_metrics_cache: dict[str, list[dict[str, any]]] = {}
        self._line_items_cache: dict[str, list[dict[str, any]]] = {}
        self._insider_trades_cache: dict[str, list[dict[str, any]]] = {}
        self._company_news_cache: dict[str, list[dict[str, any]]] = {}
        self._load()

    def _load(self):
        """Load cached data from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    self._prices_cache = data.get("prices", {})
                    self._financial_metrics_cache = data.get("financial_metrics", {})
                    self._line_items_cache = data.get("line_items", {})
                    self._insider_trades_cache = data.get("insider_trades", {})
                    self._company_news_cache = data.get("company_news", {})
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_file}: {e}")

    def _save(self):
        """Save cached data to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump({
                    "prices": self._prices_cache,
                    "financial_metrics": self._financial_metrics_cache,
                    "line_items": self._line_items_cache,
                    "insider_trades": self._insider_trades_cache,
                    "company_news": self._company_news_cache,
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save cache to {self.cache_file}: {e}")

    def _merge_data(self, existing: list[dict] | None, new_data: list[dict], key_field: str) -> list[dict]:
        """Merge existing and new data, avoiding duplicates based on a key field."""
        if not existing:
            return new_data

        # Create a set of existing keys for O(1) lookup
        existing_keys = {item[key_field] for item in existing}

        # Only add items that don't exist yet
        merged = existing.copy()
        merged.extend([item for item in new_data if item[key_field] not in existing_keys])
        
        # Sort merged data so newer items come first, preserving API order convention
        if merged and key_field in merged[0]:
            merged.sort(key=lambda x: x.get(key_field, ""), reverse=True)
            
        return merged

    def get_prices(self, ticker: str) -> list[dict[str, any]] | None:
        """Get cached price data if available."""
        return self._prices_cache.get(ticker)

    def set_prices(self, ticker: str, data: list[dict[str, any]]):
        """Append new price data to cache."""
        self._prices_cache[ticker] = self._merge_data(self._prices_cache.get(ticker), data, key_field="time")
        self._save()

    def get_financial_metrics(self, ticker: str) -> list[dict[str, any]]:
        """Get cached financial metrics if available."""
        return self._financial_metrics_cache.get(ticker)

    def set_financial_metrics(self, ticker: str, data: list[dict[str, any]]):
        """Append new financial metrics to cache."""
        self._financial_metrics_cache[ticker] = self._merge_data(self._financial_metrics_cache.get(ticker), data, key_field="report_period")
        self._save()

    def get_line_items(self, ticker: str) -> list[dict[str, any]] | None:
        """Get cached line items if available."""
        return self._line_items_cache.get(ticker)

    def set_line_items(self, ticker: str, data: list[dict[str, any]]):
        """Append new line items to cache."""
        self._line_items_cache[ticker] = self._merge_data(self._line_items_cache.get(ticker), data, key_field="report_period")
        self._save()

    def get_insider_trades(self, ticker: str) -> list[dict[str, any]] | None:
        """Get cached insider trades if available."""
        return self._insider_trades_cache.get(ticker)

    def set_insider_trades(self, ticker: str, data: list[dict[str, any]]):
        """Append new insider trades to cache."""
        self._insider_trades_cache[ticker] = self._merge_data(self._insider_trades_cache.get(ticker), data, key_field="filing_date")
        self._save()

    def get_company_news(self, ticker: str) -> list[dict[str, any]] | None:
        """Get cached company news if available."""
        return self._company_news_cache.get(ticker)

    def set_company_news(self, ticker: str, data: list[dict[str, any]]):
        """Append new company news to cache."""
        self._company_news_cache[ticker] = self._merge_data(self._company_news_cache.get(ticker), data, key_field="date")
        self._save()


# Global cache instance
_cache = Cache()


def get_cache() -> Cache:
    """Get the global cache instance."""
    return _cache
