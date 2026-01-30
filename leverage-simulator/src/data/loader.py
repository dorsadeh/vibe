"""Unified data loader with caching and calendar alignment."""

from datetime import datetime
from typing import Optional
import logging

import pandas as pd
import numpy as np

from .providers import DataProvider, YFinanceProvider, StooqProvider, fetch_with_fallback
from .fred import FREDRateProvider
from .cache import DataCache

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for price and rate data.

    Features:
    - Fetches from yfinance with Stooq fallback
    - Local parquet caching
    - Calendar alignment across assets
    - Interest rate integration from FRED
    """

    DEFAULT_ASSETS = ["SPY", "GLD", "TLT", "BIL"]

    def __init__(
        self,
        cache_dir: str = "data",
        cache_max_age_days: int = 1,
        rate_series: str = "DGS3MO",
        rate_fallback: str = "EFFR",
    ):
        """
        Initialize data loader.

        Args:
            cache_dir: Directory for cached data
            cache_max_age_days: Max cache age before refresh
            rate_series: Primary interest rate series
            rate_fallback: Fallback rate series
        """
        self.cache = DataCache(cache_dir, cache_max_age_days)
        self.primary_provider = YFinanceProvider()
        self.fallback_provider = StooqProvider()
        self.rate_provider = FREDRateProvider(rate_series, rate_fallback)

    def load_prices(
        self,
        symbols: list[str],
        start_date: str,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Load price data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' (default: today)
            force_refresh: Force re-fetch even if cached

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        prices = {}

        for symbol in symbols:
            # Check cache first
            if not force_refresh and self.cache.is_cached(
                symbol, "prices", required_end_date=end_date
            ):
                cached = self.cache.get(symbol, "prices")
                if cached is not None:
                    # Filter to requested date range
                    cached = cached.loc[start_date:end_date]
                    prices[symbol] = cached
                    continue

            # Fetch fresh data
            try:
                df = fetch_with_fallback(
                    symbol,
                    start_date,
                    end_date,
                    self.primary_provider,
                    self.fallback_provider,
                )
                self.cache.put(symbol, df, "prices")
                prices[symbol] = df
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                raise

        return prices

    def load_rates(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load interest rate data.

        Args:
            start_date: Start date
            end_date: End date
            force_refresh: Force re-fetch

        Returns:
            DataFrame with 'rate' column (annualized percentage)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        cache_key = f"{self.rate_provider.primary_series}"

        # Check cache
        if not force_refresh and self.cache.is_cached(
            cache_key, "rates", required_end_date=end_date
        ):
            cached = self.cache.get(cache_key, "rates")
            if cached is not None:
                return cached.loc[start_date:end_date]

        # Fetch fresh
        df = self.rate_provider.fetch(start_date, end_date)
        self.cache.put(cache_key, df, "rates")

        return df

    def load_aligned(
        self,
        symbols: list[str] = None,
        start_date: str = "2004-01-01",
        end_date: Optional[str] = None,
        force_refresh: bool = False,
        include_rates: bool = True,
    ) -> pd.DataFrame:
        """
        Load and align all data to a common trading calendar.

        Args:
            symbols: List of symbols (default: SPY, GLD, TLT, BIL)
            start_date: Start date
            end_date: End date
            force_refresh: Force re-fetch all data
            include_rates: Include interest rate series

        Returns:
            DataFrame with MultiIndex columns: (symbol, field)
            where field is one of: open, high, low, close, adj_close, volume
            Plus 'rate' column if include_rates=True
        """
        if symbols is None:
            symbols = self.DEFAULT_ASSETS

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Load all price data
        prices = self.load_prices(symbols, start_date, end_date, force_refresh)

        # Find common trading dates (intersection of all calendars)
        common_dates = None
        for symbol, df in prices.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))

        if not common_dates:
            raise ValueError("No common trading dates across assets")

        common_dates = sorted(common_dates)
        date_index = pd.DatetimeIndex(common_dates)

        logger.info(
            f"Aligned {len(symbols)} assets to {len(date_index)} common trading days"
        )

        # Build multi-level DataFrame
        data_dict = {}

        for symbol in symbols:
            df = prices[symbol].loc[date_index]

            # Standardize column names to lowercase
            data_dict[(symbol, "open")] = df["Open"]
            data_dict[(symbol, "high")] = df["High"]
            data_dict[(symbol, "low")] = df["Low"]
            data_dict[(symbol, "close")] = df["Close"]
            data_dict[(symbol, "adj_close")] = df["Adj Close"]
            data_dict[(symbol, "volume")] = df["Volume"]

        result = pd.DataFrame(data_dict, index=date_index)
        result.columns = pd.MultiIndex.from_tuples(
            result.columns, names=["symbol", "field"]
        )

        # Add rates if requested
        if include_rates:
            rates = self.load_rates(start_date, end_date, force_refresh)
            # Align rates to trading calendar
            aligned_rates = rates.reindex(date_index, method="ffill")
            aligned_rates = aligned_rates.bfill()  # Handle any leading NaN

            result[("_rates", "rate")] = aligned_rates["rate"]

        # Check for any remaining NaN
        nan_count = result.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Data contains {nan_count} NaN values after alignment")
            # Forward-fill then back-fill any gaps
            result = result.ffill().bfill()

        return result

    def get_returns(
        self,
        aligned_data: pd.DataFrame,
        symbols: list[str] = None,
        use_adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate daily returns from aligned price data.

        Args:
            aligned_data: DataFrame from load_aligned()
            symbols: Symbols to calculate returns for (default: all)
            use_adjusted: Use adjusted close (True) or close (False)

        Returns:
            DataFrame with daily returns for each symbol
        """
        if symbols is None:
            # Get all symbols except _rates
            symbols = [
                s for s in aligned_data.columns.get_level_values(0).unique()
                if not s.startswith("_")
            ]

        price_field = "adj_close" if use_adjusted else "close"

        returns_dict = {}
        for symbol in symbols:
            prices = aligned_data[(symbol, price_field)]
            returns_dict[symbol] = prices.pct_change()

        returns = pd.DataFrame(returns_dict, index=aligned_data.index)

        # First row will be NaN, fill with 0
        returns = returns.fillna(0)

        return returns


def quick_load(
    start_date: str = "2004-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function for quick data loading with defaults.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Aligned DataFrame with SPY, GLD, TLT, BIL and rates
    """
    loader = DataLoader()
    return loader.load_aligned(start_date=start_date, end_date=end_date)
