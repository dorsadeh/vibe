"""Data providers for fetching historical price data."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import logging

import pandas as pd
import yfinance as yf
import requests

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
            Index: DatetimeIndex (date only, no time)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider using yfinance library."""

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"[{self.name}] Fetching {symbol} from {start_date} to {end_date}")

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            raise ValueError(f"No data returned for {symbol} from {self.name}")

        # Standardize column names
        df = df.rename(columns={
            "Stock Splits": "Stock_Splits",
            "Capital Gains": "Capital_Gains",
        })

        # Keep only OHLCV + Adj Close
        columns_to_keep = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        available_cols = [c for c in columns_to_keep if c in df.columns]
        df = df[available_cols]

        # Ensure index is date only (no time component)
        df.index = pd.to_datetime(df.index).date
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "Date"

        logger.info(f"[{self.name}] Fetched {len(df)} rows for {symbol}")
        return df


class StooqProvider(DataProvider):
    """Stooq data provider as fallback."""

    BASE_URL = "https://stooq.com/q/d/l/"

    @property
    def name(self) -> str:
        return "stooq"

    def _symbol_to_stooq(self, symbol: str) -> str:
        """Convert standard ticker to Stooq format."""
        # US stocks need .US suffix
        us_etfs = {"SPY", "GLD", "TLT", "BIL", "QQQ", "IWM", "EFA", "VTI"}
        if symbol.upper() in us_etfs:
            return f"{symbol.lower()}.us"
        return symbol.lower()

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch data from Stooq."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"[{self.name}] Fetching {symbol} from {start_date} to {end_date}")

        stooq_symbol = self._symbol_to_stooq(symbol)

        # Format dates for Stooq API
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        params = {
            "s": stooq_symbol,
            "d1": start_dt.strftime("%Y%m%d"),
            "d2": end_dt.strftime("%Y%m%d"),
            "i": "d",  # daily
        }

        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))

        if df.empty or len(df) < 2:
            raise ValueError(f"No data returned for {symbol} from {self.name}")

        # Standardize column names (Stooq uses different casing)
        df.columns = df.columns.str.strip()
        column_map = {
            "Date": "Date",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }

        # Rename if needed
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Set date index
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df = df.sort_index()

        # Stooq doesn't provide Adj Close, use Close as proxy
        # Note: This is a limitation - dividends not reflected
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
            logger.warning(
                f"[{self.name}] {symbol}: Using Close as Adj Close (no dividend adjustment)"
            )

        # Filter to date range
        df = df.loc[start_date:end_date]

        logger.info(f"[{self.name}] Fetched {len(df)} rows for {symbol}")
        return df


def get_provider(
    primary: str = "yfinance",
    fallback: str = "stooq",
) -> tuple[DataProvider, Optional[DataProvider]]:
    """
    Get data provider instances.

    Args:
        primary: Primary provider name
        fallback: Fallback provider name

    Returns:
        Tuple of (primary_provider, fallback_provider)
    """
    providers = {
        "yfinance": YFinanceProvider,
        "stooq": StooqProvider,
    }

    primary_provider = providers.get(primary)
    if primary_provider is None:
        raise ValueError(f"Unknown provider: {primary}")

    fallback_provider = providers.get(fallback)

    return primary_provider(), fallback_provider() if fallback_provider else None


def fetch_with_fallback(
    symbol: str,
    start_date: str,
    end_date: Optional[str] = None,
    primary: DataProvider = None,
    fallback: DataProvider = None,
) -> pd.DataFrame:
    """
    Fetch data with automatic fallback on failure.

    Args:
        symbol: Ticker symbol
        start_date: Start date
        end_date: End date
        primary: Primary data provider
        fallback: Fallback data provider

    Returns:
        DataFrame with OHLCV data
    """
    if primary is None:
        primary, fallback = get_provider()

    try:
        return primary.fetch(symbol, start_date, end_date)
    except Exception as e:
        logger.warning(f"Primary provider failed for {symbol}: {e}")
        if fallback is not None:
            logger.info(f"Trying fallback provider for {symbol}")
            return fallback.fetch(symbol, start_date, end_date)
        raise
