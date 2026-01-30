"""FRED data provider for interest rate series."""

from datetime import datetime
from io import StringIO
from typing import Optional
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class FREDRateProvider:
    """
    Fetches interest rate series from FRED (Federal Reserve Economic Data).

    Uses direct CSV download (no API key required for public series).

    Supported series:
    - DGS3MO: 3-Month Treasury Bill Rate (secondary market)
    - EFFR: Effective Federal Funds Rate
    - SOFR: Secured Overnight Financing Rate (shorter history)
    - FEDFUNDS: Federal Funds Effective Rate (monthly)
    - DTB3: 3-Month Treasury Bill (secondary market, daily)
    """

    BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    SUPPORTED_SERIES = {
        "DGS3MO": "3-Month Treasury Bill Rate",
        "EFFR": "Effective Federal Funds Rate",
        "FEDFUNDS": "Federal Funds Effective Rate (monthly)",
        "SOFR": "Secured Overnight Financing Rate",
        "DTB3": "3-Month Treasury Bill (secondary market, daily)",
    }

    def __init__(self, primary_series: str = "DGS3MO", fallback_series: str = "EFFR"):
        """
        Initialize FRED rate provider.

        Args:
            primary_series: Primary rate series to use
            fallback_series: Fallback series if primary has gaps
        """
        if primary_series not in self.SUPPORTED_SERIES:
            raise ValueError(
                f"Unknown series: {primary_series}. "
                f"Supported: {list(self.SUPPORTED_SERIES.keys())}"
            )
        self.primary_series = primary_series
        self.fallback_series = fallback_series

    def fetch(
        self,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch interest rate series from FRED.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)

        Returns:
            DataFrame with 'rate' column (annualized percentage, e.g., 5.0 = 5%)
            Index: DatetimeIndex
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            f"[FRED] Fetching {self.primary_series} from {start_date} to {end_date}"
        )

        # Fetch primary series
        try:
            primary_df = self._fetch_series(self.primary_series, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to fetch primary series {self.primary_series}: {e}")
            raise

        # Fetch fallback series if configured
        fallback_df = None
        if self.fallback_series:
            try:
                fallback_df = self._fetch_series(
                    self.fallback_series, start_date, end_date
                )
            except Exception as e:
                logger.warning(f"Failed to fetch fallback series: {e}")

        # Combine: use primary, fill gaps with fallback
        df = primary_df.copy()
        df.columns = ["rate"]

        if fallback_df is not None:
            fallback_df.columns = ["fallback_rate"]
            df = df.join(fallback_df, how="outer")
            # Fill NaN in primary with fallback values
            df["rate"] = df["rate"].fillna(df["fallback_rate"])
            df = df.drop(columns=["fallback_rate"])

        # Forward-fill remaining gaps (rates don't change on weekends/holidays)
        df["rate"] = df["rate"].ffill()

        # Back-fill any initial NaN (for very early dates)
        df["rate"] = df["rate"].bfill()

        # Convert from percentage to decimal if needed
        # FRED rates are typically in percentage form (e.g., 5.0 = 5%)
        # We'll keep them as percentages and convert during interest calculation

        logger.info(f"[FRED] Fetched {len(df)} rate observations")
        return df

    def _fetch_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch a single series from FRED via CSV download."""
        params = {
            "id": series_id,
            "cosd": start_date,
            "coed": end_date,
        }

        logger.debug(f"[FRED] Requesting {series_id}")
        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        # Parse CSV
        df = pd.read_csv(StringIO(response.text))

        if df.empty:
            raise ValueError(f"No data returned for {series_id}")

        # FRED CSV has 'DATE' and series_id as columns
        df.columns = ["Date", series_id]

        # Convert date column
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

        # Handle missing values (FRED uses '.' for missing)
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

        # Ensure index is DatetimeIndex
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "Date"

        return df

    def get_aligned_rates(
        self,
        dates: pd.DatetimeIndex,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """
        Get rates aligned to a specific set of trading dates.

        Args:
            dates: DatetimeIndex to align to (e.g., from price data)
            start_date: Start date for fetching
            end_date: End date for fetching

        Returns:
            Series of rates aligned to the provided dates
        """
        # Fetch full rate series
        rate_df = self.fetch(start_date, end_date)

        # Reindex to match trading dates, forward-fill
        aligned = rate_df.reindex(dates, method="ffill")

        # Back-fill any initial NaN
        aligned = aligned.bfill()

        return aligned["rate"]
