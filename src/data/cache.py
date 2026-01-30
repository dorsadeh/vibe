"""Local data caching with parquet format."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """
    Local file cache for price and rate data.

    Uses parquet format for efficient storage and fast reads.
    Implements cache invalidation based on data freshness.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data",
        max_age_days: int = 1,
    ):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory for cached files
            max_age_days: Max age before cache is considered stale
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days

    def _get_cache_path(self, key: str, data_type: str = "prices") -> Path:
        """Get cache file path for a given key."""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{data_type}_{safe_key}.parquet"

    def _get_metadata_path(self, key: str, data_type: str = "prices") -> Path:
        """Get metadata file path for a given key."""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{data_type}_{safe_key}.meta"

    def is_cached(
        self,
        key: str,
        data_type: str = "prices",
        required_end_date: Optional[str] = None,
    ) -> bool:
        """
        Check if valid cache exists for a key.

        Args:
            key: Cache key (e.g., symbol name)
            data_type: Type of data ('prices' or 'rates')
            required_end_date: If set, cache must extend to this date

        Returns:
            True if valid cache exists
        """
        cache_path = self._get_cache_path(key, data_type)
        meta_path = self._get_metadata_path(key, data_type)

        if not cache_path.exists():
            return False

        # Check metadata for freshness
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = {}
                    for line in f:
                        k, v = line.strip().split("=", 1)
                        meta[k] = v

                cached_time = datetime.fromisoformat(meta.get("cached_at", ""))
                age = datetime.now() - cached_time

                # If cache is too old, consider stale
                if age > timedelta(days=self.max_age_days):
                    logger.debug(f"Cache for {key} is stale ({age.days} days old)")
                    return False

                # Check if cache extends to required date
                if required_end_date:
                    cached_end = meta.get("end_date", "")
                    if cached_end < required_end_date:
                        logger.debug(
                            f"Cache for {key} ends at {cached_end}, "
                            f"need {required_end_date}"
                        )
                        return False

            except Exception as e:
                logger.warning(f"Failed to read cache metadata for {key}: {e}")
                return False

        return True

    def get(
        self,
        key: str,
        data_type: str = "prices",
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache.

        Args:
            key: Cache key
            data_type: Type of data

        Returns:
            DataFrame if cache hit, None otherwise
        """
        cache_path = self._get_cache_path(key, data_type)

        if not cache_path.exists():
            logger.debug(f"Cache miss for {key}")
            return None

        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Cache hit for {key}: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache for {key}: {e}")
            return None

    def put(
        self,
        key: str,
        data: pd.DataFrame,
        data_type: str = "prices",
    ) -> None:
        """
        Store data in cache.

        Args:
            key: Cache key
            data: DataFrame to cache
            data_type: Type of data
        """
        cache_path = self._get_cache_path(key, data_type)
        meta_path = self._get_metadata_path(key, data_type)

        try:
            # Write parquet
            data.to_parquet(cache_path, engine="pyarrow")

            # Write metadata
            with open(meta_path, "w") as f:
                f.write(f"cached_at={datetime.now().isoformat()}\n")
                f.write(f"rows={len(data)}\n")
                if len(data) > 0:
                    f.write(f"start_date={data.index.min().strftime('%Y-%m-%d')}\n")
                    f.write(f"end_date={data.index.max().strftime('%Y-%m-%d')}\n")

            logger.info(f"Cached {len(data)} rows for {key}")

        except Exception as e:
            logger.error(f"Failed to cache data for {key}: {e}")
            raise

    def invalidate(self, key: str, data_type: str = "prices") -> None:
        """Remove cached data for a key."""
        cache_path = self._get_cache_path(key, data_type)
        meta_path = self._get_metadata_path(key, data_type)

        if cache_path.exists():
            cache_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

        logger.info(f"Invalidated cache for {key}")

    def clear_all(self) -> None:
        """Clear all cached data."""
        for path in self.cache_dir.glob("*.parquet"):
            path.unlink()
        for path in self.cache_dir.glob("*.meta"):
            path.unlink()
        logger.info("Cleared all cached data")
