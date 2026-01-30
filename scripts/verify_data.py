#!/usr/bin/env python3
"""
Verification script for Phase 1: Data Layer

Run this to verify:
1. Price data fetching for SPY, GLD, TLT, BIL
2. Interest rate fetching from FRED
3. Local caching in parquet format
4. Calendar alignment across assets
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from src.data import DataLoader


def main():
    print("=" * 60)
    print("Phase 1 Verification: Data Layer")
    print("=" * 60)

    # Initialize loader
    loader = DataLoader(
        cache_dir="data",
        cache_max_age_days=1,
        rate_series="DGS3MO",
        rate_fallback="EFFR",
    )

    # Test parameters
    start_date = "2004-01-01"
    end_date = "2024-12-31"
    symbols = ["SPY", "GLD", "TLT", "BIL"]

    print(f"\n1. Loading aligned data for {symbols}")
    print(f"   Date range: {start_date} to {end_date}")

    try:
        data = loader.load_aligned(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            force_refresh=False,  # Use cache if available
            include_rates=True,
        )
        print(f"   ✓ Loaded {len(data)} trading days")
    except Exception as e:
        print(f"   ✗ Failed to load data: {e}")
        return 1

    # Inspect the data
    print("\n2. Data structure:")
    print(f"   Shape: {data.shape}")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    print(f"   Columns: {data.columns.tolist()[:10]}...")

    print("\n3. Sample data (first 5 rows):")
    print(data.head())

    print("\n4. Sample data (last 5 rows):")
    print(data.tail())

    print("\n5. Missing values check:")
    nan_counts = data.isna().sum()
    if nan_counts.sum() == 0:
        print("   ✓ No missing values")
    else:
        print("   ✗ Missing values found:")
        print(nan_counts[nan_counts > 0])

    print("\n6. Price statistics:")
    for symbol in symbols:
        adj_close = data[(symbol, "adj_close")]
        print(f"   {symbol}:")
        print(f"      Min: ${adj_close.min():.2f}")
        print(f"      Max: ${adj_close.max():.2f}")
        print(f"      Latest: ${adj_close.iloc[-1]:.2f}")

    print("\n7. Interest rate statistics:")
    rates = data[("_rates", "rate")]
    print(f"   Min rate: {rates.min():.2f}%")
    print(f"   Max rate: {rates.max():.2f}%")
    print(f"   Current rate: {rates.iloc[-1]:.2f}%")
    print(f"   Mean rate: {rates.mean():.2f}%")

    print("\n8. Calculate returns:")
    returns = loader.get_returns(data, symbols)
    print(f"   Shape: {returns.shape}")
    print("\n   Annualized returns (approx):")
    for symbol in symbols:
        ann_return = returns[symbol].mean() * 252 * 100
        ann_vol = returns[symbol].std() * (252 ** 0.5) * 100
        print(f"   {symbol}: {ann_return:.1f}% return, {ann_vol:.1f}% vol")

    print("\n9. Cache verification:")
    cache_dir = Path("data")
    parquet_files = list(cache_dir.glob("*.parquet"))
    meta_files = list(cache_dir.glob("*.meta"))
    print(f"   Parquet files: {len(parquet_files)}")
    print(f"   Metadata files: {len(meta_files)}")
    for f in parquet_files:
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name}: {size_kb:.1f} KB")

    print("\n" + "=" * 60)
    print("Phase 1 Verification: COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
