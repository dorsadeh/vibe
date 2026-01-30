#!/usr/bin/env python3
"""
Verification script for Phase 2: Technical Indicators

Run this to verify indicators compute correctly on real data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.WARNING)

from src.data import DataLoader
from src.indicators import compute_indicators, IndicatorCalculator


def main():
    print("=" * 60)
    print("Phase 2 Verification: Technical Indicators")
    print("=" * 60)

    # Load data (use cache)
    loader = DataLoader(cache_dir="data")
    data = loader.load_aligned(
        symbols=["SPY", "GLD", "TLT", "BIL"],
        start_date="2004-01-01",
        end_date="2024-12-31",
    )

    print(f"\n1. Loaded {len(data)} trading days")

    # Compute indicators
    print("\n2. Computing indicators for SPY...")
    calc = IndicatorCalculator(
        sma_windows=[20, 50, 200],
        ema_windows=[12, 26],
        rsi_period=14,
        slope_window=20,
    )

    spy_indicators = calc.compute(data[("SPY", "adj_close")], prefix="SPY.")

    print(f"   Generated {len(spy_indicators.columns)} indicator columns:")
    for col in spy_indicators.columns:
        print(f"   - {col}")

    print("\n3. Sample indicator values (last 5 days):")
    print(spy_indicators.tail())

    print("\n4. Computing all indicators for all assets...")
    all_indicators = compute_indicators(
        data,
        symbols=["SPY", "GLD", "TLT"],
        sma_windows=[20, 50, 200],
        rsi_period=14,
    )

    print(f"   Total columns: {len(all_indicators.columns)}")
    print(f"   Shape: {all_indicators.shape}")

    print("\n5. Indicator statistics for SPY:")
    spy_cols = [c for c in all_indicators.columns if c.startswith("SPY.")]
    for col in spy_cols:
        series = all_indicators[col].dropna()
        print(f"   {col}:")
        print(f"      Min: {series.min():.2f}, Max: {series.max():.2f}, Last: {series.iloc[-1]:.2f}")

    print("\n6. Current signal check (latest values):")
    latest = all_indicators.iloc[-1]

    spy_close = latest["SPY.close"]
    spy_sma200 = latest["SPY.SMA_200"]
    spy_sma50 = latest["SPY.SMA_50"]
    spy_rsi = latest["SPY.RSI_14"]
    spy_slope = latest["SPY.SMA_200_slope"]

    print(f"   SPY Close: ${spy_close:.2f}")
    print(f"   SPY SMA(200): ${spy_sma200:.2f}")
    print(f"   SPY > SMA(200): {spy_close > spy_sma200}")
    print(f"   SMA(50) > SMA(200) (Golden Cross): {spy_sma50 > spy_sma200}")
    print(f"   RSI(14): {spy_rsi:.1f}")
    print(f"   RSI < 70 (not overbought): {spy_rsi < 70}")
    print(f"   SMA(200) slope (20d): {spy_slope:.2f}%")
    print(f"   Slope > 0 (trending up): {spy_slope > 0}")

    print("\n7. NaN check:")
    nan_counts = all_indicators.isna().sum()
    warmup_cols = nan_counts[nan_counts > 0]
    if len(warmup_cols) > 0:
        print(f"   {len(warmup_cols)} columns have NaN (expected for warmup period)")
        print(f"   Max NaN count: {warmup_cols.max()} (SMA_200 needs 200 days)")
    else:
        print("   No NaN values")

    # Check data after warmup
    valid_from = 250  # After SMA_200 warmup
    after_warmup = all_indicators.iloc[valid_from:]
    remaining_nan = after_warmup.isna().sum().sum()
    print(f"   NaN after 250-day warmup: {remaining_nan}")

    print("\n" + "=" * 60)
    print("Phase 2 Verification: COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
