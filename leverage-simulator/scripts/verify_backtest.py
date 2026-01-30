#!/usr/bin/env python3
"""
Verification script for Phase 4: Portfolio Simulation

Run this to verify the backtest engine works end-to-end.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO)

from src.data import DataLoader
from src.simulation import (
    BacktestConfig,
    BacktestEngine,
    LeverageConfig,
    PortfolioConfig,
    RebalanceConfig,
)


def main():
    print("=" * 60)
    print("Phase 4 Verification: Portfolio Simulation")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    loader = DataLoader(cache_dir="data")
    data = loader.load_aligned(
        symbols=["SPY", "GLD", "TLT", "BIL"],
        start_date="2004-01-01",
        end_date="2024-12-31",
    )
    returns = loader.get_returns(data, symbols=["SPY", "GLD", "TLT", "BIL"])
    print(f"   Loaded {len(data)} trading days")

    # Configure backtest
    print("\n2. Configuring backtest...")
    config = BacktestConfig(
        initial_equity=100_000,
        leverage=LeverageConfig(
            max_leverage=1.5,
            broker_spread=0.015,
            day_count=360,
            maintenance_margin=0.25,
            risk_off_to_cash=True,
        ),
        portfolio=PortfolioConfig(
            mode="fixed_weights",
            weights={"SPY": 0.6, "GLD": 0.2, "TLT": 0.2},
            risky_assets=["SPY", "GLD", "TLT"],
            cash_asset="BIL",
        ),
        rebalance=RebalanceConfig(
            frequency="monthly",
        ),
        signal_asset="SPY",
        leverage_rule="(SPY.close > SPY.SMA_200) AND (SPY.RSI_14 < 70)",
        target_leverage=1.5,
        sma_windows=[20, 50, 200],
        rsi_period=14,
    )

    print(f"   Rule: {config.leverage_rule}")
    print(f"   Target leverage: {config.target_leverage}x")
    print(f"   Portfolio mode: {config.portfolio.mode}")
    print(f"   Weights: {config.portfolio.weights}")

    # Run backtest
    print("\n3. Running backtest...")
    engine = BacktestEngine(data, returns, config)
    result = engine.run()

    print(f"   Simulation period: {result.equity_curve.index[0]} to {result.equity_curve.index[-1]}")
    print(f"   Trading days: {len(result.equity_curve)}")

    # Display results
    print("\n4. Equity curve:")
    print(f"   Start: ${config.initial_equity:,.2f}")
    print(f"   End:   ${result.equity_curve.iloc[-1]:,.2f}")
    print(f"   Min:   ${result.equity_curve.min():,.2f}")
    print(f"   Max:   ${result.equity_curve.max():,.2f}")

    # Calculate simple return
    total_return = (result.equity_curve.iloc[-1] / config.initial_equity - 1) * 100
    print(f"   Total return: {total_return:.1f}%")

    print("\n5. Leverage statistics:")
    lev = result.leverage_series
    print(f"   Mean leverage: {lev.mean():.2f}x")
    print(f"   % time leveraged (>1x): {(lev > 1).mean() * 100:.1f}%")
    print(f"   % time in cash (0x): {(lev == 0).mean() * 100:.1f}%")

    print("\n6. Interest costs:")
    print(f"   Total interest paid: ${result.cumulative_interest.iloc[-1]:,.2f}")
    print(f"   Avg daily interest: ${result.interest_series.mean():.2f}")

    print("\n7. Drawdown:")
    max_dd = result.drawdown_series.min() * 100
    print(f"   Max drawdown: {max_dd:.1f}%")

    print("\n8. Margin events:")
    print(f"   Total margin calls: {len(result.margin_events)}")
    if result.margin_events:
        print(f"   First: {result.margin_events[0]}")
        print(f"   Last: {result.margin_events[-1]}")

    print("\n9. Rebalancing:")
    print(f"   Total rebalances: {len(result.rebalance_dates)}")

    print("\n10. Sample equity curve (last 10 days):")
    print(result.equity_curve.tail(10))

    # Test momentum rotation mode
    print("\n" + "=" * 60)
    print("Testing Momentum Rotation Mode")
    print("=" * 60)

    config_momentum = BacktestConfig(
        initial_equity=100_000,
        leverage=LeverageConfig(
            max_leverage=1.5,
            broker_spread=0.015,
            risk_off_to_cash=True,
        ),
        portfolio=PortfolioConfig(
            mode="momentum_rotation",
            risky_assets=["SPY", "GLD", "TLT"],
            cash_asset="BIL",
            momentum_lookback=126,
            momentum_top_n=1,
        ),
        rebalance=RebalanceConfig(frequency="monthly"),
        leverage_rule="(SPY.close > SPY.SMA_200)",
        target_leverage=1.5,
    )

    engine_mom = BacktestEngine(data, returns, config_momentum)
    result_mom = engine_mom.run()

    print(f"\nMomentum rotation results:")
    print(f"   End equity: ${result_mom.equity_curve.iloc[-1]:,.2f}")
    total_return_mom = (result_mom.equity_curve.iloc[-1] / config.initial_equity - 1) * 100
    print(f"   Total return: {total_return_mom:.1f}%")
    print(f"   Max drawdown: {result_mom.drawdown_series.min() * 100:.1f}%")

    # Show which assets were held over time
    print("\n   Asset allocation over time (sample):")
    weights_mom = result_mom.weights
    sample_dates = weights_mom.index[::252]  # Yearly samples
    for date in sample_dates[:5]:
        row = weights_mom.loc[date]
        held = [f"{k}:{v:.0%}" for k, v in row.items() if v > 0]
        print(f"   {date.date()}: {', '.join(held)}")

    print("\n" + "=" * 60)
    print("Phase 4 Verification: COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
