"""Portfolio simulation module."""

from .leverage import (
    LeverageConfig,
    LeverageSimulator,
    DailyState,
    calculate_daily_interest,
    calculate_margin_ratio,
    check_margin_call,
    apply_leverage,
)

from .portfolio import (
    PortfolioConfig,
    RebalanceConfig,
    AllocationStrategy,
    FixedWeightStrategy,
    MomentumRotationStrategy,
    create_strategy,
    RebalanceScheduler,
)

from .backtest import (
    BacktestConfig,
    BacktestResult,
    BacktestEngine,
    run_backtest,
)

__all__ = [
    # Leverage
    "LeverageConfig",
    "LeverageSimulator",
    "DailyState",
    "calculate_daily_interest",
    "calculate_margin_ratio",
    "check_margin_call",
    "apply_leverage",
    # Portfolio
    "PortfolioConfig",
    "RebalanceConfig",
    "AllocationStrategy",
    "FixedWeightStrategy",
    "MomentumRotationStrategy",
    "create_strategy",
    "RebalanceScheduler",
    # Backtest
    "BacktestConfig",
    "BacktestResult",
    "BacktestEngine",
    "run_backtest",
]
