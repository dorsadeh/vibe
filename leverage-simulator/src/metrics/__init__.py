"""Performance metrics module."""

from .performance import (
    PerformanceMetrics,
    calculate_returns,
    calculate_cagr,
    calculate_volatility,
    calculate_max_drawdown,
    calculate_drawdown_series,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_rolling_sharpe,
    calculate_rolling_drawdown,
    compute_metrics,
    compute_benchmark_metrics,
    format_metrics,
    compare_metrics,
)

__all__ = [
    "PerformanceMetrics",
    "calculate_returns",
    "calculate_cagr",
    "calculate_volatility",
    "calculate_max_drawdown",
    "calculate_drawdown_series",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_rolling_sharpe",
    "calculate_rolling_drawdown",
    "compute_metrics",
    "compute_benchmark_metrics",
    "format_metrics",
    "compare_metrics",
]
