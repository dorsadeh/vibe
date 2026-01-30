"""Performance metrics for backtest analysis."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class PerformanceMetrics:
    """Summary performance metrics."""

    # Returns
    total_return: float
    """Total return as decimal (e.g., 3.92 = 392%)."""

    cagr: float
    """Compound Annual Growth Rate as decimal."""

    # Risk
    volatility: float
    """Annualized volatility as decimal."""

    max_drawdown: float
    """Maximum drawdown as decimal (negative value)."""

    # Risk-adjusted
    sharpe_ratio: float
    """Sharpe ratio (annualized)."""

    sortino_ratio: float
    """Sortino ratio (downside deviation)."""

    calmar_ratio: float
    """Calmar ratio (CAGR / |max drawdown|)."""

    # Leverage-specific
    pct_time_leveraged: float
    """Percentage of time with leverage > 1."""

    total_interest_paid: float
    """Total interest paid in dollars."""

    avg_leverage: float
    """Average leverage factor."""

    margin_calls: int
    """Number of margin call events."""

    # Additional
    win_rate: float
    """Percentage of positive days."""

    best_day: float
    """Best single day return."""

    worst_day: float
    """Worst single day return."""

    trading_days: int
    """Number of trading days in backtest."""

    years: float
    """Number of years in backtest."""


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """Calculate daily returns from equity curve."""
    return equity_curve.pct_change().fillna(0)


def calculate_cagr(
    equity_curve: pd.Series,
    trading_days_per_year: int = 252,
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        equity_curve: Portfolio equity over time
        trading_days_per_year: Trading days per year (default 252)

    Returns:
        CAGR as decimal
    """
    if len(equity_curve) < 2:
        return 0.0

    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    n_days = len(equity_curve)
    n_years = n_days / trading_days_per_year

    if start_value <= 0 or end_value <= 0 or n_years <= 0:
        return 0.0

    return (end_value / start_value) ** (1 / n_years) - 1


def calculate_volatility(
    returns: pd.Series,
    trading_days_per_year: int = 252,
) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Daily returns
        trading_days_per_year: Trading days per year

    Returns:
        Annualized volatility as decimal
    """
    return returns.std() * np.sqrt(trading_days_per_year)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        equity_curve: Portfolio equity over time

    Returns:
        Maximum drawdown as decimal (negative value)
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Calculate drawdown series."""
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: pd.Series = None,
    trading_days_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Daily returns
        risk_free_rate: Daily risk-free rate (annualized percentage)
        trading_days_per_year: Trading days per year

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    if risk_free_rate is not None:
        # Convert annual percentage to daily decimal
        daily_rf = risk_free_rate / 100 / trading_days_per_year
        # Align indices
        aligned_rf = daily_rf.reindex(returns.index, method="ffill").fillna(0)
        excess_returns = returns - aligned_rf
    else:
        excess_returns = returns

    std = excess_returns.std()
    if std == 0 or np.isclose(std, 0, atol=1e-10):
        return 0.0

    return (excess_returns.mean() / std) * np.sqrt(trading_days_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: pd.Series = None,
    trading_days_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).

    Args:
        returns: Daily returns
        risk_free_rate: Daily risk-free rate (annualized percentage)
        trading_days_per_year: Trading days per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    if risk_free_rate is not None:
        daily_rf = risk_free_rate / 100 / trading_days_per_year
        aligned_rf = daily_rf.reindex(returns.index, method="ffill").fillna(0)
        excess_returns = returns - aligned_rf
    else:
        excess_returns = returns

    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float("inf") if excess_returns.mean() > 0 else 0.0

    downside_std = np.sqrt((downside_returns ** 2).mean())

    return (excess_returns.mean() / downside_std) * np.sqrt(trading_days_per_year)


def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """
    Calculate Calmar ratio (CAGR / |max drawdown|).

    Args:
        cagr: CAGR as decimal
        max_drawdown: Max drawdown as decimal (negative)

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return float("inf") if cagr > 0 else 0.0
    return cagr / abs(max_drawdown)


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.

    Args:
        returns: Daily returns
        window: Rolling window in days
        trading_days_per_year: Trading days per year

    Returns:
        Rolling Sharpe ratio series
    """
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    # Avoid division by zero
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(trading_days_per_year)
    return rolling_sharpe


def calculate_rolling_drawdown(
    equity_curve: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Calculate rolling maximum drawdown.

    Args:
        equity_curve: Portfolio equity over time
        window: Rolling window in days

    Returns:
        Rolling max drawdown series
    """
    def rolling_max_dd(x):
        running_max = x.cummax()
        drawdown = (x - running_max) / running_max
        return drawdown.min()

    return equity_curve.rolling(window=window).apply(rolling_max_dd, raw=False)


def compute_metrics(
    equity_curve: pd.Series,
    leverage_series: pd.Series,
    cumulative_interest: pd.Series,
    margin_events: list,
    risk_free_rate: pd.Series = None,
    trading_days_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Compute all performance metrics.

    Args:
        equity_curve: Portfolio equity over time
        leverage_series: Leverage factor over time
        cumulative_interest: Cumulative interest paid
        margin_events: List of margin call dates
        risk_free_rate: Risk-free rate series (annual percentage)
        trading_days_per_year: Trading days per year

    Returns:
        PerformanceMetrics dataclass
    """
    returns = calculate_returns(equity_curve)

    # Basic metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = calculate_cagr(equity_curve, trading_days_per_year)
    volatility = calculate_volatility(returns, trading_days_per_year)
    max_drawdown = calculate_max_drawdown(equity_curve)

    # Risk-adjusted
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, trading_days_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, trading_days_per_year)
    calmar = calculate_calmar_ratio(cagr, max_drawdown)

    # Leverage metrics
    pct_leveraged = (leverage_series > 1).mean()
    total_interest = cumulative_interest.iloc[-1]
    avg_leverage = leverage_series.mean()

    # Additional
    win_rate = (returns > 0).mean()
    best_day = returns.max()
    worst_day = returns.min()
    trading_days = len(equity_curve)
    years = trading_days / trading_days_per_year

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        pct_time_leveraged=pct_leveraged,
        total_interest_paid=total_interest,
        avg_leverage=avg_leverage,
        margin_calls=len(margin_events),
        win_rate=win_rate,
        best_day=best_day,
        worst_day=worst_day,
        trading_days=trading_days,
        years=years,
    )


def compute_benchmark_metrics(
    prices: pd.Series,
    initial_equity: float = 100_000,
    trading_days_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Compute metrics for a buy-and-hold benchmark.

    Args:
        prices: Price series (adjusted close)
        initial_equity: Starting equity
        trading_days_per_year: Trading days per year

    Returns:
        PerformanceMetrics for benchmark
    """
    # Calculate shares and equity curve
    shares = initial_equity / prices.iloc[0]
    equity_curve = prices * shares

    returns = prices.pct_change().fillna(0)

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = calculate_cagr(equity_curve, trading_days_per_year)
    volatility = calculate_volatility(returns, trading_days_per_year)
    max_drawdown = calculate_max_drawdown(equity_curve)

    sharpe = calculate_sharpe_ratio(returns, None, trading_days_per_year)
    sortino = calculate_sortino_ratio(returns, None, trading_days_per_year)
    calmar = calculate_calmar_ratio(cagr, max_drawdown)

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        pct_time_leveraged=0.0,
        total_interest_paid=0.0,
        avg_leverage=1.0,
        margin_calls=0,
        win_rate=(returns > 0).mean(),
        best_day=returns.max(),
        worst_day=returns.min(),
        trading_days=len(prices),
        years=len(prices) / trading_days_per_year,
    )


def format_metrics(metrics: PerformanceMetrics, name: str = "Strategy") -> str:
    """
    Format metrics for display.

    Args:
        metrics: PerformanceMetrics to format
        name: Strategy name for header

    Returns:
        Formatted string
    """
    lines = [
        f"\n{'='*50}",
        f"  {name} Performance Summary",
        f"{'='*50}",
        "",
        "  Returns:",
        f"    Total Return:    {metrics.total_return*100:>8.1f}%",
        f"    CAGR:            {metrics.cagr*100:>8.2f}%",
        "",
        "  Risk:",
        f"    Volatility:      {metrics.volatility*100:>8.1f}%",
        f"    Max Drawdown:    {metrics.max_drawdown*100:>8.1f}%",
        "",
        "  Risk-Adjusted:",
        f"    Sharpe Ratio:    {metrics.sharpe_ratio:>8.2f}",
        f"    Sortino Ratio:   {metrics.sortino_ratio:>8.2f}",
        f"    Calmar Ratio:    {metrics.calmar_ratio:>8.2f}",
        "",
        "  Leverage:",
        f"    Avg Leverage:    {metrics.avg_leverage:>8.2f}x",
        f"    % Time Levered:  {metrics.pct_time_leveraged*100:>8.1f}%",
        f"    Interest Paid:   ${metrics.total_interest_paid:>10,.0f}",
        f"    Margin Calls:    {metrics.margin_calls:>8d}",
        "",
        "  Additional:",
        f"    Win Rate:        {metrics.win_rate*100:>8.1f}%",
        f"    Best Day:        {metrics.best_day*100:>8.2f}%",
        f"    Worst Day:       {metrics.worst_day*100:>8.2f}%",
        f"    Trading Days:    {metrics.trading_days:>8d}",
        f"    Years:           {metrics.years:>8.1f}",
        f"{'='*50}",
    ]
    return "\n".join(lines)


def compare_metrics(
    strategy: PerformanceMetrics,
    benchmark: PerformanceMetrics,
) -> str:
    """
    Compare strategy vs benchmark.

    Returns formatted comparison string.
    """
    lines = [
        "\n" + "=" * 60,
        "  Strategy vs Benchmark Comparison",
        "=" * 60,
        "",
        f"  {'Metric':<20} {'Strategy':>15} {'Benchmark':>15}",
        f"  {'-'*50}",
        f"  {'CAGR':<20} {strategy.cagr*100:>14.2f}% {benchmark.cagr*100:>14.2f}%",
        f"  {'Volatility':<20} {strategy.volatility*100:>14.1f}% {benchmark.volatility*100:>14.1f}%",
        f"  {'Max Drawdown':<20} {strategy.max_drawdown*100:>14.1f}% {benchmark.max_drawdown*100:>14.1f}%",
        f"  {'Sharpe Ratio':<20} {strategy.sharpe_ratio:>15.2f} {benchmark.sharpe_ratio:>15.2f}",
        f"  {'Calmar Ratio':<20} {strategy.calmar_ratio:>15.2f} {benchmark.calmar_ratio:>15.2f}",
        "",
        f"  {'Alpha (vs bench)':<20} {(strategy.cagr - benchmark.cagr)*100:>14.2f}%",
        "=" * 60,
    ]
    return "\n".join(lines)
