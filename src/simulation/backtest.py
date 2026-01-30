"""Main backtest simulation engine."""

from dataclasses import dataclass, field
from typing import Optional
import logging

import pandas as pd
import numpy as np

from .leverage import LeverageConfig, LeverageSimulator, DailyState
from .portfolio import (
    PortfolioConfig,
    RebalanceConfig,
    create_strategy,
    RebalanceScheduler,
)
from ..rules import RulesEngine
from ..indicators import compute_indicators

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Complete configuration for backtest."""

    # Initial conditions
    initial_equity: float = 100_000.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Leverage settings
    leverage: LeverageConfig = field(default_factory=LeverageConfig)

    # Portfolio settings
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    # Rebalancing settings
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)

    # Leverage rule
    signal_asset: str = "SPY"
    leverage_rule: str = "(SPY.close > SPY.SMA_200) AND (SPY.RSI_14 < 70)"
    target_leverage: float = 1.5

    # Indicator settings
    sma_windows: list[int] = field(default_factory=lambda: [20, 50, 200])
    ema_windows: list[int] = field(default_factory=lambda: [12, 26])
    rsi_period: int = 14

    # Transaction costs
    transaction_cost_bps: float = 0.0
    """Transaction cost in basis points (e.g., 10 = 0.1%)."""


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Time series
    equity_curve: pd.Series
    leverage_series: pd.Series
    interest_series: pd.Series
    cumulative_interest: pd.Series
    drawdown_series: pd.Series
    positions: pd.DataFrame
    weights: pd.DataFrame

    # Events
    margin_events: list[pd.Timestamp]
    rebalance_dates: list[pd.Timestamp]

    # Summary stats (computed later by metrics module)
    config: BacktestConfig = None


class BacktestEngine:
    """
    Main backtest simulation engine.

    Combines:
    - Data loading
    - Indicator computation
    - Rule evaluation
    - Portfolio allocation
    - Leverage simulation
    - Result aggregation
    """

    def __init__(
        self,
        aligned_data: pd.DataFrame,
        returns: pd.DataFrame,
        config: BacktestConfig,
    ):
        """
        Initialize backtest engine.

        Args:
            aligned_data: Aligned price data from DataLoader
            returns: Daily returns from DataLoader
            config: Backtest configuration
        """
        self.data = aligned_data
        self.returns = returns
        self.config = config

        # Extract date range
        if config.start_date:
            self.data = self.data.loc[config.start_date:]
            self.returns = self.returns.loc[config.start_date:]
        if config.end_date:
            self.data = self.data.loc[:config.end_date]
            self.returns = self.returns.loc[:config.end_date]

        self.dates = self.data.index

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize simulation components."""
        # Compute indicators
        symbols = self.config.portfolio.risky_assets + [self.config.portfolio.cash_asset]
        symbols = list(set(symbols))  # Remove duplicates

        self.indicators = compute_indicators(
            self.data,
            symbols=symbols,
            sma_windows=self.config.sma_windows,
            ema_windows=self.config.ema_windows,
            rsi_period=self.config.rsi_period,
        )

        # Initialize rules engine
        self.rules_engine = RulesEngine(self.indicators)

        # Initialize portfolio strategy
        self.strategy = create_strategy(self.config.portfolio)

        # Initialize rebalance scheduler
        self.rebalance_scheduler = RebalanceScheduler(self.config.rebalance)

        # Initialize leverage simulator
        self.leverage_sim = LeverageSimulator(self.config.leverage)

    def run(self) -> BacktestResult:
        """
        Run the backtest simulation.

        Returns:
            BacktestResult with all time series and events
        """
        logger.info(f"Running backtest from {self.dates[0]} to {self.dates[-1]}")
        logger.info(f"Rule: {self.config.leverage_rule}")

        # Evaluate leverage rule for all dates
        leverage_signal = self.rules_engine.evaluate(self.config.leverage_rule)

        # Initialize state
        equity = self.config.initial_equity
        self.leverage_sim.reset()
        self.rebalance_scheduler.reset()

        # Storage for results
        daily_states: list[DailyState] = []
        rebalance_dates: list[pd.Timestamp] = []
        current_weights: dict[str, float] = {}

        # Get prices for momentum calculation
        prices = pd.DataFrame({
            symbol: self.data[(symbol, "adj_close")]
            for symbol in self.config.portfolio.risky_assets
            if (symbol, "adj_close") in self.data.columns
        })

        # Skip warmup period (need enough data for indicators)
        warmup = max(self.config.sma_windows) + 50  # Extra buffer
        start_idx = warmup

        logger.info(f"Skipping {warmup} days for indicator warmup")

        for i, date in enumerate(self.dates):
            if i < start_idx:
                continue

            # Check if we should rebalance
            should_rebalance = self.rebalance_scheduler.should_rebalance(
                date, self.dates
            )

            if should_rebalance or not current_weights:
                # Get new target weights
                current_weights = self.strategy.get_weights(
                    date,
                    prices.loc[:date],
                    self.returns.loc[:date],
                )
                self.rebalance_scheduler.mark_rebalanced(date)
                rebalance_dates.append(date)

            # Get daily returns for this date
            daily_returns = {}
            for symbol in list(current_weights.keys()) + [self.config.portfolio.cash_asset]:
                if symbol in self.returns.columns:
                    daily_returns[symbol] = self.returns.loc[date, symbol]

            # Get current interest rate
            rate = self.data.loc[date, ("_rates", "rate")]

            # Get leverage signal for this date
            signal = leverage_signal.loc[date]

            # Handle NaN signal (during warmup)
            if pd.isna(signal):
                signal = False

            # Simulate one day
            state = self.leverage_sim.step(
                date=date,
                equity=equity,
                target_leverage=self.config.target_leverage,
                leverage_signal=signal,
                interest_rate_pct=rate,
                returns=daily_returns,
                target_weights=current_weights,
            )

            daily_states.append(state)
            equity = state.equity

        # Convert to result format
        return self._build_result(daily_states, rebalance_dates)

    def _build_result(
        self,
        daily_states: list[DailyState],
        rebalance_dates: list[pd.Timestamp],
    ) -> BacktestResult:
        """Build BacktestResult from daily states."""
        if not daily_states:
            raise ValueError("No simulation data generated")

        dates = [s.date for s in daily_states]
        index = pd.DatetimeIndex(dates)

        # Build time series
        equity_curve = pd.Series(
            [s.equity for s in daily_states],
            index=index,
            name="equity",
        )

        leverage_series = pd.Series(
            [s.leverage_factor for s in daily_states],
            index=index,
            name="leverage",
        )

        interest_series = pd.Series(
            [s.daily_interest for s in daily_states],
            index=index,
            name="daily_interest",
        )

        cumulative_interest = pd.Series(
            [s.cumulative_interest for s in daily_states],
            index=index,
            name="cumulative_interest",
        )

        # Calculate drawdown
        running_max = equity_curve.cummax()
        drawdown_series = (equity_curve - running_max) / running_max
        drawdown_series.name = "drawdown"

        # Build positions DataFrame
        all_symbols = set()
        for s in daily_states:
            all_symbols.update(s.positions.keys())

        positions_data = {symbol: [] for symbol in all_symbols}
        weights_data = {symbol: [] for symbol in all_symbols}

        for s in daily_states:
            for symbol in all_symbols:
                positions_data[symbol].append(s.positions.get(symbol, 0.0))
                weights_data[symbol].append(s.weights.get(symbol, 0.0))

        positions = pd.DataFrame(positions_data, index=index)
        weights = pd.DataFrame(weights_data, index=index)

        return BacktestResult(
            equity_curve=equity_curve,
            leverage_series=leverage_series,
            interest_series=interest_series,
            cumulative_interest=cumulative_interest,
            drawdown_series=drawdown_series,
            positions=positions,
            weights=weights,
            margin_events=self.leverage_sim.margin_events,
            rebalance_dates=rebalance_dates,
            config=self.config,
        )


def run_backtest(
    aligned_data: pd.DataFrame,
    returns: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        aligned_data: Aligned price data from DataLoader
        returns: Daily returns from DataLoader
        config: Backtest configuration

    Returns:
        BacktestResult
    """
    engine = BacktestEngine(aligned_data, returns, config)
    return engine.run()
