"""Portfolio allocation modes: fixed weights and momentum rotation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class PortfolioConfig:
    """Configuration for portfolio allocation."""

    mode: str = "fixed_weights"
    """'fixed_weights' or 'momentum_rotation'."""

    weights: dict[str, float] = None
    """Fixed weights for assets (must sum to 1.0)."""

    risky_assets: list[str] = None
    """List of risky assets for rotation/allocation."""

    cash_asset: str = "BIL"
    """Cash/risk-off asset."""

    momentum_lookback: int = 126
    """Lookback period for momentum calculation (trading days)."""

    momentum_top_n: int = 1
    """Number of top assets to hold in momentum mode."""

    def __post_init__(self):
        if self.weights is None:
            self.weights = {"SPY": 0.6, "GLD": 0.2, "TLT": 0.2}
        if self.risky_assets is None:
            self.risky_assets = ["SPY", "GLD", "TLT"]


class AllocationStrategy(ABC):
    """Base class for allocation strategies."""

    @abstractmethod
    def get_weights(
        self,
        date: pd.Timestamp,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Get target weights for the given date.

        Args:
            date: Current date
            prices: Historical prices up to date
            returns: Historical returns up to date

        Returns:
            Dict of symbol -> weight (should sum to 1.0 for risky portion)
        """
        pass


class FixedWeightStrategy(AllocationStrategy):
    """Fixed weight allocation strategy."""

    def __init__(self, weights: dict[str, float]):
        """
        Initialize with fixed weights.

        Args:
            weights: Dict of symbol -> weight (must sum to 1.0)
        """
        total = sum(weights.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self.weights = weights

    def get_weights(
        self,
        date: pd.Timestamp,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """Return fixed weights regardless of date/prices."""
        return self.weights.copy()


class MomentumRotationStrategy(AllocationStrategy):
    """
    Momentum-based rotation strategy.

    Ranks assets by trailing return and invests in top N.
    """

    def __init__(
        self,
        risky_assets: list[str],
        lookback: int = 126,
        top_n: int = 1,
    ):
        """
        Initialize momentum strategy.

        Args:
            risky_assets: List of assets to rank
            lookback: Lookback period for return calculation
            top_n: Number of top assets to hold
        """
        self.risky_assets = risky_assets
        self.lookback = lookback
        self.top_n = top_n

    def get_weights(
        self,
        date: pd.Timestamp,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Calculate momentum scores and return weights for top assets.

        Returns equal weight across top N assets.
        """
        # Calculate trailing returns for each asset
        momentum_scores = {}

        for symbol in self.risky_assets:
            if symbol not in returns.columns:
                continue

            # Get returns up to current date
            symbol_returns = returns.loc[:date, symbol]

            if len(symbol_returns) < self.lookback:
                # Not enough history, use what we have
                total_return = (1 + symbol_returns).prod() - 1
            else:
                # Use lookback period
                recent_returns = symbol_returns.iloc[-self.lookback:]
                total_return = (1 + recent_returns).prod() - 1

            momentum_scores[symbol] = total_return

        if not momentum_scores:
            # Fallback to equal weight
            weight = 1.0 / len(self.risky_assets)
            return {s: weight for s in self.risky_assets}

        # Rank by momentum
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top N
        top_assets = [symbol for symbol, _ in ranked[:self.top_n]]

        # Equal weight among top assets
        weight = 1.0 / len(top_assets)
        weights = {symbol: weight for symbol in top_assets}

        return weights


def create_strategy(config: PortfolioConfig) -> AllocationStrategy:
    """
    Factory function to create allocation strategy from config.

    Args:
        config: Portfolio configuration

    Returns:
        AllocationStrategy instance
    """
    if config.mode == "fixed_weights":
        return FixedWeightStrategy(config.weights)
    elif config.mode == "momentum_rotation":
        return MomentumRotationStrategy(
            risky_assets=config.risky_assets,
            lookback=config.momentum_lookback,
            top_n=config.momentum_top_n,
        )
    else:
        raise ValueError(f"Unknown portfolio mode: {config.mode}")


@dataclass
class RebalanceConfig:
    """Configuration for rebalancing."""

    frequency: str = "monthly"
    """'daily', 'weekly', 'monthly', or 'N_days' (e.g., '21_days')."""

    day_of_week: int = 0
    """For weekly: day of week (0=Monday)."""

    day_of_month: int = 1
    """For monthly: day of month (1=first trading day)."""


class RebalanceScheduler:
    """Determines when to rebalance the portfolio."""

    def __init__(self, config: RebalanceConfig):
        self.config = config
        self._last_rebalance: Optional[pd.Timestamp] = None
        self._days_since_rebalance = 0

    def reset(self):
        """Reset scheduler state."""
        self._last_rebalance = None
        self._days_since_rebalance = 0

    def should_rebalance(self, date: pd.Timestamp, dates: pd.DatetimeIndex) -> bool:
        """
        Check if rebalancing should occur on this date.

        Args:
            date: Current date
            dates: Full trading calendar

        Returns:
            True if should rebalance
        """
        # First day always rebalances
        if self._last_rebalance is None:
            return True

        self._days_since_rebalance += 1

        if self.config.frequency == "daily":
            return True

        elif self.config.frequency == "weekly":
            # Rebalance on specified day of week
            if date.dayofweek == self.config.day_of_week:
                return True

        elif self.config.frequency == "monthly":
            # Rebalance on first trading day of month
            if self._last_rebalance is not None:
                if date.month != self._last_rebalance.month:
                    return True

        elif self.config.frequency.endswith("_days"):
            # Extract N from "N_days"
            n_days = int(self.config.frequency.split("_")[0])
            if self._days_since_rebalance >= n_days:
                return True

        return False

    def mark_rebalanced(self, date: pd.Timestamp):
        """Mark that rebalancing occurred."""
        self._last_rebalance = date
        self._days_since_rebalance = 0
