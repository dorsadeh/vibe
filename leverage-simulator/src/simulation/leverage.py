"""Leverage and financing calculations."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class LeverageConfig:
    """Configuration for leverage and financing."""

    max_leverage: float = 1.5
    """Maximum leverage factor (e.g., 1.5 = 150% exposure)."""

    broker_spread: float = 0.015
    """Spread added to base rate (e.g., 0.015 = 1.5%)."""

    day_count: int = 360
    """Day count convention for interest (360 or 252)."""

    maintenance_margin: float = 0.25
    """Maintenance margin threshold (e.g., 0.25 = 25%)."""

    risk_off_to_cash: bool = True
    """If True, go 100% BIL when leverage OFF. If False, stay invested at 1.0x."""


def calculate_daily_interest(
    borrowed_amount: float,
    annual_rate_pct: float,
    broker_spread: float = 0.015,
    day_count: int = 360,
) -> float:
    """
    Calculate daily interest cost on borrowed amount.

    Args:
        borrowed_amount: Amount borrowed (equity * (leverage - 1))
        annual_rate_pct: Annual interest rate in percentage (e.g., 5.0 = 5%)
        broker_spread: Broker spread in decimal (e.g., 0.015 = 1.5%)
        day_count: Day count convention (360 or 252)

    Returns:
        Daily interest cost in dollars
    """
    if borrowed_amount <= 0:
        return 0.0

    # Convert rate from percentage to decimal and add spread
    annual_rate = (annual_rate_pct / 100) + broker_spread

    # Daily rate
    daily_rate = annual_rate / day_count

    return borrowed_amount * daily_rate


def calculate_margin_ratio(
    equity: float,
    gross_exposure: float,
) -> float:
    """
    Calculate margin ratio (equity / gross exposure).

    Args:
        equity: Current portfolio equity
        gross_exposure: Total gross exposure (equity * leverage)

    Returns:
        Margin ratio (1.0 = no leverage, 0.5 = 2x leverage)
    """
    if gross_exposure <= 0:
        return 1.0
    return equity / gross_exposure


def check_margin_call(
    equity: float,
    gross_exposure: float,
    maintenance_margin: float = 0.25,
) -> bool:
    """
    Check if margin call is triggered.

    Args:
        equity: Current portfolio equity
        gross_exposure: Total gross exposure
        maintenance_margin: Minimum margin ratio required

    Returns:
        True if margin call triggered (need to deleverage)
    """
    margin_ratio = calculate_margin_ratio(equity, gross_exposure)
    return margin_ratio < maintenance_margin


def apply_leverage(
    equity: float,
    leverage_factor: float,
    max_leverage: float = 2.0,
) -> tuple[float, float]:
    """
    Calculate gross exposure and borrowed amount.

    Args:
        equity: Current portfolio equity
        leverage_factor: Desired leverage (e.g., 1.5)
        max_leverage: Maximum allowed leverage

    Returns:
        Tuple of (gross_exposure, borrowed_amount)
    """
    # Cap leverage at max
    actual_leverage = min(leverage_factor, max_leverage)

    gross_exposure = equity * actual_leverage
    borrowed = gross_exposure - equity

    return gross_exposure, max(0, borrowed)


@dataclass
class DailyState:
    """State of portfolio at end of day."""

    date: pd.Timestamp
    equity: float
    gross_exposure: float
    borrowed: float
    leverage_factor: float
    leverage_on: bool
    daily_interest: float
    cumulative_interest: float
    margin_ratio: float
    margin_call: bool
    positions: dict[str, float]  # symbol -> position value
    weights: dict[str, float]    # symbol -> weight


class LeverageSimulator:
    """
    Simulates leveraged portfolio with financing costs.

    Handles:
    - Leverage application based on signals
    - Daily interest accrual
    - Margin monitoring and forced deleveraging
    """

    def __init__(self, config: LeverageConfig):
        """Initialize simulator with config."""
        self.config = config
        self.cumulative_interest = 0.0
        self.margin_events: list[pd.Timestamp] = []

    def reset(self):
        """Reset simulator state."""
        self.cumulative_interest = 0.0
        self.margin_events = []

    def step(
        self,
        date: pd.Timestamp,
        equity: float,
        target_leverage: float,
        leverage_signal: bool,
        interest_rate_pct: float,
        returns: dict[str, float],
        target_weights: dict[str, float],
    ) -> DailyState:
        """
        Simulate one day.

        Args:
            date: Current date
            equity: Portfolio equity at start of day
            target_leverage: Target leverage when signal is ON
            leverage_signal: True = leverage ON, False = OFF
            interest_rate_pct: Current interest rate (percentage)
            returns: Daily returns for each asset
            target_weights: Target weights for risky assets

        Returns:
            DailyState with end-of-day values
        """
        # Determine actual leverage
        if leverage_signal:
            leverage_factor = target_leverage
        else:
            leverage_factor = 1.0 if not self.config.risk_off_to_cash else 0.0

        # Apply leverage
        gross_exposure, borrowed = apply_leverage(
            equity, leverage_factor, self.config.max_leverage
        )

        # Calculate positions based on weights
        positions = {}
        weights = {}

        if leverage_factor > 0:
            # Allocate to risky assets
            for symbol, weight in target_weights.items():
                positions[symbol] = gross_exposure * weight
                weights[symbol] = weight
        else:
            # Risk-off: 100% cash (BIL)
            positions["BIL"] = equity
            weights["BIL"] = 1.0

        # Calculate daily return on positions
        portfolio_return = 0.0
        for symbol, position in positions.items():
            if symbol in returns:
                portfolio_return += position * returns[symbol]

        # Calculate daily interest cost
        daily_interest = calculate_daily_interest(
            borrowed,
            interest_rate_pct,
            self.config.broker_spread,
            self.config.day_count,
        )

        # Update equity
        new_equity = equity + portfolio_return - daily_interest
        self.cumulative_interest += daily_interest

        # Check margin
        margin_ratio = calculate_margin_ratio(new_equity, gross_exposure)
        margin_call = check_margin_call(
            new_equity, gross_exposure, self.config.maintenance_margin
        )

        if margin_call:
            self.margin_events.append(date)
            # Force deleverage to 1.0x
            gross_exposure = new_equity
            borrowed = 0.0
            leverage_factor = 1.0

        return DailyState(
            date=date,
            equity=new_equity,
            gross_exposure=gross_exposure,
            borrowed=borrowed,
            leverage_factor=leverage_factor,
            leverage_on=leverage_signal,
            daily_interest=daily_interest,
            cumulative_interest=self.cumulative_interest,
            margin_ratio=margin_ratio,
            margin_call=margin_call,
            positions=positions,
            weights=weights,
        )
