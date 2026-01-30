"""Tests for leverage and financing calculations."""

import pytest
import pandas as pd
import numpy as np

from src.simulation import (
    calculate_daily_interest,
    calculate_margin_ratio,
    check_margin_call,
    apply_leverage,
    LeverageConfig,
    LeverageSimulator,
)


class TestInterestCalculation:
    """Tests for interest calculation."""

    def test_no_borrowing_no_interest(self):
        """Zero borrowed amount = zero interest."""
        interest = calculate_daily_interest(0, 5.0)
        assert interest == 0.0

    def test_negative_borrowing_no_interest(self):
        """Negative borrowed (cash) = zero interest."""
        interest = calculate_daily_interest(-1000, 5.0)
        assert interest == 0.0

    def test_basic_interest_calculation_360(self):
        """Test basic interest with 360-day convention."""
        # $100,000 borrowed at 5% + 1.5% spread = 6.5% annual
        # Daily rate = 6.5% / 360 = 0.01806%
        # Daily interest = $100,000 * 0.0001806 = $18.06

        interest = calculate_daily_interest(
            borrowed_amount=100_000,
            annual_rate_pct=5.0,
            broker_spread=0.015,
            day_count=360,
        )

        expected = 100_000 * (0.05 + 0.015) / 360
        assert interest == pytest.approx(expected)
        assert interest == pytest.approx(18.056, rel=0.01)

    def test_basic_interest_calculation_252(self):
        """Test basic interest with 252-day convention."""
        interest = calculate_daily_interest(
            borrowed_amount=100_000,
            annual_rate_pct=5.0,
            broker_spread=0.015,
            day_count=252,
        )

        expected = 100_000 * (0.05 + 0.015) / 252
        assert interest == pytest.approx(expected)
        # Should be higher than 360-day convention
        interest_360 = calculate_daily_interest(100_000, 5.0, 0.015, 360)
        assert interest > interest_360

    def test_zero_rate(self):
        """Test with zero base rate (only spread applies)."""
        interest = calculate_daily_interest(
            borrowed_amount=100_000,
            annual_rate_pct=0.0,
            broker_spread=0.015,
            day_count=360,
        )

        expected = 100_000 * 0.015 / 360
        assert interest == pytest.approx(expected)

    def test_annual_interest_approximation(self):
        """Verify daily interest * 360 ≈ annual interest."""
        borrowed = 100_000
        rate = 5.0
        spread = 0.015

        daily = calculate_daily_interest(borrowed, rate, spread, 360)
        annual = daily * 360

        expected_annual = borrowed * (rate / 100 + spread)
        assert annual == pytest.approx(expected_annual, rel=0.01)


class TestMarginCalculations:
    """Tests for margin calculations."""

    def test_margin_ratio_no_leverage(self):
        """1x leverage = margin ratio of 1.0."""
        ratio = calculate_margin_ratio(equity=100_000, gross_exposure=100_000)
        assert ratio == 1.0

    def test_margin_ratio_2x_leverage(self):
        """2x leverage = margin ratio of 0.5."""
        ratio = calculate_margin_ratio(equity=100_000, gross_exposure=200_000)
        assert ratio == 0.5

    def test_margin_ratio_1_5x_leverage(self):
        """1.5x leverage = margin ratio of 0.667."""
        ratio = calculate_margin_ratio(equity=100_000, gross_exposure=150_000)
        assert ratio == pytest.approx(0.667, rel=0.01)

    def test_margin_ratio_zero_exposure(self):
        """Zero exposure returns 1.0."""
        ratio = calculate_margin_ratio(equity=100_000, gross_exposure=0)
        assert ratio == 1.0

    def test_margin_call_triggered(self):
        """Margin call when ratio below threshold."""
        # 4x leverage = 25% margin = exactly at threshold
        # Slightly below should trigger
        triggered = check_margin_call(
            equity=24_000,  # Just below 25%
            gross_exposure=100_000,
            maintenance_margin=0.25,
        )
        assert triggered == True

    def test_margin_call_not_triggered(self):
        """No margin call when ratio above threshold."""
        triggered = check_margin_call(
            equity=30_000,  # 30% margin
            gross_exposure=100_000,
            maintenance_margin=0.25,
        )
        assert triggered == False

    def test_margin_call_at_threshold(self):
        """At exactly threshold, not triggered (need to go below)."""
        triggered = check_margin_call(
            equity=25_000,  # Exactly 25%
            gross_exposure=100_000,
            maintenance_margin=0.25,
        )
        assert triggered == False


class TestApplyLeverage:
    """Tests for leverage application."""

    def test_no_leverage(self):
        """1x leverage = no borrowing."""
        gross, borrowed = apply_leverage(
            equity=100_000,
            leverage_factor=1.0,
        )
        assert gross == 100_000
        assert borrowed == 0

    def test_1_5x_leverage(self):
        """1.5x leverage = 50% borrowed."""
        gross, borrowed = apply_leverage(
            equity=100_000,
            leverage_factor=1.5,
        )
        assert gross == 150_000
        assert borrowed == 50_000

    def test_2x_leverage(self):
        """2x leverage = 100% borrowed."""
        gross, borrowed = apply_leverage(
            equity=100_000,
            leverage_factor=2.0,
        )
        assert gross == 200_000
        assert borrowed == 100_000

    def test_leverage_capped_at_max(self):
        """Leverage capped at max_leverage."""
        gross, borrowed = apply_leverage(
            equity=100_000,
            leverage_factor=3.0,
            max_leverage=2.0,
        )
        assert gross == 200_000  # Capped at 2x
        assert borrowed == 100_000

    def test_zero_leverage(self):
        """0x leverage = no exposure, no borrowing."""
        gross, borrowed = apply_leverage(
            equity=100_000,
            leverage_factor=0.0,
        )
        assert gross == 0
        assert borrowed == 0


class TestLeverageSimulator:
    """Tests for LeverageSimulator."""

    @pytest.fixture
    def config(self):
        return LeverageConfig(
            max_leverage=2.0,
            broker_spread=0.015,
            day_count=360,
            maintenance_margin=0.25,
            risk_off_to_cash=True,
        )

    @pytest.fixture
    def simulator(self, config):
        return LeverageSimulator(config)

    def test_leverage_on_with_gains(self, simulator):
        """Test leverage ON with positive returns."""
        date = pd.Timestamp("2020-01-01")

        state = simulator.step(
            date=date,
            equity=100_000,
            target_leverage=1.5,
            leverage_signal=True,
            interest_rate_pct=5.0,
            returns={"SPY": 0.01, "GLD": 0.005, "TLT": 0.002},
            target_weights={"SPY": 0.6, "GLD": 0.2, "TLT": 0.2},
        )

        # Check leverage applied
        assert state.leverage_factor == 1.5
        assert state.leverage_on == True
        assert state.gross_exposure == 150_000
        assert state.borrowed == 50_000

        # Check positions
        assert state.positions["SPY"] == pytest.approx(90_000)  # 150k * 0.6
        assert state.positions["GLD"] == pytest.approx(30_000)  # 150k * 0.2
        assert state.positions["TLT"] == pytest.approx(30_000)  # 150k * 0.2

        # Check interest accrued
        assert state.daily_interest > 0
        assert state.cumulative_interest == state.daily_interest

        # Check equity increased (gains - interest)
        # Return on 150k exposure with weighted returns
        expected_return = 150_000 * (0.6*0.01 + 0.2*0.005 + 0.2*0.002)
        expected_interest = simulator.config.broker_spread
        # Equity should be close to 100k + returns - interest
        assert state.equity > 100_000  # Should have gained

    def test_leverage_off_to_cash(self, simulator):
        """Test leverage OFF goes to BIL."""
        date = pd.Timestamp("2020-01-01")

        state = simulator.step(
            date=date,
            equity=100_000,
            target_leverage=1.5,
            leverage_signal=False,  # OFF
            interest_rate_pct=5.0,
            returns={"SPY": 0.01, "BIL": 0.0001},
            target_weights={"SPY": 1.0},
        )

        # Should be in cash
        assert state.leverage_factor == 0.0
        assert state.leverage_on == False
        assert state.borrowed == 0
        assert "BIL" in state.positions
        assert state.positions["BIL"] == pytest.approx(100_000)

        # No interest when not leveraged
        assert state.daily_interest == 0

    def test_margin_event_logged(self, simulator):
        """Test margin event is logged on breach."""
        # First get into leveraged position
        date1 = pd.Timestamp("2020-01-01")
        state1 = simulator.step(
            date=date1,
            equity=100_000,
            target_leverage=2.0,
            leverage_signal=True,
            interest_rate_pct=5.0,
            returns={"SPY": 0.0},
            target_weights={"SPY": 1.0},
        )

        # Now simulate big loss that breaches margin
        # At 2x leverage, a 40% drop puts us at 20% margin (below 25%)
        date2 = pd.Timestamp("2020-01-02")
        state2 = simulator.step(
            date=date2,
            equity=state1.equity,
            target_leverage=2.0,
            leverage_signal=True,
            interest_rate_pct=5.0,
            returns={"SPY": -0.40},  # 40% drop
            target_weights={"SPY": 1.0},
        )

        # Should have margin event
        assert state2.margin_call == True
        assert len(simulator.margin_events) > 0
        assert date2 in simulator.margin_events

        # Should be deleveraged to 1.0x
        assert state2.leverage_factor == 1.0

    def test_reset(self, simulator):
        """Test simulator reset."""
        date = pd.Timestamp("2020-01-01")
        simulator.step(
            date=date,
            equity=100_000,
            target_leverage=1.5,
            leverage_signal=True,
            interest_rate_pct=5.0,
            returns={"SPY": 0.01},
            target_weights={"SPY": 1.0},
        )

        assert simulator.cumulative_interest > 0

        simulator.reset()

        assert simulator.cumulative_interest == 0
        assert len(simulator.margin_events) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
