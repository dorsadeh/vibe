"""Tests for performance metrics."""

import pytest
import pandas as pd
import numpy as np

from src.metrics import (
    calculate_cagr,
    calculate_volatility,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_rolling_sharpe,
    compute_metrics,
)


@pytest.fixture
def equity_curve():
    """Sample equity curve: $100k growing to ~$200k over 2 years."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=504, freq="D")  # ~2 years

    # 10% annual return with 15% vol
    daily_return = 0.10 / 252
    daily_vol = 0.15 / np.sqrt(252)

    returns = np.random.normal(daily_return, daily_vol, len(dates))
    equity = 100_000 * np.cumprod(1 + returns)

    return pd.Series(equity, index=dates)


@pytest.fixture
def returns_series(equity_curve):
    """Daily returns from equity curve."""
    return equity_curve.pct_change().fillna(0)


class TestCAGR:
    """Tests for CAGR calculation."""

    def test_cagr_doubling_in_one_year(self):
        """Doubling in one year = 100% CAGR."""
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        equity = pd.Series(np.linspace(100_000, 200_000, 252), index=dates)

        cagr = calculate_cagr(equity, trading_days_per_year=252)

        # Should be close to 100%
        assert cagr == pytest.approx(1.0, rel=0.05)

    def test_cagr_no_change(self):
        """No change = 0% CAGR."""
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        equity = pd.Series([100_000] * 252, index=dates)

        cagr = calculate_cagr(equity)
        assert cagr == pytest.approx(0.0)

    def test_cagr_loss(self):
        """Loss should give negative CAGR."""
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        equity = pd.Series(np.linspace(100_000, 80_000, 252), index=dates)

        cagr = calculate_cagr(equity)
        assert cagr < 0


class TestVolatility:
    """Tests for volatility calculation."""

    def test_volatility_zero_for_constant(self):
        """Constant returns = zero volatility."""
        returns = pd.Series([0.001] * 252)
        vol = calculate_volatility(returns)
        assert vol == pytest.approx(0.0)

    def test_volatility_annualization(self):
        """Check annualization is correct."""
        np.random.seed(42)
        daily_returns = pd.Series(np.random.normal(0, 0.01, 252))

        vol = calculate_volatility(daily_returns)

        # Daily vol ~1%, annual should be ~15.8% (1% * sqrt(252))
        expected = daily_returns.std() * np.sqrt(252)
        assert vol == pytest.approx(expected)


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_no_drawdown(self):
        """Monotonically increasing = 0 drawdown."""
        equity = pd.Series([100, 110, 120, 130, 140, 150])
        dd = calculate_max_drawdown(equity)
        assert dd == 0.0

    def test_simple_drawdown(self):
        """Simple peak to trough."""
        # Peak at 100, trough at 80 = 20% drawdown
        equity = pd.Series([80, 90, 100, 90, 80, 85, 90])
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(-0.20)

    def test_multiple_drawdowns(self):
        """Return worst of multiple drawdowns."""
        # First DD: 100 -> 90 = 10%
        # Second DD: 110 -> 77 = 30%
        equity = pd.Series([100, 90, 95, 110, 100, 90, 80, 77, 85])
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(-0.30)


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_positive_excess_return(self, returns_series):
        """Positive returns should give positive Sharpe."""
        sharpe = calculate_sharpe_ratio(returns_series)
        assert sharpe > 0

    def test_sharpe_with_risk_free_rate(self, returns_series):
        """Sharpe with risk-free rate should be lower."""
        rf = pd.Series(2.0, index=returns_series.index)  # 2% annual

        sharpe_no_rf = calculate_sharpe_ratio(returns_series, None)
        sharpe_with_rf = calculate_sharpe_ratio(returns_series, rf)

        # With positive risk-free rate, Sharpe should be lower
        assert sharpe_with_rf < sharpe_no_rf

    def test_sharpe_zero_volatility(self):
        """Zero volatility should return 0 (not inf)."""
        returns = pd.Series([0.001] * 252)
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_sortino_only_gains(self):
        """Only positive returns = infinite Sortino."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.008, 0.012])
        sortino = calculate_sortino_ratio(returns)
        assert sortino == float("inf")

    def test_sortino_higher_than_sharpe(self, returns_series):
        """Sortino typically higher than Sharpe (smaller denominator)."""
        sharpe = calculate_sharpe_ratio(returns_series)
        sortino = calculate_sortino_ratio(returns_series)

        # Sortino uses downside vol which is typically smaller
        # So Sortino is typically higher
        assert sortino > sharpe or np.isclose(sortino, sharpe)


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_calmar_basic(self):
        """Basic Calmar calculation."""
        # 10% CAGR, 20% max DD = 0.5 Calmar
        calmar = calculate_calmar_ratio(cagr=0.10, max_drawdown=-0.20)
        assert calmar == pytest.approx(0.5)

    def test_calmar_no_drawdown(self):
        """No drawdown = infinite Calmar."""
        calmar = calculate_calmar_ratio(cagr=0.10, max_drawdown=0.0)
        assert calmar == float("inf")

    def test_calmar_negative_return(self):
        """Negative CAGR with drawdown."""
        calmar = calculate_calmar_ratio(cagr=-0.10, max_drawdown=-0.30)
        assert calmar < 0


class TestRollingMetrics:
    """Tests for rolling metrics."""

    def test_rolling_sharpe_length(self, returns_series):
        """Rolling Sharpe should have same length as input."""
        window = 126
        rolling = calculate_rolling_sharpe(returns_series, window=window)

        assert len(rolling) == len(returns_series)
        # First (window-1) values should be NaN
        assert rolling.iloc[:window - 1].isna().all()

    def test_rolling_sharpe_values(self, returns_series):
        """Rolling Sharpe values should be reasonable."""
        rolling = calculate_rolling_sharpe(returns_series, window=126)
        valid = rolling.dropna()

        # Sharpe typically between -3 and 3
        assert valid.min() > -5
        assert valid.max() < 5


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_compute_metrics_output(self, equity_curve):
        """Test that compute_metrics returns all fields."""
        leverage = pd.Series(1.5, index=equity_curve.index)
        interest = pd.Series(range(len(equity_curve)), index=equity_curve.index) * 10

        metrics = compute_metrics(
            equity_curve=equity_curve,
            leverage_series=leverage,
            cumulative_interest=interest,
            margin_events=[],
        )

        # Check all fields exist
        assert hasattr(metrics, "cagr")
        assert hasattr(metrics, "volatility")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "calmar_ratio")
        assert hasattr(metrics, "total_interest_paid")
        assert hasattr(metrics, "avg_leverage")

        # Check reasonable values
        assert 0 < metrics.cagr < 1  # Between 0% and 100% CAGR
        assert 0 < metrics.volatility < 1  # Reasonable vol
        assert metrics.max_drawdown < 0  # Drawdowns are negative
        assert metrics.avg_leverage == 1.5
        assert metrics.trading_days == len(equity_curve)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
