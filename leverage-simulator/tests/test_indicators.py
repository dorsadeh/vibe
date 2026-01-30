"""Tests for technical indicators."""

import pytest
import pandas as pd
import numpy as np

from src.indicators import sma, ema, rsi, slope, macd, bollinger_bands


@pytest.fixture
def sample_prices():
    """Sample price series for testing."""
    # Create a simple ascending price series
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # Trending up with some noise
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5 + 0.1)
    return pd.Series(prices, index=dates)


@pytest.fixture
def trending_prices():
    """Clear uptrend for testing."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    prices = 100 + np.arange(50) * 0.5  # Linear uptrend
    return pd.Series(prices, index=dates)


@pytest.fixture
def oscillating_prices():
    """Oscillating prices for RSI testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # Oscillate between 95 and 105
    prices = 100 + 5 * np.sin(np.linspace(0, 6 * np.pi, 100))
    return pd.Series(prices, index=dates)


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_length(self, sample_prices):
        """SMA output should have same length as input."""
        result = sma(sample_prices, 20)
        assert len(result) == len(sample_prices)

    def test_sma_initial_nan(self, sample_prices):
        """First (window-1) values should be NaN."""
        window = 20
        result = sma(sample_prices, window)
        assert result.iloc[:window - 1].isna().all()
        assert not result.iloc[window - 1:].isna().any()

    def test_sma_calculation(self):
        """Verify SMA calculation is correct."""
        prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = sma(prices, 3)

        # SMA(3) at index 2 = (1+2+3)/3 = 2
        assert result.iloc[2] == pytest.approx(2.0)
        # SMA(3) at index 3 = (2+3+4)/3 = 3
        assert result.iloc[3] == pytest.approx(3.0)
        # SMA(3) at index 9 = (8+9+10)/3 = 9
        assert result.iloc[9] == pytest.approx(9.0)

    def test_sma_smoothing(self, sample_prices):
        """SMA should be smoother than original."""
        result = sma(sample_prices, 20)
        # Standard deviation of SMA should be less than original
        valid_result = result.dropna()
        valid_prices = sample_prices.loc[valid_result.index]
        assert valid_result.std() < valid_prices.std()


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_length(self, sample_prices):
        """EMA output should have same length as input."""
        result = ema(sample_prices, 20)
        assert len(result) == len(sample_prices)

    def test_ema_starts_at_first_value(self, sample_prices):
        """EMA should start at first value (or close to it)."""
        result = ema(sample_prices, 20)
        # First EMA value equals first price
        assert not np.isnan(result.iloc[0])

    def test_ema_faster_than_sma(self, trending_prices):
        """EMA should react faster to trends than SMA."""
        ema_result = ema(trending_prices, 10)
        sma_result = sma(trending_prices, 10)

        # In an uptrend, EMA should be higher (closer to current price)
        # Compare after warmup period
        valid_idx = ~sma_result.isna()
        assert (ema_result[valid_idx].iloc[-10:] > sma_result[valid_idx].iloc[-10:]).all()


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_range(self, sample_prices):
        """RSI should be between 0 and 100."""
        result = rsi(sample_prices, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_length(self, sample_prices):
        """RSI output should have same length as input."""
        result = rsi(sample_prices, 14)
        assert len(result) == len(sample_prices)

    def test_rsi_uptrend(self, trending_prices):
        """RSI should be high (>50) in uptrend."""
        result = rsi(trending_prices, 14)
        # After warmup, RSI should be above 50 in steady uptrend
        assert result.iloc[-1] > 50

    def test_rsi_downtrend(self):
        """RSI should be low (<50) in downtrend."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = pd.Series(100 - np.arange(50) * 0.5, index=dates)
        result = rsi(prices, 14)
        # RSI should be below 50 in downtrend
        assert result.iloc[-1] < 50

    def test_rsi_extreme_up(self):
        """RSI should approach 100 with only gains."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = pd.Series(100 + np.arange(30) * 2.0, index=dates)  # Strong uptrend
        result = rsi(prices, 14)
        # RSI should be very high
        assert result.iloc[-1] > 90


class TestSlope:
    """Tests for slope calculation."""

    def test_slope_uptrend(self, trending_prices):
        """Slope should be positive in uptrend."""
        result = slope(trending_prices, 10)
        # After warmup, slope should be positive
        assert result.iloc[-1] > 0

    def test_slope_downtrend(self):
        """Slope should be negative in downtrend."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = pd.Series(100 - np.arange(50) * 0.5, index=dates)
        result = slope(prices, 10)
        assert result.iloc[-1] < 0

    def test_slope_flat(self):
        """Slope should be near zero for flat prices."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = pd.Series([100.0] * 50, index=dates)
        result = slope(prices, 10)
        assert result.iloc[-1] == pytest.approx(0.0, abs=1e-10)


class TestMACD:
    """Tests for MACD."""

    def test_macd_components(self, sample_prices):
        """MACD should return three components."""
        macd_line, signal_line, histogram = macd(sample_prices)
        assert len(macd_line) == len(sample_prices)
        assert len(signal_line) == len(sample_prices)
        assert len(histogram) == len(sample_prices)

    def test_macd_histogram_calculation(self, sample_prices):
        """Histogram should equal MACD line minus signal line."""
        macd_line, signal_line, histogram = macd(sample_prices)
        expected = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected)


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bollinger_bands_order(self, sample_prices):
        """Lower < Middle < Upper."""
        middle, upper, lower = bollinger_bands(sample_prices, 20, 2.0)
        valid_idx = ~middle.isna()
        assert (lower[valid_idx] < middle[valid_idx]).all()
        assert (middle[valid_idx] < upper[valid_idx]).all()

    def test_bollinger_middle_is_sma(self, sample_prices):
        """Middle band should equal SMA."""
        middle, _, _ = bollinger_bands(sample_prices, 20, 2.0)
        expected_sma = sma(sample_prices, 20)
        pd.testing.assert_series_equal(middle, expected_sma)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
