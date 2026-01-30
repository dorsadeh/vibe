"""Technical indicators: SMA, EMA, RSI, and slope calculations."""

from typing import Optional
import pandas as pd
import numpy as np


def sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        series: Price series
        window: Lookback window (e.g., 20, 50, 200)

    Returns:
        SMA series (NaN for first window-1 values)
    """
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int, adjust: bool = True) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        series: Price series
        span: EMA span/period (e.g., 12, 26)
        adjust: Use adjust=True for standard EMA calculation

    Returns:
        EMA series
    """
    return series.ewm(span=span, adjust=adjust).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Uses the standard Wilder smoothing method (EMA with alpha=1/period).

    Args:
        series: Price series
        period: RSI period (default 14)

    Returns:
        RSI series (0-100 scale)
    """
    # Calculate price changes
    delta = series.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss

    # Handle division by zero (no losses = RSI of 100)
    rsi_values = 100 - (100 / (1 + rs))

    # Where avg_loss is 0, RSI should be 100
    rsi_values = rsi_values.replace([np.inf, -np.inf], 100)

    return rsi_values


def slope(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate the slope (rate of change) of a series over N periods.

    Returns the annualized percentage change, useful for trend strength.

    Args:
        series: Any time series (typically a moving average)
        window: Lookback window for slope calculation

    Returns:
        Slope as percentage change over window (not annualized)
    """
    return series.pct_change(periods=window) * 100


def slope_linear(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate linear regression slope over a rolling window.

    More robust than simple pct_change for noisy data.

    Args:
        series: Any time series
        window: Lookback window

    Returns:
        Linear regression slope (points per day)
    """
    def _linreg_slope(y):
        if len(y) < window or np.isnan(y).any():
            return np.nan
        x = np.arange(len(y))
        # Linear regression slope: cov(x,y) / var(x)
        return np.cov(x, y)[0, 1] / np.var(x)

    return series.rolling(window=window, min_periods=window).apply(_linreg_slope, raw=True)


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence.

    Args:
        series: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = ema(series, fast_period)
    slow_ema = ema(series, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Args:
        series: Price series
        window: SMA window
        num_std: Number of standard deviations for bands

    Returns:
        Tuple of (middle_band, upper_band, lower_band)
    """
    middle = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return middle, upper, lower


class IndicatorCalculator:
    """
    Calculator for computing all indicators on price data.

    Computes indicators for a given asset and stores them in a DataFrame.
    """

    def __init__(
        self,
        sma_windows: list[int] = None,
        ema_windows: list[int] = None,
        rsi_period: int = 14,
        slope_window: int = 20,
    ):
        """
        Initialize indicator calculator.

        Args:
            sma_windows: List of SMA periods (default: [20, 50, 200])
            ema_windows: List of EMA periods (default: [12, 26])
            rsi_period: RSI period (default: 14)
            slope_window: Slope calculation window (default: 20)
        """
        self.sma_windows = sma_windows or [20, 50, 200]
        self.ema_windows = ema_windows or [12, 26]
        self.rsi_period = rsi_period
        self.slope_window = slope_window

    def compute(
        self,
        prices: pd.Series,
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Compute all configured indicators for a price series.

        Args:
            prices: Price series (typically adjusted close)
            prefix: Prefix for column names (e.g., 'SPY_')

        Returns:
            DataFrame with indicator columns
        """
        result = pd.DataFrame(index=prices.index)

        # Include the price itself
        result[f"{prefix}close"] = prices

        # SMAs
        for window in self.sma_windows:
            result[f"{prefix}SMA_{window}"] = sma(prices, window)

        # EMAs
        for window in self.ema_windows:
            result[f"{prefix}EMA_{window}"] = ema(prices, window)

        # RSI
        result[f"{prefix}RSI_{self.rsi_period}"] = rsi(prices, self.rsi_period)

        # Slope of key SMAs
        for window in self.sma_windows:
            sma_col = f"{prefix}SMA_{window}"
            result[f"{prefix}SMA_{window}_slope"] = slope(
                result[sma_col], self.slope_window
            )

        return result

    def compute_all(
        self,
        aligned_data: pd.DataFrame,
        symbols: list[str],
        price_field: str = "adj_close",
    ) -> pd.DataFrame:
        """
        Compute indicators for multiple assets.

        Args:
            aligned_data: DataFrame from DataLoader.load_aligned()
            symbols: List of symbols to compute indicators for
            price_field: Price field to use (default: adj_close)

        Returns:
            DataFrame with all indicators for all symbols
        """
        result = pd.DataFrame(index=aligned_data.index)

        for symbol in symbols:
            prices = aligned_data[(symbol, price_field)]
            indicators = self.compute(prices, prefix=f"{symbol}.")
            result = pd.concat([result, indicators], axis=1)

        return result


def compute_indicators(
    aligned_data: pd.DataFrame,
    symbols: list[str],
    sma_windows: list[int] = None,
    ema_windows: list[int] = None,
    rsi_period: int = 14,
    slope_window: int = 20,
) -> pd.DataFrame:
    """
    Convenience function to compute indicators.

    Args:
        aligned_data: DataFrame from DataLoader.load_aligned()
        symbols: List of symbols
        sma_windows: SMA periods (default: [20, 50, 200])
        ema_windows: EMA periods (default: [12, 26])
        rsi_period: RSI period
        slope_window: Slope window

    Returns:
        DataFrame with all indicators
    """
    calc = IndicatorCalculator(
        sma_windows=sma_windows,
        ema_windows=ema_windows,
        rsi_period=rsi_period,
        slope_window=slope_window,
    )
    return calc.compute_all(aligned_data, symbols)
