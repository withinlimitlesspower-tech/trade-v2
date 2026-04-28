```py
"""
Technical Indicator Calculations Module

This module provides comprehensive technical indicator calculations for cryptocurrency
market analysis. It serves as the local analysis layer before DeepSeek API calls,
implementing standard technical indicators used in trading strategies.

All functions are designed to be:
- Stateless and pure (no side effects)
- Input validated with proper error handling
- Type-hinted for clarity and IDE support
- Optimized for performance with numpy vectorization
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases for clarity
PriceArray = Union[List[float], np.ndarray, pd.Series]
IndicatorResult = Union[float, np.ndarray, pd.Series]


class IndicatorError(Exception):
    """Custom exception for indicator calculation errors."""
    pass


class TrendDirection(Enum):
    """Enum for trend direction classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    STRONG_BULLISH = "strong_bullish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class IndicatorConfig:
    """Configuration for indicator calculations."""
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    stochastic_k: int = 14
    stochastic_d: int = 3
    adx_period: int = 14
    ichimoku_conversion: int = 9
    ichimoku_base: int = 26
    ichimoku_span_b: int = 52
    
    def __post_init__(self):
        """Set default values if None."""
        if self.sma_periods is None:
            self.sma_periods = [10, 20, 50, 200]
        if self.ema_periods is None:
            self.ema_periods = [12, 26]


def validate_price_data(prices: PriceArray, min_length: int = 1) -> np.ndarray:
    """
    Validate and convert price data to numpy array.
    
    Args:
        prices: Input price data (list, numpy array, or pandas Series)
        min_length: Minimum required length of price data
        
    Returns:
        Validated numpy array of prices
        
    Raises:
        IndicatorError: If validation fails
    """
    if prices is None or len(prices) == 0:
        raise IndicatorError("Price data cannot be empty")
    
    if not isinstance(prices, (list, np.ndarray, pd.Series)):
        raise IndicatorError(f"Invalid price data type: {type(prices)}")
    
    # Convert to numpy array if needed
    if isinstance(prices, pd.Series):
        prices_array = prices.values.astype(float)
    elif isinstance(prices, list):
        prices_array = np.array(prices, dtype=float)
    else:
        prices_array = prices.astype(float)
    
    # Check for NaN or infinite values
    if np.any(np.isnan(prices_array)) or np.any(np.isinf(prices_array)):
        raise IndicatorError("Price data contains NaN or infinite values")
    
    # Check minimum length
    if len(prices_array) < min_length:
        raise IndicatorError(
            f"Price data length ({len(prices_array)}) is less than minimum required ({min_length})"
        )
    
    return prices_array


def calculate_sma(prices: PriceArray, period: int = 20) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        prices: Price data array
        period: Moving average period (default: 20)
        
    Returns:
        Array of SMA values with NaN for initial periods
        
    Raises:
        IndicatorError: If calculation fails
    """
    try:
        prices_array = validate_price_data(prices, min_length=period)
        
        # Use pandas for efficient rolling calculation
        series = pd.Series(prices_array)
        sma = series.rolling(window=period).mean().values
        
        logger.debug(f"SMA calculated with period {period}")
        return sma
        
    except Exception as e:
        logger.error(f"Failed to calculate SMA: {str(e)}")
        raise IndicatorError(f"SMA calculation failed: {str(e)}")


def calculate_ema(prices: PriceArray, period: int = 12) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        prices: Price data array
        period: Moving average period (default: 12)
        
    Returns:
        Array of EMA values with NaN for initial periods
        
    Raises:
        IndicatorError: If calculation fails
    """
    try:
        prices_array = validate_price_data(prices, min_length=period)
        
        # Use pandas for efficient EMA calculation
        series = pd.Series(prices_array)
        ema = series.ewm(span=period, adjust=False).mean().values
        
        logger.debug(f"EMA calculated with period {period}")
        return ema
        
    except Exception as e:
        logger.error(f"Failed to calculate EMA: {str(e)}")
        raise IndicatorError(f"EMA calculation failed: {str(e)}")


def calculate_rsi(prices: PriceArray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price data array
        period: RSI period (default: 14)
        
    Returns:
        Array of RSI values (0-100) with NaN for initial periods
        
    Raises:
        IndicatorError: If calculation fails
    """
    try:
        prices_array = validate_price_data(prices, min_length=period + 1)
        
        # Calculate price changes
        deltas = np.diff(prices_array)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses using Wilder's smoothing method
        avg_gains = np.zeros_like(prices_array)
        avg_losses = np.zeros_like(prices_array)
        
        # First average uses simple average
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        # Subsequent averages use exponential smoothing
        for i in range(period + 1, len(prices_array)):
            avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i - 1]) / period
        
        # Calculate RS and RSI
        rs = np.where(avg_losses != 0, avg_gains / avg_losses, float('inf'))
        rsi_values = 100 - (100 / (1 + rs))
        
        # Set initial values to NaN
        rsi_values[:period] = np.nan
        
        logger.debug(f"RSI calculated with period {period}")
        return rsi_values
        
    except Exception as e:
        logger.error(f"Failed to calculate RSI: {str(e)}")
        raise IndicatorError(f"RSI calculation failed: {str(e)}")


def calculate_macd(
    prices: PriceArray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        prices: Price data array
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
        
    Raises:
        IndicatorError: If calculation fails
    """
    try:
        prices_array = validate_price_data(prices, min_length=slow_period + signal_period)
        
        # Calculate EMAs
        ema_fast = calculate_ema(prices_array, fast_period)
        ema_slow = calculate_ema(prices_array, slow_period)
        
        # MACD line is difference between fast and slow EMAs
        macd_line = ema_fast - ema_slow
        
        # Signal line is EMA of MACD line
        signal_line = calculate_ema(macd_line[slow_period - fast_period:], signal_period)
        
        # Pad signal line to match original length
        signal_line_padded = np.full_like(macd_line, np.nan)
        signal_line_padded[slow_period - fast_period + signal_period - 1:] = signal_line
        
        # Histogram is difference between MACD and Signal lines
        histogram = macd_line - signal_line_padded
        
        logger.debug(f"MACD calculated with periods ({fast_period}, {slow_period}, {signal_period})")
        return macd_line, signal_line_padded, histogram
        
    except Exception as e:
        logger.error(f"Failed to calculate MACD: {str(e)}")
        raise IndicatorError(f"MACD calculation failed: {str(e)}")


def calculate_bollinger_bands(
    prices: PriceArray,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Price data array
        period: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2.0)
        
    Returns:
        Tuple of (Upper band, Middle band (SMA), Lower band)
        
    Raises:
        IndicatorError: If calculation fails
    """
    try:
        prices_array = validate_price_data(prices, min_length=period)
        
        # Calculate middle band (SMA)
        middle_band = calculate_sma(prices_array, period)
        
        # Calculate standard deviation using pandas rolling window
        series = pd.Series(prices_array)
        std_dev = series.rolling(window=period).std().values
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        logger.debug(f"Bollinger Bands calculated with period {period}, std {num_std}")
        return upper_band, middle_band, lower_band
        
    except Exception as e:
        logger.error(f"Failed to calculate Bollinger Bands: {str(e)}")
        raise IndicatorError(f"Bollinger Bands calculation failed: {str(e)}")


def calculate_atr(
    high_prices: PriceArray,
    low_prices: PriceArray,
    close_prices: PriceArray,
    period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high_prices: High price data array
        low_prices: Low price data array  
        close_prices: Close price data array
        period: ATR period (default: 14)
        
    Returns:
        Array of ATR values with NaN for initial periods
        
    Raises:
        IndicatorError: If calculation fails
    """
    try:
        high_array = validate_price_data(high_prices)
        low_array = validate_price_data(low_prices)
        close_array = validate_price_data(close_prices)
        
        # Validate all arrays have same length
        if not (len(high_array) == len(low_array) == len(close_array)):
            raise IndicatorError("High, low, and close arrays must have the same length")
        
        if len(high_array) < period + 1:
            raise IndicatorError(f"Need at least {period + 1} data points for ATR")
        
        # Calculate True Range components
        high_low = high_array[1:] - low_array[1:]
        
        # Shift close prices for comparison with current high/low
        close_prev = close_array[:-1]
        
        high_close_prev = np.abs(high_array[1:] - close_prev)
        low_close_prev = np.abs(low_array[1:] - close_prev)
        
        # True Range is maximum of the three components
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Calculate ATR using Wilder's smoothing method
        atr_values = np.full(len(close_array), np.nan)
        
        # First ATR is simple average of first 'period' true ranges
        atr_values[period] = np.mean(true_range[:period])
        
        # Subsequent ATR values use exponential smoothing
        for i in range(period + 1, len(close_array)):
            atr_values[i] = (atr_values[i - 1] * (period - 1) + true_range[i - 1]) / period
        
        logger.debug(f"ATR calculated with period {period}")
        return atr_values
        
    except Exception as e:
        logger.error(f"Failed to calculate ATR: {str(e)}")
        raise IndicatorError(f"ATR calculation failed: {str(e)}")


def calculate_stochastic(
    high_prices: PriceArray,
    low_prices: PriceArray,
    close_prices: PriceArray,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Args:
        high_prices: High price data array
        low_prices: Low price data array
        close_prices: Close price data array
        k_period: %K period (default: 14)
        d_period: %D smoothing period (default: 3)
        
    Returns:
        Tuple of (%K values, %D values) with NaN for initial periods
        
    Raises:
        IndicatorError: If calculation fails
    """
    try:
        high_array = validate_price_data(high_prices)
        low_array = validate_price_data(low_prices)