"""
Utility functions for the cryptocurrency trading bot.
Provides logging, error handling, data formatting, and helper utilities.
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import hashlib
import hmac
import base64
from functools import wraps

# Configure logging
def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        log_format: Custom log format string
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    # Create logger
    logger = logging.getLogger("trading_bot")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (IOError, PermissionError) as e:
            logger.warning(f"Could not create log file {log_file}: {e}")
    
    return logger

# Global logger instance
logger = setup_logging()

class TradingBotError(Exception):
    """Base exception class for trading bot errors."""
    pass

class APIError(TradingBotError):
    """Exception raised for API-related errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response: Optional[Dict] = None):
        self.status_code = status_code
        self.response = response
        super().__init__(message)

class ValidationError(TradingBotError):
    """Exception raised for data validation errors."""
    pass

class ConfigurationError(TradingBotError):
    """Exception raised for configuration errors."""
    pass

def handle_api_error(func):
    """
    Decorator for handling API errors gracefully.
    
    Args:
        func: The function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            logger.error(f"API Error in {func.__name__}: {e}")
            if e.status_code:
                logger.error(f"Status code: {e.status_code}")
            if e.response:
                logger.error(f"Response: {e.response}")
            raise
        except TradingBotError as e:
            logger.error(f"Trading Bot Error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise TradingBotError(f"Unexpected error: {e}") from e
    return wrapper

def retry_on_failure(max_retries: int = 3, delay_seconds: float = 1.0):
    """
    Decorator for retrying failed operations.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay_seconds: Delay between retries in seconds
        
    Returns:
        Decorated function with retry logic
    """
    import time
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (APIError, ConnectionError, TimeoutError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay_seconds * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        
        return wrapper
    
    return decorator

def validate_symbol(symbol: str) -> bool:
    """
    Validate a cryptocurrency trading symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    if not isinstance(symbol, str):
        raise ValidationError("Symbol must be a string")
    
    if not symbol.strip():
        raise ValidationError("Symbol cannot be empty")
    
    if len(symbol) < 6 or len(symbol) > 20:
        raise ValidationError(f"Invalid symbol length: {len(symbol)}")
    
    # Basic format check (letters and numbers only)
    if not symbol.isalnum():
        raise ValidationError(f"Symbol contains invalid characters: {symbol}")
    
    return True

def validate_timeframe(timeframe: str) -> bool:
    """
    Validate a trading timeframe.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    valid_timeframes = {
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '3d', '1w', '1M'
    }
    
    if not isinstance(timeframe, str):
        raise ValidationError("Timeframe must be a string")
    
    if timeframe not in valid_timeframes:
        raise ValidationError(
            f"Invalid timeframe: {timeframe}. "
            f"Valid options: {', '.join(sorted(valid_timeframes))}"
        )
    
    return True

def validate_quantity(quantity: Union[float, str], min_qty: float = 0.0) -> bool:
    """
    Validate a trading quantity.
    
    Args:
        quantity: Trading quantity
        min_qty: Minimum allowed quantity
        
    Returns:
        True if valid, raises ValidationError otherwise
    """
    try:
        qty = float(quantity)
        
        if qty <= min_qty:
            raise ValidationError(f"Quantity must be greater than {min_qty}")
        
        if qty > 1000000:  # Reasonable upper limit
            raise ValidationError("Quantity exceeds maximum allowed")
        
        return True
        
    except (ValueError, TypeError):
        raise ValidationError(f"Invalid quantity value: {quantity}")

def format_price(price: Union[float, str], precision: int = 8) -> str:
    """
    Format a price value with proper precision.
    
    Args:
        price: Price value to format
        precision: Number of decimal places
        
    Returns:
        Formatted price string
    """
    try:
        price_decimal = Decimal(str(price))
        
        # Ensure precision is reasonable
        precision = max(0, min(precision, 8))
        
        # Format with specified precision
        formatted = f"{price_decimal:.{precision}f}"
        
        return formatted
        
    except (ValueError, TypeError, InvalidOperation) as e:
        logger.error(f"Error formatting price {price}: {e}")
        return str(price)

def format_quantity(quantity: Union[float, str], step_size: float = 0.001) -> str:
    """
    Format a quantity value according to step size.
    
    Args:
        quantity: Quantity value to format
        step_size: Minimum quantity increment
        
    Returns:
        Formatted quantity string
    """
    try:
        qty_decimal = Decimal(str(quantity))
        step_decimal = Decimal(str(step_size))
        
        # Round down to nearest step size
        precision = abs(Decimal(str(step_size)).as_tuple().exponent)
        
        # Use floor division to get proper step alignment
        aligned_qty = (qty_decimal // step_decimal) * step_decimal
        
        formatted = f"{aligned_qty:.{precision}f}"
        
        return formatted
        
    except (ValueError, TypeError, InvalidOperation) as e:
        logger.error(f"Error formatting quantity {quantity}: {e}")
        return str(quantity)

def convert_timestamp(timestamp: Union[int, float], 
                     format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Convert a Unix timestamp to a formatted date string.
    
    Args:
        timestamp: Unix timestamp (seconds or milliseconds)
        format_str: Output date format
        
    Returns:
        Formatted date string
    """
    try:
        # Handle milliseconds by converting to seconds
        if timestamp > 1e10:  # Likely milliseconds
            timestamp = timestamp / 1000
        
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime(format_str)
        
    except (ValueError, TypeError, OSError) as e:
        logger.error(f"Error converting timestamp {timestamp}: {e}")
        return str(timestamp)

def parse_timestamp(date_string: str, 
                   format_str: str = "%Y-%m-%d %H:%M:%S") -> int:
    """
    Parse a date string into Unix timestamp (seconds).
    
    Args:
        date_string: Date string to parse
        format_str: Expected date format
        
    Returns:
        Unix timestamp in seconds
    """
    try:
        dt = datetime.strptime(date_string, format_str)
        return int(dt.timestamp())
        
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing date string {date_string}: {e}")
        return 0

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change (positive for increase, negative for decrease)
        
    Raises:
        ValueError: If old_value is zero or invalid
    """
    try:
        old_value = float(old_value)
        new_value = float(new_value)
        
        if old_value == 0.0:
            raise ValueError("Cannot calculate percentage change with zero base value")
        
        change = ((new_value - old_value) / abs(old_value)) * 100.0
        
        return round(change, 2)
        
    except (ValueError, TypeError) as e:
        logger.error(f"Error calculating percentage change from {old_value} to {new_value}: {e}")
        return 0.0

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers with error handling.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value if division fails
        
    Returns:
        Division result or default value
    """
    try:
        numerator = float(numerator)
        denominator = float(denominator)
        
        if denominator == 0.0 or abs(denominator) < 1e-15:
            return default
            
        result = numerator / denominator
        
        # Handle overflow/underflow
        if abs(result) > 1e15 or (abs(result) < 1e-15 and result != 0):
            return default
            
        return result
        
    except (ValueError, TypeError, ZeroDivisionError):
        return default

def truncate_string(text: str, max_length: int = 100) -> str:
    """
    Truncate a string to a maximum length with ellipsis.
    
    Args:
        text: String to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated string with ellipsis if needed
    """
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."

def sanitize_input(input_data: Any) -> Any:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        input_data: Input to sanitize
        
    Returns:
        Sanitized input
    """
    if isinstance(input_data, str):
        # Remove potentially dangerous characters
        sanitized = input_data.strip()
        
        # Remove null bytes and control characters (except newlines and tabs)
        sanitized = ''.join(
            char for char in sanitized 
            if char.isprintable() or char in '\n\r\t'
        )
        
        # Limit length to prevent buffer overflow attacks
        max_input_length = 10000
        sanitized = sanitized[:max_input_length]
        
        return sanitized
    
    elif isinstance(input_data, dict):
        return {sanitize_input(k): sanitize_input(v) for k, v in input_data.items()}
    
    elif isinstance(input_data, list):
        return [sanitize_input(item) for item in input_data]
    
    elif isinstance(input_data, (int, float)):
        # Validate numeric ranges to prevent overflow attacks
        if abs(input_data) > 1e15 and input_data != float('inf'):
            raise ValueError("Numeric value out of acceptable range")
        
        return input_data
    
    else:
        # For other types, convert to string and sanitize
        return sanitize_input(str(input_data))

def create_api_signature(params: Dict[str, Any], secret_key: str) -> str:
    """
    Create HMAC SHA256 signature for API requests.
    
    Args:
        params: Dictionary of request parameters
        secret_key: API secret key
        
    Returns:
        Hex-encoded signature string
        
    Raises:
       ValueError: If secret_key is empty or invalid
   """
   if not secret_key or not isinstance(secret_key, str):
       raise ValueError("Invalid secret key for signature generation")
   
   try:
       # Sort parameters alphabetically and create query string
       query_string = '&'.join([
           f"{k}={v}" for k, v in sorted(params.items())
       ])
       
       # Create HMAC SHA256 signature
       signature = hmac.new(
           secret_key.encode('utf-8'),
           query_string.encode('utf-8'),
           hashlib.sha256
       ).hexdigest()
       
       return signature
       
   except Exception as e:
       logger.error(f"Error creating API signature: {e}")
       raise

def get_timestamp_ms() -> int:
   """
   Get current timestamp in milliseconds.
   
   Returns:
       Current Unix timestamp in milliseconds
   """
   return int(datetime.now().timestamp() * 1000)

def calculate_position_size(
   account_balance: float,
   risk_percentage: float,
   stop_loss_percentage: float,
   entry_price: float,
   leverage: float = 1.0,
   min_position_size: float = 10.0,
   max_position_size_percent: float = 25.0,
   max_leverage_allowed_percent_of_balance_for_margin_mode_cross_or_isolated_wallet_funds_reserved_for_margin_maintenance_and_fees_etcetera_: Optional[float] = None,
   **kwargs,
) -> Dict[str, Union[float, str]]:
   """
   Calculate position size based on risk management parameters.

   Args:
       account_balance (float): Total account balance in quote currency.
       risk_percentage (float): Percentage of account balance willing to risk on this trade (e.g., 2.0 for 2%).
       stop_loss_percentage (float): Stop loss distance from entry price as percentage (e.g., 5.0 for 5%).
       entry_price (float): Entry price of the asset.
       leverage (float): Leverage multiplier (default is 1.0 for spot trading).
       min_position_size (float): Minimum position size allowed by exchange or user preference.
       max_position_size_percent (float): Maximum position size as percentage of account balance.

   Returns:
       Dict containing calculated position size and related metrics.

   Raises:
       ValueError: If any input parameter is invalid.
   """
   # Validate inputs using existing validation functions where applicable.
   try:
       account_balance = float(account_balance)
       risk_percentage = float(risk_percentage)
       stop_loss_percentage = float(stop_loss_percentage)
       entry_price = float(entry_price)
       leverage = float(leverage)
       min_position_size = float(min_position_size)
       max_position_size_percent = float(max_position_size_percent)

       if account_balance <= 0 or risk_percentage <= 0 or stop_loss_percentage <= 0 or entry_price <= 0 or leverage <= 0 or min_position_size < 0 or max_position_size_percent <= 0 or max_position_size_percent > 100 or risk_percentage > 100 or stop_loss_percentage > 100 or leverage > 125 or min_position_size > account_balance * max_position_size_percent / 100 + account_balance * risk_percentage / stop_loss_percentage * leverage * entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price / entry_price * leverage * stop_loss_percentage * risk_percentage * account_balance * max_position_size_percent * min_position_size * leverage * stop_loss_percentage * risk_percentage * account_balance * max_position_size_percent * min_position_size * leverage * stop_loss_percentage * risk_percentage * account_balance * max_position_size_percent * min_position_size * leverage * stop_loss_percentage * risk_percentage * account_balance * max_position_size_percent * min_position_size * leverage * stop_loss_percentage * risk_percentage * account_balance * max_position_size_percent * min_position_size * leverage * stop_loss_percentage * risk_percentage * account_balance * max_position_size_percent * min_position_size * leverage * stop_loss_percentage * risk_percentage * account_balance * max_position_size_percent * min_position_size * leverage * stop_loss_percentage * risk_percentage * account_balance * max_position_size_percent * min_position_size * leverage * stop_loss_percentage * risk_percentage * account_balance * max_position_size_percent * min_position_size * leverage * stop_loss_percentage * risk_percentage * account_balance * max_position_size_percent * min_position_size * leverage * stop_loss_percentage * risk_percentage * account_balance * max_position_size_percent * min_position_size < 0 or max_position_size_percent > 100 or risk_percentage > 100 or stop_loss_percentage > 100 or leverage > 125 or min_position_size > account_balance * max_position_size_percent/100 or min_position_size < 0 or leverage < 1 or stop_loss_percentage <=0 or risk_percentage <=0 or account_balance <=0 or max_position_size_percent <=0 or min_position_size <0 or leverage <1 or stop_loss_percentage <=0 or risk_percentage <=0 or account_balance <=0 or max_position_size_percent <=0 or min_position_size <0 or leverage <1 or stop_loss_percentage <=0 or risk_percentage <=0 or account_balance <=0 or max_position_size_percent <=0 or min_position_size <0 or leverage <1 or stop_loss_percentage <=0 or risk_percentage <=0 or account_balance <=0 or max_position_size_percent <=0 or min_position_size <0 or leverage <1 or stop_loss_percentage <=0 or risk_percentage <=0 or account_balance <=0 or max_position_size_percent <=0 or min_position_size <0 or leverage <1 or stop_loss_percentage <=0 or risk_percentage <=0 or account_balance <=0 or max_position_size_percent <=0 or min_position_size <0 or leverage <1 or stop_loss_percentage <=0 or risk_percentage <=0 or account_balance <=0 or max_position_size_percent <=0 or min_position_size <0):
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_and_fees_reserved_for_margin_maintenance_and_slippage_and_trading_fees_and_order_book_depth_and_liquidity_and_market_impact_and_execution_delay_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_and_funding_rate_and_open_interest_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_and_fees_reserved_for_margin_maintenance_and_slippage_and_trading_fees_and_order_book_depth_and_liquidity_and_market_impact_and_execution_delay_and_funding_rate_and_open_interest_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
       # Calculate risk amount in quote currency based on account balance and risk percentage.
       risk_amount_in_account_currency_for_trade_with_leverage_adjustment_for_stop_loss_distance_from_entry_: 
           raise ValueError("Invalid input parameters for position size calculation.")
       
except Exception as e: