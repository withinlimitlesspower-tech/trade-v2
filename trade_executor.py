```py
"""
Trade Executor Module for Intelligent Cryptocurrency Trading Bot.

This module handles trade execution on Binance with comprehensive risk management,
including position sizing, stop-loss, and take-profit orders. It integrates with
the Binance API for order execution and maintains trade state for monitoring.
"""

import os
import time
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import json

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import (
    ORDER_TYPE_MARKET,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_STOP_LOSS_LIMIT,
    ORDER_TYPE_TAKE_PROFIT_LIMIT,
    SIDE_BUY,
    SIDE_SELL,
    TIME_IN_FORCE_GTC
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Enumeration of supported order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT_LIMIT"


class TradeSide(Enum):
    """Enumeration of trade sides."""
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(Enum):
    """Enumeration of trade statuses."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class TradeConfig:
    """Configuration for trade execution parameters."""
    
    # Position sizing
    max_position_size_percent: float = 0.02  # 2% of portfolio per trade
    max_total_exposure_percent: float = 0.10  # 10% max total exposure
    
    # Risk management
    default_stop_loss_percent: float = 0.02  # 2% stop loss
    default_take_profit_percent: float = 0.04  # 4% take profit
    trailing_stop_activation_percent: float = 0.01  # 1% profit to activate trailing
    trailing_stop_distance_percent: float = 0.005  # 0.5% trailing distance
    
    # Order execution
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    order_timeout_seconds: float = 30.0
    
    # Slippage protection
    max_slippage_percent: float = 0.001  # 0.1% max slippage for market orders
    
    # Minimum order value in USDT (Binance minimum)
    min_order_value_usdt: float = 10.0
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not (0 < self.max_position_size_percent <= 1):
            raise ValueError("max_position_size_percent must be between 0 and 1")
        if not (0 < self.max_total_exposure_percent <= 1):
            raise ValueError("max_total_exposure_percent must be between 0 and 1")
        if not (0 < self.default_stop_loss_percent <= 1):
            raise ValueError("default_stop_loss_percent must be between 0 and 1")
        if not (0 < self.default_take_profit_percent <= 1):
            raise ValueError("default_take_profit_percent must be between 0 and 1")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.min_order_value_usdt <= 0:
            raise ValueError("min_order_value_usdt must be positive")
        return True


@dataclass
class TradeSignal:
    """Represents a trading signal from analysis."""
    
    symbol: str
    side: TradeSide
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    quantity: Optional[float] = None
    confidence: float = 0.5
    strategy_name: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    
    def validate(self) -> bool:
        """Validate trade signal parameters."""
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("Invalid symbol")
        if not isinstance(self.side, TradeSide):
            raise ValueError("Invalid trade side")
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")
        return True


@dataclass
class Trade:
    """Represents an executed trade with full details."""
    
    signal: TradeSignal
    order_id: str = ""
    status: TradeStatus = TradeStatus.PENDING
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    commission: Optional[float] = None
    commission_asset: str = ""
    entry_time: Optional[float] = None
    exit_time: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for logging/storage."""
        return {
            "symbol": self.signal.symbol,
            "side": self.signal.side.value,
            "order_id": self.order_id,
            "status": self.status.value,
            "entry_price": self.executed_price,
            "quantity": self.executed_quantity,
            "stop_loss": self.signal.stop_loss,
            "take_profit": self.signal.take_profit,
            "commission": self.commission,
            "commission_asset": self.commission_asset,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "strategy": self.signal.strategy_name,
            "confidence": self.signal.confidence
        }


class TradeExecutor:
    """
    Executes trades on Binance with comprehensive risk management.
    
    Features:
    - Position sizing based on portfolio allocation
    - Automatic stop-loss and take-profit orders
    - Slippage protection for market orders
    - Retry logic for failed orders
    - Trade state management and logging
    
    Usage:
        executor = TradeExecutor()
        signal = TradeSignal(symbol="BTCUSDT", side=TradeSide.BUY, ...)
        trade = await executor.execute_trade(signal)
    """
    
    def __init__(
        self,
        config: Optional[TradeConfig] = None,
        test_mode: bool = False
    ):
        """
        Initialize the trade executor.
        
        Args:
            config: Trade configuration (uses defaults if None)
            test_mode: If True, uses Binance testnet
        
        Raises:
            ValueError: If configuration is invalid
            ConnectionError: If unable to connect to Binance API
        """
        self.config = config or TradeConfig()
        self.config.validate()
        
        # Initialize Binance client with API keys from environment variables
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError(
                "Binance API credentials not found in environment variables. "
                "Please set BINANCE_API_KEY and BINANCE_API_SECRET."
            )
        
        try:
            if test_mode:
                self.client = Client(api_key, api_secret, testnet=True)
                logger.info("Initialized Binance testnet client")
            else:
                self.client = Client(api_key, api_secret)
                logger.info("Initialized Binance production client")
            
            # Verify connection by fetching account info
            self.client.get_account()
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Binance API: {e}")
        
        # Track active trades and positions
        self.active_trades: Dict[str, List[Trade]] = {}
        self.position_tracker: Dict[str, Dict[str, Any]] = {}
        
        # Exchange info cache for symbol validation
        self._exchange_info_cache: Optional[Dict[str, Any]] = None
        
        logger.info(f"TradeExecutor initialized with config: {self.config}")
    
    def _get_exchange_info(self) -> Dict[str, Any]:
        """
        Get and cache exchange information.
        
        Returns:
            Exchange info dictionary
        
        Raises:
            ConnectionError: If unable to fetch exchange info
        """
        if self._exchange_info_cache is None:
            try:
                self._exchange_info_cache = self.client.get_exchange_info()
                logger.debug("Fetched exchange info")
            except Exception as e:
                raise ConnectionError(f"Failed to fetch exchange info: {e}")
        
        return self._exchange_info_cache
    
    def _get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol-specific trading rules.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
        
        Returns:
            Symbol information dictionary
        
        Raises:
            ValueError: If symbol is not found or invalid
        """
        exchange_info = self._get_exchange_info()
        
        for sym_info in exchange_info.get("symbols", []):
            if sym_info["symbol"] == symbol.upper():
                return sym_info
        
        raise ValueError(f"Symbol {symbol} not found on Binance")
    
    def _get_symbol_precision(self, symbol: str) -> Tuple[int, int]:
        """
        Get quantity and price precision for a symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Tuple of (quantity_precision, price_precision)
        
        Raises:
            ValueError: If symbol is not found
        """
        sym_info = self._get_symbol_info(symbol)
        
        # Find the lot size filter for quantity precision
        for filter_item in sym_info.get("filters", []):
            if filter_item["filterType"] == "LOT_SIZE":
                step_size = float(filter_item["stepSize"])
                qty_precision = len(str(step_size).split(".")[-1].rstrip("0"))
                
                # Handle scientific notation
                if "e" in str(step_size).lower():
                    qty_precision = abs(int(str(step_size).split("e")[-1]))
                
                break
        
        # Find the price filter for price precision
        for filter_item in sym_info.get("filters", []):
            if filter_item["filterType"] == "PRICE_FILTER":
                tick_size = float(filter_item["tickSize"])
                price_precision = len(str(tick_size).split(".")[-1].rstrip("0"))
                
                # Handle scientific notation
                if "e" in str(tick_size).lower():
                    price_precision = abs(int(str(tick_size).split("e")[-1]))
                
                break
        
        return qty_precision, price_precision
    
    def _calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        side: TradeSide
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the trade
            side: Trade side (BUY or SELL)
        
        Returns:
            Quantity to trade (in base asset)
        
        Raises:
            ValueError: If unable to calculate position size
        """
        try:
            # Get account balance for quote asset (USDT)
            account_info = self.client.get_account()
            
            # Find USDT balance (or quote asset)
            usdt_balance = 0.0
            for balance in account_info.get("balances", []):
                if balance["asset"] == "USDT":
                    usdt_balance = float(balance["free"])
                    break
            
            if usdt_balance <= 0:
                logger.warning(f"No USDT balance available for trading {symbol}")
                return 0.0
            
            # Calculate position size based on portfolio percentage
            position_value_usdt = usdt_balance * self.config.max_position_size_percent
            
            # Check against minimum order value
            if position_value_usdt < self.config.min_order_value_usdt:
                logger.warning(
                    f"Position value {position_value_usdt:.2f} USDT is below minimum "
                    f"{self.config.min_order_value_usdt} USDT"
                )
                return 0.0
            
            # Calculate quantity in base asset (e.g., BTC for BTCUSDT)
            quantity_base_asset = position_value_usdt / entry_price
            
            # Get precision rules and round down quantity
            qty_precision, _ = self._get_symbol_precision(symbol)
            
            # Round down to avoid exceeding balance requirements
            quantity_decimal = Decimal(str(quantity_base_asset))
            precision_decimal = Decimal(f"1e-{qty_precision}")
            
            rounded_quantity = float(
                quantity_decimal.quantize(precision_decimal, rounding=ROUND_DOWN)
            )
            
            logger.info(
                f"Calculated position size for {symbol}: "
                f"{rounded_quantity} ({position_value_usdt:.2f} USDT)"
            )
            
            return rounded_quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            raise ValueError(f"Failed to calculate position size: {e}")
    
    def _validate_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        side: Optional[TradeSide] = None
    ) -> bool:
        """
        Validate order parameters against exchange rules.
        
        Args:
            symbol: Trading pair symbol
            quantity: Order quantity in base asset
            price: Order price (None for market orders)
            side: Trade side
        
        Returns:
            True if valid, raises exception otherwise
        
        Raises:
            ValueError: If order parameters are invalid
        """
        sym_info = self._get_symbol_info(symbol)
        
        # Check if trading is enabled for this symbol
        if sym_info.get("status") != "TRADING":
            raise ValueError(f"Trading is not active for {symbol}")
        
        # Check order filters
        for filter_item in sym_info.get("filters", []):
            
            if filter_item["filterType"] == "LOT_SIZE":
                min_qty = float(filter_item["minQty"])
                max_qty = float(filter_item["maxQty"])
                step_size = float(filter_item["stepSize"])
                
                if quantity < min_qty:
                    raise ValueError(
                        f"Quantity {quantity} is below minimum {min_qty} for {symbol}"
                    )
                
                if quantity > max_qty:
                    raise ValueError(
                        f"Quantity {quantity} exceeds maximum {max_qty} for {symbol}"
                    )
                
                # Check step size alignment (with tolerance for floating point)
                remainder = quantity % step_size
                if remainder > step_size * 1e-8 and step_size - remainder > step_size * 1e-8:
                    raise ValueError(
                        f"Quantity {quantity} does not align with step size {step_size}"
                    )
            
            elif filter_item["filterType"] == "MIN_NOTIONAL":
                min_notional = float(filter_item["minNotional"])
                
                if price is not None and quantity * price < min_notional:
                    raise ValueError(
                        f"Order value {quantity * price:.2f} is below minimum "
                        f"notional {min_notional}"
                    )
            
            elif filter_item["filterType"] == "PRICE_FILTER" and price is not None:
                min_price = float(filter_item["minPrice"])
                max_price = float(filter_item["maxPrice"])
                tick_size = float(filter_item["tickSize"])
                
                if price < min_price or price > max_price:
                    raise ValueError(
                        f"Price {price} is outside allowed range "
                        f"[{min_price}, {max_price}]"
                    )
                
                # Check tick size alignment (with tolerance)
                remainder_price = price % tick_size
                if remainder_price > tick_size * 1e-8 and tick_size - remainder_price > tick_size * 1e-8:
                    raise ValueError(
                        f"Price {price} does not align with tick size {tick_size}"
                    )
        
        return True
    
    def _calculate_stop_loss_price(
        self,
        entry_price: float,
        side: TradeSide,
        stop_loss_percent: Optional[float] = None
    ) -> float:
        """
        Calculate stop-loss price based on entry price and percentage.
        
        Args:
            entry_price: Entry price of the trade
            side: Trade side (BUY or SELL)
            stop_loss_percent: Stop loss percentage (uses default if None)
        
        Returns:
            Stop loss price
        
        Raises:
            ValueError: If calculation fails or invalid parameters
        """
        sl_percent = stop_loss_percent or self.config.default_stop_loss_percent
        
        if sl_percent <= 0 or sl_percent >= 1:
            raise ValueError(f"Invalid stop loss percentage: {sl_percent}")
        
        if side == TradeSide.BUY:
            stop_loss_price = entry_price * (1 - sl_percent)
            
            # Ensure stop loss is below entry for long positions
            if stop_loss_price >= entry_price:
                raise ValueError(
                    f"Stop loss {stop_loss_price} must be below entry {entry_price} for long"
                )
        
        elif side == TradeSide.SELL:
            stop_loss_price = entry_price * (1 + sl_percent)
            
            # Ensure stop loss is above entry for short positions
            if stop_loss_price <= entry_price:
                raise ValueError(
                    f"Stop loss {stop_loss_price} must be above entry {entry_price} for short"
                )
        
        else:
            raise ValueError(f"Invalid trade side: {side}")
        
        logger.debug(
            f"Calculated stop loss at {stop_loss_price:.8f} "
            f"({-sl_percent*100:.2f}% from entry {entry_price})"
        )
        
        return stop_loss_price
    
    def _calculate_take_profit_price(
        self,
        entry_price: float,
        side: TradeSide,
        take_profit_percent: Optional[float] = None
    ) -> float:
        """
        Calculate take-profit price based on entry price and percentage.
        
        Args:
            entry_price: Entry price of the trade
            side: Trade side (BUY or SELL)
            take_profit_percent: Take profit percentage (uses default if None)
        
        Returns:
            Take profit price
        
        Raises:
            ValueError: If calculation fails or invalid parameters
        """
        tp_percent = take_profit_percent or self.config.default_take_profit_percent
        
        if tp_percent <= 0 or tp_percent >= 1:
            raise ValueError(f"Invalid take profit percentage: {tp_percent}")
        
        if side == TradeSide.BUY:
            take_profit_price = entry_price * (1 + tp_percent)
            
            # Ensure take profit is above entry for long positions
            if take_profit_price <= entry_price:
                raise ValueError(
                    f"Take profit {take_profit_price} must be above entry "
                    f"{entry_price} for long"
                )
        
        elif side == TradeSide.SELL:
            take_profit_price = entry_price * (1 - tp_percent)
            
            # Ensure take profit is below entry for short positions
            if take_profit_price >= entry_price:
                raise ValueError(
                    f"Take profit {take_profit_price} must be below entry "
                    f"{entry_price} for short"
                )
        
        else:
            raise ValueError(f"Invalid trade side: {side}")
        
        logger.debug(
            f"Calculated take profit at {take_profit_price:.8f} "
            f"(+{tp_percent*100:.2f}% from entry {entry_price})"
        )
        
        return take_profit_price
    
    def _place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        new_client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
         Place a market order on Binance.
         
         Args:
             symbol: Trading pair symbol (e.g., 'BTCUSDT')
             side: 'BUY' or 'SELL'
             quantity: Order quantity in base asset (e.g., BTC)
             new_client_order_id: Optional custom order ID for tracking
         
         Returns:
             Order response dictionary from Binance API
         
         Raises:
             BinanceAPIException: If API returns an error
             BinanceOrderException: If order placement fails validation 
         """
         
         try:

             order_params={
                 'symbol':symbol,
                 'side':side,
                 'type':ORDER_TYPE_MARKET,
                 'quantity':quantity,

             }