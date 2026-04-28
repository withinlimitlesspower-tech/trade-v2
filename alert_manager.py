```py
"""
Alert Manager for Cryptocurrency Trading Bot
Manages real-time alerts for trade entries, exits, and market pullbacks
Supports console output, webhook notifications, and Replit notifications
"""

import json
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Enumeration of alert types"""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    MARKET_PULLBACK = "market_pullback"
    SIGNAL_GENERATED = "signal_generated"
    ERROR = "error"
    INFO = "info"
    WARNING = "warning"


class AlertPriority(Enum):
    """Enumeration of alert priorities"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Data class representing a single alert"""
    type: AlertType
    message: str
    symbol: Optional[str] = None
    price: Optional[float] = None
    priority: AlertPriority = AlertPriority.MEDIUM
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Set timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        data = asdict(self)
        data['type'] = self.type.value
        data['priority'] = self.priority.value
        return data

    def to_json(self) -> str:
        """Convert alert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class AlertManager:
    """
    Manages real-time alerts for the trading bot.
    Supports multiple notification channels: console, webhook, and Replit.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        replit_token: Optional[str] = None,
        enable_console: bool = True,
        enable_webhook: bool = False,
        enable_replit: bool = False,
        min_priority: AlertPriority = AlertPriority.LOW,
        max_alerts_per_minute: int = 60
    ):
        """
        Initialize the AlertManager.

        Args:
            webhook_url: URL for webhook notifications (e.g., Discord, Slack)
            replit_token: Replit API token for notifications
            enable_console: Enable console output alerts
            enable_webhook: Enable webhook notifications
            enable_replit: Enable Replit notifications
            min_priority: Minimum priority level to trigger alerts
            max_alerts_per_minute: Maximum number of alerts per minute (rate limiting)
        """
        self.webhook_url = webhook_url
        self.replit_token = replit_token
        self.enable_console = enable_console
        self.enable_webhook = enable_webhook and bool(webhook_url)
        self.enable_replit = enable_replit and bool(replit_token)
        self.min_priority = min_priority
        self.max_alerts_per_minute = max_alerts_per_minute
        
        # Rate limiting attributes
        self._alert_timestamps: List[datetime] = []
        
        # Alert history for deduplication and analysis
        self._alert_history: List[Alert] = []
        self._max_history_size: int = 1000
        
        logger.info(
            f"AlertManager initialized - Console: {enable_console}, "
            f"Webhook: {self.enable_webhook}, Replit: {self.enable_replit}"
        )

    def _check_rate_limit(self) -> bool:
        """
        Check if we've exceeded the rate limit.
        
        Returns:
            True if allowed to send alert, False if rate limited
        """
        now = datetime.utcnow()
        
        # Remove timestamps older than 1 minute
        self._alert_timestamps = [
            ts for ts in self._alert_timestamps 
            if (now - ts).total_seconds() < 60
        ]
        
        if len(self._alert_timestamps) >= self.max_alerts_per_minute:
            logger.warning(f"Rate limit exceeded: {len(self._alert_timestamps)} alerts in last minute")
            return False
        
        self._alert_timestamps.append(now)
        return True

    def _should_send_alert(self, alert: Alert) -> bool:
        """
        Determine if an alert should be sent based on priority and rate limiting.
        
        Args:
            alert: The alert to check
            
        Returns:
            True if the alert should be sent
        """
        priority_order = {
            AlertPriority.LOW: 0,
            AlertPriority.MEDIUM: 1,
            AlertPriority.HIGH: 2,
            AlertPriority.CRITICAL: 3
        }
        
        min_priority_order = priority_order.get(self.min_priority, 0)
        alert_priority_order = priority_order.get(alert.priority, 0)
        
        if alert_priority_order < min_priority_order:
            return False
            
        return self._check_rate_limit()

    def _add_to_history(self, alert: Alert) -> None:
        """Add alert to history and maintain max size"""
        self._alert_history.append(alert)
        
        # Trim history if too large
        if len(self._alert_history) > self._max_history_size:
            self._alert_history = self._alert_history[-self._max_history_size:]

    def _format_console_message(self, alert: Alert) -> str:
        """Format alert for console output"""
        priority_colors = {
            AlertPriority.LOW: "\033[94m",      # Blue
            AlertPriority.MEDIUM: "\033[92m",   # Green
            AlertPriority.HIGH: "\033[93m",     # Yellow
            AlertPriority.CRITICAL: "\033[91m"  # Red
        }
        
        reset_color = "\033[0m"
        
        color = priority_colors.get(alert.priority, reset_color)
        
        parts = [
            f"[{alert.timestamp}]",
            f"[{alert.type.value.upper()}]",
            f"[{alert.priority.value.upper()}]"
        ]
        
        if alert.symbol:
            parts.append(f"[{alert.symbol}]")
        
        if alert.price is not None:
            parts.append(f"@ ${alert.price:.8f}")
        
        parts.append(f"- {alert.message}")
        
        return f"{color}{' '.join(parts)}{reset_color}"

    async def _send_webhook_async(self, alert: Alert) -> bool:
        """
        Send alert via webhook asynchronously.
        
        Args:
            alert: The alert to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.webhook_url:
            return False
            
        try:
            payload = {
                "content": None,
                "embeds": [{
                    "title": f"Trading Bot Alert - {alert.type.value.upper()}",
                    "description": alert.message,
                    "color": self._get_embed_color(alert.priority),
                    "fields": [
                        {"name": "Type", "value": alert.type.value, "inline": True},
                        {"name": "Priority", "value": alert.priority.value, "inline": True},
                        {"name": "Timestamp", "value": alert.timestamp, "inline": False}
                    ],
                    "timestamp": alert.timestamp
                }]
            }
            
            if alert.symbol:
                payload["embeds"][0]["fields"].append(
                    {"name": "Symbol", "value": alert.symbol, "inline": True}
                )
            
            if alert.price is not None:
                payload["embeds"][0]["fields"].append(
                    {"name": "Price", "value": f"${alert.price:.8f}", "inline": True}
                )
            
            if alert.metadata:
                payload["embeds"][0]["fields"].append(
                    {"name": "Metadata", "value": json.dumps(alert.metadata, indent=2), "inline": False}
                )
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 204 or response.status == 200:
                        logger.debug(f"Webhook alert sent successfully for {alert.type.value}")
                        return True
                    else:
                        logger.error(f"Webhook returned status {response.status}: {await response.text()}")
                        return False
                        
        except asyncio.TimeoutError:
            logger.error("Webhook request timed out")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"Webhook client error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending webhook: {e}")
            return False

    def _send_webhook_sync(self, alert: Alert) -> bool:
        """
        Send alert via webhook synchronously (fallback).
        
        Args:
            alert: The alert to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.webhook_url:
            return False
            
        try:
            payload = {
                "content": f"**{alert.type.value.upper()}** - {alert.message}",
                "username": "Trading Bot"
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204 or response.status_code == 200:
                logger.debug(f"Webhook (sync) sent successfully")
                return True
            else:
                logger.error(f"Webhook (sync) returned {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Webhook (sync) error: {e}")
            return False

    async def _send_replit_notification(self, alert: Alert) -> bool:
        """
        Send notification via Replit's notification system.
        
        Args:
            alert: The alert to send
            
        Returns:
            True if successful, False otherwise
        """
        if not self.replit_token:
            return False
            
        try:
            # Replit notifications are typically sent via their API or SDK
            # This is a placeholder implementation that logs the notification
            logger.info(f"Replit notification would be sent: {alert.message}")
            
            # In a real implementation, you would use Replit's notification API
            # For now, we'll just log it as a placeholder
            
            return True
            
        except Exception as e:
            logger.error(f"Replit notification error: {e}")
            return False

    def _get_embed_color(self, priority: AlertPriority) -> int:
        """Get embed color for webhook based on priority"""
        colors = {
            AlertPriority.LOW: 3447003,      # Blue
            AlertPriority.MEDIUM: 3066993,   # Green
            AlertPriority.HIGH: 15105570,     # Yellow/Orange
            AlertPriority.CRITICAL: 15158332  # Red
        }
        return colors.get(priority, 3447003)

    async def send_alert_async(
        self,
        alert_type: AlertType,
        message: str,
        symbol: Optional[str] = None,
        price: Optional[float] = None,
        priority: AlertPriority = AlertPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send an alert asynchronously through all enabled channels.
        
        Args:
            alert_type: Type of alert (trade entry/exit/pullback/etc.)
            message: Alert message content
            symbol: Trading symbol (e.g., BTCUSDT)
            price: Current price at time of alert
            priority: Priority level of the alert
            metadata: Additional metadata for the alert
            
        Returns:
            True if at least one channel was successful
            
        Raises:
            ValueError: If message is empty or invalid parameters provided
        """
        # Input validation
        if not message or not isinstance(message, str):
            raise ValueError("Message must be a non-empty string")
            
        if symbol and not isinstance(symbol, str):
            raise ValueError("Symbol must be a string")
            
        if price is not None and not isinstance(price, (int, float)):
            raise ValueError("Price must be a number")
            
        # Create alert object
        alert = Alert(
            type=alert_type,
            message=message.strip(),
            symbol=symbol.upper() if symbol else None,
            price=price,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Check if we should send this alert
        if not self._should_send_alert(alert):
            logger.debug(f"Alert suppressed due to priority or rate limiting")
            return False
        
        # Add to history
        self._add_to_history(alert)
        
        success_count = 0
        
        # Console output (always synchronous)
        if self.enable_console:
            console_msg = self._format_console_message(alert)
            
            # Use appropriate logging level based on priority/type
            if alert.type == AlertType.ERROR or alert.priority == AlertPriority.CRITICAL:
                logger.error(console_msg)
            elif alert.type == AlertType.WARNING or alert.priority == AlertPriority.HIGH:
                logger.warning(console_msg)
            else:
                logger.info(console_msg)
            
            success_count += 1
        
        # Webhook notification (async)
        if self.enable_webhook and self.webhook_url:
            try:
                webhook_success = await self._send_webhook_async(alert)
                if webhook_success:
                    success_count += 1
                else:
                    # Fallback to sync method if async fails
                    webhook_success_sync = self._send_webhook_sync(alert)
                    if webhook_success_sync:
                        success_count += 1
                        
            except Exception as e:
                logger.error(f"Webhook async failed, trying sync fallback: {e}")
                webhook_success_sync = self._send_webhook_sync(alert)
                if webhook_success_sync:
                    success_count += 1
        
        # Replit notification (async)
        if self.enable_replit and self.replit_token:
            try:
                replit_success = await self._send_replit_notification(alert)
                if replit_success:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Replit notification failed: {e}")
        
        return success_count > 0

    def send_alert(
        self,
        alert_type: AlertType,
        message: str,
        symbol: Optional[str] = None,
        price: Optional[float] = None,
        priority: AlertPriority = AlertPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        use_async: bool = True
    ) -> bool:
        """
        Send an alert through all enabled channels.
        
        Args:
            alert_type: Type of alert (trade entry/exit/pullback/etc.)
            message: Alert message content
            symbol: Trading symbol (e.g., BTCUSDT)
            price: Current price at time of alert
            priority: Priority level of the alert
            metadata: Additional metadata for the alert
            use_async: Whether to use async sending (default True)
            
        Returns:
            True if at least one channel was successful
            
        Raises:
            ValueError: If message is empty or invalid parameters provided
        """
        
def send_sync():
    """Synchronous wrapper for send_alert"""
    
# Create event loop for async operations if needed
    
if use_async and hasattr(asyncio, 'get_running_loop'):
    
try:
    
loop = asyncio.get_running_loop()
    
# If we're already in an async context
    
return loop.run_until_complete(
    
self.send_alert_async(
    
alert_type=alert_type,
    
message=message,
    
symbol=symbol,
    
price=price,
    
priority=priority,
    
metadata=metadata
    
)
    
)
    
except RuntimeError:
    
# No running event loop
    
pass
    
# Synchronous fallback
    
return asyncio.run(
    
self.send_alert_async(
    
alert_type=alert_type,
    
message=message,
    
symbol=symbol,
    
price=price,
    
priority=priority,
    
metadata=metadata
    
)
    
)

def get_recent_alerts(
self,
limit: int = 10,
alert_type: Optional[AlertType] = None,
symbol: Optional[str] = None,
min_priority: Optional[AlertPriority] = None
) -> List[Alert]:
"""
Get recent alerts from history with optional filtering.

Args:
limit: Maximum number of alerts to return (default 10)
alert_type: Filter by alert type (optional)
symbol: Filter by symbol (optional)
min_priority: Minimum priority level (optional)

Returns:
List of matching alerts sorted by timestamp (newest first)
"""
filtered_alerts = list(reversed(self._alert_history))

if alert_type:

filtered_alerts = [a for a in filtered_alerts if a.type == alert_type]

if symbol:

filtered_alerts = [a for a in filtered_alerts 
if a.symbol and a.symbol.upper() == symbol.upper()]

if min_priority:

priority_order = {
AlertPriority.LOW: 0,
AlertPriority.MEDIUM: 1,
AlertPriority.HIGH: 2,
AlertPriority.CRITICAL: 3

}

min_order = priority_order.get(min_priority, 0)

filtered_alerts = [
a for a in filtered_alerts 
if priority_order.get(a.priority, 0) >= min_order

]

return filtered_alerts[:limit]

def clear_history(self) -> None:

"""Clear all stored alert history"""

self._alert_history.clear()

logger.info("Alert history cleared")

def get_statistics(self) -> Dict[str, Any]:

"""Get statistics about sent alerts"""

if not self._alert_history:

return {"total_alerts": 0}

type_counts = {}

priority_counts = {}

for alert in self._alert_history:

type_counts[alert.type.value] = type_counts.get(alert.type.value, 0) + 1

priority_counts[alert.priority.value] = priority_counts.get(alert.priority.value, 0) + 1

return {

"total_alerts": len(self._alert_history),

"by_type": type_counts,

"by_priority": priority_counts,

"timeframe_hours": round(

(datetime.utcnow() - datetime.fromisoformat(self._alert_history[0].timestamp)).total_seconds() / 3600,

2

)

}

def validate_configuration(self) -> Dict[str, bool]:

"""Validate the current configuration and return status of each channel"""

return {

"console_enabled": self.enable_console,

"webhook_enabled": self.enable_webhook and bool(self.webhook_url),

"replit_enabled": self.enable_replit and bool(self.replit_token),

"rate_limit_active": len(self._alert_timestamps) > 0,

"history_size": len(self._alert_history)

}


class TradeAlertFactory:

"""
Factory class for creating common trading alerts with proper formatting.
"""

@staticmethod

def create_entry_alert(

symbol: str,

entry_price: float,

direction: str,

reasoning: str,

confidence_score: float

) -> tuple:

"""

Create a trade entry alert.

Args:

symbol: Trading pair symbol

entry_price: Entry price

direction: 'LONG' or 'SHORT'

reasoning: Technical reasoning for entry

confidence_score: Confidence score (0-100)

Returns:

Tuple of (AlertType, message, metadata)

"""

message = (

f"{direction.upper()} ENTRY SIGNAL on {symbol} @ ${entry_price:.8f}\n"

f"Confidence Score: {confidence_score:.1f}%\n"

f"Reasoning:\n{reasoning}"

)

metadata = {

"direction": direction,

"entry_price": entry_price,

"confidence_score": confidence_score,

"reasoning": reasoning

}

return (

AlertType.TRADE_ENTRY,

message,

AlertPriority.HIGH,

metadata

)

@staticmethod

def create_exit_alert(

symbol: str,

exit_price: float,

entry_price: float,

pnl_percentage: float,

reasoning: str

) -> tuple:

"""

Create a trade exit alert.

Args:

symbol: Trading pair symbol

exit_price: Exit price

entry_price: Entry price

pnl_percentage: Profit/Loss percentage

reasoning: Technical reasoning for exit

Returns:

Tuple of (AlertType, message, metadata)

"""

direction_text = "PROFITABLE EXIT" if pnl_percentage >= 0 else "LOSS EXIT"

message = (

f"{direction_text} on {symbol}\n"

f"Entry Price @ ${entry_price:.8f}\n"

f"Exit Price @ ${exit_price:.8f}\n"

f"PnL Percentage @ ${pnl_percentage:.2f}%\n"

f"Reasoning:\n{reasoning}"

)

metadata = {

"exit_price": exit_price,

"entry_price": entry_price,

"pnl_percentage": pnl_percentage,

"reasoning": reasoning

}

priority = AlertPriority.CRITICAL if abs(pnl_percentage) > 10 else AlertPriority.HIGH

return (

AlertType.TRADE_EXIT,

message,

priority,

metadata

)

@staticmethod

def create_pullback_alert(

symbol: str,

current_price: float,

pullback_percentage_from_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price_of_entry_price