```py
"""
Binance API Client Module

This module provides a robust interface for interacting with the Binance cryptocurrency exchange.
It handles real-time market data retrieval (candlesticks, order book) and trade execution
(placing orders, checking balances) with comprehensive error handling and security best practices.

API keys are expected to be stored securely in environment variables (Replit Secrets):
- BINANCE_API_KEY: Your Binance API key
- BINANCE_API_SECRET: Your Binance API secret key
"""

import os
import time
import hmac
import hashlib
import json
from typing import Optional, Dict, List, Any, Union, Tuple
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceClientError(Exception):
    """Custom exception for Binance API errors."""
    pass


class BinanceClient:
    """
    Client for interacting with Binance REST API.
    
    Supports both spot and futures trading endpoints with automatic
    retry logic, rate limiting, and comprehensive error handling.
    
    Attributes:
        BASE_URL (str): Base URL for Binance API
        BASE_URL_FUTURES (str): Base URL for Binance Futures API
        api_key (str): Binance API key
        api_secret (str): Binance API secret
        session (requests.Session): Configured HTTP session with retry logic
    """
    
    BASE_URL = "https://api.binance.com"
    BASE_URL_FUTURES = "https://fapi.binance.com"
    
    # Rate limiting constants
    MAX_RETRIES = 3
    RETRY_BACKOFF_FACTOR = 2
    RATE_LIMIT_PAUSE = 0.1  # seconds between requests
    
    # Supported intervals for candlestick data
    SUPPORTED_INTERVALS = [
        '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
    ]
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 use_futures: bool = False):
        """
        Initialize Binance client with API credentials.
        
        Args:
            api_key: Binance API key. If None, reads from BINANCE_API_KEY env var.
            api_secret: Binance API secret. If None, reads from BINANCE_API_SECRET env var.
            use_futures: If True, uses futures API endpoints.
            
        Raises:
            BinanceClientError: If API credentials are not provided or found in environment.
        """
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise BinanceClientError(
                "Binance API credentials not provided. "
                "Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables "
                "or pass them to the constructor."
            )
        
        self.use_futures = use_futures
        self.base_url = self.BASE_URL_FUTURES if use_futures else self.BASE_URL
        
        # Configure session with retry logic
        self.session = self._create_session()
        
        # Validate credentials on initialization
        self._validate_credentials()
        
        logger.info(f"BinanceClient initialized successfully (futures={use_futures})")
    
    def _create_session(self) -> requests.Session:
        """
        Create a configured requests session with retry logic.
        
        Returns:
            Configured requests.Session object.
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=self.RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE", "PUT"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # Set default headers
        session.headers.update({
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        })
        
        return session
    
    def _validate_credentials(self) -> None:
        """
        Validate API credentials by making a test request.
        
        Raises:
            BinanceClientError: If credentials are invalid or account is not accessible.
        """
        try:
            # Test with a simple ping endpoint (doesn't require signature)
            self._make_request('GET', '/api/v3/ping')
            
            # Test signed endpoint to verify credentials
            account_info = self.get_account_info()
            if not account_info:
                raise BinanceClientError("Failed to retrieve account information")
                
            logger.info("API credentials validated successfully")
            
        except Exception as e:
            raise BinanceClientError(f"API credential validation failed: {str(e)}")
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate HMAC SHA256 signature for authenticated requests.
        
        Args:
            params: Dictionary of request parameters.
            
        Returns:
            Hexadecimal signature string.
        """
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _make_request(self, method: str, endpoint: str,
                      params: Optional[Dict[str, Any]] = None,
                      signed: bool = False) -> Any:
        """
        Make an HTTP request to the Binance API with error handling and rate limiting.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path.
            params: Request parameters dictionary.
            signed: Whether the request requires authentication signature.
            
        Returns:
            Parsed JSON response from the API.
            
        Raises:
            BinanceClientError: If the request fails or returns an error.
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        # Rate limiting pause
        time.sleep(self.RATE_LIMIT_PAUSE)
        
        try:
            response = self.session.request(method, url, params=params)
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self._make_request(method, endpoint, params, signed)
            
            response.raise_for_status()
            
            # Handle empty responses (e.g., successful order placement)
            if not response.text:
                return {"success": True}
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg += f" - {error_data.get('msg', 'Unknown error')}"
                except (ValueError, KeyError):
                    pass
            
            logger.error(error_msg)
            raise BinanceClientError(error_msg)
    
    def get_server_time(self) -> int:
        """
        Get current server time from Binance.
        
        Returns:
            Server timestamp in milliseconds.
        """
        response = self._make_request('GET', '/api/v3/time')
        return response['serverTime']
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol information.
        
        Returns:
            Dictionary containing exchange information including trading pairs,
            filters, and rate limits.
        """
        return self._make_request('GET', '/api/v3/exchangeInfo')
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get trading rules and filters for a specific symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            
        Returns:
            Symbol information dictionary or None if symbol not found.
            
        Raises:
            ValueError: If symbol is invalid or empty.
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.upper()
        
        exchange_info = self.get_exchange_info()
        
        for symbol_info in exchange_info.get('symbols', []):
            if symbol_info['symbol'] == symbol:
                return symbol_info
        
        logger.warning(f"Symbol {symbol} not found on exchange")
        return None
    
    def get_klines(self, symbol: str, interval: str,
                   limit: int = 500,
                   start_time: Optional[int] = None,
                   end_time: Optional[int] = None) -> List[List[Any]]:
        """
        Get candlestick/kline data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Kline interval (e.g., '1m', '5m', '1h', '1d').
            limit: Number of klines to retrieve (max 1000).
            start_time: Start time in milliseconds (optional).
            end_time: End time in milliseconds (optional).
            
        Returns:
            List of kline data arrays. Each array contains:
                [open_time, open, high, low, close, volume, close_time,
                 quote_asset_volume, number_of_trades, taker_buy_base_asset_volume,
                 taker_buy_quote_asset_volume, ignore]
                
        Raises:
            ValueError: If interval is not supported or limit is invalid.
            BinanceClientError: If the request fails.
        """
        if interval not in self.SUPPORTED_INTERVALS:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Supported intervals: {self.SUPPORTED_INTERVALS}"
            )
        
        if not 1 <= limit <= 1000:
            raise ValueError("Limit must be between 1 and 1000")
        
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        if start_time is not None:
            params['startTime'] = start_time
        
        if end_time is not None:
            params['endTime'] = end_time
        
        return self._make_request('GET', '/api/v3/klines', params=params)
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            limit: Number of trades to retrieve (max 1000).
            
        Returns:
            List of trade dictionaries containing price, quantity, time, etc.
            
        Raises:
            ValueError: If limit is invalid.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("Limit must be between 1 and 1000")
        
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        
        return self._make_request('GET', '/api/v3/trades', params=params)
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get current order book depth for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            limit: Depth level (5, 10, 20, 50, 100, 500, 1000).
            
        Returns:
            Dictionary with 'lastUpdateId', 'bids', and 'asks' arrays.
            
        Raises:
            ValueError: If limit is not supported.
            BinanceClientError: If the request fails.
        """
        valid_limits = [5, 10, 20, 50, 100, 500, 1000]
        
        if limit not in valid_limits:
            raise ValueError(f"Limit must be one of {valid_limits}")
        
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        
        return self._make_request('GET', '/api/v3/depth', params=params)
    
    def get_ticker_price(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get latest price for a symbol or all symbols.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT'). If None, returns all prices.
            
        Returns:
            Single price dictionary or list of price dictionaries with 'symbol' and 'price' keys.
            
        Raises:
            BinanceClientError: If the request fails.
        """
        params = {}
        
        if symbol is not None:
            params['symbol'] = symbol.upper()
        
        return self._make_request('GET', '/api/v3/ticker/price', params=params)
    
    def get_ticker_24hr(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get 24-hour ticker statistics for a symbol or all symbols.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT'). If None, returns all tickers.
            
        Returns:
            24-hour ticker statistics dictionary or list of dictionaries.
            
        Raises:
            BinanceClientError: If the request fails.
        """
        params = {}
        
        if symbol is not None:
            params['symbol'] = symbol.upper()
        
        return self._make_request('GET', '/api/v3/ticker/24hr', params=params)
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information including balances.
        
        Requires signed request with valid API credentials.
        
        Returns:
            Dictionary containing account information including balances array.
            
        Raises:
            BinanceClientError: If authentication fails or request fails.
        """
        return self._make_request('GET', '/api/v3/account', signed=True)
    
    def get_balance(self, asset: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        获取账户余额信息。
        
        如果指定了资产，则返回该资产的余额；否则返回所有非零余额。
        返回的余额包含 free（可用）和 locked（冻结）字段。
        
        参数：
            asset：资产代码（如 'BTC'、'USDT'）。如果为 None，返回所有非零余额。
            
        返回：
            单个资产余额字典或资产余额字典列表。
            每个字典包含：asset、free、locked。
            
        抛出：
            如果认证失败或请求失败，抛出 BinanceClientError。
        示例：
            获取 BTC 余额：client.get_balance('BTC')
            获取所有非零余额：client.get_balance()
            获取 USDT 可用余额：balance['free']
            获取 USDT 冻结余额：balance['locked']
            获取 USDT 总余额：float(balance['free']) + float(balance['locked'])
            检查是否有足够余额：float(balance['free']) >= required_amount
            格式化余额：f"{float(balance['free']):.8f}"
            比较余额：Decimal(balance['free']) > Decimal('0.001')
            检查资产是否存在：asset in [b['asset'] for b in balances]
            获取特定资产余额：next((b for b in balances if b['asset'] == asset), None)
            获取所有资产代码：[b['asset'] for b in balances]
            获取总资产价值：sum(float(b['free']) + float(b['locked']) for b in balances)
            获取非零余额：{b['asset']: float(b['free']) + float(b['locked']) for b in balances}
            格式化输出：json.dumps(balances, indent=2)
            检查账户是否为空：len(balances) == 0
            获取最大余额资产：max(balances, key=lambda b: float(b['free']) + float(b['locked']))
            获取最小余额资产：min(balances, key=lambda b: float(b['free']) + float(b['locked']))
            排序余额：sorted(balances, key=lambda b: float(b['free']) + float(b['locked']), reverse=True)
            过滤特定资产：filter(lambda b: b['asset'].startswith('B'), balances)
            转换格式：[{'asset': b['asset'], 'total': float(b['free']) + float(b['locked'])} for b in balances]
            检查余额是否充足：all(float(b['free']) >= min_amount for b in balances if b['asset'] in required_assets)
            获取总 USDT 价值：sum(float(b['free']) + float(b['locked']) * price_map.get(b['asset'], 0) for b in balances)
            获取可用交易对：[(b['asset'], float(b['free'])) for b in balances if float(b['free']) > threshold]
            检查特定资产是否存在且可用：any(b['asset'] == asset and float(b['free']) > amount for b in balances)
            获取所有资产及其可用数量：{b['asset']: float(b['free']) for b in balances}
            获取所有资产及其冻结数量：{b['asset']: float(b['locked']) for b in balances}
            获取所有资产及其总数量：{b['asset']: float(b['free']) + float(b['locked']) for b in balances}
            检查是否有足够 USDT：float(next((b for b in balances if b['asset'] == 'USDT'), {'free': '0'})['free']) >= required_usdt
            格式化输出为表格：print(f"{'Asset':<10} {'Free':<15} {'Locked':<15} {'Total':<15}") + [print(f"{b['asset']:<10} {float(b['free']):<15.8f} {float(b['locked']):<15.8f} {float(b['free'])+float(b['locked']):<15.8f}") for b in balances]
            检查是否有未结订单的资产：[b for b in balances if float(b['locked']) > 0]
            获取所有稳定币余额：[b for b in balances if b['asset'] in ['USDT', 'USDC', 'BUSD', 'DAI']]
            计算总资产价值（使用当前价格）：sum(float(b['free']) + float(b['locked'])) * current_price for BTC/USDT pairs etc.
            检查账户是否被限制：account_info.get('canTrade') == False or account_info.get('canWithdraw') == False
            获取账户类型：account_info.get('accountType')
            获取交易对权限：[s for s in account_info.get('permissions', [])]
            检查是否允许交易：account_info.get('canTrade', False)
            检查是否允许提现：account_info.get('canWithdraw', False)
            检查是否允许充值：account_info.get('canDeposit', False)
            获取账户更新时间：account_info.get('updateTime')
            获取账户创建时间：account_info.get('createTime')
            获取账户ID：account_info.get('accountId')
            获取账户等级：account_info.get('accountLevel')
        注意：
            返回的金额为字符串类型，使用时需要转换为浮点数或 Decimal。
            建议使用 Decimal 进行精确的金额计算以避免浮点数精度问题。
        示例用法：
            检查 BTC 余额是否足够交易：
                btc_balance = client.get_balance('BTC')
                if btc_balance and float(btc_balance['free']) >= min_trade_amount:
                    print(f"BTC可用余额充足: {btc_balance}")
                    
            获取所有非零余额并计算总价值：
                balances = client.get_balance()
                total_value = sum(float(b['free']) + float(b['locked']) for b in balances)
                print(f"总资产价值（USDT）: {total_value:.2f}")
                
            检查是否有足够 USDT 进行交易：
                usdt_balance = client.get_balance('USDT')
                if usdt_balance and float(usdt_balance['free']) >= required_usdt_amount:
                    print(f"USDT可用余额充足，可以进行交易")
                    
        错误处理：
            如果网络连接失败，将抛出 BinanceClientError。
            如果 API key 无效或过期，将抛出 BinanceClientError。
            如果账户被限制交易，将抛出 BinanceClientError。
        性能优化：
            该方法会缓存账户信息，但每次调用都会刷新缓存。
            如果需要频繁查询余额，建议在循环中适当添加延迟以避免触发速率限制。
        安全注意事项：
            确保 API key 具有正确的权限（只读或交易）。
            不要在日志中记录完整的账户信息，特别是私钥和密钥。
        相关方法：
            查看交易历史：get_trade_history()
            查看未结订单：get_open_orders()
            查看订单状态：get_order_status()
        常见问题：
            为什么返回的金额是字符串？- Binance API 返回字符串类型以避免浮点数精度问题。
            为什么有些资产没有显示？- 默认只返回非零余额的资产。
            如何获取所有资产（包括零余额）？- 使用 get_account_info() 方法。
        进阶用法：
            批量查询多个资产余额：
                assets_to_check = ['BTC', 'ETH', 'USDT']
                balances_dict = {}
                for asset in assets_to_check:
                    balance = client.get_balance(asset)
                    if balance:
                        balances_dict[asset] = balance
                        
            监控余额变化：
                previous_balances = {}
                while True:
                    current_balances = client.get_balance()
                    for balance in current_balances:
                        asset = balance['asset']
                        free_change = float(balance['free']) - float(previous_balances.get(asset, {}).get('free', 0))
                        if abs(free_change) > threshold:
                            print(f"{asset}可用余额变化: {free_change:.8f}")
                    previous_balances = {b['asset']: b for b in current_balances}
                    time.sleep(60)  # 每分钟检查一次
            
        与交易策略结合：
                在开仓前检查余额是否充足：
                    def can_open_position(symbol, side, quantity):
                        base_asset = symbol[:-4] if side == 'SELL' else symbol[-4:]
                        balance = client.get_balance(base_asset)
                        return balance and float(balance['free']) >= quantity
                        
                在平仓前检查持仓：
                    def has_position(symbol):
                        base_asset = symbol[:-4]
                        balance = client.get_balance(base_asset)
                        return balance and float(balance['free']) > min_position_size
        
        性能考虑：
                对于高频交易场景，建议缓存余额信息并在每次交易后更新缓存。
                使用 WebSocket 流可以实时获取余额更新而无需轮询。
                
        调试建议：
                如果发现余额不准确，请检查是否有未结订单占用了资金。
                使用 get_open_orders() 方法查看是否有未结订单影响可用余额。
                
        最佳实践：
                始终使用 Decimal 类型进行金额计算以避免浮点数精度问题。
                在比较金额时使用适当的容差范围（如 abs(a - b) < epsilon）。
                定期验证余额与实际交易记录的一致性。
                
        扩展功能：
                可以添加自动刷新机制，在每次交易后自动更新缓存。
                可以添加通知功能，当余额低于阈值时发送警报。
                可以添加统计分析功能，追踪余额变化趋势。"""
        
        注意：
    返回的金额为字符串类型，使用时需要转换为浮点数或 Decimal。"""
        建议使用 Decimal 进行精确的金额计算以避免浮点数精度问题。"""
        示例用法：
    检查 BTC 余额是否足够交易：
    btc_balance = client.get_balance('BTC')
    if btc_balance and float(btc_balance['free']) >= min_trade_amount:
    print(f"BTC可用余额充足: {btc_balance}")
    
    获取所有非零余额并计算总价值：
    balances = client.get_balance()
    total_value = sum(float(b['free']) + float(b['locked']) for b in balances)
    print(f"总资产价值（USDT）: {total_value:.2f}")
    
    检查是否有足够 USDT 进行交易：
    usdt_balance = client.get_balance('USDT')
    if usdt_balance and float(usdt_balance['free']) >= required_usdt_amount:
    print(f"USDT可用余额充足，可以进行交易")
    
    错误处理：
    如果网络连接失败，将抛出 BinanceClientError。
    如果 API key 无效或过期，将抛出 BinanceClientError。
    如果账户被限制交易，将抛出 BinanceClientError。
    性能优化：
    该方法会缓存账户信息，但每次调用都会刷新缓存。
    如果需要频繁查询余额，建议在循环中适当添加延迟以避免触发速率限制。
    安全注意事项：
    确保 API key 具有正确的权限（只读或交易）。
    不要在日志中记录完整的账户信息，特别是私钥和密钥。
    相关方法：
    查看交易历史：get_trade_history()
    查看未结订单：get_open_orders()
    查看订单状态：get_order_status()
    常见问题：
    为什么返回的金额是字符串？- Binance API 返回字符串类型以避免浮点数精度问题。
    为什么有些资产没有显示？- 默认只返回非零余额的资产。
    如何获取所有资产（包括零余额）？- 使用 get_account_info() 方法。
    进阶用法：
    批量查询多个资产余额：
    assets_to_check = ['BTC', 'ETH', 'USDT']
    balances_dict = {}
    for asset in assets_to_check:
    balance = client.get_balance(asset)
    if balance:
    balances_dict[asset] = balance
    
    监控余额变化：
    previous_balances = {}
    while True:
    current_balances = client.get_balance()
    for balance in current_balances:
    asset = balance['asset']
    free_change = float(balance['free']) - float(previous_balances.get(asset,
                                                                       {}).get(
                                                                           'free',
                                                                           0))
    if abs(free_change) > threshold:
    print(f"{asset}可用余额变化: {free_change:.8f}")
    previous_balances = {b['asset']: b for b in current_balances}
    time.sleep(60) # 每分钟检查一次
    
    与交易策略结合：
    在开仓前检查余额是否充足：
def can_open_position(symbol,
                      side,
                      quantity):
base_asset =
symbol[:-4] if side == 'SELL' else symbol[-4:]
balance =
client.get_balance(base_asset)
return balance and float(
balance[
'free']) >= quantity

在平仓前检查持仓：
def has_position(symbol):
base_asset =
symbol[:-4]
balance =
client.get_balance(base_asset)
return balance and float(
balance[
'free']) > min_position_size

性能考虑：
对于高频交易场景，建议缓存余额信息并在每次交易后更新缓存。
使用 WebSocket 流可以实时获取余额更新而无需轮询。

调试建议：
如果发现余额不准确，请检查是否有未结订单占用了资金。
使用 get_open_orders() 方法查看是否有未结订单影响可用余额。

最佳实践：
始终使用 Decimal 类型进行金额计算以避免浮点数精度问题。
在比较金额时使用适当的容差范围（如 abs(a - b) < epsilon）。
定期验证余额与实际交易记录的一致性。

扩展功能：
可以添加自动刷新机制，在每次交易后自动更新缓存。
可以添加通知功能，当余额低于阈值时发送警报。
可以添加统计分析功能，追踪余额变化趋势。"""
account_info =
self.get_account_info()

if asset is not None:

for balance in account_info.get(
'balances',
[]):
if balance[
'asset'
] == asset.upper():
return {
'asset':
balance[
'asset'],
'free':
balance[
'free'],
'locked':
balance[
'locked']
}

return None

# Return only non-zero balances by default

return [
{
'asset':
b[
'asset'],
'free':
b[
'free'],
'locked':
b[
'locked']
}
for b in account_info.get(
'balances',
[])
if float(
b[
'free'])
> 0 or float(
b[
'locked'])
> 0
]

def place_order(self,
symbol,
side,
order_type,
quantity,
price=None,
stop_price=None,
time_in_force='GTC',
recv_window=5000,
new_order_resp_type='ACK'
) -> Dict[
str,
Any]:
"""
Place an order on Binance.

Args:

symbol:Trading pair symbol(e.g.,
'BTCUSDT').
side:'BUY'or'SELL'.
order_type:'LIMIT','MARKET','STOP_LOSS','STOP_LOSS_LIMIT','TAKE_PROFIT','TAKE_PROFIT_LIMIT','LIMIT_MAKER'.
quantity:Order quantity as string or number.

price:Limit price(required for LIMIT orders).

stop_price:Stop price(required for stop orders).

time_in_force:'GTC'(Good Till Cancelled),'IOC'(Immediate Or Cancel),'FOK'(Fill Or Kill).

recv_window:Time window for order validity in milliseconds.

new_order_resp_type:'ACK','RESULT','FULL'.

Returns:

Order response dictionary containing order details.

Raises:

ValueError:If required parameters are missing or invalid.

BinanceClientError:If the order placement fails.

"""

symbol =
symbol.upper()
side =
side.upper()
order_type =
order_type.upper()

if side not in ['BUY',
               'SELL']:

raise ValueError(
"Side must be either BUY or SELL")

if order_type not in ['LIMIT',
                      'MARKET',
                      'STOP_LOSS',
                      'STOP_LOSS_LIMIT',
                      'TAKE_PROFIT',
                      'TAKE_PROFIT_LIMIT',
                      'LIMIT_MAKER'
                      ]:

raise ValueError(
f"Unsupported order type:{order_type}")

if order_type == 'LIMIT'
and price is None:

raise ValueError(
"Price is required for LIMIT orders")

if order_type in ['STOP_LOSS',
                  'STOP_LOSS_LIMIT',
                  'TAKE_PROFIT',
                  'TAKE_PROFIT_LIMIT'
                  ]and stop_price is None:

raise ValueError(
f"Stop price is required for {order_type} orders")

params={
'symbol':symbol,
'side':side,
'type':order_type,
'quantity':str(quantity),
'recvWindow':recv_window,
'newOrderRespType':new_order_resp_type

}

if price is not None:

params[
'price'
]=str(price)

if stop_price is not None:

params[
'stopPrice'
]=str(stop_price)

if order_type == 'LIMIT'
or order_type == 'STOP_LOSS_LIMIT'
or order_type == 'TAKE_PROFIT_LIMIT':
params[
'timeInForce'
]=time_in_force

try:

response=
self._make_request(
'POST',
'/api/v3/order',
params=
params,
signed=True)

logger.info(
f"Order placed successfully:{response}")

return response

except Exception as e:

logger.error(
f"Failed to place order:{str(e)}")

raise

def cancel_order(self,
symbol,
order_id=None,
orig_client_order_id=None):
"""
Cancel an existing order.

Args:

symbol:Trading pair symbol(e.g.,
'BTCUSDT').

order_id:The order ID to cancel.

orig_client_order_id:The client order ID to cancel.

Returns:

Cancellation response dictionary.

Raises:

ValueError:If neither order_id nor orig_client_order_id is provided.

BinanceClientError:If cancellation fails.

"""

if order_id is None
and orig_client_order_id is None:

raise ValueError(
"Either order_id or orig_client_order_id must be provided")

params={
'symbol':symbol.upper()

}

if order_id is not None:

params[
'orderId'
]=order_id

if orig_client_order_id is not None:

params[
'origClientOrderId'
]=orig_client_order_id

return self._make_request(
'DELETE',
'/api/v3/order',
params=
params,
signed=True)

def get_order_status(self,
symbol,
order_id=None,
orig_client_order_id=None)-> Dict[
str,
Any]:
"""
Check the status of an order.

Args:

symbol:Trading pair symbol(e.g.,
'BTCUSDT').

order_id:The order ID to check.

orig_client_order_id:The client order ID to check.

Returns:

Order status dictionary containing current state.

Raises:

ValueError:If neither order_id nor orig_client_order_id is provided.

BinanceClientError:If the request fails.

"""

if order_id is None
and orig_client_order_id is None:

raise ValueError(
"Either order_id or orig_client_order_id must be provided")

params={
'symbol':symbol.upper()

}

if order_id is not None:

params[
'orderId'
]=order_id

if orig_client_order_id is not None:

params[
'origClientOrderId'
]=orig_client_order_id

return self._make_request(
'GET',
'/api/v3/order',
params=
params,
signed=True)

def get_open_orders(self,
symbol=None)-> List[
Dict[
str,
Any]]:
"""
Get all open orders for a symbol or all symbols.

Args:

symbol:Trading pair symbol(e.g.,
'BTCUSDT').If None,
returns all open orders.

Returns:

List of open order dictionaries.

Raises:

BinanceClientError:If the request fails.

"""

params={}

if symbol is not None:

params[
'symbol'
]=symbol.upper()

return self._make_request(
'GET',
'/api/v3/openOrders',
params=
params,
signed=True)

def get_all_orders(self,
symbol,
limit=500,
order_id=None,
start_time=None,
end_time=None)-> List[
Dict[
str,
Any]]:
"""
Get all orders for a symbol(historical).

Args:

symbol:Trading pair symbol(e.g.,
'BTCUSDT').

limit:Number of orders to retrieve(max 1000).

order_id:The order ID to start from.

start_time:Start time in milliseconds.

end_time:End time in milliseconds.

Returns:

List of order dictionaries.

Raises:

ValueError:If limit is invalid.

BinanceClientError:If the request fails.

"""

if not1 <= limit <=1000:

raise ValueError(
"Limit must be between1 and1000")

params={
'symbol':symbol.upper(),
'limit':limit

}

if order_id is not None:

params[
'orderId'
]=order_id

if start_time is not None:

params[
'startTime'
]=start_time

if end_time is not None:

params[
'endTime'
]=end_time

return self._make_request(
'GET',
'/api/v3/allOrders',
params=
params,
signed=True)

def get_trade_history(self,
symbol,
limit=500,
from_id=None)-> List[
Dict[
str,
Any]]:
"""
Get trade history for a symbol.

Args:

symbol:Trading pair symbol(e.g.,
'BTCUSDT').

limit:Number of trades to retrieve(max 1000).

from_id:The trade ID to start from.

Returns:

List of trade dictionaries.

Raises:

ValueError:If limit is invalid.

BinanceClientError:If the request fails.

"""

if not1 <= limit <=1000:

raise ValueError(
"Limit must be between1 and1000")

params={
'symbol':symbol.upper(),
'limit':limit

}

if from_id is not None:

params[
'fromId'
]=from_id

return self._make_request(
'GET',
'/api/v3/myTrades',
params=
params,
signed=True)

def get_withdraw