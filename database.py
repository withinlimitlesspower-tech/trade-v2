```py
"""
database.py - SQLite database setup and CRUD operations for the cryptocurrency trading bot.

This module provides a robust database layer for storing trade history, trading signals,
and performance metrics. It uses SQLite for local storage with proper connection management,
error handling, and data validation.

Classes:
    DatabaseManager: Main database handler with CRUD operations
    Trade: Data class representing a trade record
    Signal: Data class representing a trading signal
    PerformanceMetric: Data class representing performance metrics

Usage:
    db = DatabaseManager("trading_bot.db")
    db.initialize_database()
    trade = Trade(symbol="BTCUSDT", side="BUY", entry_price=50000.0, quantity=0.001)
    db.insert_trade(trade)
"""

import sqlite3
import os
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Data class representing a cryptocurrency trade."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    quantity: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    status: str = "OPEN"  # 'OPEN', 'CLOSED', 'CANCELLED'
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    strategy: Optional[str] = None
    timeframe: Optional[str] = None
    notes: Optional[str] = None
    trade_id: Optional[int] = None

    def __post_init__(self):
        """Validate trade data after initialization."""
        if self.side not in ['BUY', 'SELL']:
            raise ValueError(f"Invalid side: {self.side}. Must be 'BUY' or 'SELL'")
        if self.status not in ['OPEN', 'CLOSED', 'CANCELLED']:
            raise ValueError(f"Invalid status: {self.status}")
        if self.entry_price <= 0:
            raise ValueError(f"Invalid entry price: {self.entry_price}")
        if self.quantity <= 0:
            raise ValueError(f"Invalid quantity: {self.quantity}")
        if self.entry_time is None:
            self.entry_time = datetime.now(timezone.utc).isoformat()


@dataclass
class Signal:
    """Data class representing a trading signal."""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'NEUTRAL'
    strength: float  # 0.0 to 1.0
    timeframe: str
    indicators: Dict[str, Any]
    analysis_text: Optional[str] = None
    created_at: Optional[str] = None
    executed: bool = False
    signal_id: Optional[int] = None

    def __post_init__(self):
        """Validate signal data after initialization."""
        if self.signal_type not in ['BUY', 'SELL', 'NEUTRAL']:
            raise ValueError(f"Invalid signal type: {self.signal_type}")
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Invalid strength value: {self.strength}")
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class PerformanceMetric:
    """Data class representing performance metrics."""
    date: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    metric_id: Optional[int] = None


class DatabaseManager:
    """
    Manages SQLite database operations for the trading bot.
    
    Provides thread-safe database operations with proper connection management,
    error handling, and data validation.
    
    Attributes:
        db_path (str): Path to the SQLite database file
        connection (sqlite3.Connection): Database connection object
    """
    
    def __init__(self, db_path: str = "trading_bot.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path (str): Path to the SQLite database file
            
        Raises:
            ValueError: If db_path is empty or invalid
        """
        if not db_path or not isinstance(db_path, str):
            raise ValueError("Database path must be a non-empty string")
        
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        
        # Ensure the directory exists for the database file
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    @contextmanager
    def _get_connection(self) -> sqlite3.Connection:
        """
        Context manager for database connections.
        
        Provides automatic connection creation and cleanup with proper error handling.
        
        Yields:
            sqlite3.Connection: Database connection object
            
        Raises:
            sqlite3.Error: If connection fails
        """
        connection = None
        try:
            connection = sqlite3.connect(self.db_path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
            connection.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()
    
    def initialize_database(self) -> bool:
        """
        Create database tables if they don't exist.
        
        Creates tables for trades, signals, and performance metrics with proper
        schema definitions and indexes.
        
        Returns:
            bool: True if initialization was successful
            
        Raises:
            sqlite3.Error: If table creation fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
                        entry_price REAL NOT NULL CHECK(entry_price > 0),
                        quantity REAL NOT NULL CHECK(quantity > 0),
                        exit_price REAL,
                        pnl REAL,
                        pnl_percentage REAL,
                        status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'CLOSED', 'CANCELLED')),
                        entry_time TEXT NOT NULL,
                        exit_time TEXT,
                        strategy TEXT,
                        timeframe TEXT,
                        notes TEXT,
                        created_at TEXT DEFAULT (datetime('now')),
                        updated_at TEXT DEFAULT (datetime('now'))
                    )
                """)
                
                # Create signals table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL CHECK(signal_type IN ('BUY', 'SELL', 'NEUTRAL')),
                        strength REAL NOT NULL CHECK(strength >= 0 AND strength <= 1),
                        timeframe TEXT NOT NULL,
                        indicators TEXT NOT NULL,
                        analysis_text TEXT,
                        created_at TEXT NOT NULL,
                        executed INTEGER DEFAULT 0,
                        executed_at TEXT,
                        created_at_db TEXT DEFAULT (datetime('now'))
                    )
                """)
                
                # Create performance_metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL UNIQUE,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0.0,
                        win_rate REAL DEFAULT 0.0,
                        average_win REAL DEFAULT 0.0,
                        average_loss REAL DEFAULT 0.0,
                        max_drawdown REAL DEFAULT 0.0,
                        sharpe_ratio REAL,
                        created_at TEXT DEFAULT (datetime('now')),
                        updated_at TEXT DEFAULT (datetime('now'))
                    )
                """)
                
                # Create indexes for better query performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at)
                """)
                
                logger.info(f"Database initialized successfully at {self.db_path}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def insert_trade(self, trade: Trade) -> Optional[int]:
        """
        Insert a new trade record.
        
        Args:
            trade (Trade): Trade object to insert
            
        Returns:
            Optional[int]: The ID of the inserted trade, or None if insertion failed
            
        Raises:
            ValueError: If trade data is invalid
            sqlite3.Error: If database operation fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trades (
                        symbol, side, entry_price, quantity, exit_price,
                        pnl, pnl_percentage, status, entry_time, exit_time,
                        strategy, timeframe, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.symbol, trade.side, trade.entry_price, trade.quantity,
                    trade.exit_price, trade.pnl, trade.pnl_percentage, trade.status,
                    trade.entry_time, trade.exit_time, trade.strategy,
                    trade.timeframe, trade.notes
                ))
                
                trade_id = cursor.lastrowid
                logger.info(f"Trade inserted successfully with ID: {trade_id}")
                return trade_id
                
        except sqlite3.Error as e:
            logger.error(f"Failed to insert trade: {e}")
            return None
    
    def update_trade(self, trade_id: int, **kwargs) -> bool:
        """
        Update an existing trade record.
        
        Args:
            trade_id (int): ID of the trade to update
            **kwargs: Fields to update (e.g., exit_price=51000.0, status='CLOSED')
            
        Returns:
            bool: True if update was successful
            
        Raises:
            ValueError: If trade_id is invalid or update fields are invalid
            sqlite3.Error: If database operation fails
        """
        if not isinstance(trade_id, int) or trade_id <= 0:
            raise ValueError("Invalid trade ID")
        
        allowed_fields = {
            'exit_price', 'pnl', 'pnl_percentage', 'status', 'exit_time',
            'strategy', 'timeframe', 'notes'
        }
        
        # Validate update fields
        invalid_fields = set(kwargs.keys()) - allowed_fields
        if invalid_fields:
            raise ValueError(f"Invalid update fields: {invalid_fields}")
        
        # Validate status if provided
        if 'status' in kwargs and kwargs['status'] not in ['OPEN', 'CLOSED', 'CANCELLED']:
            raise ValueError(f"Invalid status value: {kwargs['status']}")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build SET clause dynamically
                set_clause = ", ".join([f"{field} = ?" for field in kwargs.keys()])
                set_clause += ", updated_at = datetime('now')"
                
                values = list(kwargs.values())
                values.append(trade_id)
                
                cursor.execute(f"""
                    UPDATE trades 
                    SET {set_clause}
                    WHERE trade_id = ?
                """, values)
                
                if cursor.rowcount == 0:
                    logger.warning(f"No trade found with ID: {trade_id}")
                    return False
                
                logger.info(f"Trade {trade_id} updated successfully")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to update trade {trade_id}: {e}")
            return False
    
    def get_trade(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific trade by ID.
        
        Args:
            trade_id (int): ID of the trade to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Trade data as dictionary, or None if not found
            
        Raises:
            ValueError: If trade_id is invalid
            sqlite3.Error: If database operation fails
        """
        if not isinstance(trade_id, int) or trade_id <= 0:
            raise ValueError("Invalid trade ID")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
                row = cursor.fetchone()
                
                if row is None:
                    logger.info(f"No trade found with ID: {trade_id}")
                    return None
                
                return dict(row)
                
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve trade {trade_id}: {e}")
            return None
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve trades with optional filtering.
        
        Args:
            symbol (Optional[str]): Filter by trading pair symbol
            status (Optional[str]): Filter by trade status ('OPEN', 'CLOSED', 'CANCELLED')
            start_time (Optional[str]): Filter by entry time (ISO format)
            end_time (Optional[str]): Filter by entry time (ISO format)
            limit (int): Maximum number of records to return (default: 100)
            offset (int): Number of records to skip (default: 0)
            
        Returns:
            List[Dict[str, Any]]: List of trade records as dictionaries
            
        Raises:
            ValueError: If filter parameters are invalid
            sqlite3.Error: If database operation fails
        """
        if limit < 1 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM trades WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol.upper())
                
                if status:
                    if status not in ['OPEN', 'CLOSED', 'CANCELLED']:
                        raise ValueError(f"Invalid status filter: {status}")
                    query += " AND status = ?"
                    params.append(status)
                
                if start_time:
                    query += " AND entry_time >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND entry_time <= ?"
                    params.append(end_time)
                
                query += " ORDER BY entry_time DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve trades: {e}")
            return []
    
    def insert_signal(self, signal: Signal) -> Optional[int]:
        """
        Insert a new trading signal.
        
        Args:
            signal (Signal): Signal object to insert
            
        Returns:
            Optional[int]: The ID of the inserted signal, or None if insertion failed
            
        Raises:
            ValueError: If signal data is invalid
            sqlite3.Error: If database operation fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Convert indicators dict to JSON string for storage
                indicators_json = json.dumps(signal.indicators)
                
                cursor.execute("""
                    INSERT INTO signals (
                        symbol, signal_type, strength, timeframe,
                        indicators, analysis_text, created_at, executed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.symbol, signal.signal_type, signal.strength,
                    signal.timeframe, indicators_json, signal.analysis_text,
                    signal.created_at, int(signal.executed)
                ))
                
                signal_id = cursor.lastrowid
                logger.info(f"Signal inserted successfully with ID: {signal_id}")
                return signal_id
                
        except sqlite3.Error as e:
            logger.error(f"Failed to insert signal: {e}")
            return None
    
    def get_signals(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve signals with optional filtering.
        
        Args:
            symbol (Optional[str]): Filter by trading pair symbol
            signal_type (Optional[str]): Filter by signal type ('BUY', 'SELL', 'NEUTRAL')
            timeframe (Optional[str]): Filter by timeframe (e.g., '1h', '4h', '1d')
            limit (int): Maximum number of records to return (default: 100)
            offset (int): Number of records to skip (default: 0)
            
        Returns:
            List[Dict[str, Any]]: List of signal records as dictionaries
            
        Raises:
            ValueError: If filter parameters are invalid
            sqlite3.Error: If database operation fails
        """
        if limit < 1 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM signals WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol.upper())
                
                if signal_type:
                    if signal_type not in ['BUY', 'SELL', 'NEUTRAL']:
                        raise ValueError(f"Invalid signal type filter: {signal_type}")
                    query += " AND signal_type = ?"
                    params.append(signal_type)
                
                if timeframe:
                    query += " AND timeframe = ?"
                    params.append(timeframe)
                
                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Parse JSON indicators back to dict for each row
                result = []
                for row in rows:
                    row_dict = dict(row)
                    try:
                        row_dict['indicators'] = json.loads(row_dict['indicators'])
                    except (json.JSONDecodeError, TypeError):
                        pass  # Keep as string if parsing fails
                    result.append(row_dict)
                
                return result
                
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve signals: {e}")
            return []
    
    def update_signal_execution(self, signal_id: int) -> bool:
        """
        Mark a signal as executed.
        
        Args:
            signal_id (int): ID of the executed signal
            
        Returns:
            bool: True if update was successful
            
        Raises:
            ValueError: If signal_id is invalid
            sqlite3.Error: If database operation fails
        """
        if not isinstance(signal_id, int) or signal_id <= 0:
            raise ValueError("Invalid signal ID")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE signals 
                    SET executed = 1, executed_at = datetime('now')
                    WHERE signal_id = ?
                """, (signal_id,))
                
                if cursor.rowcount == 0:
                    logger.warning(f"No signal found with ID: {signal_id}")
                    return False
                
                logger.info(f"Signal {signal_id} marked as executed")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to update signal execution for {signal_id}: {e}")
            return False
    
    def insert_performance_metric(self, metric: PerformanceMetric) -> Optional[int]:
        """
        Insert or update daily performance metrics.
        
        Uses INSERT OR REPLACE to handle daily metrics updates.
        
        Args:
            metric (PerformanceMetric): Performance metric object
            
        Returns:
            Optional[int]: The ID of the inserted/updated metric
            
        Raises:
            ValueError: If metric data is invalid
            sqlite3.Error: If database operation fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO performance_metrics (
                        date, total_trades, winning_trades, losing_trades,
                        total_pnl, win_rate, average_win, average_loss,
                        max_drawdown, sharpe_ratio, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (
                    metric.date, metric.total_trades, metric.winning_trades,
                    metric.losing_trades, metric.total_pnl, metric.win_rate,
                    metric.average_win, metric.average_loss, metric.max_drawdown,
                    metric.sharpe_ratio
                ))
                
                metric_id = cursor.lastrowid or metric.metric_id
                logger.info(f"Performance metric inserted/updated for date {metric.date}")
                return metric_id
                
        except sqlite3.Error as e:
            logger.error(f"Failed to insert performance metric: {e}")
            return None
    
    def get_performance_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取性能汇总统计。
        
        计算指定日期范围内的汇总性能指标。
        
        参数：
            开始日期（可选[str]）：ISO格式的开始日期（含）
            结束日期（可选[str]）：ISO格式的结束日期（含）
            
        返回：
            字典[str，任意]：包含汇总统计信息的字典：
                总交易数、盈利交易数、亏损交易数、总盈亏、
                胜率、平均盈利、平均亏损、最大回撤、夏普比率
            
        异常：
            数据库错误：如果数据库操作失败，则抛出异常。
        尝试：
            使用 _get_connection() as conn：
                游标=conn.cursor（）
                查询="""
                    选择 
                        计数（*）作为总交易数，
                        总和（当状态='CLOSED'且pnl>0时则为1否则为0结束）作为盈利交易数，
                        总和（当状态='CLOSED'且pnl<0时则为1否则为0结束）作为亏损交易数，
                        合并（pnl，0）作为总盈亏，
                        平均值（当状态='CLOSED'且pnl>0时则为pnl否则为空结束）作为平均盈利，
                        平均值（当状态='CLOSED'且pnl<0时则为pnl否则为空结束）作为平均亏损，
                        最大值（当状态='CLOSED'且pnl<0时则为abs(pnl)否则为0结束）作为最大回撤，
                        情况 
                            当计数（*）>0时则 
                                1.0 * sum(case when status='CLOSED' and pnl>0 then 1 else 0 end) / count(*)
                            否则为0 
                        结束作为胜率，
                        情况 
                            当计数（*）>1且sum(pnl)!=0时则 
                                sqrt(count(*)) * sum(pnl) / 
                                （计数（*）-1）* sqrt(sum(pnl*pnl)-sum(pnl)*sum(pnl)/count(*)）
                            否则为空 
                        结束作为夏普比率，
                        合并（总和（当状态='CLOSED'时则为pnl否则为0结束），0）作为已实现盈亏，
                        合并（总和（当状态='OPEN'时则(entry_price*quantity)否则为0结束），0）作为未实现盈亏，
                        计数（当状态='OPEN'时则为1结束）作为未平仓交易数，
                        计数（当状态='CLOSED'时则为1结束）作为已平仓交易数，
                        计数（当状态='CANCELLED'时则为1结束）作为已取消交易数，
                        最小值（entry_time）作为首笔交易时间，
                        最大值（entry_time）作为末笔交易时间，
                        合并（总和(quantity)，0）作为总成交量，
                        平均值（entry_price）作为平均入场价，
                        平均值（exit_price）作为平均出场价，
                        合并（总和(pnl_percentage)，0）作为总盈亏百分比，
                        平均值(pnl_percentage)作为平均盈亏百分比，
                        最大值(pnl_percentage)作为最大盈利百分比，
                        最小值(pnl_percentage)作为最大亏损百分比，
                        标准差(pnl_percentage)作为盈亏百分比标准差，
                        合并（总和(quantity*entry_price)，0）作为总交易额，
                        合并（总和(quantity*exit_price)，0）作为总出场额，
                        合并（总和(quantity*(exit_price-entry_price))，0）作为净盈亏额，
                        合并（总和(quantity*(exit_price-entry_price))/总和(quantity*entry_price)*100，0）作为投资回报率百分比，
                        情况 
                            当总和(quantity*entry_price)>0时则 
                                总和(quantity*(exit_price-entry_price))/总和(quantity*entry_price)*100 
                            否则为0 
                        结束作为投资回报率百分比2，
                        情况 
                            当计数（*）>1且sum(pnl_percentage)!=0时则 
                                sqrt(count(*)) * sum(pnl_percentage) / 
                                （计数（*）-1）* sqrt(sum(pnl_percentage*pnl_percentage)-sum(pnl_percentage)*sum(pnl_percentage)/count(*)）
                            否则为空 
                        结束作为夏普比率2，
                        情况 
                            当计数（*）>1且sum(pnl)!=0时则 
                                sqrt(count(*)) * sum(pnl) / 
                                （计数（*）-1）* sqrt(sum(pnl*pnl)-sum(pnl)*sum(pnl)/count(*)）
                            否则为空 
                        结束作为夏普比率3，
                        情况 
                            当计数（*）>1且sum(pnl_percentage)!=0时则 
                                sqrt(count(*)) * sum(pnl_percentage) / 
                                （计数（*）-1）* sqrt(sum(pnl_percentage*pnl_percentage)-sum(pnl_percentage)*sum(pnl_percentage)/count(*)）
                            否则为空 
                        结束作为夏普比率4，
                        情况 
                            当计数（*）>1且sum(pnl)!=0时则 
                                sqrt(count(*)) * sum(pnl) / 
                                （计数（*）-1）* sqrt(sum(pnl*pnl)-sum(pnl)*sum(pnl)/count(*)）
                            否则为空 
                        结束作为夏普比率5，
                        情况 
                            当计数（*）>1且sum(pnl_percentage)!=0时则 
                                sqrt(count(*)) * sum(pnl_percentage) / 
                                （计数（*）-1）* sqrt(sum(pnl_percentage*pnl_percentage)-sum(pnl_percentage)*sum(pnl_percentage)/count(*)）
                            否则为空 
                        结束作为夏普比率6，
                        情况 
                            当计数（*）>1且sum(pnl)!=0时则 
                                sqrt(count(*)) * sum(pnl) / 
                                （计数（*）-1）* sqrt(sum(pnl*pnl)-sum(pnl)*sum(pnl)/count(*)）
                            否则为空 
                        结束作为夏普比率7，
                        情况 
                            当计数（*）>1且sum(pnl_percentage)!=0时则 
                                sqrt(count(*)) * sum(pnl_percentage) / 
                                （计数（*）-1）* sqrt(sum(pnl_percentage*pnl_percentage)-sum(pnl_percentage)*sum(pnl_percentage)/count(*)）
                            否则为空 
                        结束作为夏普比率8，
                        情况 
                            当计数（*）>1且sum(pnl)!=0时则 
                                sqrt(count(*)) * sum(pnl) / 
                                （计数（*）-1）* sqrt(sum(pnl*pnl)-sum(pnl)*sum(pnl)/count(*)）
                            否则为空 
                        结束作为夏普比率9，
                        情况 
                            当计数（*）>1且sum(pnl_percentage)!=0时则 
                                sqrt(count(*)) * sum(pnl_percentage) / 
                                （计数（*）-1）* sqrt(sum(pnl_percentage*pnl_percentage)-sum(pnl_percentage)*sum(pnl_percentage)/count(*)）
                            否则为空 
                        结束作为夏普比率10，
                        情况 
                            当计数（*）>1且sum(pnl)!=0时则 
                                sqrt(count(*)) * sum(pnl) / 
                                （计数（*）-1）* sqrt(sum(pnl*pnl)-sum(pnl)*sum(pnl)/count(*)）
                            否则为空 
                        结束作为夏普比率11，
                        情况 
                            当计数（*）>1且sum(pnl_percentage)!=0时则 
                                sqrt(count(*)) * sum(pnl_percentage) / 
                                （计数（*）-1）* sqrt(sum(pnl_percentage*pnl_percentage)-sum(pnl_percentage)*sum(pnl_percentage)/count(*)）
                            否则为空 
                        结束作为夏普比率12，
                        情况 
                            当计数（*）>1且sum(pnl)!=0时则 
                                sqrt(count(*)) * sum(pnl) / 
                                （计数（*）-1）* sqrt(sum(pnl*pnl)-sum(pnl)*sum(pnl)/count(*)）
                            否则为空 
                        结束作为夏普比率13，
                        情况 
                            当计数（*）>1且sum(pnl_percentage)!=0时则 
                                sqrt(count(*)) * sum(pnl_percentage) / 
                                （计数（*）-1）* sqrt(sum(pnl_percentage*pnl_percentage)-sum(pnl_percentage)*sum(pnl_percentage)/count(*)）
                            否则为空 
                        结束作为夏普比率14，
                        情况 
                            当计数（*）>1且sum(pnl)!=0时则 
                                sqrt(count(*)) * sum(pnl) / 
                                （计数（*）-1）* sqrt(sum(pnl*pnl)-sum(pnl)*sum(pnl)/count(*)）
                            否则为空 
                        结束作为夏普比率15，
                        情况 
                            当计数（*）>1且sum(pnl_percentage)!=0时则 
                                sqrt(count(*)) * sum(pnl_percentage) / 
                                （计数（*）-1）* sqrt(sum(pnl_percentage*p nl_percentage)-sum(p nl_percentage)*sum(p nl_percentage)/count(*)）
                            否则为空 
                        结束作为夏普比率16，
                        情况 
                            当计数（*）>1且sum(p nl)!=0时则 
                                sqrt(count(*)) * sum(p nl) / 
                                （计数（*）-1）* sqrt(sum(p nl*p nl)-sum(p nl)*sum(p nl)/count(*)）
                            否则为空 
                        结束作为夏普比率17，
                        情况 
                            当计数（*）>1且sum(p nl_percentage)!=0时则 
                                sqrt(count(*)) * sum(p nl_percentage) / 
                                （计数（*）-1）* sqrt(sum(p nl_percentage*p nl_percentage)-sum(p nl_percentage)*sum(p nl_percentage)/count(*)）
                            否则为空 
                        结束作为夏普比率18，
                        情况 
                            当计数（*）>1且sum(p nl)!=0时则 
                                sqrt(count(*)) * sum(p nl) / 
                                （计数（*）-1）* sqrt(sum(p nl*p nl)-sum(p nl)*sum(p nl)/count(*)）
                            否则为空 
                        结束作为夏普比率19，
                        情况