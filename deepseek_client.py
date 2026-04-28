"""
DeepSeek API Client for Cryptocurrency Trading Bot.

This module provides an interface to interact with DeepSeek API for advanced
technical and fundamental analysis, including trendlines, patterns, and indicators.
It handles API authentication, request formatting, response parsing, and error handling.

Typical usage example:
    client = DeepSeekClient()
    analysis = client.analyze_market(market_data)
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepSeekAPIError(Exception):
    """Custom exception for DeepSeek API errors."""
    pass


class DeepSeekRateLimitError(DeepSeekAPIError):
    """Exception raised when API rate limit is exceeded."""
    pass


class DeepSeekAuthenticationError(DeepSeekAPIError):
    """Exception raised for authentication failures."""
    pass


class DeepSeekClient:
    """
    Client for interacting with DeepSeek API for cryptocurrency analysis.
    
    This client handles authentication, request management, and response parsing
    for technical and fundamental analysis of cryptocurrency markets.
    
    Attributes:
        api_key: DeepSeek API key from environment variables
        base_url: Base URL for DeepSeek API endpoints
        session: Requests session with retry logic
        max_retries: Maximum number of retry attempts for failed requests
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize DeepSeek client with API credentials and configuration.

        Args:
            api_key: DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env var.
            base_url: Base URL for DeepSeek API endpoints.
            max_retries: Maximum number of retry attempts for failed requests.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.base_url = base_url.rstrip('/')
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize session with retry logic
        self.session = self._create_session()
        
        logger.info("DeepSeek client initialized successfully")

    def _create_session(self) -> requests.Session:
        """
        Create requests session with retry strategy and headers.

        Returns:
            Configured requests Session object.
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # Set default headers
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "CryptoTradingBot/1.0"
        })
        
        return session

    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to DeepSeek API with error handling.

        Args:
            endpoint: API endpoint path.
            method: HTTP method (GET or POST).
            data: Request payload for POST requests.
            params: Query parameters for GET requests.

        Returns:
            Parsed JSON response from API.

        Raises:
            DeepSeekAuthenticationError: If authentication fails.
            DeepSeekRateLimitError: If rate limit is exceeded.
            DeepSeekAPIError: For other API errors.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            if method.upper() == "GET":
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )
            else:
                response = self.session.post(
                    url,
                    json=data,
                    params=params,
                    timeout=self.timeout
                )
            
            # Handle specific HTTP status codes
            if response.status_code == 401:
                raise DeepSeekAuthenticationError(
                    "Authentication failed. Check your API key."
                )
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                raise DeepSeekRateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after} seconds."
                )
            
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout for {url}: {str(e)}")
            raise DeepSeekAPIError(f"Request timed out after {self.timeout}s") from e
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {str(e)}")
            raise DeepSeekAPIError(f"Failed to connect to DeepSeek API") from e
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {response.status_code} for {url}: {str(e)}")
            try:
                error_detail = response.json().get("error", {}).get("message", str(e))
            except (json.JSONDecodeError, AttributeError):
                error_detail = str(e)
            raise DeepSeekAPIError(f"API request failed: {error_detail}") from e
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {url}: {str(e)}")
            raise DeepSeekAPIError("Invalid response format from API") from e

    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """
        Validate market data structure before sending to API.

        Args:
            market_data: Dictionary containing market data to validate.

        Returns:
            True if valid, raises ValueError otherwise.

        Raises:
            ValueError: If market data is invalid or missing required fields.
        """
        required_fields = ["symbol", "price", "volume"]
        
        if not isinstance(market_data, dict):
            raise ValueError("Market data must be a dictionary")
        
        missing_fields = [field for field in required_fields if field not in market_data]
        if missing_fields:
            raise ValueError(f"Missing required fields in market data: {missing_fields}")
        
        # Validate data types
        if not isinstance(market_data.get("price"), (int, float)):
            raise ValueError("Price must be a numeric value")
        
        if not isinstance(market_data.get("volume"), (int, float)):
            raise ValueError("Volume must be a numeric value")
        
        return True

    def analyze_market(
        self,
        market_data: Dict[str, Any],
        analysis_type: str = "technical",
        timeframe: str = "1h",
        include_patterns: bool = True,
        include_indicators: bool = True,
        include_trendlines: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis using DeepSeek AI.

        Analyzes cryptocurrency market data for technical patterns, indicators,
        and trendlines. Returns structured analysis results.

        Args:
            market_data: Dictionary containing market information including:
                - symbol: Trading pair symbol (e.g., "BTCUSDT")
                - price: Current price
                - volume: 24h volume
                - high_24h: 24h high price (optional)
                - low_24h: 24h low price (optional)
                - price_change_24h: 24h price change percentage (optional)
                - market_cap: Market capitalization (optional)
                - historical_data: List of historical price/volume data (optional)
            analysis_type: Type of analysis ("technical", "fundamental", or "both")
            timeframe: Analysis timeframe ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w")
            include_patterns: Whether to include chart pattern detection
            include_indicators: Whether to include technical indicator analysis
            include_trendlines: Whether to include trendline analysis

        Returns:
            Dictionary containing analysis results with keys:
                - symbol: Analyzed trading pair
                - timestamp: Analysis timestamp
                - analysis_type: Type of analysis performed
                - timeframe: Timeframe analyzed
                - patterns: Detected chart patterns (if requested)
                - indicators: Technical indicator values (if requested)
                - trendlines: Trendline analysis (if requested)
                - sentiment: Market sentiment assessment
                - recommendation: Trading recommendation
                - confidence_score: Confidence level of analysis (0-100)
                - risk_level: Risk assessment ("low", "medium", "high")

        Raises:
            ValueError: If market data validation fails.
            DeepSeekAPIError: If API request fails.
        """
        # Validate input data
        self._validate_market_data(market_data)
        
        if analysis_type not in ["technical", "fundamental", "both"]:
            raise ValueError("analysis_type must be 'technical', 'fundamental', or 'both'")
        
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        if timeframe not in valid_timeframes:
            raise ValueError(f"timeframe must be one of {valid_timeframes}")
        
        # Prepare analysis request payload
        payload = {
            "symbol": market_data["symbol"],
            "price": market_data["price"],
            "volume": market_data["volume"],
            "analysis_type": analysis_type,
            "timeframe": timeframe,
            "include_patterns": include_patterns,
            "include_indicators": include_indicators,
            "include_trendlines": include_trendlines,
            "market_context": {
                "high_24h": market_data.get("high_24h"),
                "low_24h": market_data.get("low_24h"),
                "price_change_24h": market_data.get("price_change_24h"),
                "market_cap": market_data.get("market_cap")
            }
        }
        
        # Add historical data if provided (limit to prevent oversized requests)
        if "historical_data" in market_data and isinstance(market_data["historical_data"], list):
            payload["historical_data"] = market_data["historical_data"][-100:]  # Last 100 data points
        
        try:
            logger.info(f"Requesting {analysis_type} analysis for {market_data['symbol']} on {timeframe} timeframe")
            
            response = self._make_request(
                endpoint="analyze/market",
                method="POST",
                data=payload
            )
            
            # Parse and structure the response
            analysis_result = self._parse_analysis_response(response)
            
            logger.info(f"Analysis completed for {market_data['symbol']}")
            
            return analysis_result
            
        except DeepSeekAPIError as e:
            logger.error(f"Analysis failed for {market_data['symbol']}: {str(e)}")
            
            # Return fallback analysis on error
            return self._get_fallback_analysis(market_data["symbol"], str(e))

    def _parse_analysis_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and structure the raw API response into a standardized format.

        Args:
            response: Raw API response dictionary.

        Returns:
            Structured analysis result dictionary.
        """
        # Extract core analysis data with defaults for missing fields
        result = {
            "symbol": response.get("symbol", ""),
            "timestamp": response.get("timestamp", datetime.utcnow().isoformat()),
            "analysis_type": response.get("analysis_type", ""),
            "timeframe": response.get("timeframe", ""),
            
            # Technical patterns detection
            "patterns": self._parse_patterns(response.get("patterns", [])),
            
            # Technical indicators calculation
            "indicators": self._parse_indicators(response.get("indicators", {})),
            
            # Trendline analysis
            "trendlines": self._parse_trendlines(response.get("trendlines", [])),
            
            # Market sentiment and recommendations
            "sentiment": response.get("sentiment", {}),
            "recommendation": response.get("recommendation", {
                "action": "hold",
                "reasoning": "Insufficient data for recommendation"
            }),
            
            # Confidence and risk metrics
            "confidence_score": min(max(response.get("confidence_score", 50), 0), 100),
            "risk_level": response.get("risk_level", "medium"),
            
            # Additional metadata
            "support_levels": response.get("support_levels", []),
            "resistance_levels": response.get("resistance_levels", []),
            
            # Raw response for debugging
            "_raw_response": response
        }
        
        return result

    def _parse_patterns(self, patterns_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse detected chart patterns from API response.

        Args:
            patterns_data: List of pattern dictionaries from API.

        Returns:
            List of parsed pattern dictionaries with standardized format.
        """
        parsed_patterns = []
        
        for pattern in patterns_data:
            parsed_pattern = {
                "name": pattern.get("name", ""),
                "type": pattern.get("type", ""),  # bullish/bearish/neutral
                "confidence": min(max(pattern.get("confidence", 0), 0), 100),
                "description": pattern.get("description", ""),
                "price_target": pattern.get("price_target"),
                "stop_loss": pattern.get("stop_loss"),
                "formation_start": pattern.get("formation_start"),
                "formation_end": pattern.get("formation_end"),
                "completion_percentage": min(max(pattern.get("completion_percentage", 0), 0), 100)
            }
            
            if parsed_pattern["name"]:
                parsed_patterns.append(parsed_pattern)
        
        return parsed_patterns

    def _parse_indicators(self, indicators_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse technical indicators from API response.

        Args:
            indicators_data: Dictionary of indicator values from API.

        Returns:
          Parsed indicators dictionary with standardized format.
      """
      parsed_indicators = {}
      
      # Define expected indicators and their parsing logic
      indicator_mapping = {
          "rsi": {"name": "RSI", "min": 0, "max": 100},
          "macd": {"name": "MACD"},
          "moving_averages": {"name": "Moving Averages"},
          "bollinger_bands": {"name": "Bollinger Bands"},
          "stochastic": {"name": "Stochastic Oscillator"},
          "volume_profile": {"name": "Volume Profile"},
          "support_resistance": {"name": "Support/Resistance Levels"},
          "fibonacci_levels": {"name": "Fibonacci Levels"}
      }
      
      for indicator_key, indicator_info in indicator_mapping.items():
          raw_value = indicators_data.get(indicator_key)
          
          if raw_value is not None:
              parsed_value = raw_value
              
              # Apply range constraints where applicable
              if indicator_info.get("min") is not None and indicator_info.get("max") is not None:
                  if isinstance(raw_value, (int, float)):
                      parsed_value = min(max(raw_value, indicator_info["min"]), indicator_info["max"])
              
              parsed_indicators[indicator_info["name"]] = parsed_value
      
      return parsed_indicators

    def _parse_trendlines(self, trendlines_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
      """
      Parse trendline analysis from API response.

      Args:
          trendlines_data: List of trendline dictionaries from API.

      Returns:
          List of parsed trendline dictionaries with standardized format.
      """
      parsed_trendlines = []
      
      for trendline in trendlines_data:
          parsed_line = {
              "type": trendline.get("type", ""),  # support/resistance/channel
              "direction": trendline.get("direction", ""),  # upward/downward/sideways
              "strength": min(max(trendline.get("strength", 0), 0), 100),
              "start_price": trendline.get("start_price"),
              "end_price": trendline.get("end_price"),
              "start_time": trendline.get("start_time"),
              "end_time": trendline.get("end_time"),
              "touches_count": max(trendline.get("touches_count", 0), 0),
              "breakout_probability": min(max(trendline.get("breakout_probability", 0), 0), 100),
              "description": trendline.get("description", "")
          }
          
          if parsed_line["type"] and parsed_line["direction"]:
              parsed_trendlines.append(parsed_line)
      
      return parsed_trendlines

    def _get_fallback_analysis(self, symbol: str, error_message: str) -> Dict[str, Any]:
      """
      Generate fallback analysis when API call fails.

      Args:
          symbol: Trading pair symbol.
          error_message: Error message from failed API call.

      Returns:
          Dictionary with safe default values and error information.
      """
      return {
          "symbol": symbol,
          "timestamp": datetime.utcnow().isoformat(),
          "analysis_type": "",
          "timeframe": "",
          "patterns": [],
          "indicators": {},
          "trendlines": [],
          "sentiment": {
              "overall_sentiment": neutral",
              "_error": f"Analysis unavailable due to API error"
          },
          recommendation {
              action":"hold",
              reasoning f"Unable to generate recommendation due to error"
          },
          confidence_score 0,
          risk_level medium",
          support_levels [],
          resistance_levels [],
          "_error error_message"
      }

   def analyze_multiple_symbols(
       self,
       symbols_list List[str],
       market_data_dict Dict[str,Dict[strAny]],
       analysis_type str ="technical",
       timeframe str="1h"
   )->Dict[str,Dict[strAny]]:
       """
       Perform analysis for multiple trading pairs simultaneously.

       Args:
           symbols_list List of trading pair symbols to analyze.
           market_data_dict Dictionary mapping symbols to their market data.
           analysis_type Type of analysis to perform.
           timeframe Timeframe for analysis.

       Returns Dictionary mapping symbols to their analysis results.
       """
       results {}
       
       for symbol in symbols_list:
           if symbol in market_data_dict:
               try result=self.analyze_market(
                   market_data_dict[symbol],
                   analysis_type=analysis_type,
                   timeframe=timeframe
               )
               results[symbol]=result
           else results[symbol]=self._get_fallback_analysis(
               symbol,
               f"No market data available for {symbol}"
           )
       
       return results

   def get_market_sentiment(
       self,
       symbol str,
       news_data Optional[List[Dict[strAny]]]=None,
       social_sentiment Optional[Dict[strAny]]=None
   )->Dict[strAny]:
       """
       Analyze overall market sentiment using fundamental and news data.

       Args:
           symbol Trading pair symbol.
           news_data List of recent news articles (optional).
           social_sentiment Social media sentiment data (optional).

       Returns Dictionary containing sentiment analysis results.
       """
       payload={
           symbol symbol",
           news news_data or [],
           social_sentiment social_sentiment or {}
       }
       
       try logger.info(f"Requesting sentiment analysis for {symbol}")
           
           response=self._make_request(
               endpoint analyze/sentiment",
               method POST",
               data payload"
           )
           
           return {
               overall_sentiment response get overall_sentiment neutral",
               sentiment_score min max response get sentiment_score 50 ,0 ,100 ,
               key_factors response get key_factors [],
               news_sentiment response get news_sentiment {},
               social_sentiment response get social_sentiment {},
               timestamp datetime utcnow isoformat()
           }
           
       except DeepSeekAPIError as e logger.error(f"Sentiment analysis failed for {symbol}: {str(e)}")
           
           return {
               overall_sentiment neutral",
               sentiment_score 50,
               key_factors [],
               _error str(e)
           }

   def get_trading_signals(
       self,
       symbol str,
       technical_analysis Dict[strAny],
       fundamental_analysis Optional[Dict[strAny]]=None,
       risk_tolerance str="medium"
   )->Dict[strAny]:
       """
       Generate trading signals based on combined technical and fundamental analysis.

       Args:
           symbol Trading pair symbol.
           technical_analysis Results from technical analysis.
           fundamental_analysis Results from fundamental analysis (optional).
           risk_tolerance Risk tolerance level low medium high.

       Returns Dictionary containing trading signals and recommendations.
       """
       payload={
           symbol symbol",
           technical_analysis technical_analysis",
           fundamental_analysis fundamental_analysis or {},
           risk_tolerance risk_tolerance"
       }
       
       try logger.info(f"Generating trading signals for {symbol}")
           
           response=self._make_request(
               endpoint generate signals",
               method POST",
               data payload"
           )
           
           return {
               signal_type response get signal_type neutral",
               entry_price response get entry_price ,
               exit_price response get exit_price ,
               stop_loss response get stop_loss ,
               take_profit response get take_profit ,
               confidence min max response get confidence 50 ,0 ,100 ,
               reasoning response get reasoning "",
               time_horizon response get time_horizon "",
               risk_reward_ratio response get risk_reward_ratio ,
               timestamp datetime utcnow isoformat()
           }
           
       except DeepSeekAPIError as e logger.error(f"Signal generation failed for {symbol}: {str(e)}")
           
           return {
               signal_type neutral",
               confidence 0,
               reasoning f"Unable to generate signals due to error"
           }

   def close(self):
       """Clean up resources by closing the session."""
       if hasattr(self session) and self session self session.close()
       
   def __enter__(self):
       """Context manager entry."""
       return self

   def __exit__(self exc_type exc_val exc_tb):
       """Context manager exit with cleanup."""
       self close()


if __name__ "__main__":
   """Example usage of the DeepSeekClient class."""
   
   # Example market data structure
   example_market_data={
       symbol BTCUSDT",
       price 45000.50,
       volume 1234567890.00,
       high_24h 46000.00,
       low_24h 44000.00,
       price_change_24h 2.5,
       market_cap 850000000000.00,
       historical_data [
           {"timestamp":"2024-01-01T00:00Z","price":44000,"volume":1000000},
           {"timestamp":"2024-01-01T01:00Z","price":44500,"volume":1200000}
       ]
   }
   
   try client=DeepSeekClient()
       
       print(f"Analyzing {example_market_data['symbol']}...")
       
       result=client.analyze_market(
           example_market_data,
           analysis_type technical",
           timeframe 1h"
       )
       
       print(json dumps(result indent=2))
       
   except Exception as e print(f"Error during example execution:{e}")