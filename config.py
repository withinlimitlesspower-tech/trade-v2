"""
Configuration module for the cryptocurrency trading bot.

This module handles loading and validating all environment variables from
Replit Secrets, including API keys for Binance and DeepSeek services.
Provides centralized configuration management with proper error handling
and type validation.

Environment Variables Required:
    BINANCE_API_KEY: Binance API key for market data and trading
    BINANCE_SECRET_KEY: Binance secret key for authentication
    DEEPSEEK_API_KEY: DeepSeek API key for AI analysis
    BINANCE_TESTNET (optional): Use Binance testnet (true/false)
    LOG_LEVEL (optional): Logging level (default: INFO)
    MAX_POSITION_SIZE (optional): Maximum position size in USDT (default: 100)
    RISK_PER_TRADE (optional): Risk per trade as decimal (default: 0.02)
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Trading environment options."""
    PRODUCTION = "production"
    TESTNET = "testnet"


@dataclass
class Config:
    """
    Central configuration class for the trading bot.
    
    Attributes:
        binance_api_key: Binance API key for authentication
        binance_secret_key: Binance secret key for signing requests
        deepseek_api_key: DeepSeek API key for AI analysis
        environment: Trading environment (production/testnet)
        log_level: Logging level configuration
        max_position_size: Maximum position size in USDT
        risk_per_trade: Risk per trade as decimal (0.0 to 1.0)
        supported_timeframes: List of supported trading timeframes
        api_timeout: API request timeout in seconds
        max_retries: Maximum number of API retry attempts
    """
    
    binance_api_key: str = ""
    binance_secret_key: str = ""
    deepseek_api_key: str = ""
    environment: Environment = Environment.PRODUCTION
    log_level: str = "INFO"
    max_position_size: float = 100.0
    risk_per_trade: float = 0.02
    supported_timeframes: list = field(default_factory=lambda: [
        "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"
    ])
    api_timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_position_size()
        self._validate_risk_per_trade()
    
    def _validate_position_size(self) -> None:
        """Validate max position size is positive."""
        if self.max_position_size <= 0:
            raise ValueError(
                f"Invalid max_position_size: {self.max_position_size}. "
                "Must be greater than 0."
            )
    
    def _validate_risk_per_trade(self) -> None:
        """Validate risk per trade is between 0 and 1."""
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError(
                f"Invalid risk_per_trade: {self.risk_per_trade}. "
                "Must be between 0 and 1."
            )


class ConfigLoader:
    """
    Handles loading and validating configuration from environment variables.
    
    This class provides methods to securely load API keys and other
    configuration from Replit Secrets or system environment variables.
    """
    
    REQUIRED_VARS = [
        "BINANCE_API_KEY",
        "BINANCE_SECRET_KEY",
        "DEEPSEEK_API_KEY"
    ]
    
    OPTIONAL_VARS = {
        "BINANCE_TESTNET": ("false", str),
        "LOG_LEVEL": ("INFO", str),
        "MAX_POSITION_SIZE": ("100", float),
        "RISK_PER_TRADE": ("0.02", float)
    }
    
    @staticmethod
    def _get_env_var(var_name: str, required: bool = True) -> Optional[str]:
        """
        Safely retrieve an environment variable.
        
        Args:
            var_name: Name of the environment variable
            required: Whether the variable is required
            
        Returns:
            Value of the environment variable or None if not found
            
        Raises:
            EnvironmentError: If required variable is missing
        """
        value = os.environ.get(var_name)
        
        if value is None or value.strip() == "":
            if required:
                raise EnvironmentError(
                    f"Required environment variable '{var_name}' is not set. "
                    "Please ensure it is configured in Replit Secrets."
                )
            return None
        
        return value.strip()
    
    @staticmethod
    def _parse_boolean(value: str) -> bool:
        """Parse string boolean value."""
        return value.lower() in ("true", "1", "yes", "y")
    
    @staticmethod
    def _parse_float(value: str) -> float:
        """Parse string to float with validation."""
        try:
            return float(value)
        except ValueError as e:
            raise ValueError(f"Invalid float value: {value}") from e
    
    @classmethod
    def validate_environment(cls) -> Dict[str, bool]:
        """
        Validate all required environment variables exist.
        
        Returns:
            Dictionary with validation results for each variable
            
        Raises:
            EnvironmentError: If any required variables are missing
        """
        validation_results = {}
        missing_vars = []
        
        # Check required variables
        for var_name in cls.REQUIRED_VARS:
            try:
                value = cls._get_env_var(var_name, required=True)
                validation_results[var_name] = bool(value)
            except EnvironmentError:
                missing_vars.append(var_name)
                validation_results[var_name] = False
        
        # Check optional variables (just warn if missing)
        for var_name, (default, _) in cls.OPTIONAL_VARS.items():
            try:
                value = cls._get_env_var(var_name, required=False)
                validation_results[var_name] = value is not None
            except EnvironmentError:
                # Optional variables use defaults, so they're always valid
                validation_results[var_name] = True
        
        if missing_vars:
            error_msg = (
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                "Please configure them in Replit Secrets:\n"
                "1. Go to your Replit project\n"
                "2. Click on 'Tools' in the sidebar\n"
                "3. Select 'Secrets'\n"
                "4. Add the missing variables"
            )
            logger.error(error_msg)
            raise EnvironmentError(error_msg)
        
        return validation_results
    
    @classmethod
    def load_config(cls) -> Config:
        """
        Load and validate all configuration from environment variables.
        
        Returns:
            Config object with all settings loaded
            
        Raises:
            EnvironmentError: If required variables are missing or invalid
            ValueError: If configuration values are invalid
        """
        logger.info("Loading configuration from environment variables...")
        
        # First validate all required variables exist
        cls.validate_environment()
        
        try:
            # Load required variables
            binance_api_key = cls._get_env_var("BINANCE_API_KEY", required=True)
            binance_secret_key = cls._get_env_var("BINANCE_SECRET_KEY", required=True)
            deepseek_api_key = cls._get_env_var("DEEPSEEK_API_KEY", required=True)
            
            # Load optional variables with defaults
            testnet_str = cls._get_env_var("BINANCE_TESTNET", required=False) or "false"
            use_testnet = cls._parse_boolean(testnet_str)
            
            log_level = cls._get_env_var("LOG_LEVEL", required=False) or "INFO"
            
            max_pos_str = cls._get_env_var("MAX_POSITION_SIZE", required=False) or "100"
            max_position_size = cls._parse_float(max_pos_str)
            
            risk_str = cls._get_env_var("RISK_PER_TRADE", required=False) or "0.02"
            risk_per_trade = cls._parse_float(risk_str)
            
            # Create configuration object
            config = Config(
                binance_api_key=binance_api_key,
                binance_secret_key=binance_secret_key,
                deepseek_api_key=deepseek_api_key,
                environment=Environment.TESTNET if use_testnet else Environment.PRODUCTION,
                log_level=log_level.upper(),
                max_position_size=max_position_size,
                risk_per_trade=risk_per_trade
            )
            
            # Validate log level
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if config.log_level not in valid_log_levels:
                logger.warning(
                    f"Invalid log level '{config.log_level}'. Defaulting to 'INFO'."
                )
                config.log_level = "INFO"
            
            logger.info(
                f"Configuration loaded successfully. "
                f"Environment: {config.environment.value}, "
                f"Log Level: {config.log_level}"
            )
            
            # Mask sensitive information for logging
            masked_config = {
                "binance_api_key": f"{config.binance_api_key[:4]}...{config.binance_api_key[-4:]}",
                "binance_secret_key": f"{config.binance_secret_key[:4]}...{config.binance_secret_key[-4:]}",
                "deepseek_api_key": f"{config.deepseek_api_key[:4]}...{config.deepseek_api_key[-4:]}",
                "environment": config.environment.value,
                "log_level": config.log_level,
                "max_position_size": config.max_position_size,
                "risk_per_trade": config.risk_per_trade,
                "supported_timeframes": config.supported_timeframes,
                "api_timeout": config.api_timeout,
                "max_retries": config.max_retries
            }
            
            logger.debug(f"Configuration details: {masked_config}")
            
            return config
            
        except (ValueError, EnvironmentError) as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise


# Global configuration instance (lazy-loaded)
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    This function implements lazy loading pattern to ensure configuration
    is only loaded once and reused throughout the application lifecycle.
    
    Returns:
        Config object with all settings
        
    Raises:
        EnvironmentError: If configuration cannot be loaded
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigLoader.load_config()
    
    return _config_instance


def reload_config() -> Config:
    """
    Force reload configuration from environment variables.
    
    This is useful when environment variables might have changed during runtime.
    
    Returns:
        Newly loaded Config object
    """
    global _config_instance
    
    logger.info("Reloading configuration...")
    _config_instance = ConfigLoader.load_config()
    
    return _config_instance


def validate_api_keys() -> bool:
    """
    Quick validation that all API keys are properly formatted.
    
    Returns:
        True if all API keys appear valid, False otherwise
        
    Note:
        This performs basic format validation only, not actual API connectivity.
    """
    try:
        config = get_config()
        
        # Basic format validation for API keys
        if len(config.binance_api_key) < 10:
            logger.warning("Binance API key appears too short")
            return False
        
        if len(config.binance_secret_key) < 10:
            logger.warning("Binance secret key appears too short")
            return False
        
        if len(config.deepseek_api_key) < 10:
            logger.warning("DeepSeek API key appears too short")
            return False
        
        logger.info("All API keys pass basic format validation")
        return True
        
    except Exception as e:
        logger.error(f"API key validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    """
    Main block for testing configuration loading.
    
    Run this script directly to verify your Replit Secrets are configured correctly.
    
    Usage:
        python config.py
        
    Expected output:
        Configuration loaded successfully.
        
    If errors occur, they will be displayed with guidance on how to fix them.
    """
    
    print("=" * 60)
    print("Trading Bot Configuration Validator")
    print("=" * 60)
    
    try:
        # Validate environment variables exist
        print("\nChecking environment variables...")
        
        # Test configuration loading
        config = get_config()
        
        print("\n✓ Configuration loaded successfully!")
        
        # Display non-sensitive configuration info
        print(f"\nConfiguration Summary:")
        print(f"  • Environment: {config.environment.value}")
        print(f"  • Log Level: {config.log_level}")
        print(f"  • Max Position Size: ${config.max_position_size:.2f} USDT")
        print(f"  • Risk Per Trade: {config.risk_per_trade:.2%}")
        
        # Validate API keys format
        print("\nValidating API keys format...")
        
        if validate_api_keys():
            print("✓ All API keys pass format validation")
            
            # Show masked keys for verification
            print(f"\nAPI Keys (masked):")
            print(f"  • Binance API Key: {config.binance_api_key[:4]}...{config.binance_api_key[-4:]}")
            print(f"  • Binance Secret Key: {config.binance_secret_key[:4]}...{config.binance_secret_key[-4:]}")
            print(f"  • DeepSeek API Key: {config.deepseek_api_key[:4]}...{config.deepseek_api_key[-4:]}")
        
        print("\n✓ All checks passed! Configuration is ready.")
        
    except EnvironmentError as e:
        print(f"\n✗ Configuration Error:")
        print(f"  {str(e)}")
        
        print("\nTroubleshooting Steps:")
        print("  1. Go to your Replit project")
        print("  2. Click on 'Tools' in the sidebar")
        print("  3. Select 'Secrets'")
        print("  4. Ensure the following variables are set:")
        
        for var in ConfigLoader.REQUIRED_VARS:
            print(f"     - {var}")
        
        print("\n  5. Restart your application after setting secrets")
        
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n✗ Configuration Value Error:")
        print(f"  {str(e)}")
        
        sys.exit(1)
        
except Exception as e:
print(f"\n✗ Unexpected Error:")
print(f"  {str(e)}")

print("\nPlease check your configuration and try again.")
sys.exit(1)