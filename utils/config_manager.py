"""
Advanced Configuration System for Portfolio Forecasting

This module provides comprehensive configuration management for all aspects
of the portfolio forecasting system including models, data sources, and UI settings.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for forecasting models."""
    
    # General model settings
    default_periods: int = 30
    confidence_interval: float = 0.95
    cross_validation_folds: int = 5
    
    # ARIMA settings
    arima_auto: bool = True
    arima_max_p: int = 5
    arima_max_d: int = 2
    arima_max_q: int = 5
    arima_seasonal: bool = True
    arima_m: int = 12
    
    # Prophet settings
    prophet_growth: str = "linear"  # "linear" or "logistic"
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_daily_seasonality: bool = False
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_prior_scale: float = 10.0
    
    # LSTM settings
    lstm_units: int = 50
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_lookback: int = 60
    
    # Ensemble settings
    ensemble_use_voting: bool = True
    ensemble_use_stacking: bool = True
    ensemble_use_blending: bool = True
    ensemble_weight_method: str = "performance_based"  # "performance_based", "equal", "inverse_variance"


@dataclass
class DataConfig:
    """Configuration for data handling."""
    
    # Data sources
    default_source: str = "yfinance"  # "yfinance", "alpha_vantage", "quandl"
    backup_sources: list = None
    
    # Data processing
    fill_method: str = "forward"  # "forward", "backward", "interpolate", "drop"
    outlier_detection: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 3.0
    
    # Frequency and periods
    default_frequency: str = "D"  # "D", "W", "M", "Q", "Y"
    min_data_points: int = 100
    max_data_points: int = 10000
    
    # Market data settings
    market_hours_only: bool = False
    adjust_splits: bool = True
    adjust_dividends: bool = True
    
    def __post_init__(self):
        if self.backup_sources is None:
            self.backup_sources = ["alpha_vantage", "quandl"]


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    
    # VaR settings
    var_confidence_levels: list = None
    var_methods: list = None
    var_lookback_days: int = 252
    
    # Monte Carlo settings
    mc_simulations: int = 10000
    mc_time_horizon: int = 252
    mc_confidence_intervals: list = None
    
    # Stress testing
    stress_scenarios: dict = None
    stress_shock_sizes: list = None
    
    # Portfolio metrics
    benchmark_symbol: str = "^GSPC"  # S&P 500
    risk_free_rate: float = 0.02  # 2% annual
    
    def __post_init__(self):
        if self.var_confidence_levels is None:
            self.var_confidence_levels = [0.95, 0.99, 0.999]
        
        if self.var_methods is None:
            self.var_methods = ["historical", "parametric", "monte_carlo"]
        
        if self.mc_confidence_intervals is None:
            self.mc_confidence_intervals = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        if self.stress_scenarios is None:
            self.stress_scenarios = {
                "Market Crash": {"equity": -0.30, "bond": -0.05, "commodity": -0.15},
                "Interest Rate Spike": {"equity": -0.10, "bond": -0.20, "commodity": 0.05},
                "Inflation Surge": {"equity": -0.05, "bond": -0.15, "commodity": 0.20},
                "Currency Crisis": {"equity": -0.15, "bond": 0.02, "fx": -0.25}
            }
        
        if self.stress_shock_sizes is None:
            self.stress_shock_sizes = [0.01, 0.02, 0.05, 0.10, 0.20]


@dataclass
class UIConfig:
    """Configuration for user interface."""
    
    # Theme and styling
    theme: str = "light"  # "light", "dark", "auto"
    color_scheme: str = "blue"  # "blue", "green", "red", "purple", "orange"
    chart_style: str = "plotly_white"  # "plotly", "plotly_white", "plotly_dark", "ggplot2"
    
    # Layout settings
    sidebar_width: int = 300
    main_content_width: str = "wide"  # "wide", "centered"
    show_tooltips: bool = True
    show_help_text: bool = True
    
    # Chart settings
    default_chart_height: int = 500
    interactive_charts: bool = True
    show_grid: bool = True
    show_legend: bool = True
    
    # Data display
    max_table_rows: int = 1000
    decimal_places: int = 4
    percentage_format: str = "0.2%"
    currency_symbol: str = "$"
    
    # Performance settings
    cache_data: bool = True
    cache_duration: int = 3600  # seconds
    lazy_loading: bool = True
    
    # Export settings
    export_formats: list = None
    default_export_format: str = "csv"
    include_metadata: bool = True
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["csv", "xlsx", "json", "parquet"]


@dataclass
class APIConfig:
    """Configuration for API integrations."""
    
    # API keys (to be set via environment variables)
    alpha_vantage_key: Optional[str] = None
    quandl_key: Optional[str] = None
    fred_key: Optional[str] = None
    news_api_key: Optional[str] = None
    
    # Rate limiting
    requests_per_minute: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Timeout settings
    connection_timeout: int = 10
    read_timeout: int = 30
    
    # Cache settings
    enable_cache: bool = True
    cache_directory: str = "data/cache"
    cache_expiry_hours: int = 24
    
    # Data quality
    validate_data: bool = True
    min_data_quality_score: float = 0.8
    auto_fallback: bool = True


@dataclass
class SystemConfig:
    """Configuration for system-wide settings."""
    
    # Logging
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    log_file: Optional[str] = "logs/portfolio_forecaster.log"
    log_rotation: bool = True
    max_log_size: str = "10MB"
    
    # Performance
    max_workers: int = 4
    memory_limit: str = "4GB"
    enable_gpu: bool = False
    
    # Storage
    data_directory: str = "data"
    models_directory: str = "models"
    exports_directory: str = "exports"
    
    # Security
    enable_auth: bool = False
    session_timeout: int = 3600
    
    # Updates
    check_updates: bool = True
    auto_update: bool = False


class ConfigManager:
    """Advanced configuration manager for the portfolio forecasting system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file
        """
        self.config_path = config_path or "config/portfolio_config.yaml"
        
        # Initialize default configurations
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.risk_config = RiskConfig()
        self.ui_config = UIConfig()
        self.api_config = APIConfig()
        self.system_config = SystemConfig()
        
        # Load configuration if file exists
        self.load_config()
        
        # Load API keys from environment
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from environment variables."""
        self.api_config.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.api_config.quandl_key = os.getenv('QUANDL_API_KEY')
        self.api_config.fred_key = os.getenv('FRED_API_KEY')
        self.api_config.news_api_key = os.getenv('NEWS_API_KEY')
    
    def load_config(self):
        """Load configuration from file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            # Create default configuration
            self.save_config()
            return
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configurations
            if 'model' in config_data:
                self.model_config = ModelConfig(**config_data['model'])
            
            if 'data' in config_data:
                self.data_config = DataConfig(**config_data['data'])
            
            if 'risk' in config_data:
                self.risk_config = RiskConfig(**config_data['risk'])
            
            if 'ui' in config_data:
                self.ui_config = UIConfig(**config_data['ui'])
            
            if 'api' in config_data:
                self.api_config = APIConfig(**config_data['api'])
            
            if 'system' in config_data:
                self.system_config = SystemConfig(**config_data['system'])
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration.")
    
    def save_config(self):
        """Save current configuration to file."""
        config_data = {
            'model': asdict(self.model_config),
            'data': asdict(self.data_config),
            'risk': asdict(self.risk_config),
            'ui': asdict(self.ui_config),
            'api': asdict(self.api_config),
            'system': asdict(self.system_config)
        }
        
        # Create config directory if it doesn't exist
        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() == '.yaml' or config_file.suffix.lower() == '.yml':
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2)
            
            print(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def update_config(self, section: str, **kwargs):
        """
        Update configuration section.
        
        Parameters:
        -----------
        section : str
            Configuration section to update
        **kwargs : dict
            Configuration parameters to update
        """
        if section == 'model':
            for key, value in kwargs.items():
                if hasattr(self.model_config, key):
                    setattr(self.model_config, key, value)
        
        elif section == 'data':
            for key, value in kwargs.items():
                if hasattr(self.data_config, key):
                    setattr(self.data_config, key, value)
        
        elif section == 'risk':
            for key, value in kwargs.items():
                if hasattr(self.risk_config, key):
                    setattr(self.risk_config, key, value)
        
        elif section == 'ui':
            for key, value in kwargs.items():
                if hasattr(self.ui_config, key):
                    setattr(self.ui_config, key, value)
        
        elif section == 'api':
            for key, value in kwargs.items():
                if hasattr(self.api_config, key):
                    setattr(self.api_config, key, value)
        
        elif section == 'system':
            for key, value in kwargs.items():
                if hasattr(self.system_config, key):
                    setattr(self.system_config, key, value)
        
        else:
            raise ValueError(f"Unknown configuration section: {section}")
    
    def get_config(self, section: str) -> Any:
        """
        Get configuration section.
        
        Parameters:
        -----------
        section : str
            Configuration section to retrieve
            
        Returns:
        --------
        Configuration object
        """
        if section == 'model':
            return self.model_config
        elif section == 'data':
            return self.data_config
        elif section == 'risk':
            return self.risk_config
        elif section == 'ui':
            return self.ui_config
        elif section == 'api':
            return self.api_config
        elif section == 'system':
            return self.system_config
        else:
            raise ValueError(f"Unknown configuration section: {section}")
    
    def validate_config(self):
        """
        Validate configuration settings.
        
        Returns:
        --------
        dict : Validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Validate model configuration
        if self.model_config.confidence_interval <= 0 or self.model_config.confidence_interval >= 1:
            validation_results['errors'].append("Confidence interval must be between 0 and 1")
            validation_results['valid'] = False
        
        if self.model_config.default_periods <= 0:
            validation_results['errors'].append("Default periods must be positive")
            validation_results['valid'] = False
        
        # Validate data configuration
        if self.data_config.min_data_points <= 0:
            validation_results['errors'].append("Minimum data points must be positive")
            validation_results['valid'] = False
        
        if self.data_config.min_data_points >= self.data_config.max_data_points:
            validation_results['errors'].append("Minimum data points must be less than maximum")
            validation_results['valid'] = False
        
        # Validate risk configuration
        for level in self.risk_config.var_confidence_levels:
            if level <= 0 or level >= 1:
                validation_results['errors'].append(f"VaR confidence level {level} must be between 0 and 1")
                validation_results['valid'] = False
        
        # Validate API configuration
        if self.api_config.requests_per_minute <= 0:
            validation_results['errors'].append("Requests per minute must be positive")
            validation_results['valid'] = False
        
        # Check for missing API keys (warnings only)
        api_keys = [
            ('Alpha Vantage', self.api_config.alpha_vantage_key),
            ('Quandl', self.api_config.quandl_key),
            ('FRED', self.api_config.fred_key),
            ('News API', self.api_config.news_api_key)
        ]
        
        for name, key in api_keys:
            if not key:
                validation_results['warnings'].append(f"{name} API key not configured")
        
        return validation_results
    
    def reset_to_defaults(self, section: Optional[str] = None):
        """
        Reset configuration to defaults.
        
        Parameters:
        -----------
        section : str, optional
            Specific section to reset, or None for all sections
        """
        if section is None or section == 'model':
            self.model_config = ModelConfig()
        
        if section is None or section == 'data':
            self.data_config = DataConfig()
        
        if section is None or section == 'risk':
            self.risk_config = RiskConfig()
        
        if section is None or section == 'ui':
            self.ui_config = UIConfig()
        
        if section is None or section == 'api':
            self.api_config = APIConfig()
            self._load_api_keys()  # Reload API keys from environment
        
        if section is None or section == 'system':
            self.system_config = SystemConfig()
    
    def get_summary(self):
        """
        Get configuration summary.
        
        Returns:
        --------
        dict : Configuration summary
        """
        return {
            'config_file': self.config_path,
            'sections': {
                'model': len(asdict(self.model_config)),
                'data': len(asdict(self.data_config)),
                'risk': len(asdict(self.risk_config)),
                'ui': len(asdict(self.ui_config)),
                'api': len(asdict(self.api_config)),
                'system': len(asdict(self.system_config))
            },
            'validation': self.validate_config(),
            'api_keys_configured': sum([
                bool(self.api_config.alpha_vantage_key),
                bool(self.api_config.quandl_key),
                bool(self.api_config.fred_key),
                bool(self.api_config.news_api_key)
            ])
        }


# Global configuration manager instance
config_manager = ConfigManager()