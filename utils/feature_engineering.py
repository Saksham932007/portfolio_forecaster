"""Feature engineering utilities for time series forecasting."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union


def create_lag_features(data: pd.Series, lags: List[int], prefix: str = 'lag') -> pd.DataFrame:
    """
    Create lag features for a time series.
    
    Args:
        data (pd.Series): Time series data
        lags (List[int]): List of lag periods to create
        prefix (str): Prefix for the lag column names
        
    Returns:
        pd.DataFrame: DataFrame with original data and lag features
    """
    result_df = pd.DataFrame(index=data.index)
    result_df['original'] = data
    
    for lag in lags:
        if lag <= 0:
            raise ValueError("Lag periods must be positive integers")
        result_df[f'{prefix}_{lag}'] = data.shift(lag)
    
    return result_df


def create_moving_averages(data: pd.Series, windows: List[int], prefix: str = 'ma') -> pd.DataFrame:
    """
    Create moving average features for a time series.
    
    Args:
        data (pd.Series): Time series data
        windows (List[int]): List of window sizes for moving averages
        prefix (str): Prefix for the moving average column names
        
    Returns:
        pd.DataFrame: DataFrame with original data and moving average features
    """
    result_df = pd.DataFrame(index=data.index)
    result_df['original'] = data
    
    for window in windows:
        if window <= 0:
            raise ValueError("Window sizes must be positive integers")
        result_df[f'{prefix}_{window}'] = data.rolling(window=window).mean()
    
    return result_df


def create_exponential_moving_averages(data: pd.Series, spans: List[int], prefix: str = 'ema') -> pd.DataFrame:
    """
    Create exponential moving average features for a time series.
    
    Args:
        data (pd.Series): Time series data
        spans (List[int]): List of span periods for exponential moving averages
        prefix (str): Prefix for the EMA column names
        
    Returns:
        pd.DataFrame: DataFrame with original data and EMA features
    """
    result_df = pd.DataFrame(index=data.index)
    result_df['original'] = data
    
    for span in spans:
        if span <= 0:
            raise ValueError("Span periods must be positive integers")
        result_df[f'{prefix}_{span}'] = data.ewm(span=span).mean()
    
    return result_df


def create_technical_indicators(data: pd.DataFrame, ticker_column: str = None) -> pd.DataFrame:
    """
    Create technical indicators from OHLCV data.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
        ticker_column (str): If specified, creates indicators for this ticker only
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators
    """
    if ticker_column:
        # If working with multi-indexed data
        try:
            ticker_data = data.xs(ticker_column, level='ticker')
        except (KeyError, AttributeError):
            # If it's already single ticker data
            ticker_data = data
    else:
        ticker_data = data
    
    result_df = ticker_data.copy()
    
    # Price-based indicators
    if all(col in ticker_data.columns for col in ['high', 'low', 'close']):
        # True Range
        result_df['true_range'] = np.maximum(
            ticker_data['high'] - ticker_data['low'],
            np.maximum(
                abs(ticker_data['high'] - ticker_data['close'].shift(1)),
                abs(ticker_data['low'] - ticker_data['close'].shift(1))
            )
        )
        
        # Average True Range (14 periods)
        result_df['atr_14'] = result_df['true_range'].rolling(window=14).mean()
        
        # Price momentum (rate of change)
        result_df['price_momentum_5'] = ticker_data['close'].pct_change(5)
        result_df['price_momentum_10'] = ticker_data['close'].pct_change(10)
    
    # Volume-based indicators
    if 'volume' in ticker_data.columns:
        # Volume moving averages
        result_df['volume_ma_10'] = ticker_data['volume'].rolling(window=10).mean()
        result_df['volume_ma_20'] = ticker_data['volume'].rolling(window=20).mean()
        
        # Volume ratio
        result_df['volume_ratio'] = ticker_data['volume'] / result_df['volume_ma_20']
    
    # Volatility indicators
    if 'close' in ticker_data.columns:
        # Rolling standard deviation (volatility)
        result_df['volatility_10'] = ticker_data['close'].rolling(window=10).std()
        result_df['volatility_20'] = ticker_data['close'].rolling(window=20).std()
        
        # Daily returns
        result_df['returns'] = ticker_data['close'].pct_change()
        
        # Rolling Sharpe ratio (simplified, assuming risk-free rate = 0)
        result_df['sharpe_20'] = (
            result_df['returns'].rolling(window=20).mean() / 
            result_df['returns'].rolling(window=20).std() * np.sqrt(252)
        )
    
    return result_df


def create_cyclical_features(data: pd.Series) -> pd.DataFrame:
    """
    Create cyclical features from datetime index.
    
    Args:
        data (pd.Series): Time series data with datetime index
        
    Returns:
        pd.DataFrame: DataFrame with cyclical features
    """
    result_df = pd.DataFrame(index=data.index)
    result_df['original'] = data
    
    # Extract datetime components
    dates = data.index
    
    # Day of week (0=Monday, 6=Sunday)
    result_df['day_of_week'] = dates.dayofweek
    result_df['day_of_week_sin'] = np.sin(2 * np.pi * dates.dayofweek / 7)
    result_df['day_of_week_cos'] = np.cos(2 * np.pi * dates.dayofweek / 7)
    
    # Month
    result_df['month'] = dates.month
    result_df['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
    result_df['month_cos'] = np.cos(2 * np.pi * dates.month / 12)
    
    # Quarter
    result_df['quarter'] = dates.quarter
    result_df['quarter_sin'] = np.sin(2 * np.pi * dates.quarter / 4)
    result_df['quarter_cos'] = np.cos(2 * np.pi * dates.quarter / 4)
    
    # Day of year
    result_df['day_of_year'] = dates.dayofyear
    result_df['day_of_year_sin'] = np.sin(2 * np.pi * dates.dayofyear / 365.25)
    result_df['day_of_year_cos'] = np.cos(2 * np.pi * dates.dayofyear / 365.25)
    
    # Weekend indicator
    result_df['is_weekend'] = (dates.dayofweek >= 5).astype(int)
    
    return result_df


def create_comprehensive_features(data: pd.Series, 
                                lag_periods: List[int] = [1, 2, 3, 5, 10],
                                ma_windows: List[int] = [5, 10, 20],
                                ema_spans: List[int] = [5, 10, 20],
                                include_cyclical: bool = True) -> pd.DataFrame:
    """
    Create a comprehensive set of features for time series forecasting.
    
    Args:
        data (pd.Series): Time series data
        lag_periods (List[int]): Lag periods to create
        ma_windows (List[int]): Moving average windows
        ema_spans (List[int]): Exponential moving average spans
        include_cyclical (bool): Whether to include cyclical features
        
    Returns:
        pd.DataFrame: DataFrame with comprehensive features
    """
    # Start with original data
    result_df = pd.DataFrame(index=data.index)
    result_df['target'] = data
    
    # Add lag features
    lag_features = create_lag_features(data, lag_periods)
    for col in lag_features.columns:
        if col != 'original':
            result_df[col] = lag_features[col]
    
    # Add moving average features
    ma_features = create_moving_averages(data, ma_windows)
    for col in ma_features.columns:
        if col != 'original':
            result_df[col] = ma_features[col]
    
    # Add exponential moving average features
    ema_features = create_exponential_moving_averages(data, ema_spans)
    for col in ema_features.columns:
        if col != 'original':
            result_df[col] = ema_features[col]
    
    # Add basic statistical features
    result_df['returns'] = data.pct_change()
    result_df['log_returns'] = np.log(data / data.shift(1))
    result_df['volatility_5'] = data.rolling(window=5).std()
    result_df['volatility_10'] = data.rolling(window=10).std()
    
    # Add trend features
    result_df['trend_5'] = data.rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    result_df['trend_10'] = data.rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Add cyclical features if requested
    if include_cyclical:
        cyclical_features = create_cyclical_features(data)
        for col in cyclical_features.columns:
            if col != 'original':
                result_df[col] = cyclical_features[col]
    
    return result_df


# Example usage and testing functions
if __name__ == "__main__":
    from data_loader import load_data, get_ticker_data
    import os
    
    # Get the path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'mock_stock_data.csv')
    
    try:
        print("Testing feature engineering functions...")
        
        # Load data
        data = load_data(data_path)
        aapl_close = get_ticker_data(data, 'AAPL', 'close')
        
        # Test lag features
        print("\n1. Testing lag features...")
        lag_features = create_lag_features(aapl_close, [1, 2, 5])
        print(f"Lag features shape: {lag_features.shape}")
        print(f"Columns: {lag_features.columns.tolist()}")
        
        # Test moving averages
        print("\n2. Testing moving averages...")
        ma_features = create_moving_averages(aapl_close, [5, 10, 20])
        print(f"MA features shape: {ma_features.shape}")
        print(f"Columns: {ma_features.columns.tolist()}")
        
        # Test exponential moving averages
        print("\n3. Testing exponential moving averages...")
        ema_features = create_exponential_moving_averages(aapl_close, [5, 10, 20])
        print(f"EMA features shape: {ema_features.shape}")
        print(f"Columns: {ema_features.columns.tolist()}")
        
        # Test cyclical features
        print("\n4. Testing cyclical features...")
        cyclical_features = create_cyclical_features(aapl_close)
        print(f"Cyclical features shape: {cyclical_features.shape}")
        print(f"Columns: {cyclical_features.columns.tolist()}")
        
        # Test comprehensive features
        print("\n5. Testing comprehensive features...")
        comprehensive_features = create_comprehensive_features(aapl_close)
        print(f"Comprehensive features shape: {comprehensive_features.shape}")
        print(f"Number of features: {len(comprehensive_features.columns)}")
        print(f"Sample columns: {comprehensive_features.columns.tolist()[:10]}...")
        
        # Show some statistics
        print(f"\n6. Feature statistics (first 50 rows to avoid NaN from lags)...")
        clean_data = comprehensive_features.iloc[50:].dropna()
        print(f"Clean data shape: {clean_data.shape}")
        print(f"Non-null values per column: {clean_data.notna().sum().min()} to {clean_data.notna().sum().max()}")
        
        print("\nAll feature engineering tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()