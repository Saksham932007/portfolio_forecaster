"""Preprocessing utilities for time series forecasting."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Optional, Dict, Any, Union
import warnings


def check_stationarity(series: pd.Series, significance_level: float = 0.05) -> Dict[str, Any]:
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.
    
    Args:
        series (pd.Series): Time series to test
        significance_level (float): Significance level for the test
        
    Returns:
        Dict: Test results including statistic, p-value, and conclusion
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        
        # Remove any NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return {
                'is_stationary': False,
                'test_statistic': None,
                'p_value': None,
                'critical_values': None,
                'conclusion': 'Insufficient data for stationarity test'
            }
        
        # Perform Augmented Dickey-Fuller test
        result = adfuller(clean_series, autolag='AIC')
        
        is_stationary = result[1] <= significance_level
        
        return {
            'is_stationary': is_stationary,
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'conclusion': 'Stationary' if is_stationary else 'Non-stationary'
        }
        
    except ImportError:
        warnings.warn("statsmodels not available, assuming non-stationary")
        return {
            'is_stationary': False,
            'test_statistic': None,
            'p_value': None,
            'critical_values': None,
            'conclusion': 'Unable to test (statsmodels not available)'
        }


def make_stationary(series: pd.Series, method: str = 'difference', 
                   difference_order: int = 1, log_transform: bool = False) -> Tuple[pd.Series, Dict]:
    """
    Make a time series stationary using differencing or other methods.
    
    Args:
        series (pd.Series): Time series to make stationary
        method (str): Method to use ('difference', 'log_difference', 'pct_change')
        difference_order (int): Order of differencing (for 'difference' method)
        log_transform (bool): Whether to apply log transform before differencing
        
    Returns:
        Tuple: (stationary_series, transformation_info)
    """
    original_series = series.copy()
    
    # Store transformation info for inverse transformation
    transform_info = {
        'method': method,
        'difference_order': difference_order,
        'log_transform': log_transform,
        'original_values': series.iloc[:difference_order].values if method == 'difference' else None,
        'original_index': series.index
    }
    
    # Apply log transform if requested
    if log_transform:
        if (series <= 0).any():
            warnings.warn("Series contains non-positive values, adding constant before log transform")
            series = series + abs(series.min()) + 1
        series = np.log(series)
        transform_info['log_constant'] = abs(original_series.min()) + 1 if (original_series <= 0).any() else 0
    
    # Apply the specified method
    if method == 'difference':
        for _ in range(difference_order):
            series = series.diff()
        stationary_series = series.dropna()
        
    elif method == 'log_difference':
        if not log_transform:
            if (series <= 0).any():
                warnings.warn("Series contains non-positive values, adding constant before log transform")
                series = series + abs(series.min()) + 1
                transform_info['log_constant'] = abs(original_series.min()) + 1
            series = np.log(series)
        stationary_series = series.diff().dropna()
        
    elif method == 'pct_change':
        stationary_series = series.pct_change().dropna()
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return stationary_series, transform_info


def invert_stationarity(stationary_series: pd.Series, transform_info: Dict, 
                       original_series: Optional[pd.Series] = None) -> pd.Series:
    """
    Invert the stationarity transformation to get back original scale values.
    
    Args:
        stationary_series (pd.Series): Stationary series to invert
        transform_info (Dict): Transformation information from make_stationary
        original_series (pd.Series, optional): Original series for reference
        
    Returns:
        pd.Series: Series in original scale
    """
    method = transform_info['method']
    result = stationary_series.copy()
    
    if method == 'difference':
        # Integrate the differences
        if original_series is not None and transform_info['original_values'] is not None:
            # Use the last known values from original series
            start_values = original_series.iloc[:transform_info['difference_order']].values
        elif transform_info['original_values'] is not None:
            start_values = transform_info['original_values']
        else:
            # Fallback: assume starting values are 0
            warnings.warn("No original values available for inversion, using zeros")
            start_values = np.zeros(transform_info['difference_order'])
        
        # Cumulative sum starting from the original values
        result = result.cumsum()
        for i, start_val in enumerate(reversed(start_values)):
            result = result + start_val
            
    elif method == 'log_difference':
        # First integrate the log differences
        if original_series is not None:
            last_log_value = np.log(original_series.iloc[-len(result)-1:].iloc[0])
        else:
            last_log_value = 0
            
        result = result.cumsum() + last_log_value
        
        # Then exponentiate
        result = np.exp(result)
        
        # Subtract log constant if it was added
        if 'log_constant' in transform_info and transform_info['log_constant'] > 0:
            result = result - transform_info['log_constant']
            
    elif method == 'pct_change':
        # Convert percentage changes back to levels
        if original_series is not None:
            start_value = original_series.iloc[-len(result)-1:].iloc[0]
        else:
            start_value = 1
            
        result = (1 + result).cumprod() * start_value
    
    # Apply inverse log transform if it was applied
    if transform_info['log_transform'] and method == 'difference':
        result = np.exp(result)
        if 'log_constant' in transform_info and transform_info['log_constant'] > 0:
            result = result - transform_info['log_constant']
    
    return result


class TimeSeriesScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler for time series data that preserves temporal structure.
    """
    
    def __init__(self, scaler_type: str = 'standard', feature_range: Tuple[float, float] = (0, 1)):
        """
        Initialize the TimeSeriesScaler.
        
        Args:
            scaler_type (str): Type of scaler ('standard', 'minmax', 'robust')
            feature_range (Tuple): Range for MinMaxScaler
        """
        self.scaler_type = scaler_type
        self.feature_range = feature_range
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, X: Union[pd.Series, pd.DataFrame, np.ndarray], y=None):
        """Fit the scaler to the data."""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=self.feature_range)
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # Convert to numpy array for fitting
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X_array = X.values.reshape(-1, 1) if X.ndim == 1 else X.values
        else:
            X_array = X.reshape(-1, 1) if X.ndim == 1 else X
            
        self.scaler.fit(X_array)
        self.is_fitted = True
        return self
    
    def transform(self, X: Union[pd.Series, pd.DataFrame, np.ndarray]) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """Transform the data using the fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")
        
        # Store original type and index
        original_type = type(X)
        original_index = getattr(X, 'index', None)
        original_columns = getattr(X, 'columns', None)
        
        # Convert to numpy array
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X_array = X.values.reshape(-1, 1) if X.ndim == 1 else X.values
        else:
            X_array = X.reshape(-1, 1) if X.ndim == 1 else X
        
        # Transform
        X_scaled = self.scaler.transform(X_array)
        
        # Convert back to original type
        if original_type == pd.Series:
            return pd.Series(X_scaled.flatten(), index=original_index)
        elif original_type == pd.DataFrame:
            return pd.DataFrame(X_scaled, index=original_index, columns=original_columns)
        else:
            return X_scaled.reshape(X.shape)
    
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame, np.ndarray]) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """Inverse transform the scaled data."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")
        
        # Store original type and index
        original_type = type(X)
        original_index = getattr(X, 'index', None)
        original_columns = getattr(X, 'columns', None)
        
        # Convert to numpy array
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X_array = X.values.reshape(-1, 1) if X.ndim == 1 else X.values
        else:
            X_array = X.reshape(-1, 1) if X.ndim == 1 else X
        
        # Inverse transform
        X_original = self.scaler.inverse_transform(X_array)
        
        # Convert back to original type
        if original_type == pd.Series:
            return pd.Series(X_original.flatten(), index=original_index)
        elif original_type == pd.DataFrame:
            return pd.DataFrame(X_original, index=original_index, columns=original_columns)
        else:
            return X_original.reshape(X.shape)


def get_scalers(data: Union[pd.DataFrame, pd.Series], scaler_type: str = 'standard') -> Dict[str, TimeSeriesScaler]:
    """
    Fit scalers for each column in the data.
    
    Args:
        data (Union[pd.DataFrame, pd.Series]): Data to fit scalers on
        scaler_type (str): Type of scaler to use
        
    Returns:
        Dict: Dictionary of fitted scalers for each column
    """
    scalers = {}
    
    if isinstance(data, pd.Series):
        scaler = TimeSeriesScaler(scaler_type=scaler_type)
        scaler.fit(data)
        scalers['data'] = scaler
    else:
        for column in data.columns:
            scaler = TimeSeriesScaler(scaler_type=scaler_type)
            scaler.fit(data[column])
            scalers[column] = scaler
    
    return scalers


def scale_data(data: Union[pd.DataFrame, pd.Series], scalers: Dict[str, TimeSeriesScaler]) -> Union[pd.DataFrame, pd.Series]:
    """
    Scale data using fitted scalers.
    
    Args:
        data (Union[pd.DataFrame, pd.Series]): Data to scale
        scalers (Dict): Dictionary of fitted scalers
        
    Returns:
        Union[pd.DataFrame, pd.Series]: Scaled data
    """
    if isinstance(data, pd.Series):
        if 'data' in scalers:
            return scalers['data'].transform(data)
        else:
            raise ValueError("No scaler found for series data")
    else:
        scaled_data = data.copy()
        for column in data.columns:
            if column in scalers:
                scaled_data[column] = scalers[column].transform(data[column])
            else:
                warnings.warn(f"No scaler found for column {column}, leaving unscaled")
        return scaled_data


def invert_scale(data: Union[pd.DataFrame, pd.Series], scalers: Dict[str, TimeSeriesScaler]) -> Union[pd.DataFrame, pd.Series]:
    """
    Inverse transform scaled data using fitted scalers.
    
    Args:
        data (Union[pd.DataFrame, pd.Series]): Scaled data to inverse transform
        scalers (Dict): Dictionary of fitted scalers
        
    Returns:
        Union[pd.DataFrame, pd.Series]: Data in original scale
    """
    if isinstance(data, pd.Series):
        if 'data' in scalers:
            return scalers['data'].inverse_transform(data)
        else:
            raise ValueError("No scaler found for series data")
    else:
        original_data = data.copy()
        for column in data.columns:
            if column in scalers:
                original_data[column] = scalers[column].inverse_transform(data[column])
            else:
                warnings.warn(f"No scaler found for column {column}, leaving as-is")
        return original_data


# Example usage and testing functions
if __name__ == "__main__":
    from data_loader import load_data, get_ticker_data
    import os
    
    # Get the path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'mock_stock_data.csv')
    
    try:
        print("Testing preprocessing functions...")
        
        # Load data
        data = load_data(data_path)
        aapl_close = get_ticker_data(data, 'AAPL', 'close')
        
        # Test stationarity check
        print("\n1. Testing stationarity check...")
        stationarity_result = check_stationarity(aapl_close)
        print(f"Is stationary: {stationarity_result['is_stationary']}")
        print(f"P-value: {stationarity_result['p_value']}")
        print(f"Conclusion: {stationarity_result['conclusion']}")
        
        # Test make stationary
        print("\n2. Testing make stationary...")
        stationary_series, transform_info = make_stationary(aapl_close, method='difference')
        print(f"Original series length: {len(aapl_close)}")
        print(f"Stationary series length: {len(stationary_series)}")
        print(f"Transform method: {transform_info['method']}")
        
        # Test stationarity of differenced series
        diff_stationarity = check_stationarity(stationary_series)
        print(f"Differenced series is stationary: {diff_stationarity['is_stationary']}")
        
        # Test invert stationarity
        print("\n3. Testing invert stationarity...")
        reconstructed = invert_stationarity(stationary_series, transform_info, aapl_close)
        print(f"Reconstructed series length: {len(reconstructed)}")
        original_subset = aapl_close.iloc[1:]  # Skip first value due to differencing
        mse = np.mean((reconstructed.values - original_subset.values) ** 2)
        print(f"Reconstruction MSE: {mse:.6f}")
        
        # Test scaling
        print("\n4. Testing scaling...")
        scaler = TimeSeriesScaler(scaler_type='standard')
        scaler.fit(aapl_close)
        scaled_data = scaler.transform(aapl_close)
        print(f"Original data range: {aapl_close.min():.2f} to {aapl_close.max():.2f}")
        print(f"Scaled data range: {scaled_data.min():.2f} to {scaled_data.max():.2f}")
        print(f"Scaled data mean: {scaled_data.mean():.6f}")
        print(f"Scaled data std: {scaled_data.std():.6f}")
        
        # Test inverse scaling
        reconstructed_scale = scaler.inverse_transform(scaled_data)
        scale_mse = np.mean((reconstructed_scale.values - aapl_close.values) ** 2)
        print(f"Scale reconstruction MSE: {scale_mse:.10f}")
        
        # Test get_scalers and scale_data functions
        print("\n5. Testing scaler utilities...")
        multi_data = data.xs('AAPL', level='ticker')[['open', 'high', 'low', 'close']]
        scalers = get_scalers(multi_data, scaler_type='minmax')
        scaled_multi = scale_data(multi_data, scalers)
        print(f"Multi-column data shape: {multi_data.shape}")
        print(f"Scaled multi-column range: {scaled_multi.min().min():.2f} to {scaled_multi.max().max():.2f}")
        
        # Test inverse scaling for multi-column
        reconstructed_multi = invert_scale(scaled_multi, scalers)
        multi_mse = np.mean((reconstructed_multi.values - multi_data.values) ** 2)
        print(f"Multi-column reconstruction MSE: {multi_mse:.10f}")
        
        print("\nAll preprocessing tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()