"""Vector Autoregression (VAR) model for multivariate time series forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings


class VARModel:
    """
    Vector Autoregression model for multivariate time series forecasting.
    
    This class implements VAR modeling with automatic data preparation,
    stationarity handling, and forecasting capabilities.
    """
    
    def __init__(self, maxlags: int = 10, trend: str = 'c'):
        """
        Initialize VAR model.
        
        Args:
            maxlags (int): Maximum number of lags to consider
            trend (str): Trend specification ('n', 'c', 'ct', 'ctt')
                'n': no constant, no trend
                'c': constant only (default)
                'ct': constant and trend
                'ctt': constant, linear and quadratic trend
        """
        self.maxlags = maxlags
        self.trend = trend
        self.model = None
        self.fitted_model = None
        self.data_info = None
        self.transformation_info = {}
        self.is_fitted = False
        
    def _check_stationarity(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Check stationarity for each series in the DataFrame.
        
        Args:
            data (pd.DataFrame): Multivariate time series data
            
        Returns:
            Dict: Stationarity status for each column
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            warnings.warn("statsmodels not available, assuming stationarity")
            return {col: True for col in data.columns}
        
        stationarity_results = {}
        for column in data.columns:
            try:
                series = data[column].dropna()
                if len(series) < 10:
                    stationarity_results[column] = False
                    continue
                    
                result = adfuller(series, autolag='AIC')
                # Series is stationary if p-value < 0.05
                stationarity_results[column] = result[1] <= 0.05
            except Exception as e:
                warnings.warn(f"Stationarity test failed for {column}: {e}")
                stationarity_results[column] = False
                
        return stationarity_results
    
    def _make_stationary(self, data: pd.DataFrame, method: str = 'difference') -> Tuple[pd.DataFrame, Dict]:
        """
        Make the multivariate series stationary.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): Method to use ('difference', 'log_difference')
            
        Returns:
            Tuple: (stationary_data, transformation_info)
        """
        stationary_data = data.copy()
        transform_info = {}
        
        for column in data.columns:
            series = data[column]
            
            if method == 'difference':
                # Store the last value for inverse transformation
                transform_info[column] = {
                    'method': method,
                    'last_value': series.iloc[-1],
                    'original_values': series.iloc[-2:].values  # Last 2 values for safety
                }
                stationary_data[column] = series.diff()
                
            elif method == 'log_difference':
                # Handle non-positive values
                if (series <= 0).any():
                    min_val = abs(series.min()) + 1
                    series = series + min_val
                    transform_info[column] = {
                        'method': method,
                        'log_constant': min_val,
                        'last_value': series.iloc[-1]
                    }
                else:
                    transform_info[column] = {
                        'method': method,
                        'log_constant': 0,
                        'last_value': series.iloc[-1]
                    }
                
                log_series = np.log(series)
                stationary_data[column] = log_series.diff()
            else:
                raise ValueError(f"Unknown method: {method}")
        
        # Drop the first row (NaN due to differencing)
        stationary_data = stationary_data.dropna()
        
        return stationary_data, transform_info
    
    def _prepare_data(self, data: Union[pd.DataFrame, Dict[str, pd.Series]], 
                     make_stationary: bool = True) -> pd.DataFrame:
        """
        Prepare data for VAR modeling.
        
        Args:
            data: Either DataFrame or dict of series
            make_stationary: Whether to make series stationary
            
        Returns:
            pd.DataFrame: Prepared data for VAR modeling
        """
        # Convert dict to DataFrame if necessary
        if isinstance(data, dict):
            # Align all series to common index
            common_index = None
            for series in data.values():
                if common_index is None:
                    common_index = series.index
                else:
                    common_index = common_index.intersection(series.index)
            
            aligned_data = {}
            for name, series in data.items():
                aligned_data[name] = series.reindex(common_index)
            
            prepared_data = pd.DataFrame(aligned_data)
        else:
            prepared_data = data.copy()
        
        # Remove any rows with NaN values
        prepared_data = prepared_data.dropna()
        
        if len(prepared_data) < self.maxlags + 10:
            raise ValueError(f"Insufficient data: need at least {self.maxlags + 10} observations")
        
        # Store original data info
        self.data_info = {
            'columns': list(prepared_data.columns),
            'original_shape': prepared_data.shape,
            'date_range': (prepared_data.index[0], prepared_data.index[-1])
        }
        
        # Check and handle stationarity
        if make_stationary:
            stationarity_status = self._check_stationarity(prepared_data)
            non_stationary_cols = [col for col, is_stat in stationarity_status.items() if not is_stat]
            
            if non_stationary_cols:
                if len(non_stationary_cols) == len(prepared_data.columns):
                    # All series are non-stationary, apply transformation to all
                    prepared_data, self.transformation_info = self._make_stationary(
                        prepared_data, method='difference'
                    )
                else:
                    # Mixed stationarity, apply transformation only to non-stationary series
                    transform_data = prepared_data[non_stationary_cols]
                    stationary_transform, transform_info = self._make_stationary(
                        transform_data, method='difference'
                    )
                    
                    # Combine transformed and already stationary data
                    prepared_data = prepared_data.drop(columns=non_stationary_cols)
                    prepared_data = pd.concat([prepared_data, stationary_transform], axis=1)
                    prepared_data = prepared_data.dropna()
                    
                    self.transformation_info = transform_info
        
        return prepared_data
    
    def fit(self, data: Union[pd.DataFrame, Dict[str, pd.Series]], 
           maxlags: Optional[int] = None, ic: str = 'aic') -> 'VARModel':
        """
        Fit the VAR model to the data.
        
        Args:
            data: Multivariate time series data
            maxlags: Maximum lags to consider (overrides instance maxlags)
            ic: Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
            
        Returns:
            self: Fitted VARModel instance
        """
        try:
            from statsmodels.tsa.api import VAR
        except ImportError:
            raise ImportError("statsmodels is required for VAR modeling")
        
        # Use provided maxlags or instance maxlags
        if maxlags is not None:
            self.maxlags = maxlags
        
        # Prepare data
        prepared_data = self._prepare_data(data, make_stationary=True)
        
        # Initialize VAR model
        self.model = VAR(prepared_data)
        
        # Select optimal lag length
        try:
            lag_order_results = self.model.select_order(maxlags=self.maxlags)
            
            if ic in lag_order_results.selected_orders:
                optimal_lags = lag_order_results.selected_orders[ic]
            else:
                # Fallback to AIC if requested IC not available
                optimal_lags = lag_order_results.selected_orders.get('aic', 1)
            
            # Ensure minimum of 1 lag
            self.optimal_lags = max(optimal_lags, 1)
            
        except Exception as e:
            warnings.warn(f"Lag selection failed: {e}, using 2 lags")
            self.optimal_lags = 2
        
        # Fit the model
        try:
            self.fitted_model = self.model.fit(self.optimal_lags, trend=self.trend)
            self.is_fitted = True
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"VAR model fitting failed: {e}")
    
    def predict(self, steps: int, alpha: float = 0.05) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Generate forecasts from the fitted VAR model.
        
        Args:
            steps: Number of steps to forecast
            alpha: Significance level for confidence intervals
            
        Returns:
            Tuple: (forecasts_df, confidence_intervals_df)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Generate forecasts
            forecast_result = self.fitted_model.forecast(
                self.fitted_model.endog, steps=steps
            )
            
            # Create forecast index
            if hasattr(self.fitted_model, 'data') and hasattr(self.fitted_model.data, 'dates') and self.fitted_model.data.dates is not None:
                last_date = self.fitted_model.data.dates[-1]
                forecast_index = pd.date_range(
                    start=last_date + pd.DateOffset(days=1),
                    periods=steps,
                    freq='D'
                )
            else:
                # Use data_info to create proper index
                last_date = self.data_info['date_range'][1]
                if isinstance(last_date, pd.Timestamp):
                    forecast_index = pd.date_range(
                        start=last_date + pd.DateOffset(days=1),
                        periods=steps,
                        freq='D'
                    )
                else:
                    # Fallback to numeric index
                    forecast_index = range(len(self.fitted_model.endog), 
                                         len(self.fitted_model.endog) + steps)
            
            # Create forecast DataFrame
            forecasts_df = pd.DataFrame(
                forecast_result,
                index=forecast_index,
                columns=self.data_info['columns']
            )
            
            # Generate confidence intervals if possible
            confidence_intervals_df = None
            try:
                # This is a simplified confidence interval calculation
                # In practice, you might want to use bootstrap or analytical methods
                residuals = self.fitted_model.resid
                residual_std = np.std(residuals, axis=0)
                
                # Approximate confidence intervals using residual standard deviation
                z_score = 1.96  # For 95% confidence interval
                margin = z_score * residual_std
                
                ci_data = []
                for i, (_, row) in enumerate(forecasts_df.iterrows()):
                    ci_row = {}
                    for j, col in enumerate(forecasts_df.columns):
                        ci_row[f'{col}_lower'] = row[col] - margin[j] * np.sqrt(i + 1)
                        ci_row[f'{col}_upper'] = row[col] + margin[j] * np.sqrt(i + 1)
                    ci_data.append(ci_row)
                
                confidence_intervals_df = pd.DataFrame(ci_data, index=forecast_index)
                
            except Exception as e:
                warnings.warn(f"Could not generate confidence intervals: {e}")
            
            # Invert transformations if applied
            if self.transformation_info:
                forecasts_df = self._invert_transformation(forecasts_df)
                # Note: CI inversion is more complex and skipped for now
            
            return forecasts_df, confidence_intervals_df
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def _invert_transformation(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Invert the transformations applied during data preparation.
        
        Args:
            forecasts: Transformed forecasts
            
        Returns:
            pd.DataFrame: Forecasts in original scale
        """
        inverted_forecasts = forecasts.copy()
        
        for column in forecasts.columns:
            if column in self.transformation_info:
                transform_info = self.transformation_info[column]
                method = transform_info['method']
                
                if method == 'difference':
                    # Integrate the differences
                    last_value = transform_info['last_value']
                    inverted_forecasts[column] = forecasts[column].cumsum() + last_value
                    
                elif method == 'log_difference':
                    # Integrate and then exponentiate
                    last_log_value = np.log(transform_info['last_value'])
                    log_forecasts = forecasts[column].cumsum() + last_log_value
                    inverted_forecasts[column] = np.exp(log_forecasts)
                    
                    # Subtract log constant if it was added
                    if transform_info.get('log_constant', 0) > 0:
                        inverted_forecasts[column] -= transform_info['log_constant']
        
        return inverted_forecasts
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the fitted VAR model.
        
        Returns:
            str: Model summary
        """
        if not self.is_fitted:
            return "Model not fitted"
        
        summary = f"VAR Model Summary\n"
        summary += f"================\n"
        summary += f"Variables: {', '.join(self.data_info['columns'])}\n"
        summary += f"Optimal lags: {self.optimal_lags}\n"
        summary += f"Trend: {self.trend}\n"
        summary += f"Sample size: {len(self.fitted_model.endog)}\n"
        summary += f"Date range: {self.data_info['date_range'][0]} to {self.data_info['date_range'][1]}\n"
        
        if self.transformation_info:
            summary += f"Transformations applied: {list(self.transformation_info.keys())}\n"
        
        # Add model diagnostics if available
        try:
            aic = self.fitted_model.aic
            bic = self.fitted_model.bic
            summary += f"AIC: {aic:.4f}\n"
            summary += f"BIC: {bic:.4f}\n"
        except:
            pass
        
        return summary


# Example usage and testing functions
if __name__ == "__main__":
    import os
    import sys
    
    # Add utils to path for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    
    from data_loader import load_data, get_multiple_tickers_data
    from evaluation import calculate_all_metrics
    
    # Get the path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'mock_stock_data.csv')
    
    try:
        print("Testing VAR Model...")
        
        # Load data
        data = load_data(data_path)
        
        # Get multiple tickers data for VAR modeling
        tickers = ['AAPL', 'MSFT']
        multi_data = get_multiple_tickers_data(data, tickers, 'close')
        
        print(f"Loaded data shape: {multi_data.shape}")
        print(f"Date range: {multi_data.index[0]} to {multi_data.index[-1]}")
        print(f"Variables: {multi_data.columns.tolist()}")
        
        # Split data for testing
        train_size = int(len(multi_data) * 0.8)
        train_data = multi_data.iloc[:train_size]
        test_data = multi_data.iloc[train_size:]
        
        print(f"\nTrain size: {len(train_data)}")
        print(f"Test size: {len(test_data)}")
        
        # Test VAR model
        print("\n1. Fitting VAR model...")
        var_model = VARModel(maxlags=5)
        var_model.fit(train_data)
        
        print(f"Model fitted successfully!")
        print(f"Optimal lags: {var_model.optimal_lags}")
        
        # Generate forecasts
        print("\n2. Generating forecasts...")
        forecasts, confidence_intervals = var_model.predict(steps=len(test_data))
        
        print(f"Forecasts shape: {forecasts.shape}")
        print(f"Forecast columns: {forecasts.columns.tolist()}")
        
        # Evaluate forecasts
        print("\n3. Evaluating forecasts...")
        for ticker in tickers:
            if ticker in forecasts.columns and ticker in test_data.columns:
                actual = test_data[ticker].values
                predicted = forecasts[ticker].values
                
                metrics = calculate_all_metrics(actual, predicted)
                print(f"\n{ticker} Forecast Performance:")
                print(f"  RMSE: {metrics['RMSE']:.4f}")
                print(f"  MAPE: {metrics['MAPE']:.2f}%")
                print(f"  R²: {metrics['R2']:.4f}")
        
        # Test model summary
        print("\n4. Model Summary:")
        print(var_model.get_model_summary())
        
        # Test with dictionary input
        print("\n5. Testing with dictionary input...")
        data_dict = {
            'AAPL_close': train_data['AAPL'],
            'MSFT_close': train_data['MSFT']
        }
        
        var_model_dict = VARModel(maxlags=3)
        var_model_dict.fit(data_dict)
        forecasts_dict, _ = var_model_dict.predict(steps=5)
        
        print(f"Dictionary input test successful!")
        print(f"Forecast shape: {forecasts_dict.shape}")
        
        print("\nAll VAR model tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()