"""Baseline forecasting models for portfolio forecasting system."""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
import warnings


def naive_forecast(series: pd.Series, forecast_steps: int = 1) -> pd.Series:
    """
    Generate naive forecasts where the forecast equals the last observed value.
    
    Args:
        series (pd.Series): Historical time series data
        forecast_steps (int): Number of steps to forecast ahead
        
    Returns:
        pd.Series: Naive forecasts
    """
    if len(series) == 0:
        raise ValueError("Series cannot be empty")
    
    # Get the last value
    last_value = series.iloc[-1]
    
    # Create forecast index
    if isinstance(series.index, pd.DatetimeIndex):
        # If datetime index, continue the time series
        last_date = series.index[-1]
        freq = pd.infer_freq(series.index)
        if freq is None:
            # Fall back to daily frequency if can't infer
            freq = 'D'
        forecast_index = pd.date_range(
            start=last_date + pd.DateOffset(days=1), 
            periods=forecast_steps, 
            freq=freq
        )
    else:
        # If numeric index, just continue the sequence
        start_idx = series.index[-1] + 1
        forecast_index = range(start_idx, start_idx + forecast_steps)
    
    # Create forecast series with constant value
    forecasts = pd.Series([last_value] * forecast_steps, index=forecast_index)
    
    return forecasts


def seasonal_naive_forecast(series: pd.Series, season_length: int, 
                          forecast_steps: int = 1) -> pd.Series:
    """
    Generate seasonal naive forecasts using values from the same season in previous periods.
    
    Args:
        series (pd.Series): Historical time series data
        season_length (int): Length of the seasonal pattern (e.g., 12 for monthly data)
        forecast_steps (int): Number of steps to forecast ahead
        
    Returns:
        pd.Series: Seasonal naive forecasts
    """
    if len(series) < season_length:
        warnings.warn(f"Series length ({len(series)}) is less than season length ({season_length}), "
                     "falling back to naive forecast")
        return naive_forecast(series, forecast_steps)
    
    forecasts = []
    
    # Generate forecasts for each step
    for step in range(forecast_steps):
        # Find the corresponding seasonal index
        seasonal_idx = (len(series) + step) % season_length
        
        # Get all historical values at this seasonal position
        seasonal_values = []
        for i in range(seasonal_idx, len(series), season_length):
            seasonal_values.append(series.iloc[i])
        
        if seasonal_values:
            # Use the most recent seasonal value
            forecast_value = seasonal_values[-1]
        else:
            # Fallback to last known value
            forecast_value = series.iloc[-1]
        
        forecasts.append(forecast_value)
    
    # Create forecast index
    if isinstance(series.index, pd.DatetimeIndex):
        last_date = series.index[-1]
        freq = pd.infer_freq(series.index)
        if freq is None:
            freq = 'D'
        forecast_index = pd.date_range(
            start=last_date + pd.DateOffset(days=1), 
            periods=forecast_steps, 
            freq=freq
        )
    else:
        start_idx = series.index[-1] + 1
        forecast_index = range(start_idx, start_idx + forecast_steps)
    
    return pd.Series(forecasts, index=forecast_index)


def arima_forecast(series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1),
                  forecast_steps: int = 1, seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                  return_confidence_intervals: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
    """
    Generate ARIMA forecasts for a time series.
    
    Args:
        series (pd.Series): Historical time series data
        order (Tuple[int, int, int]): ARIMA order (p, d, q)
        forecast_steps (int): Number of steps to forecast ahead
        seasonal_order (Tuple[int, int, int, int], optional): Seasonal ARIMA order (P, D, Q, s)
        return_confidence_intervals (bool): Whether to return confidence intervals
        
    Returns:
        Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]: Forecasts and optionally confidence intervals
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        warnings.warn("statsmodels not available, falling back to naive forecast")
        forecasts = naive_forecast(series, forecast_steps)
        if return_confidence_intervals:
            # Create dummy confidence intervals
            ci_df = pd.DataFrame({
                'lower': forecasts * 0.95,
                'upper': forecasts * 1.05
            }, index=forecasts.index)
            return forecasts, ci_df
        return forecasts
    
    try:
        # Remove any NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < max(order) + 10:
            warnings.warn("Insufficient data for ARIMA model, falling back to naive forecast")
            forecasts = naive_forecast(series, forecast_steps)
            if return_confidence_intervals:
                ci_df = pd.DataFrame({
                    'lower': forecasts * 0.95,
                    'upper': forecasts * 1.05
                }, index=forecasts.index)
                return forecasts, ci_df
            return forecasts
        
        # Fit the model
        if seasonal_order is not None:
            # Use SARIMAX for seasonal models
            model = SARIMAX(clean_series, order=order, seasonal_order=seasonal_order)
        else:
            # Use ARIMA for non-seasonal models
            model = ARIMA(clean_series, order=order)
        
        fitted_model = model.fit()
        
        # Generate forecasts
        forecast_result = fitted_model.forecast(steps=forecast_steps, alpha=0.05)
        
        # Create forecast index
        if isinstance(series.index, pd.DatetimeIndex):
            last_date = series.index[-1]
            freq = pd.infer_freq(series.index)
            if freq is None:
                freq = 'D'
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(days=1), 
                periods=forecast_steps, 
                freq=freq
            )
        else:
            start_idx = series.index[-1] + 1
            forecast_index = range(start_idx, start_idx + forecast_steps)
        
        if return_confidence_intervals:
            # Get prediction intervals
            pred_result = fitted_model.get_prediction(
                start=len(clean_series), 
                end=len(clean_series) + forecast_steps - 1
            )
            pred_ci = pred_result.conf_int()
            
            forecasts = pd.Series(forecast_result, index=forecast_index)
            ci_df = pd.DataFrame({
                'lower': pred_ci.iloc[:, 0].values,
                'upper': pred_ci.iloc[:, 1].values
            }, index=forecast_index)
            
            return forecasts, ci_df
        else:
            forecasts = pd.Series(forecast_result, index=forecast_index)
            return forecasts
            
    except Exception as e:
        warnings.warn(f"ARIMA fitting failed: {e}, falling back to naive forecast")
        forecasts = naive_forecast(series, forecast_steps)
        if return_confidence_intervals:
            ci_df = pd.DataFrame({
                'lower': forecasts * 0.95,
                'upper': forecasts * 1.05
            }, index=forecasts.index)
            return forecasts, ci_df
        return forecasts


def auto_arima_forecast(series: pd.Series, forecast_steps: int = 1,
                       max_p: int = 3, max_d: int = 2, max_q: int = 3,
                       seasonal: bool = False, m: int = 1) -> pd.Series:
    """
    Automatically select ARIMA order and generate forecasts.
    
    Args:
        series (pd.Series): Historical time series data
        forecast_steps (int): Number of steps to forecast ahead
        max_p (int): Maximum AR order to test
        max_d (int): Maximum differencing order to test
        max_q (int): Maximum MA order to test
        seasonal (bool): Whether to include seasonal components
        m (int): Seasonal period length
        
    Returns:
        pd.Series: ARIMA forecasts with automatically selected order
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        warnings.warn("statsmodels not available, falling back to naive forecast")
        return naive_forecast(series, forecast_steps)
    
    # Remove any NaN values
    clean_series = series.dropna()
    
    if len(clean_series) < 20:
        warnings.warn("Insufficient data for auto ARIMA, falling back to naive forecast")
        return naive_forecast(series, forecast_steps)
    
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal_order = None
    
    # Grid search for best parameters
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    if seasonal and m > 1:
                        # Test seasonal models
                        for P in range(2):
                            for D in range(2):
                                for Q in range(2):
                                    try:
                                        from statsmodels.tsa.statespace.sarimax import SARIMAX
                                        model = SARIMAX(clean_series, 
                                                       order=(p, d, q),
                                                       seasonal_order=(P, D, Q, m))
                                        fitted = model.fit(disp=False)
                                        if fitted.aic < best_aic:
                                            best_aic = fitted.aic
                                            best_order = (p, d, q)
                                            best_seasonal_order = (P, D, Q, m)
                                    except:
                                        continue
                    else:
                        # Test non-seasonal models
                        model = ARIMA(clean_series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_seasonal_order = None
                            
                except Exception:
                    continue
    
    # Generate forecast with best model
    return arima_forecast(series, order=best_order, forecast_steps=forecast_steps,
                         seasonal_order=best_seasonal_order)


def moving_average_forecast(series: pd.Series, window: int = 10, 
                          forecast_steps: int = 1) -> pd.Series:
    """
    Generate forecasts using moving average.
    
    Args:
        series (pd.Series): Historical time series data
        window (int): Moving average window size
        forecast_steps (int): Number of steps to forecast ahead
        
    Returns:
        pd.Series: Moving average forecasts
    """
    if len(series) < window:
        warnings.warn(f"Series length ({len(series)}) is less than window size ({window}), "
                     "falling back to naive forecast")
        return naive_forecast(series, forecast_steps)
    
    # Calculate the moving average of the last 'window' values
    forecast_value = series.iloc[-window:].mean()
    
    # Create forecast index
    if isinstance(series.index, pd.DatetimeIndex):
        last_date = series.index[-1]
        freq = pd.infer_freq(series.index)
        if freq is None:
            freq = 'D'
        forecast_index = pd.date_range(
            start=last_date + pd.DateOffset(days=1), 
            periods=forecast_steps, 
            freq=freq
        )
    else:
        start_idx = series.index[-1] + 1
        forecast_index = range(start_idx, start_idx + forecast_steps)
    
    # Return constant forecast
    forecasts = pd.Series([forecast_value] * forecast_steps, index=forecast_index)
    
    return forecasts


# Example usage and testing functions
if __name__ == "__main__":
    import os
    import sys
    
    # Add utils to path for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    
    from data_loader import load_data, get_ticker_data
    from evaluation import calculate_all_metrics
    
    # Get the path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'mock_stock_data.csv')
    
    try:
        print("Testing baseline forecasting models...")
        
        # Load data
        data = load_data(data_path)
        aapl_close = get_ticker_data(data, 'AAPL', 'close')
        
        # Split data for testing
        train_size = int(len(aapl_close) * 0.8)
        train_data = aapl_close.iloc[:train_size]
        test_data = aapl_close.iloc[train_size:]
        forecast_steps = len(test_data)
        
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        # Test naive forecast
        print("\n1. Testing naive forecast...")
        naive_pred = naive_forecast(train_data, forecast_steps)
        naive_metrics = calculate_all_metrics(test_data.values, naive_pred.values)
        print(f"Naive RMSE: {naive_metrics['RMSE']:.4f}")
        print(f"Naive MAPE: {naive_metrics['MAPE']:.2f}%")
        
        # Test seasonal naive
        print("\n2. Testing seasonal naive forecast...")
        seasonal_pred = seasonal_naive_forecast(train_data, season_length=30, 
                                               forecast_steps=forecast_steps)
        seasonal_metrics = calculate_all_metrics(test_data.values, seasonal_pred.values)
        print(f"Seasonal Naive RMSE: {seasonal_metrics['RMSE']:.4f}")
        print(f"Seasonal Naive MAPE: {seasonal_metrics['MAPE']:.2f}%")
        
        # Test moving average
        print("\n3. Testing moving average forecast...")
        ma_pred = moving_average_forecast(train_data, window=10, forecast_steps=forecast_steps)
        ma_metrics = calculate_all_metrics(test_data.values, ma_pred.values)
        print(f"Moving Average RMSE: {ma_metrics['RMSE']:.4f}")
        print(f"Moving Average MAPE: {ma_metrics['MAPE']:.2f}%")
        
        # Test ARIMA forecast
        print("\n4. Testing ARIMA forecast...")
        arima_pred = arima_forecast(train_data, order=(1, 1, 1), forecast_steps=forecast_steps)
        arima_metrics = calculate_all_metrics(test_data.values, arima_pred.values)
        print(f"ARIMA RMSE: {arima_metrics['RMSE']:.4f}")
        print(f"ARIMA MAPE: {arima_metrics['MAPE']:.2f}%")
        
        # Test ARIMA with confidence intervals
        print("\n5. Testing ARIMA with confidence intervals...")
        arima_pred_ci, confidence_intervals = arima_forecast(
            train_data, order=(1, 1, 1), forecast_steps=min(10, forecast_steps),
            return_confidence_intervals=True
        )
        print(f"ARIMA with CI forecast shape: {arima_pred_ci.shape}")
        print(f"Confidence intervals shape: {confidence_intervals.shape}")
        print(f"Average CI width: {(confidence_intervals['upper'] - confidence_intervals['lower']).mean():.4f}")
        
        # Compare all models
        print("\n6. Model comparison summary:")
        print(f"{'Model':<20} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 40)
        print(f"{'Naive':<20} {naive_metrics['RMSE']:<10.4f} {naive_metrics['MAPE']:<10.2f}")
        print(f"{'Seasonal Naive':<20} {seasonal_metrics['RMSE']:<10.4f} {seasonal_metrics['MAPE']:<10.2f}")
        print(f"{'Moving Average':<20} {ma_metrics['RMSE']:<10.4f} {ma_metrics['MAPE']:<10.2f}")
        print(f"{'ARIMA':<20} {arima_metrics['RMSE']:<10.4f} {arima_metrics['MAPE']:<10.2f}")
        
        print("\nAll baseline model tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()