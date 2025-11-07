"""Evaluation metrics for forecasting models."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings


def rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: RMSE value
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: MAPE value as percentage
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        warnings.warn("All true values are zero, MAPE cannot be calculated")
        return np.inf
    
    if not mask.all():
        warnings.warn("Some true values are zero, excluding them from MAPE calculation")
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mae(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: MAE value
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Square Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: MSE value
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    return np.mean((y_true - y_pred) ** 2)


def smape(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: SMAPE value as percentage
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    mask = denominator != 0
    if not mask.any():
        warnings.warn("All values are zero, SMAPE cannot be calculated")
        return 0.0
    
    if not mask.all():
        warnings.warn("Some values sum to zero, excluding them from SMAPE calculation")
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        denominator = denominator[mask]
    
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def r2_score(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: R-squared value
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def directional_accuracy(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: Directional accuracy as percentage
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        raise ValueError("Need at least 2 values to calculate directional accuracy")
    
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Calculate direction of change
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Calculate accuracy
    correct_directions = np.sum(true_direction == pred_direction)
    total_directions = len(true_direction)
    
    return (correct_directions / total_directions) * 100


def theil_u_statistic(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Theil's U statistic for forecast accuracy.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: Theil's U statistic
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        raise ValueError("Need at least 2 values to calculate Theil's U statistic")
    
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Calculate forecast errors
    forecast_error = np.sum((y_pred[1:] - y_true[1:]) ** 2)
    
    # Calculate naive forecast errors (using previous period as forecast)
    naive_error = np.sum((y_true[:-1] - y_true[1:]) ** 2)
    
    if naive_error == 0:
        return 0.0 if forecast_error == 0 else np.inf
    
    return np.sqrt(forecast_error / naive_error)


def calculate_all_metrics(y_true: Union[np.ndarray, pd.Series], 
                         y_pred: Union[np.ndarray, pd.Series],
                         include_directional: bool = True) -> Dict[str, float]:
    """
    Calculate all available metrics for forecast evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        include_directional: Whether to include directional accuracy and Theil's U
        
    Returns:
        Dict: Dictionary containing all calculated metrics
    """
    metrics = {}
    
    try:
        metrics['RMSE'] = rmse(y_true, y_pred)
    except Exception as e:
        metrics['RMSE'] = np.nan
        warnings.warn(f"Could not calculate RMSE: {e}")
    
    try:
        metrics['MAE'] = mae(y_true, y_pred)
    except Exception as e:
        metrics['MAE'] = np.nan
        warnings.warn(f"Could not calculate MAE: {e}")
    
    try:
        metrics['MSE'] = mse(y_true, y_pred)
    except Exception as e:
        metrics['MSE'] = np.nan
        warnings.warn(f"Could not calculate MSE: {e}")
    
    try:
        metrics['MAPE'] = mape(y_true, y_pred)
    except Exception as e:
        metrics['MAPE'] = np.nan
        warnings.warn(f"Could not calculate MAPE: {e}")
    
    try:
        metrics['SMAPE'] = smape(y_true, y_pred)
    except Exception as e:
        metrics['SMAPE'] = np.nan
        warnings.warn(f"Could not calculate SMAPE: {e}")
    
    try:
        metrics['R2'] = r2_score(y_true, y_pred)
    except Exception as e:
        metrics['R2'] = np.nan
        warnings.warn(f"Could not calculate R2: {e}")
    
    if include_directional:
        try:
            metrics['Directional_Accuracy'] = directional_accuracy(y_true, y_pred)
        except Exception as e:
            metrics['Directional_Accuracy'] = np.nan
            warnings.warn(f"Could not calculate directional accuracy: {e}")
        
        try:
            metrics['Theil_U'] = theil_u_statistic(y_true, y_pred)
        except Exception as e:
            metrics['Theil_U'] = np.nan
            warnings.warn(f"Could not calculate Theil's U: {e}")
    
    return metrics


def compare_forecasts(true_values: Union[np.ndarray, pd.Series],
                     forecasts: Dict[str, Union[np.ndarray, pd.Series]],
                     metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare multiple forecasts using various metrics.
    
    Args:
        true_values: True values
        forecasts: Dictionary of {model_name: predictions}
        metrics: List of metrics to calculate (default: all)
        
    Returns:
        pd.DataFrame: Comparison table with models as rows and metrics as columns
    """
    if metrics is None:
        metrics = ['RMSE', 'MAE', 'MAPE', 'SMAPE', 'R2', 'Directional_Accuracy', 'Theil_U']
    
    results = {}
    
    for model_name, predictions in forecasts.items():
        model_metrics = calculate_all_metrics(true_values, predictions)
        results[model_name] = {metric: model_metrics.get(metric, np.nan) for metric in metrics}
    
    return pd.DataFrame(results).T


def forecast_accuracy_summary(y_true: Union[np.ndarray, pd.Series],
                            y_pred: Union[np.ndarray, pd.Series],
                            model_name: str = "Model") -> str:
    """
    Generate a formatted summary of forecast accuracy.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model for the summary
        
    Returns:
        str: Formatted accuracy summary
    """
    metrics = calculate_all_metrics(y_true, y_pred)
    
    summary = f"\n{model_name} Forecast Accuracy Summary:\n"
    summary += "=" * (len(model_name) + 30) + "\n"
    
    # Main error metrics
    summary += f"RMSE:               {metrics.get('RMSE', 'N/A'):.4f}\n"
    summary += f"MAE:                {metrics.get('MAE', 'N/A'):.4f}\n"
    summary += f"MAPE:               {metrics.get('MAPE', 'N/A'):.2f}%\n"
    summary += f"SMAPE:              {metrics.get('SMAPE', 'N/A'):.2f}%\n"
    
    # Model fit metrics
    summary += f"R-squared:          {metrics.get('R2', 'N/A'):.4f}\n"
    
    # Directional metrics
    if 'Directional_Accuracy' in metrics and not np.isnan(metrics['Directional_Accuracy']):
        summary += f"Directional Acc:    {metrics['Directional_Accuracy']:.2f}%\n"
    
    if 'Theil_U' in metrics and not np.isnan(metrics['Theil_U']):
        summary += f"Theil's U:          {metrics['Theil_U']:.4f}\n"
    
    # Interpretation
    summary += "\nInterpretation:\n"
    if metrics.get('R2', 0) > 0.8:
        summary += "• Excellent model fit (R² > 0.8)\n"
    elif metrics.get('R2', 0) > 0.6:
        summary += "• Good model fit (R² > 0.6)\n"
    elif metrics.get('R2', 0) > 0.3:
        summary += "• Moderate model fit (R² > 0.3)\n"
    else:
        summary += "• Poor model fit (R² ≤ 0.3)\n"
    
    if metrics.get('Theil_U', float('inf')) < 1:
        summary += "• Better than naive forecast (Theil's U < 1)\n"
    elif metrics.get('Theil_U', 0) == 1:
        summary += "• Equal to naive forecast (Theil's U = 1)\n"
    elif metrics.get('Theil_U', 0) > 1:
        summary += "• Worse than naive forecast (Theil's U > 1)\n"
    
    return summary


# Example usage and testing functions
if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    # Generate sample data
    np.random.seed(42)
    n = 100
    
    # Create realistic time series with trend and noise
    time_index = np.arange(n)
    true_values = 100 + 0.5 * time_index + 10 * np.sin(0.1 * time_index) + np.random.normal(0, 5, n)
    
    # Create predictions with some error
    pred_values = true_values + np.random.normal(0, 3, n)
    
    # Test individual metrics
    print("\n1. Testing individual metrics...")
    print(f"RMSE: {rmse(true_values, pred_values):.4f}")
    print(f"MAE: {mae(true_values, pred_values):.4f}")
    print(f"MAPE: {mape(true_values, pred_values):.2f}%")
    print(f"SMAPE: {smape(true_values, pred_values):.2f}%")
    print(f"R²: {r2_score(true_values, pred_values):.4f}")
    print(f"Directional Accuracy: {directional_accuracy(true_values, pred_values):.2f}%")
    print(f"Theil's U: {theil_u_statistic(true_values, pred_values):.4f}")
    
    # Test calculate_all_metrics
    print("\n2. Testing calculate_all_metrics...")
    all_metrics = calculate_all_metrics(true_values, pred_values)
    for metric, value in all_metrics.items():
        if isinstance(value, float):
            if 'Accuracy' in metric or 'MAPE' in metric or 'SMAPE' in metric:
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value:.4f}")
    
    # Test compare_forecasts
    print("\n3. Testing compare_forecasts...")
    
    # Create multiple forecast scenarios
    forecasts = {
        'Model_A': pred_values,
        'Model_B': true_values + np.random.normal(0, 2, n),  # Better model
        'Model_C': true_values + np.random.normal(0, 8, n),  # Worse model
        'Naive': np.concatenate([[true_values[0]], true_values[:-1]])  # Naive forecast
    }
    
    comparison = compare_forecasts(true_values, forecasts)
    print(comparison.round(4))
    
    # Test forecast_accuracy_summary
    print("\n4. Testing forecast_accuracy_summary...")
    summary = forecast_accuracy_summary(true_values, pred_values, "Test Model")
    print(summary)
    
    # Test edge cases
    print("\n5. Testing edge cases...")
    
    # Test with pandas Series
    true_series = pd.Series(true_values)
    pred_series = pd.Series(pred_values)
    pd_rmse = rmse(true_series, pred_series)
    print(f"RMSE with pandas Series: {pd_rmse:.4f}")
    
    # Test with zeros (for MAPE)
    try:
        zero_true = np.array([0, 1, 2, 0, 4])
        zero_pred = np.array([0.1, 1.1, 1.9, 0.2, 3.8])
        zero_mape = mape(zero_true, zero_pred)
        print(f"MAPE with some zeros: {zero_mape:.2f}%")
    except Exception as e:
        print(f"MAPE with zeros failed as expected: {e}")
    
    print("\nAll evaluation tests completed!")