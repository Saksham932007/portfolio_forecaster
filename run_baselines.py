"""Baseline model runner script for testing and evaluation."""

import os
import sys
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt  # Commented out for now
from typing import Dict, List

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'utils'))
sys.path.append(os.path.join(project_root, 'models'))

from data_loader import load_data, get_ticker_data, split_train_test
from evaluation import calculate_all_metrics, compare_forecasts, forecast_accuracy_summary
from baseline import naive_forecast, seasonal_naive_forecast, moving_average_forecast, arima_forecast


def run_baseline_comparison(ticker: str = 'AAPL', test_size: float = 0.2, 
                          column: str = 'close', verbose: bool = True) -> Dict:
    """
    Run a comprehensive comparison of baseline forecasting models.
    
    Args:
        ticker (str): Ticker symbol to analyze
        test_size (float): Proportion of data for testing
        column (str): Column to forecast
        verbose (bool): Whether to print detailed results
        
    Returns:
        Dict: Results containing forecasts and metrics for all models
    """
    # Load data
    data_path = os.path.join(project_root, 'data', 'mock_stock_data.csv')
    data = load_data(data_path)
    
    # Get ticker data
    ticker_series = get_ticker_data(data, ticker, column)
    
    # Split into train/test
    train_data, test_data = split_train_test(ticker_series, test_size=test_size)
    
    forecast_steps = len(test_data)
    
    if verbose:
        print(f"Baseline Model Comparison for {ticker} ({column})")
        print("=" * 60)
        print(f"Total data points: {len(ticker_series)}")
        print(f"Training data: {len(train_data)}")
        print(f"Test data: {len(test_data)} (forecasting {forecast_steps} steps)")
        print(f"Date range: {ticker_series.index[0]} to {ticker_series.index[-1]}")
        print()
    
    # Initialize results storage
    forecasts = {}
    metrics = {}
    
    # 1. Naive Forecast
    if verbose:
        print("1. Running Naive Forecast...")
    try:
        naive_pred = naive_forecast(train_data, forecast_steps)
        forecasts['Naive'] = naive_pred
        metrics['Naive'] = calculate_all_metrics(test_data.values, naive_pred.values)
        if verbose:
            print(f"   ✓ Naive forecast completed")
    except Exception as e:
        if verbose:
            print(f"   ✗ Naive forecast failed: {e}")
        forecasts['Naive'] = None
        metrics['Naive'] = None
    
    # 2. Seasonal Naive Forecast
    if verbose:
        print("2. Running Seasonal Naive Forecast...")
    try:
        seasonal_pred = seasonal_naive_forecast(train_data, season_length=30, 
                                              forecast_steps=forecast_steps)
        forecasts['Seasonal_Naive'] = seasonal_pred
        metrics['Seasonal_Naive'] = calculate_all_metrics(test_data.values, seasonal_pred.values)
        if verbose:
            print(f"   ✓ Seasonal naive forecast completed")
    except Exception as e:
        if verbose:
            print(f"   ✗ Seasonal naive forecast failed: {e}")
        forecasts['Seasonal_Naive'] = None
        metrics['Seasonal_Naive'] = None
    
    # 3. Moving Average Forecast (different windows)
    for window in [5, 10, 20]:
        model_name = f'MA_{window}'
        if verbose:
            print(f"3.{window//5}. Running Moving Average Forecast (window={window})...")
        try:
            ma_pred = moving_average_forecast(train_data, window=window, 
                                            forecast_steps=forecast_steps)
            forecasts[model_name] = ma_pred
            metrics[model_name] = calculate_all_metrics(test_data.values, ma_pred.values)
            if verbose:
                print(f"   ✓ Moving average (window={window}) forecast completed")
        except Exception as e:
            if verbose:
                print(f"   ✗ Moving average (window={window}) forecast failed: {e}")
            forecasts[model_name] = None
            metrics[model_name] = None
    
    # 4. ARIMA Forecasts (different orders)
    arima_orders = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)]
    for i, order in enumerate(arima_orders, 1):
        model_name = f'ARIMA_{order[0]}_{order[1]}_{order[2]}'
        if verbose:
            print(f"4.{i}. Running ARIMA{order} Forecast...")
        try:
            arima_pred = arima_forecast(train_data, order=order, 
                                      forecast_steps=forecast_steps)
            forecasts[model_name] = arima_pred
            metrics[model_name] = calculate_all_metrics(test_data.values, arima_pred.values)
            if verbose:
                print(f"   ✓ ARIMA{order} forecast completed")
        except Exception as e:
            if verbose:
                print(f"   ✗ ARIMA{order} forecast failed: {e}")
            forecasts[model_name] = None
            metrics[model_name] = None
    
    # Filter out failed models
    valid_forecasts = {k: v for k, v in forecasts.items() if v is not None}
    valid_metrics = {k: v for k, v in metrics.items() if v is not None}
    
    if verbose:
        print(f"\nSuccessfully completed {len(valid_forecasts)} models")
        print("\nModel Performance Summary:")
        print("-" * 80)
        
        # Create comparison DataFrame
        comparison_df = compare_forecasts(test_data, valid_forecasts, 
                                        metrics=['RMSE', 'MAE', 'MAPE', 'R2'])
        print(comparison_df.round(4))
        
        # Find best model by RMSE
        if len(comparison_df) > 0:
            best_model = comparison_df['RMSE'].idxmin()
            print(f"\nBest performing model: {best_model}")
            print(f"Best RMSE: {comparison_df.loc[best_model, 'RMSE']:.4f}")
            
            # Detailed summary for best model
            if best_model in valid_forecasts:
                print(forecast_accuracy_summary(test_data.values, 
                                              valid_forecasts[best_model].values,
                                              f"Best Model ({best_model})"))
    else:
        comparison_df = None
    
    # Return comprehensive results
    comparison_df = compare_forecasts(test_data, valid_forecasts) if valid_forecasts else None
    results = {
        'ticker': ticker,
        'column': column,
        'train_data': train_data,
        'test_data': test_data,
        'forecasts': valid_forecasts,
        'metrics': valid_metrics,
        'comparison': comparison_df,
        'best_model': comparison_df['RMSE'].idxmin() if comparison_df is not None and len(comparison_df) > 0 else None
    }
    
    return results


def plot_forecast_comparison(results: Dict, save_plot: bool = False, 
                           plot_path: str = None, show_confidence: bool = False):
    """
    Plot forecast comparison for visual evaluation.
    
    Args:
        results (Dict): Results from run_baseline_comparison
        save_plot (bool): Whether to save the plot
        plot_path (str): Path to save the plot
        show_confidence (bool): Whether to show confidence intervals (if available)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return
    
    plt.figure(figsize=(15, 8))
    
    train_data = results['train_data']
    test_data = results['test_data']
    forecasts = results['forecasts']
    
    # Plot training data
    plt.plot(train_data.index, train_data.values, 'b-', label='Training Data', linewidth=2)
    
    # Plot test data (actual)
    plt.plot(test_data.index, test_data.values, 'g-', label='Actual', linewidth=2)
    
    # Plot forecasts
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        if forecast is not None:
            color = colors[i % len(colors)]
            plt.plot(forecast.index, forecast.values, '--', 
                    label=f'{model_name}', color=color, linewidth=1.5)
    
    plt.axvline(x=test_data.index[0], color='black', linestyle=':', 
                label='Train/Test Split', alpha=0.7)
    
    plt.title(f'Baseline Model Forecasts Comparison\n{results["ticker"]} - {results["column"].title()} Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plot:
        if plot_path is None:
            plot_path = f'baseline_forecast_comparison_{results["ticker"]}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    
    plt.show()


def run_multiple_tickers(tickers: List[str] = ['AAPL', 'MSFT', 'GOOG'], 
                        column: str = 'close', test_size: float = 0.2) -> Dict:
    """
    Run baseline comparison for multiple tickers.
    
    Args:
        tickers (List[str]): List of tickers to analyze
        column (str): Column to forecast
        test_size (float): Proportion of data for testing
        
    Returns:
        Dict: Results for all tickers
    """
    all_results = {}
    
    print("Running Baseline Model Comparison for Multiple Tickers")
    print("=" * 60)
    
    for ticker in tickers:
        print(f"\n{'='*20} {ticker} {'='*20}")
        try:
            results = run_baseline_comparison(ticker, test_size, column, verbose=False)
            all_results[ticker] = results
            
            # Print summary for this ticker
            if results['comparison'] is not None:
                best_model = results['best_model']
                best_rmse = results['comparison'].loc[best_model, 'RMSE']
                best_mape = results['comparison'].loc[best_model, 'MAPE']
                print(f"Best model for {ticker}: {best_model}")
                print(f"RMSE: {best_rmse:.4f}, MAPE: {best_mape:.2f}%")
            else:
                print(f"No successful models for {ticker}")
                
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")
            all_results[ticker] = None
    
    # Overall summary
    print(f"\n{'='*20} OVERALL SUMMARY {'='*20}")
    successful_tickers = [k for k, v in all_results.items() if v is not None]
    print(f"Successfully processed {len(successful_tickers)}/{len(tickers)} tickers")
    
    # Best model across all tickers
    if successful_tickers:
        best_models = {}
        for ticker in successful_tickers:
            if all_results[ticker]['best_model']:
                best_models[ticker] = all_results[ticker]['best_model']
        
        if best_models:
            from collections import Counter
            model_counts = Counter(best_models.values())
            most_common_model = model_counts.most_common(1)[0]
            print(f"Most successful model: {most_common_model[0]} (best for {most_common_model[1]}/{len(best_models)} tickers)")
    
    return all_results


if __name__ == "__main__":
    print("Portfolio Forecaster - Baseline Model Runner")
    print("=" * 50)
    
    # Run single ticker analysis
    print("\n1. Single Ticker Analysis (AAPL)")
    results_single = run_baseline_comparison('AAPL', test_size=0.2, verbose=True)
    
    # Plot the results
    print("\n2. Generating forecast comparison plot...")
    try:
        plot_forecast_comparison(results_single, save_plot=True)
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    # Run multiple ticker analysis
    print("\n3. Multiple Ticker Analysis")
    results_multi = run_multiple_tickers(['AAPL', 'MSFT', 'GOOG'], test_size=0.2)
    
    print(f"\nBaseline model evaluation completed!")
    print(f"Results available for: {list(results_multi.keys())}")