"""Backtesting framework for evaluating forecasting models."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import warnings
import os
import sys

# Add project paths for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'utils'))
sys.path.append(os.path.join(project_root, 'models'))

from data_loader import load_data, get_ticker_data, split_train_test
from evaluation import calculate_all_metrics, compare_forecasts
from baseline import naive_forecast, arima_forecast, moving_average_forecast
from var_model import VARModel
from deep_learning_model import create_deep_learning_forecaster


class SimpleBacktester:
    """
    Simple backtesting framework for time series forecasting models.
    
    Implements a simple train/test split approach suitable for hackathon
    demonstration purposes.
    """
    
    def __init__(self, test_size: float = 0.2):
        """
        Initialize the backtester.
        
        Args:
            test_size: Proportion of data to use for testing
        """
        self.test_size = test_size
        self.results = {}
        self.models = {}
        
    def add_baseline_models(self):
        """Add baseline forecasting models to the backtest."""
        self.models.update({
            'naive': {
                'function': naive_forecast,
                'params': {'forecast_steps': None}  # Will be set during backtest
            },
            'moving_average_5': {
                'function': moving_average_forecast,
                'params': {'window': 5, 'forecast_steps': None}
            },
            'moving_average_10': {
                'function': moving_average_forecast,
                'params': {'window': 10, 'forecast_steps': None}
            },
            'arima_111': {
                'function': arima_forecast,
                'params': {'order': (1, 1, 1), 'forecast_steps': None}
            }
        })
    
    def add_var_model(self, maxlags: int = 5):
        """
        Add VAR model to the backtest.
        
        Args:
            maxlags: Maximum lags for VAR model
        """
        self.models['var'] = {
            'type': 'multivariate',
            'class': VARModel,
            'params': {'maxlags': maxlags}
        }
    
    def add_deep_learning_models(self, max_epochs: int = 5):
        """
        Add deep learning models to the backtest.
        
        Args:
            max_epochs: Maximum training epochs for DL models
        """
        self.models.update({
            'deepar': {
                'type': 'deep_learning',
                'model_type': 'deepar',
                'params': {'max_epochs': max_epochs}
            },
            'tft': {
                'type': 'deep_learning',
                'model_type': 'tft',
                'params': {'max_epochs': max_epochs}
            }
        })
    
    def _run_univariate_model(self, model_name: str, model_config: Dict, 
                             train_data: pd.Series, test_data: pd.Series) -> Dict:
        """
        Run a univariate forecasting model.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
            train_data: Training data
            test_data: Test data
            
        Returns:
            Dict: Results including predictions and metrics
        """
        try:
            # Set forecast steps
            forecast_steps = len(test_data)
            params = model_config['params'].copy()
            params['forecast_steps'] = forecast_steps
            
            # Generate forecasts
            forecasts = model_config['function'](train_data, **params)
            
            # Calculate metrics
            metrics = calculate_all_metrics(test_data.values, forecasts.values)
            
            return {
                'model_name': model_name,
                'forecasts': forecasts,
                'metrics': metrics,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'model_name': model_name,
                'forecasts': None,
                'metrics': None,
                'success': False,
                'error': str(e)
            }
    
    def _run_var_model(self, model_name: str, model_config: Dict,
                       train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """
        Run VAR model for multivariate forecasting.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
            train_data: Training data
            test_data: Test data
            
        Returns:
            Dict: Results including predictions and metrics
        """
        try:
            # Initialize and fit VAR model
            var_model = model_config['class'](**model_config['params'])
            var_model.fit(train_data)
            
            # Generate forecasts
            forecasts, _ = var_model.predict(steps=len(test_data))
            
            # Calculate metrics for each variable
            all_metrics = {}
            for column in test_data.columns:
                if column in forecasts.columns:
                    metrics = calculate_all_metrics(
                        test_data[column].values, 
                        forecasts[column].values
                    )
                    all_metrics[column] = metrics
            
            return {
                'model_name': model_name,
                'forecasts': forecasts,
                'metrics': all_metrics,
                'success': True,
                'error': None,
                'model_summary': var_model.get_model_summary()
            }
            
        except Exception as e:
            return {
                'model_name': model_name,
                'forecasts': None,
                'metrics': None,
                'success': False,
                'error': str(e)
            }
    
    def _run_deep_learning_model(self, model_name: str, model_config: Dict,
                                data: pd.DataFrame, target_column: str) -> Dict:
        """
        Run deep learning model.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
            data: Full dataset (will be split internally)
            target_column: Target column name
            
        Returns:
            Dict: Results including predictions and metadata
        """
        try:
            # Create forecaster
            forecaster = create_deep_learning_forecaster(
                max_prediction_length=int(len(data) * self.test_size),
                max_encoder_length=min(60, int(len(data) * 0.3))
            )
            
            # Prepare data for deep learning
            dl_data = data.reset_index()
            if 'time_idx' not in dl_data.columns:
                dl_data['time_idx'] = range(len(dl_data))
            if 'ticker' not in dl_data.columns:
                dl_data['ticker'] = 'default'
            
            # Run the appropriate model
            if model_config['model_type'] == 'deepar':
                predictions, metadata = forecaster.fit_and_predict_deepar(
                    dl_data, target_column, **model_config['params']
                )
            elif model_config['model_type'] == 'tft':
                predictions, metadata = forecaster.fit_and_predict_tft(
                    dl_data, target_column, **model_config['params']
                )
            else:
                raise ValueError(f"Unknown deep learning model type: {model_config['model_type']}")
            
            # Calculate metrics (compare with test portion)
            test_size_actual = len(predictions)
            test_actual = data[target_column].iloc[-test_size_actual:].values
            
            metrics = calculate_all_metrics(test_actual, predictions)
            
            return {
                'model_name': model_name,
                'forecasts': predictions,
                'metrics': metrics,
                'success': True,
                'error': None,
                'metadata': metadata
            }
            
        except Exception as e:
            return {
                'model_name': model_name,
                'forecasts': None,
                'metrics': None,
                'success': False,
                'error': str(e)
            }
    
    def run_backtest(self, data: pd.Series, ticker: str = 'Unknown') -> Dict:
        """
        Run backtest on univariate time series data.
        
        Args:
            data: Time series data
            ticker: Ticker symbol for identification
            
        Returns:
            Dict: Comprehensive backtest results
        """
        print(f"Running backtest for {ticker}...")
        print(f"Data points: {len(data)}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Split data
        train_data, test_data = split_train_test(data, self.test_size)
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        results = {
            'ticker': ticker,
            'data_info': {
                'total_points': len(data),
                'train_size': len(train_data),
                'test_size': len(test_data),
                'date_range': (data.index[0], data.index[-1])
            },
            'model_results': {},
            'summary': {}
        }
        
        # Run each model
        for model_name, model_config in self.models.items():
            print(f"  Running {model_name}...")
            
            if model_config.get('type') == 'multivariate':
                print(f"    Skipping {model_name} (requires multivariate data)")
                continue
            elif model_config.get('type') == 'deep_learning':
                # Prepare DataFrame for deep learning
                dl_data = pd.DataFrame({'close': data})
                result = self._run_deep_learning_model(
                    model_name, model_config, dl_data, 'close'
                )
            else:
                # Standard univariate model
                result = self._run_univariate_model(
                    model_name, model_config, train_data, test_data
                )
            
            results['model_results'][model_name] = result
            
            if result['success']:
                print(f"    ✓ {model_name} completed")
            else:
                print(f"    ✗ {model_name} failed: {result['error']}")
        
        # Generate summary
        results['summary'] = self._generate_summary(results['model_results'], test_data)
        
        return results
    
    def run_multivariate_backtest(self, data: pd.DataFrame, tickers: List[str]) -> Dict:
        """
        Run backtest on multivariate time series data.
        
        Args:
            data: Multivariate time series DataFrame
            tickers: List of ticker symbols
            
        Returns:
            Dict: Comprehensive backtest results
        """
        print(f"Running multivariate backtest for {tickers}...")
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Split data
        train_size = int(len(data) * (1 - self.test_size))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        results = {
            'tickers': tickers,
            'data_info': {
                'shape': data.shape,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'date_range': (data.index[0], data.index[-1])
            },
            'model_results': {},
            'summary': {}
        }
        
        # Run multivariate models
        for model_name, model_config in self.models.items():
            if model_config.get('type') != 'multivariate':
                continue
                
            print(f"  Running {model_name}...")
            
            result = self._run_var_model(model_name, model_config, train_data, test_data)
            results['model_results'][model_name] = result
            
            if result['success']:
                print(f"    ✓ {model_name} completed")
            else:
                print(f"    ✗ {model_name} failed: {result['error']}")
        
        # Generate summary
        results['summary'] = self._generate_multivariate_summary(results['model_results'], tickers)
        
        return results
    
    def _generate_summary(self, model_results: Dict, test_data: pd.Series) -> Dict:
        """Generate summary statistics for univariate backtest."""
        successful_models = {k: v for k, v in model_results.items() if v['success']}
        
        if not successful_models:
            return {'note': 'No successful models', 'best_model': None}
        
        # Create comparison DataFrame
        forecasts_dict = {}
        for model_name, result in successful_models.items():
            if result['forecasts'] is not None:
                forecasts_dict[model_name] = result['forecasts']
        
        if forecasts_dict:
            comparison_df = compare_forecasts(test_data, forecasts_dict)
            best_model = comparison_df['RMSE'].idxmin()
            best_rmse = comparison_df.loc[best_model, 'RMSE']
            best_mape = comparison_df.loc[best_model, 'MAPE']
            
            return {
                'num_successful_models': len(successful_models),
                'best_model': best_model,
                'best_rmse': best_rmse,
                'best_mape': best_mape,
                'comparison': comparison_df,
                'model_ranking': comparison_df.sort_values('RMSE').index.tolist()
            }
        
        return {'note': 'No valid forecasts generated'}
    
    def _generate_multivariate_summary(self, model_results: Dict, tickers: List[str]) -> Dict:
        """Generate summary statistics for multivariate backtest."""
        successful_models = {k: v for k, v in model_results.items() if v['success']}
        
        if not successful_models:
            return {'note': 'No successful models'}
        
        summary = {'num_successful_models': len(successful_models)}
        
        # Summarize performance by ticker
        for ticker in tickers:
            ticker_performance = {}
            for model_name, result in successful_models.items():
                if result['metrics'] and ticker in result['metrics']:
                    metrics = result['metrics'][ticker]
                    ticker_performance[model_name] = {
                        'RMSE': metrics.get('RMSE', np.nan),
                        'MAPE': metrics.get('MAPE', np.nan)
                    }
            summary[f'{ticker}_performance'] = ticker_performance
        
        return summary
    
    def print_results(self, results: Dict):
        """Print formatted backtest results."""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}")
        
        if 'ticker' in results:
            print(f"Ticker: {results['ticker']}")
        elif 'tickers' in results:
            print(f"Tickers: {', '.join(results['tickers'])}")
        
        data_info = results['data_info']
        print(f"Total data points: {data_info.get('total_points', data_info.get('shape', 'N/A'))}")
        print(f"Train size: {data_info['train_size']}")
        print(f"Test size: {data_info['test_size']}")
        print(f"Date range: {data_info['date_range'][0]} to {data_info['date_range'][1]}")
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"{'-'*60}")
        
        for model_name, result in results['model_results'].items():
            if result['success']:
                if isinstance(result['metrics'], dict) and 'RMSE' in result['metrics']:
                    # Univariate metrics
                    print(f"{model_name:20} RMSE: {result['metrics']['RMSE']:.4f} "
                          f"MAPE: {result['metrics']['MAPE']:.2f}%")
                elif isinstance(result['metrics'], dict):
                    # Multivariate metrics
                    print(f"{model_name:20} (multivariate)")
                    for ticker, metrics in result['metrics'].items():
                        print(f"  {ticker:18} RMSE: {metrics['RMSE']:.4f} "
                              f"MAPE: {metrics['MAPE']:.2f}%")
            else:
                print(f"{model_name:20} FAILED: {result['error'][:50]}...")
        
        # Print summary
        summary = results['summary']
        if 'best_model' in summary and summary['best_model']:
            print(f"\nBEST MODEL: {summary['best_model']}")
            print(f"Best RMSE: {summary['best_rmse']:.4f}")
            print(f"Best MAPE: {summary['best_mape']:.2f}%")


def main():
    """Main function demonstrating backtesting framework."""
    print("Portfolio Forecaster - Backtesting Framework")
    print("=" * 50)
    
    # Load data
    data_path = os.path.join(project_root, 'data', 'mock_stock_data.csv')
    data = load_data(data_path)
    
    # Create backtester
    backtester = SimpleBacktester(test_size=0.2)
    
    # Add all model types
    backtester.add_baseline_models()
    backtester.add_var_model()
    backtester.add_deep_learning_models(max_epochs=3)  # Quick training for demo
    
    print(f"Added {len(backtester.models)} models: {list(backtester.models.keys())}")
    
    # Test 1: Univariate backtest (AAPL)
    print(f"\n{'='*50}")
    print("TEST 1: UNIVARIATE BACKTEST (AAPL)")
    print(f"{'='*50}")
    
    aapl_data = get_ticker_data(data, 'AAPL', 'close')
    aapl_results = backtester.run_backtest(aapl_data, 'AAPL')
    backtester.print_results(aapl_results)
    
    # Test 2: Multivariate backtest (AAPL + MSFT)
    print(f"\n{'='*50}")
    print("TEST 2: MULTIVARIATE BACKTEST (AAPL + MSFT)")
    print(f"{'='*50}")
    
    from data_loader import get_multiple_tickers_data
    multi_data = get_multiple_tickers_data(data, ['AAPL', 'MSFT'], 'close')
    multi_results = backtester.run_multivariate_backtest(multi_data, ['AAPL', 'MSFT'])
    backtester.print_results(multi_results)
    
    print(f"\n{'='*50}")
    print("BACKTESTING COMPLETED!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()