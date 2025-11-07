"""Data loading utilities for the portfolio forecasting system."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load stock data from CSV file and set up proper indexing.
    
    Args:
        csv_path (str): Path to the CSV file containing stock data
        
    Returns:
        pd.DataFrame: DataFrame with date and ticker as multi-index
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Set multi-index (date, ticker)
    df = df.set_index(['date', 'ticker'])
    
    # Sort the index for better performance
    df = df.sort_index()
    
    return df


def get_ticker_data(data: pd.DataFrame, ticker: str, column: str = 'close') -> pd.Series:
    """
    Extract time series data for a specific ticker and column.
    
    Args:
        data (pd.DataFrame): Multi-indexed DataFrame with (date, ticker)
        ticker (str): Ticker symbol to extract
        column (str): Column to extract (default: 'close')
        
    Returns:
        pd.Series: Time series data for the specified ticker and column
    """
    try:
        # Extract data for the specific ticker
        ticker_data = data.xs(ticker, level='ticker')
        
        # Return the specified column
        return ticker_data[column]
        
    except KeyError:
        raise ValueError(f"Ticker '{ticker}' not found in the data")


def get_multiple_tickers_data(data: pd.DataFrame, tickers: List[str], column: str = 'close') -> pd.DataFrame:
    """
    Extract time series data for multiple tickers.
    
    Args:
        data (pd.DataFrame): Multi-indexed DataFrame with (date, ticker)
        tickers (List[str]): List of ticker symbols to extract
        column (str): Column to extract (default: 'close')
        
    Returns:
        pd.DataFrame: DataFrame with dates as index and tickers as columns
    """
    result_data = {}
    
    for ticker in tickers:
        try:
            ticker_series = get_ticker_data(data, ticker, column)
            result_data[ticker] = ticker_series
        except ValueError:
            print(f"Warning: Ticker '{ticker}' not found in data, skipping...")
            continue
    
    # Create DataFrame with tickers as columns
    result_df = pd.DataFrame(result_data)
    
    return result_df


def split_train_test(data: pd.Series, test_size: float = 0.2) -> tuple:
    """
    Split time series data into train and test sets.
    
    Args:
        data (pd.Series): Time series data to split
        test_size (float): Proportion of data to use for testing (default: 0.2)
        
    Returns:
        tuple: (train_data, test_data)
    """
    split_point = int(len(data) * (1 - test_size))
    
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    return train_data, test_data


def get_data_summary(data: pd.DataFrame) -> Dict:
    """
    Get summary statistics for the loaded data.
    
    Args:
        data (pd.DataFrame): Multi-indexed DataFrame with stock data
        
    Returns:
        Dict: Summary statistics including date range, tickers, and basic stats
    """
    # Get unique tickers
    tickers = data.index.get_level_values('ticker').unique().tolist()
    
    # Get date range
    dates = data.index.get_level_values('date').unique()
    date_range = (dates.min(), dates.max())
    
    # Get basic statistics for each ticker's close price
    close_stats = {}
    for ticker in tickers:
        ticker_data = get_ticker_data(data, ticker, 'close')
        close_stats[ticker] = {
            'mean': ticker_data.mean(),
            'std': ticker_data.std(),
            'min': ticker_data.min(),
            'max': ticker_data.max(),
            'count': len(ticker_data)
        }
    
    summary = {
        'tickers': tickers,
        'num_tickers': len(tickers),
        'date_range': date_range,
        'num_days': len(dates),
        'total_records': len(data),
        'columns': data.columns.tolist(),
        'close_price_stats': close_stats
    }
    
    return summary


# Example usage and testing functions
if __name__ == "__main__":
    # Test the data loading functions
    import os
    
    # Get the path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'mock_stock_data.csv')
    
    try:
        # Load the data
        print("Loading data...")
        data = load_data(data_path)
        print(f"Data shape: {data.shape}")
        print(f"Index levels: {data.index.names}")
        
        # Get summary
        summary = get_data_summary(data)
        print(f"\nData Summary:")
        print(f"Tickers: {summary['tickers']}")
        print(f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
        print(f"Number of days: {summary['num_days']}")
        
        # Test single ticker extraction
        print(f"\nTesting single ticker extraction (AAPL)...")
        aapl_close = get_ticker_data(data, 'AAPL', 'close')
        print(f"AAPL close price series shape: {aapl_close.shape}")
        print(f"First 5 values:\n{aapl_close.head()}")
        
        # Test multiple tickers extraction
        print(f"\nTesting multiple tickers extraction...")
        multi_data = get_multiple_tickers_data(data, ['AAPL', 'MSFT', 'GOOG'], 'close')
        print(f"Multi-ticker data shape: {multi_data.shape}")
        print(f"Columns: {multi_data.columns.tolist()}")
        
        # Test train/test split
        print(f"\nTesting train/test split...")
        train, test = split_train_test(aapl_close)
        print(f"Train size: {len(train)}, Test size: {len(test)}")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")