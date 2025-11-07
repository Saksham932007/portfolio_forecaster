"""Deep learning forecasting models using PyTorch Forecasting framework."""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import os


class DeepLearningForecaster:
    """
    Base class for deep learning time series forecasting using PyTorch Forecasting.
    
    This class provides a unified interface for various deep learning forecasting
    models including DeepAR, Temporal Fusion Transformer, and others.
    """
    
    def __init__(self, max_prediction_length: int = 30, max_encoder_length: int = 60):
        """
        Initialize the deep learning forecaster.
        
        Args:
            max_prediction_length: Maximum prediction horizon
            max_encoder_length: Maximum length of input sequence
        """
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.training_cutoff = None
        self.time_series_dataset = None
        self.validation_dataset = None
        self.model = None
        self.trainer = None
        self.is_fitted = False
        
        # Check if required packages are available
        self._check_dependencies()
    
    def _check_dependencies(self) -> bool:
        """
        Check if required deep learning packages are available.
        
        Returns:
            bool: True if all dependencies are available
        """
        try:
            import torch
            import pytorch_lightning
            import pytorch_forecasting
            return True
        except ImportError as e:
            self.dependencies_available = False
            warnings.warn(f"Deep learning dependencies not available: {e}")
            return False
    
    def create_timeseries_dataset(self, data: pd.DataFrame, target_column: str, 
                                 group_id_column: str = 'ticker',
                                 time_idx_column: str = 'time_idx',
                                 prediction_horizon: Optional[int] = None) -> 'TimeSeriesDataSet':
        """
        Convert pandas DataFrame to PyTorch Forecasting TimeSeriesDataSet.
        
        Args:
            data: Input DataFrame with time series data
            target_column: Name of the target variable column
            group_id_column: Name of the group identifier column
            time_idx_column: Name of the time index column
            prediction_horizon: Prediction horizon override
            
        Returns:
            TimeSeriesDataSet: Prepared dataset for training
        """
        if not self._check_dependencies():
            raise ImportError("PyTorch Forecasting dependencies not available")
        
        try:
            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
        except ImportError:
            raise ImportError("pytorch_forecasting not available")
        
        # Use provided horizon or default
        prediction_length = prediction_horizon or self.max_prediction_length
        
        # Prepare data format
        prepared_data = data.copy()
        
        # Ensure time_idx column exists
        if time_idx_column not in prepared_data.columns:
            if isinstance(prepared_data.index, pd.DatetimeIndex):
                # Create time index from datetime index
                prepared_data = prepared_data.reset_index()
                prepared_data[time_idx_column] = range(len(prepared_data))
            else:
                prepared_data[time_idx_column] = range(len(prepared_data))
        
        # Ensure group_id column exists
        if group_id_column not in prepared_data.columns:
            prepared_data[group_id_column] = 'default_group'
        
        # Set training cutoff (80% of data)
        max_time_idx = prepared_data[time_idx_column].max()
        self.training_cutoff = max_time_idx - prediction_length
        
        # Identify categorical and continuous variables
        categorical_vars = [group_id_column]
        continuous_vars = [col for col in prepared_data.columns 
                          if col not in [time_idx_column, target_column, group_id_column]
                          and prepared_data[col].dtype in ['float64', 'int64']]
        
        # Create TimeSeriesDataSet
        self.time_series_dataset = TimeSeriesDataSet(
            prepared_data[prepared_data[time_idx_column] <= self.training_cutoff],
            time_idx=time_idx_column,
            target=target_column,
            group_ids=[group_id_column],
            min_encoder_length=max(1, self.max_encoder_length // 2),
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=prediction_length,
            static_categoricals=categorical_vars,
            time_varying_known_categoricals=[],
            time_varying_known_reals=[],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=continuous_vars + [target_column],
            target_normalizer=GroupNormalizer(groups=categorical_vars, transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        # Create validation dataset
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.time_series_dataset, 
            prepared_data, 
            predict=True, 
            stop_randomization=True
        )
        
        return self.time_series_dataset
    
    def build_deepar_model(self, train_dataset: 'TimeSeriesDataSet') -> 'DeepAR':
        """
        Build a DeepAR model from the training dataset.
        
        Args:
            train_dataset: Training TimeSeriesDataSet
            
        Returns:
            DeepAR: Configured DeepAR model
        """
        try:
            from pytorch_forecasting import DeepAR
        except ImportError:
            raise ImportError("pytorch_forecasting.DeepAR not available")
        
        # Configure DeepAR model
        model = DeepAR.from_dataset(
            train_dataset,
            learning_rate=0.03,
            hidden_size=30,
            rnn_layers=2,
            dropout=0.1,
            loss=None,  # Use default loss
        )
        
        return model
    
    def build_tft_model(self, train_dataset: 'TimeSeriesDataSet') -> 'TemporalFusionTransformer':
        """
        Build a Temporal Fusion Transformer model from the training dataset.
        
        Args:
            train_dataset: Training TimeSeriesDataSet
            
        Returns:
            TemporalFusionTransformer: Configured TFT model
        """
        try:
            from pytorch_forecasting import TemporalFusionTransformer
        except ImportError:
            raise ImportError("pytorch_forecasting.TemporalFusionTransformer not available")
        
        # Configure TFT model
        model = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # 7 quantiles by default
            loss=None,  # Use default quantile loss
            reduce_on_plateau_patience=4,
        )
        
        return model
    
    def train_model(self, model, train_dataloader, val_dataloader, 
                   max_epochs: int = 30, gpus: Optional[int] = None) -> 'pytorch_lightning.Trainer':
        """
        Train a PyTorch Forecasting model.
        
        Args:
            model: The forecasting model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            max_epochs: Maximum number of training epochs
            gpus: Number of GPUs to use (None for CPU)
            
        Returns:
            pytorch_lightning.Trainer: Trained model trainer
        """
        try:
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
        except ImportError:
            raise ImportError("pytorch_lightning not available")
        
        # Configure trainer
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, verbose=False, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        
        # Set up trainer
        trainer_kwargs = {
            "max_epochs": max_epochs,
            "callbacks": callbacks,
            "enable_model_summary": True,
            "enable_progress_bar": True,
            "logger": False,  # Disable logging for cleaner output
        }
        
        # Add GPU configuration if specified
        if gpus is not None and gpus > 0:
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = gpus
        
        self.trainer = pl.Trainer(**trainer_kwargs)
        
        # Train the model
        self.trainer.fit(
            model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader
        )
        
        self.model = model
        self.is_fitted = True
        
        return self.trainer
    
    def predict(self, model, dataloader, return_index: bool = True, 
               return_decoder_lengths: bool = True) -> Union[np.ndarray, Tuple]:
        """
        Generate predictions from a trained model.
        
        Args:
            model: Trained forecasting model
            dataloader: Data loader for prediction
            return_index: Whether to return prediction index
            return_decoder_lengths: Whether to return decoder lengths
            
        Returns:
            Predictions (and optionally index and decoder lengths)
        """
        if not self.is_fitted:
            warnings.warn("Model not trained, predictions may be unreliable")
        
        # Generate predictions
        predictions = model.predict(
            dataloader, 
            return_index=return_index,
            return_decoder_lengths=return_decoder_lengths
        )
        
        return predictions
    
    def create_dataloaders(self, batch_size: int = 64, num_workers: int = 0) -> Tuple:
        """
        Create data loaders for training and validation.
        
        Args:
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            
        Returns:
            Tuple: (train_dataloader, val_dataloader)
        """
        if self.time_series_dataset is None:
            raise ValueError("TimeSeriesDataSet not created. Call create_timeseries_dataset first.")
        
        train_dataloader = self.time_series_dataset.to_dataloader(
            train=True, 
            batch_size=batch_size, 
            num_workers=num_workers
        )
        
        val_dataloader = self.validation_dataset.to_dataloader(
            train=False, 
            batch_size=batch_size * 10, 
            num_workers=num_workers
        )
        
        return train_dataloader, val_dataloader
    
    def fit_and_predict_deepar(self, data: pd.DataFrame, target_column: str,
                              group_id_column: str = 'ticker',
                              max_epochs: int = 10) -> Tuple[np.ndarray, Dict]:
        """
        Complete workflow: fit DeepAR model and generate predictions.
        
        Args:
            data: Input DataFrame
            target_column: Target variable column name
            group_id_column: Group identifier column name
            max_epochs: Maximum training epochs
            
        Returns:
            Tuple: (predictions, metadata)
        """
        # Create dataset
        train_dataset = self.create_timeseries_dataset(data, target_column, group_id_column)
        
        # Build model
        model = self.build_deepar_model(train_dataset)
        
        # Create data loaders
        train_dataloader, val_dataloader = self.create_dataloaders()
        
        # Train model
        trainer = self.train_model(model, train_dataloader, val_dataloader, max_epochs)
        
        # Generate predictions
        predictions = self.predict(model, val_dataloader)
        
        # Metadata
        metadata = {
            'model_type': 'DeepAR',
            'prediction_length': self.max_prediction_length,
            'encoder_length': self.max_encoder_length,
            'training_epochs': trainer.current_epoch,
            'dataset_shape': train_dataset.index.shape
        }
        
        return predictions, metadata
    
    def fit_and_predict_tft(self, data: pd.DataFrame, target_column: str,
                           group_id_column: str = 'ticker',
                           max_epochs: int = 10) -> Tuple[np.ndarray, Dict]:
        """
        Complete workflow: fit TFT model and generate predictions.
        
        Args:
            data: Input DataFrame
            target_column: Target variable column name
            group_id_column: Group identifier column name
            max_epochs: Maximum training epochs
            
        Returns:
            Tuple: (predictions, metadata)
        """
        # Create dataset
        train_dataset = self.create_timeseries_dataset(data, target_column, group_id_column)
        
        # Build model
        model = self.build_tft_model(train_dataset)
        
        # Create data loaders
        train_dataloader, val_dataloader = self.create_dataloaders()
        
        # Train model
        trainer = self.train_model(model, train_dataloader, val_dataloader, max_epochs)
        
        # Generate predictions
        predictions = self.predict(model, val_dataloader)
        
        # Metadata
        metadata = {
            'model_type': 'TFT',
            'prediction_length': self.max_prediction_length,
            'encoder_length': self.max_encoder_length,
            'training_epochs': trainer.current_epoch,
            'dataset_shape': train_dataset.index.shape
        }
        
        return predictions, metadata


# Mock implementations for when PyTorch Forecasting is not available
class MockDeepLearningForecaster(DeepLearningForecaster):
    """
    Mock implementation of DeepLearningForecaster for when dependencies are not available.
    """
    
    def __init__(self, *args, **kwargs):
        self.max_prediction_length = kwargs.get('max_prediction_length', 30)
        self.max_encoder_length = kwargs.get('max_encoder_length', 60)
        self.dependencies_available = False
        warnings.warn("PyTorch Forecasting not available, using mock implementation")
    
    def fit_and_predict_deepar(self, data: pd.DataFrame, target_column: str, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Mock DeepAR implementation using simple forecasting."""
        # Generate simple predictions (naive forecast)
        target_series = data[target_column]
        last_value = target_series.iloc[-1]
        predictions = np.full(self.max_prediction_length, last_value)
        
        metadata = {
            'model_type': 'Mock_DeepAR',
            'prediction_length': self.max_prediction_length,
            'note': 'PyTorch Forecasting not available, using naive forecast'
        }
        
        return predictions, metadata
    
    def fit_and_predict_tft(self, data: pd.DataFrame, target_column: str, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Mock TFT implementation using simple forecasting."""
        # Generate simple predictions (moving average)
        target_series = data[target_column]
        window_size = min(10, len(target_series))
        last_values = target_series.iloc[-window_size:]
        avg_value = last_values.mean()
        predictions = np.full(self.max_prediction_length, avg_value)
        
        metadata = {
            'model_type': 'Mock_TFT',
            'prediction_length': self.max_prediction_length,
            'note': 'PyTorch Forecasting not available, using moving average forecast'
        }
        
        return predictions, metadata


# Factory function to create the appropriate forecaster
def create_deep_learning_forecaster(**kwargs) -> Union[DeepLearningForecaster, MockDeepLearningForecaster]:
    """
    Create a DeepLearningForecaster instance, falling back to mock if dependencies unavailable.
    
    Returns:
        DeepLearningForecaster or MockDeepLearningForecaster
    """
    try:
        import torch
        import pytorch_lightning
        import pytorch_forecasting
        return DeepLearningForecaster(**kwargs)
    except ImportError:
        return MockDeepLearningForecaster(**kwargs)


# Example usage and testing functions
if __name__ == "__main__":
    import os
    import sys
    
    # Add utils to path for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    
    try:
        from data_loader import load_data, get_ticker_data
        from evaluation import calculate_all_metrics
        
        # Get the path to the data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(current_dir), 'data', 'mock_stock_data.csv')
        
        print("Testing Deep Learning Forecaster...")
        
        # Load data
        data = load_data(data_path)
        
        # Prepare data for deep learning
        aapl_data = data.xs('AAPL', level='ticker').reset_index()
        aapl_data['time_idx'] = range(len(aapl_data))
        aapl_data['ticker'] = 'AAPL'
        
        print(f"Data shape: {aapl_data.shape}")
        print(f"Columns: {aapl_data.columns.tolist()}")
        print(f"Date range: {aapl_data['date'].min()} to {aapl_data['date'].max()}")
        
        # Create forecaster
        forecaster = create_deep_learning_forecaster(
            max_prediction_length=20,
            max_encoder_length=40
        )
        
        print(f"\nUsing forecaster: {type(forecaster).__name__}")
        
        # Test DeepAR
        print("\n1. Testing DeepAR...")
        try:
            deepar_predictions, deepar_metadata = forecaster.fit_and_predict_deepar(
                aapl_data, 
                target_column='close',
                max_epochs=3  # Quick test
            )
            
            print(f"DeepAR predictions shape: {deepar_predictions.shape}")
            print(f"DeepAR metadata: {deepar_metadata}")
            
        except Exception as e:
            print(f"DeepAR test failed: {e}")
        
        # Test TFT
        print("\n2. Testing TFT...")
        try:
            tft_predictions, tft_metadata = forecaster.fit_and_predict_tft(
                aapl_data, 
                target_column='close',
                max_epochs=3  # Quick test
            )
            
            print(f"TFT predictions shape: {tft_predictions.shape}")
            print(f"TFT metadata: {tft_metadata}")
            
        except Exception as e:
            print(f"TFT test failed: {e}")
        
        print("\nDeep learning forecaster tests completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()