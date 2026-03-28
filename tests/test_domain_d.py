"""
Tests for Domain D: Time Series Forecasting
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain_d_time_series.data_generator import (
    TimeSeriesGenerator, create_timeseries_dataset
)
from src.domain_d_time_series.approach_01_statistical import ARIMAForecaster
from src.domain_d_time_series.approach_03_tree_based import TreeBasedForecaster
from src.core.metrics import compute_timeseries_metrics


class TestDataGenerator:
    """Test time series data generation."""
    
    def test_series_generation(self):
        generator = TimeSeriesGenerator(seed=42)
        series = generator.generate_series(n_samples=500, n_features=1)
        
        assert series.shape == (500, 1)
        assert not np.isnan(series).any()
    
    def test_multivariate_generation(self):
        generator = TimeSeriesGenerator(seed=42)
        series = generator.generate_multivariate(n_samples=200, n_series=3)
        
        assert series.shape == (200, 3)
    
    def test_dataset_creation(self):
        dataset = create_timeseries_dataset(
            n_samples=500, lookback=24, forecast_horizon=12
        )
        
        assert 'train' in dataset
        assert dataset['train']['X'].shape[1] == 24  # Lookback
        assert dataset['train']['y'].shape[1] == 12  # Horizon


class TestARIMAForecaster:
    """Test ARIMA forecaster."""
    
    def test_training(self):
        dataset = create_timeseries_dataset(n_samples=300, lookback=24, forecast_horizon=12)
        
        model = ARIMAForecaster({'order': (1, 1, 1)})
        model.train(
            dataset['train']['X'], 
            dataset['train']['y'],
            train_series=dataset['train_series'],
            forecast_horizon=12
        )
        
        assert model.is_trained
    
    def test_prediction_shape(self):
        dataset = create_timeseries_dataset(n_samples=300, lookback=24, forecast_horizon=12)
        
        model = ARIMAForecaster()
        model.train(
            dataset['train']['X'],
            dataset['train']['y'],
            train_series=dataset['train_series'],
            forecast_horizon=12
        )
        
        predictions = model.predict(dataset['test']['X'][:5])
        
        assert predictions.shape[0] == 5
        assert predictions.shape[1] == 12


class TestTreeBasedForecaster:
    """Test tree-based forecaster."""
    
    def test_training(self):
        dataset = create_timeseries_dataset(n_samples=300, lookback=24, forecast_horizon=12)
        
        model = TreeBasedForecaster({'n_estimators': 10})
        model.train(
            dataset['train']['X'],
            dataset['train']['y'],
            forecast_horizon=12
        )
        
        assert model.is_trained
        assert model.model is not None
    
    def test_prediction(self):
        dataset = create_timeseries_dataset(n_samples=300, lookback=24, forecast_horizon=12)
        
        model = TreeBasedForecaster({'n_estimators': 10})
        model.train(
            dataset['train']['X'],
            dataset['train']['y'],
            forecast_horizon=12
        )
        
        predictions = model.predict(dataset['test']['X'][:5])
        
        assert predictions.shape == dataset['test']['y'][:5].shape


class TestTimeSeriesMetrics:
    """Test time series metrics."""
    
    def test_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        metrics = compute_timeseries_metrics(y_true, y_pred)
        
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
    
    def test_mase_calculation(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        naive = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # Shift by 1
        
        metrics = compute_timeseries_metrics(y_true, y_pred, naive)
        
        assert 'mase' in metrics
        assert metrics['mase'] < 1  # Better than naive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])