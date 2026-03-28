"""
Synthetic data generator for Time Series Forecasting domain.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TimeSeriesComponents:
    """Components of a time series."""
    trend: np.ndarray
    seasonality: np.ndarray
    noise: np.ndarray
    combined: np.ndarray


class TimeSeriesGenerator:
    """Generate synthetic time series data."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate_series(
        self,
        n_samples: int = 1000,
        trend_type: str = 'linear',
        seasonal_periods: List[int] = [7, 365],
        noise_std: float = 0.1,
        n_features: int = 1
    ) -> np.ndarray:
        """Generate a single time series with multiple components."""
        
        t = np.arange(n_samples)
        series = np.zeros((n_samples, n_features))
        
        for f in range(n_features):
            # Trend component
            if trend_type == 'linear':
                slope = self.rng.uniform(-0.01, 0.01)
                trend = slope * t
            elif trend_type == 'quadratic':
                a = self.rng.uniform(-0.0001, 0.0001)
                b = self.rng.uniform(-0.01, 0.01)
                trend = a * t**2 + b * t
            elif trend_type == 'exponential':
                rate = self.rng.uniform(0.0001, 0.001)
                trend = np.exp(rate * t) - 1
            else:
                trend = np.zeros(n_samples)
            
            # Seasonal components
            seasonality = np.zeros(n_samples)
            for period in seasonal_periods:
                amplitude = self.rng.uniform(0.5, 2.0)
                phase = self.rng.uniform(0, 2 * np.pi)
                seasonality += amplitude * np.sin(2 * np.pi * t / period + phase)
            
            # Noise
            noise = self.rng.randn(n_samples) * noise_std
            
            # Combine
            series[:, f] = trend + seasonality + noise
            
            # Add some level shifts
            n_shifts = self.rng.randint(0, 3)
            for _ in range(n_shifts):
                shift_point = self.rng.randint(n_samples // 4, 3 * n_samples // 4)
                shift_amount = self.rng.uniform(-1, 1)
                series[shift_point:, f] += shift_amount
        
        return series
    
    def generate_multivariate(
        self,
        n_samples: int = 1000,
        n_series: int = 5,
        correlation: float = 0.5
    ) -> np.ndarray:
        """Generate correlated multivariate time series."""
        
        # Generate independent series
        independent = np.zeros((n_samples, n_series))
        for i in range(n_series):
            independent[:, i] = self.generate_series(n_samples, n_features=1).flatten()
        
        # Add correlation
        corr_matrix = np.eye(n_series) * (1 - correlation) + correlation
        L = np.linalg.cholesky(corr_matrix)
        correlated = independent @ L.T
        
        return correlated
    
    def generate_with_anomalies(
        self,
        n_samples: int = 1000,
        anomaly_ratio: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate series with labeled anomalies."""
        
        series = self.generate_series(n_samples, n_features=1).flatten()
        anomaly_labels = np.zeros(n_samples, dtype=int)
        
        n_anomalies = int(n_samples * anomaly_ratio)
        anomaly_indices = self.rng.choice(
            range(10, n_samples - 10), n_anomalies, replace=False
        )
        
        for idx in anomaly_indices:
            anomaly_type = self.rng.choice(['spike', 'dip', 'shift'])
            
            if anomaly_type == 'spike':
                series[idx] += self.rng.uniform(3, 5) * np.std(series)
            elif anomaly_type == 'dip':
                series[idx] -= self.rng.uniform(3, 5) * np.std(series)
            elif anomaly_type == 'shift':
                series[idx:idx+5] += self.rng.uniform(2, 3) * np.std(series)
                anomaly_labels[idx:idx+5] = 1
            
            anomaly_labels[idx] = 1
        
        return series, anomaly_labels


def create_timeseries_dataset(
    n_samples: int = 2000,
    n_features: int = 1,
    forecast_horizon: int = 24,
    lookback: int = 168,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict:
    """Create train/val/test split for time series forecasting."""
    
    generator = TimeSeriesGenerator(seed)
    
    # Generate series
    series = generator.generate_series(
        n_samples=n_samples,
        trend_type='linear',
        seasonal_periods=[24, 168],  # Daily and weekly
        noise_std=0.2,
        n_features=n_features
    )
    
    # Create sliding windows
    X, y = [], []
    for i in range(lookback, len(series) - forecast_horizon):
        X.append(series[i - lookback:i])
        y.append(series[i:i + forecast_horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split
    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)
    
    train_X, train_y = X[:n_train], y[:n_train]
    val_X, val_y = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    test_X, test_y = X[n_train + n_val:], y[n_train + n_val:]
    
    # Also provide raw series for statistical methods
    train_end = n_train + lookback
    val_end = train_end + n_val
    
    return {
        'train': {'X': train_X, 'y': train_y},
        'val': {'X': val_X, 'y': val_y},
        'test': {'X': test_X, 'y': test_y},
        'raw_series': series,
        'train_series': series[:train_end],
        'val_series': series[train_end:val_end],
        'test_series': series[val_end:],
        'lookback': lookback,
        'forecast_horizon': forecast_horizon,
        'n_features': n_features
    }


def add_noise_to_series(series: np.ndarray, noise_level: float) -> np.ndarray:
    """Add noise to time series."""
    if noise_level == 0:
        return series
    noise = np.random.randn(*series.shape) * noise_level * np.std(series)
    return series + noise


if __name__ == "__main__":
    dataset = create_timeseries_dataset(n_samples=1000, lookback=48, forecast_horizon=12)
    print(f"Train X shape: {dataset['train']['X'].shape}")
    print(f"Train y shape: {dataset['train']['y'].shape}")
    print(f"Test X shape: {dataset['test']['X'].shape}")