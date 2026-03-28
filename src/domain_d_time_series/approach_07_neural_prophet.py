"""
Approach 7: Prophet-style Forecasting

Philosophy: Decompose series into trend, seasonality, and holidays.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from ..core.base_model import BaseApproach


class ProphetStyleForecaster(BaseApproach):
    """Prophet-inspired forecaster using decomposition."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Prophet-Style", config)
        
        self.seasonality_periods = config.get('seasonality_periods', [24, 168]) if config else [24, 168]
        
        self.trend_params = None
        self.seasonality_params = {}
        self.train_mean = 0
        self.forecast_horizon = 24
        
        self.metrics.interpretability_score = 0.9
        self.metrics.maintenance_complexity = 0.3
    
    def _fit_trend(self, y: np.ndarray) -> None:
        """Fit linear trend."""
        t = np.arange(len(y))
        self.trend_params = np.polyfit(t, y, 1)
    
    def _fit_seasonality(self, y: np.ndarray, period: int) -> np.ndarray:
        """Fit seasonal component via averaging."""
        seasonal = np.zeros(period)
        counts = np.zeros(period)
        
        for i, val in enumerate(y):
            idx = i % period
            seasonal[idx] += val
            counts[idx] += 1
        
        seasonal /= np.maximum(counts, 1)
        seasonal -= seasonal.mean()
        
        return seasonal
    
    def train(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              train_series: np.ndarray = None, forecast_horizon: int = 24) -> None:
        
        self.forecast_horizon = forecast_horizon
        
        if train_series is not None:
            series = train_series.flatten()
        elif X_train is not None:
            series = X_train.reshape(-1)
        else:
            raise ValueError("Need train_series or X_train")
        
        self.train_mean = series.mean()
        series_centered = series - self.train_mean
        
        # Fit trend
        self._fit_trend(series_centered)
        
        # Remove trend
        t = np.arange(len(series))
        trend = np.polyval(self.trend_params, t)
        detrended = series_centered - trend
        
        # Fit seasonality
        for period in self.seasonality_periods:
            if len(series) >= period * 2:
                self.seasonality_params[period] = self._fit_seasonality(detrended, period)
        
        self.series_len = len(series)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        
        for i in range(len(X)):
            # Start forecasting from end of series
            start_idx = self.series_len + i
            forecast = np.zeros(self.forecast_horizon)
            
            for h in range(self.forecast_horizon):
                t = start_idx + h
                
                # Trend
                pred = np.polyval(self.trend_params, t)
                
                # Seasonality
                for period, seasonal in self.seasonality_params.items():
                    pred += seasonal[t % period]
                
                forecast[h] = pred + self.train_mean
            
            predictions.append(forecast.reshape(-1, 1))
        
        return np.array(predictions)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Decompose series into trend + seasonality + residual',
            'inductive_bias': 'Time series have additive/multiplicative components',
            'strengths': 'Interpretable components, handles missing data, robust',
            'weaknesses': 'Additive assumption, manual seasonality periods',
            'best_for': 'Business forecasting with known seasonality'
        }
    
    def get_model_size(self) -> float:
        return 0.01
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        errors = np.mean((y_test - y_pred) ** 2, axis=(1, 2))
        worst_indices = np.argsort(-errors)[:n_cases]
        for idx in worst_indices:
            failures.append({'index': int(idx), 'mse': float(errors[idx])})
        return failures