"""
Approach 2: Exponential Smoothing Methods

Philosophy: Weighted averages with exponentially decaying weights.
"""

from typing import Dict, List, Any, Optional
import numpy as np

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from ..core.base_model import BaseApproach


class ExpSmoothingForecaster(BaseApproach):
    """Exponential Smoothing (Holt-Winters) forecaster."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Exponential Smoothing", config)
        
        self.seasonal_periods = config.get('seasonal_periods', 24) if config else 24
        self.trend = config.get('trend', 'add') if config else 'add'
        self.seasonal = config.get('seasonal', 'add') if config else 'add'
        
        self.fitted_model = None
        self.train_series = None
        self.forecast_horizon = 24
        
        self.metrics.interpretability_score = 0.9
        self.metrics.maintenance_complexity = 0.2
    
    def train(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              train_series: np.ndarray = None, forecast_horizon: int = 24) -> None:
        
        self.forecast_horizon = forecast_horizon
        
        if train_series is not None:
            series = train_series.flatten() if train_series.ndim > 1 else train_series
        elif X_train is not None:
            series = X_train.reshape(-1)[:500]  # Use subset
        else:
            raise ValueError("Need train_series or X_train")
        
        self.train_series = series
        
        if not HAS_STATSMODELS:
            self.is_trained = True
            return
        
        try:
            model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods
            )
            self.fitted_model = model.fit(optimized=True)
        except Exception as e:
            print(f"Exponential smoothing failed: {e}")
            self.fitted_model = None
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not HAS_STATSMODELS or self.fitted_model is None:
            return np.tile(X[:, -1, :], (1, self.forecast_horizon, 1))
        
        predictions = []
        for i in range(len(X)):
            try:
                forecast = self.fitted_model.forecast(self.forecast_horizon)
                predictions.append(forecast.reshape(-1, 1))
            except:
                predictions.append(np.tile(X[i, -1], (self.forecast_horizon, 1)))
        
        return np.array(predictions)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Recent observations matter more; exponential decay of weights',
            'inductive_bias': 'Smooth trends and repeating seasonal patterns',
            'strengths': 'Simple, handles trend and seasonality, fast',
            'weaknesses': 'Fixed seasonal period, sensitive to outliers',
            'best_for': 'Series with clear trend and seasonality'
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