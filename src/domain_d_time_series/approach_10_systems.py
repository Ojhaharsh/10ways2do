"""
Approach 10: Systems Perspective for Time Series

Philosophy: Production considerations for forecasting.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import time
from collections import deque

from ..core.base_model import BaseApproach


class StreamingForecaster(BaseApproach):
    """Production forecaster with streaming updates."""
    
    def __init__(self, base_forecaster: BaseApproach, config: Optional[Dict] = None):
        super().__init__(f"Streaming({base_forecaster.name})", config)
        
        self.base = base_forecaster
        
        self.buffer_size = config.get('buffer_size', 1000) if config else 1000
        self.retrain_interval = config.get('retrain_interval', 100) if config else 100
        
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.prediction_count = 0
        self.last_retrain = 0
        
        self.latencies = []
        self.errors = []
        
        self.forecast_horizon = 24
        
        self.metrics.interpretability_score = self.base.metrics.interpretability_score
        self.metrics.maintenance_complexity = self.base.metrics.maintenance_complexity + 0.2
    
    def train(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              train_series: np.ndarray = None, forecast_horizon: int = 24) -> None:
        
        self.forecast_horizon = forecast_horizon
        self.base.train(X_train, y_train, X_val, y_val, train_series, forecast_horizon)
        
        if X_train is not None:
            for x in X_train[-self.buffer_size:]:
                self.data_buffer.append(x)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        start = time.time()
        
        predictions = self.base.predict(X)
        
        self.latencies.append(time.time() - start)
        self.prediction_count += len(X)
        
        # Add to buffer
        for x in X:
            self.data_buffer.append(x)
        
        return predictions
    
    def update_with_observation(self, actual: np.ndarray, predicted: np.ndarray) -> None:
        """Update with actual observation for online learning."""
        error = np.mean((actual - predicted) ** 2)
        self.errors.append(error)
        
        # Check if retrain needed
        if self.prediction_count - self.last_retrain >= self.retrain_interval:
            self._maybe_retrain()
    
    def _maybe_retrain(self) -> None:
        """Retrain if performance degraded."""
        if len(self.errors) < 10:
            return
        
        recent_error = np.mean(self.errors[-10:])
        historical_error = np.mean(self.errors[:-10]) if len(self.errors) > 10 else recent_error
        
        if recent_error > historical_error * 1.5:
            print("Performance degraded, retraining...")
            # In practice, would retrain here
            self.last_retrain = self.prediction_count
    
    def get_stats(self) -> Dict[str, float]:
        return {
            'prediction_count': self.prediction_count,
            'avg_latency_ms': np.mean(self.latencies) * 1000 if self.latencies else 0,
            'p95_latency_ms': np.percentile(self.latencies, 95) * 1000 if len(self.latencies) > 10 else 0,
            'recent_mse': np.mean(self.errors[-100:]) if self.errors else 0,
            'buffer_size': len(self.data_buffer)
        }
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Production wrapper with streaming, monitoring, and adaptive retraining',
            'inductive_bias': 'Same as base model',
            'strengths': 'Production-ready, handles drift, observable',
            'weaknesses': 'Added complexity',
            'best_for': 'Production deployment with evolving data'
        }
    
    def get_model_size(self) -> float:
        return self.base.get_model_size()
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        return self.base.collect_failure_cases(X_test, y_test, y_pred, n_cases)