"""
Approach 8: Ensemble Methods for Time Series

Philosophy: Combine multiple forecasters.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from ..core.base_model import BaseApproach
from .approach_01_statistical import ARIMAForecaster
from .approach_02_exponential_smoothing import ExpSmoothingForecaster
from .approach_03_tree_based import TreeBasedForecaster


class EnsembleForecaster(BaseApproach):
    """Ensemble of time series forecasters."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Ensemble", config)
        
        self.models = [
            ExpSmoothingForecaster(),
            TreeBasedForecaster({'n_estimators': 50}),
        ]
        
        self.weights = None
        self.forecast_horizon = 24
        
        self.metrics.interpretability_score = 0.5
        self.metrics.maintenance_complexity = 0.6
    
    def train(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              train_series: np.ndarray = None, forecast_horizon: int = 24) -> None:
        
        self.forecast_horizon = forecast_horizon
        
        for model in self.models:
            try:
                model.train(X_train, y_train, X_val, y_val, train_series, forecast_horizon)
            except Exception as e:
                print(f"Model {model.name} training failed: {e}")
        
        # Learn weights if validation data available
        if X_val is not None and y_val is not None:
            self._learn_weights(X_val, y_val)
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        self.is_trained = True
    
    def _learn_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Learn weights based on validation performance."""
        errors = []
        
        for model in self.models:
            try:
                pred = model.predict(X_val)
                mse = np.mean((y_val - pred) ** 2)
                errors.append(1.0 / (mse + 1e-8))
            except:
                errors.append(0.001)
        
        errors = np.array(errors)
        self.weights = errors / errors.sum()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            try:
                pred = model.predict(X)
                predictions.append(pred * weight)
            except:
                pass
        
        if predictions:
            return np.sum(predictions, axis=0)
        else:
            return np.zeros((len(X), self.forecast_horizon, 1))
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Combine diverse forecasters to reduce error',
            'inductive_bias': 'Different methods capture different patterns',
            'strengths': 'More robust, reduces variance, handles uncertainty',
            'weaknesses': 'Slower, more complex, may average out good predictions',
            'best_for': 'Production systems requiring reliability'
        }
    
    def get_model_size(self) -> float:
        return sum(m.get_model_size() for m in self.models)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        errors = np.mean((y_test - y_pred) ** 2, axis=(1, 2))
        worst_indices = np.argsort(-errors)[:n_cases]
        for idx in worst_indices:
            failures.append({'index': int(idx), 'mse': float(errors[idx])})
        return failures