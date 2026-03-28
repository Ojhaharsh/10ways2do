"""
Approach 9: Hybrid Time Series Methods

Philosophy: Combine statistical decomposition with ML.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from ..core.base_model import BaseApproach
from .approach_07_neural_prophet import ProphetStyleForecaster
from .approach_04_rnn_lstm import LSTMForecaster


class HybridForecaster(BaseApproach):
    """Hybrid statistical + ML forecaster."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Hybrid (Decomp + ML)", config)
        
        self.decomposer = ProphetStyleForecaster()
        self.residual_model = LSTMForecaster({'epochs': 20, 'hidden_dim': 32})
        
        self.forecast_horizon = 24
        
        self.metrics.interpretability_score = 0.6
        self.metrics.maintenance_complexity = 0.7
    
    def train(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              train_series: np.ndarray = None, forecast_horizon: int = 24) -> None:
        
        self.forecast_horizon = forecast_horizon
        
        # Train decomposition model
        self.decomposer.train(X_train, y_train, X_val, y_val, train_series, forecast_horizon)
        
        # Get residuals
        if X_train is not None:
            decomp_pred = self.decomposer.predict(X_train)
            residuals = y_train - decomp_pred
            
            # Train ML model on residuals
            try:
                self.residual_model.train(X_train, residuals, forecast_horizon=forecast_horizon)
            except Exception as e:
                print(f"Residual model training failed: {e}")
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Decomposition prediction
        decomp_pred = self.decomposer.predict(X)
        
        # Residual prediction
        try:
            residual_pred = self.residual_model.predict(X)
            return decomp_pred + residual_pred
        except:
            return decomp_pred
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Use statistical methods for structure, ML for residuals',
            'inductive_bias': 'Combination of interpretable structure and flexible learning',
            'strengths': 'Best of both worlds, interpretable trend/seasonality',
            'weaknesses': 'Complex pipeline, error propagation',
            'best_for': 'When interpretability and accuracy both matter'
        }
    
    def get_model_size(self) -> float:
        return self.decomposer.get_model_size() + self.residual_model.get_model_size()
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        errors = np.mean((y_test - y_pred) ** 2, axis=(1, 2))
        worst_indices = np.argsort(-errors)[:n_cases]
        for idx in worst_indices:
            failures.append({'index': int(idx), 'mse': float(errors[idx])})
        return failures