"""
Approach 3: Tree-Based Methods for Time Series

Philosophy: Use gradient boosting on lagged features.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from ..core.base_model import BaseApproach


class TreeBasedForecaster(BaseApproach):
    """Gradient Boosting / XGBoost forecaster."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Tree-Based (XGBoost)", config)
        
        self.n_estimators = config.get('n_estimators', 100) if config else 100
        self.max_depth = config.get('max_depth', 5) if config else 5
        self.use_xgb = config.get('use_xgb', HAS_XGB) if config else HAS_XGB
        
        self.model = None
        self.forecast_horizon = 24
        
        self.metrics.interpretability_score = 0.6
        self.metrics.maintenance_complexity = 0.4
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Flatten windows and add time features."""
        n_samples = X.shape[0]
        
        # Flatten the lookback window
        flat = X.reshape(n_samples, -1)
        
        # Add statistical features
        features = [
            flat,
            np.mean(X, axis=1),
            np.std(X, axis=1),
            np.min(X, axis=1),
            np.max(X, axis=1),
            X[:, -1, :],  # Last value
            X[:, -7, :] if X.shape[1] >= 7 else X[:, 0, :],  # Value 7 steps ago
        ]
        
        return np.hstack([f.reshape(n_samples, -1) for f in features])
    
    def train(self, X_train: np.ndarray = None, y_train: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              train_series: np.ndarray = None, forecast_horizon: int = 24) -> None:
        
        self.forecast_horizon = forecast_horizon
        
        if X_train is None:
            raise ValueError("Need X_train for tree-based methods")
        
        X_features = self._prepare_features(X_train)
        y_flat = y_train.reshape(len(y_train), -1)
        
        if self.use_xgb and HAS_XGB:
            base_model = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:
            base_model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
        
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_features, y_flat)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_features = self._prepare_features(X)
        predictions = self.model.predict(X_features)
        
        # Reshape to (n_samples, horizon, n_features)
        n_features = X.shape[2] if X.ndim == 3 else 1
        return predictions.reshape(len(X), self.forecast_horizon, n_features)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Learn non-linear mappings from lagged features to future values',
            'inductive_bias': 'Tree splits capture threshold effects and interactions',
            'strengths': 'Handles non-linearity, feature importance, robust',
            'weaknesses': 'No native sequence modeling, needs feature engineering',
            'best_for': 'Complex non-linear relationships, tabular-like features'
        }
    
    def get_model_size(self) -> float:
        if self.model is None:
            return 0.0
        return len(pickle.dumps(self.model)) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        errors = np.mean((y_test - y_pred) ** 2, axis=(1, 2))
        worst_indices = np.argsort(-errors)[:n_cases]
        for idx in worst_indices:
            failures.append({'index': int(idx), 'mse': float(errors[idx])})
        return failures