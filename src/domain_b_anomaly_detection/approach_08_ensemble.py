"""
Approach 8: Ensemble Methods for Anomaly Detection

Philosophy: Combine multiple detectors for robustness.
- Reduces individual method biases
- Better coverage of anomaly types
"""

from typing import Dict, List, Any, Optional
import numpy as np

from ..core.base_model import BaseApproach
from .approach_01_statistical import StatisticalAnomalyDetector
from .approach_02_distance_based import KNNAnomalyDetector, LOFDetector
from .approach_03_tree_based import IsolationForestDetector


class EnsembleAnomalyDetector(BaseApproach):
    """Ensemble of multiple anomaly detectors."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Ensemble", config)
        
        self.combination = config.get('combination', 'average') if config else 'average'
        self.threshold = config.get('threshold', 0.5) if config else 0.5
        
        # Initialize detectors
        self.detectors = [
            StatisticalAnomalyDetector({'method': 'zscore'}),
            KNNAnomalyDetector({'k': 5}),
            LOFDetector({'n_neighbors': 20}),
            IsolationForestDetector({'n_estimators': 100})
        ]
        
        self.weights = None
        
        self.metrics.interpretability_score = 0.5
        self.metrics.maintenance_complexity = 0.7
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        
        # Train all detectors
        for detector in self.detectors:
            detector.train(X_train, y_train, X_val, y_val)
        
        # Learn weights if validation data available
        if X_val is not None and y_val is not None:
            self._learn_weights(X_val, y_val)
        else:
            self.weights = np.ones(len(self.detectors)) / len(self.detectors)
        
        self.is_trained = True
    
    def _learn_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Learn weights based on validation performance."""
        from sklearn.metrics import f1_score
        
        scores = []
        for detector in self.detectors:
            y_pred = detector.predict(X_val)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            scores.append(f1 + 0.01)  # Add small value to avoid zero weights
        
        self.weights = np.array(scores) / sum(scores)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores > self.threshold).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        all_scores = np.zeros((len(X), len(self.detectors)))
        
        for i, detector in enumerate(self.detectors):
            all_scores[:, i] = detector.score(X)
        
        if self.combination == 'average':
            combined = np.average(all_scores, axis=1, weights=self.weights)
        elif self.combination == 'max':
            combined = np.max(all_scores, axis=1)
        elif self.combination == 'vote':
            votes = (all_scores > 0.5).astype(float)
            combined = np.average(votes, axis=1, weights=self.weights)
        else:
            combined = np.mean(all_scores, axis=1)
        
        return combined
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Combine multiple detection strategies for robustness',
            'inductive_bias': 'Different methods catch different anomaly types',
            'strengths': 'More robust, better coverage, reduced false positives',
            'weaknesses': 'Slower, more complex, may average out strong signals',
            'best_for': 'Production systems requiring high reliability'
        }
    
    def get_model_size(self) -> float:
        return sum(d.get_model_size() for d in self.detectors)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        wrong = y_test != y_pred
        
        # Get individual detector predictions for analysis
        individual_preds = np.zeros((len(X_test), len(self.detectors)))
        for i, detector in enumerate(self.detectors):
            individual_preds[:, i] = detector.predict(X_test)
        
        for idx in np.where(wrong)[0][:n_cases]:
            failures.append({
                'index': int(idx),
                'type': 'false_positive' if y_pred[idx] == 1 else 'false_negative',
                'individual_predictions': individual_preds[idx].tolist(),
                'detector_names': [d.name for d in self.detectors]
            })
        return failures