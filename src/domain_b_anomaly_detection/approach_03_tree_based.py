"""
Approach 3: Tree-Based Anomaly Detection

Philosophy: Anomalies are easier to isolate.
- Isolation Forest
- Extended Isolation Forest
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle

from ..core.base_model import BaseApproach


class IsolationForestDetector(BaseApproach):
    """Isolation Forest anomaly detector."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Tree-Based (Isolation Forest)", config)
        
        self.n_estimators = config.get('n_estimators', 100) if config else 100
        self.contamination = config.get('contamination', 0.05) if config else 0.05
        self.max_samples = config.get('max_samples', 'auto') if config else 'auto'
        
        self.model = None
        self.scaler = StandardScaler()
        
        self.metrics.interpretability_score = 0.6
        self.metrics.maintenance_complexity = 0.3
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return (predictions == -1).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        scores = -self.model.score_samples(X_scaled)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Anomalies require fewer splits to isolate in random trees',
            'inductive_bias': 'Anomalies are few and different from normal points',
            'strengths': 'Linear complexity, handles high dimensions, no distance metric',
            'weaknesses': 'Random splits may miss axis-aligned anomalies, less local',
            'best_for': 'High-dimensional data, large datasets, fast inference needed'
        }
    
    def get_model_size(self) -> float:
        if self.model is None:
            return 0.0
        return len(pickle.dumps(self.model)) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        wrong = y_test != y_pred
        scores = self.score(X_test)
        
        for idx in np.where(wrong)[0][:n_cases]:
            failures.append({
                'index': int(idx),
                'type': 'false_positive' if y_pred[idx] == 1 else 'false_negative',
                'isolation_score': float(scores[idx]),
                'reason': 'Point was too easy/hard to isolate'
            })
        return failures