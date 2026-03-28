"""
Approach 9: Hybrid Anomaly Detection

Philosophy: Multi-stage detection pipeline.
- Different methods for different anomaly types
- Cascaded detection for efficiency
"""

from typing import Dict, List, Any, Optional
import numpy as np

from ..core.base_model import BaseApproach
from .approach_01_statistical import StatisticalAnomalyDetector
from .approach_03_tree_based import IsolationForestDetector
from .approach_04_autoencoder import AutoencoderDetector


class HybridAnomalyDetector(BaseApproach):
    """
    Multi-stage hybrid anomaly detector.
    
    Stage 1: Fast statistical filter for obvious anomalies
    Stage 2: Tree-based for medium complexity
    Stage 3: Deep learning for subtle anomalies
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Hybrid (Multi-Stage)", config)
        
        self.stat_threshold = config.get('stat_threshold', 4.0) if config else 4.0
        
        self.stage1 = StatisticalAnomalyDetector({'threshold': self.stat_threshold})
        self.stage2 = IsolationForestDetector({'contamination': 0.05})
        self.stage3 = AutoencoderDetector({'epochs': 30})
        
        self.stage_weights = [0.3, 0.3, 0.4]
        
        self.metrics.interpretability_score = 0.5
        self.metrics.maintenance_complexity = 0.8
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        
        self.stage1.train(X_train, y_train)
        self.stage2.train(X_train, y_train)
        self.stage3.train(X_train, y_train)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores > 0.5).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        score1 = self.stage1.score(X)
        score2 = self.stage2.score(X)
        score3 = self.stage3.score(X)
        
        combined = (
            self.stage_weights[0] * score1 +
            self.stage_weights[1] * score2 +
            self.stage_weights[2] * score3
        )
        
        return combined
    
    def predict_cascade(self, X: np.ndarray) -> np.ndarray:
        """
        Cascaded prediction for efficiency.
        Only runs later stages if needed.
        """
        results = np.zeros(len(X))
        
        # Stage 1: Fast statistical check
        score1 = self.stage1.score(X)
        obvious_anomalies = score1 > 0.8
        results[obvious_anomalies] = 1
        
        # Stage 2: Tree-based for uncertain cases
        uncertain = (score1 > 0.3) & (~obvious_anomalies)
        if uncertain.any():
            score2 = self.stage2.score(X[uncertain])
            results[np.where(uncertain)[0][score2 > 0.5]] = 1
        
        # Stage 3: Deep learning for subtle cases
        still_uncertain = (score1 > 0.2) & (results == 0)
        if still_uncertain.any():
            score3 = self.stage3.score(X[still_uncertain])
            results[np.where(still_uncertain)[0][score3 > 0.6]] = 1
        
        return results.astype(int)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Use cascaded detection for efficiency and coverage',
            'inductive_bias': 'Different anomalies require different detection strategies',
            'strengths': 'Efficient (fast methods first), good coverage, interpretable stages',
            'weaknesses': 'Complex to tune, cascading errors possible',
            'best_for': 'Production with mixed anomaly types and latency requirements'
        }
    
    def get_model_size(self) -> float:
        return self.stage1.get_model_size() + self.stage2.get_model_size() + self.stage3.get_model_size()
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        wrong = y_test != y_pred
        
        s1 = self.stage1.score(X_test)
        s2 = self.stage2.score(X_test)
        s3 = self.stage3.score(X_test)
        
        for idx in np.where(wrong)[0][:n_cases]:
            failures.append({
                'index': int(idx),
                'type': 'false_positive' if y_pred[idx] == 1 else 'false_negative',
                'stage_scores': {
                    'statistical': float(s1[idx]),
                    'isolation_forest': float(s2[idx]),
                    'autoencoder': float(s3[idx])
                }
            })
        return failures