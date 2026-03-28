"""
Approach 2: Distance-Based Anomaly Detection

Philosophy: Anomalies are far from normal points.
- KNN distance
- Local Outlier Factor (LOF)
- DBSCAN-based
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pickle

from ..core.base_model import BaseApproach


class KNNAnomalyDetector(BaseApproach):
    """K-Nearest Neighbors based anomaly detection."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Distance-Based (KNN)", config)
        
        self.k = config.get('k', 5) if config else 5
        self.threshold_percentile = config.get('threshold_percentile', 95) if config else 95
        
        self.knn = None
        self.scaler = StandardScaler()
        self.threshold = None
        
        self.metrics.interpretability_score = 0.7
        self.metrics.maintenance_complexity = 0.3
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        
        if y_train is not None:
            X_normal = X_train[y_train == 0]
        else:
            X_normal = X_train
        
        X_scaled = self.scaler.fit_transform(X_normal)
        
        self.knn = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree')
        self.knn.fit(X_scaled)
        
        distances, _ = self.knn.kneighbors(X_scaled)
        mean_distances = distances.mean(axis=1)
        self.threshold = np.percentile(mean_distances, self.threshold_percentile)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores > 0.5).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        distances, _ = self.knn.kneighbors(X_scaled)
        mean_distances = distances.mean(axis=1)
        scores = mean_distances / (self.threshold * 2)
        return np.clip(scores, 0, 1)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Anomalies are points far from their nearest neighbors',
            'inductive_bias': 'Distance in feature space indicates anomaly',
            'strengths': 'Non-parametric, captures local density',
            'weaknesses': 'Sensitive to k, curse of dimensionality, O(n) inference',
            'best_for': 'Moderate dimensions, when local context matters'
        }
    
    def get_model_size(self) -> float:
        if self.knn is None:
            return 0.0
        return len(pickle.dumps(self.knn)) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        wrong = y_test != y_pred
        X_scaled = self.scaler.transform(X_test)
        distances, _ = self.knn.kneighbors(X_scaled)
        
        for idx in np.where(wrong)[0][:n_cases]:
            failures.append({
                'index': int(idx),
                'type': 'false_positive' if y_pred[idx] == 1 else 'false_negative',
                'mean_distance': float(distances[idx].mean()),
                'threshold': float(self.threshold)
            })
        return failures


class LOFDetector(BaseApproach):
    """Local Outlier Factor detector."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Distance-Based (LOF)", config)
        
        self.n_neighbors = config.get('n_neighbors', 20) if config else 20
        self.contamination = config.get('contamination', 0.05) if config else 0.05
        
        self.lof = None
        self.scaler = StandardScaler()
        
        self.metrics.interpretability_score = 0.6
        self.metrics.maintenance_complexity = 0.3
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True
        )
        self.lof.fit(X_scaled)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        predictions = self.lof.predict(X_scaled)
        return (predictions == -1).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        scores = -self.lof.score_samples(X_scaled)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Anomalies have lower local density than neighbors',
            'inductive_bias': 'Compares point density to neighborhood density',
            'strengths': 'Handles varying densities, local perspective',
            'weaknesses': 'Computationally expensive, sensitive to n_neighbors',
            'best_for': 'Datasets with clusters of varying densities'
        }
    
    def get_model_size(self) -> float:
        if self.lof is None:
            return 0.0
        return len(pickle.dumps(self.lof)) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        wrong = y_test != y_pred
        scores = self.score(X_test)
        
        for idx in np.where(wrong)[0][:n_cases]:
            failures.append({
                'index': int(idx),
                'type': 'false_positive' if y_pred[idx] == 1 else 'false_negative',
                'lof_score': float(scores[idx])
            })
        return failures