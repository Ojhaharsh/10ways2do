"""
Approach 7: Graph-Based Anomaly Detection

Philosophy: Model relationships between entities.
- Build graph from data
- Anomalies have unusual graph properties
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import connected_components
import networkx as nx

from ..core.base_model import BaseApproach


class GraphAnomalyDetector(BaseApproach):
    """Graph-based anomaly detection using node properties."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Graph-Based", config)
        
        self.k = config.get('k', 10) if config else 10
        self.threshold_percentile = config.get('threshold_percentile', 5) if config else 5
        
        self.scaler = StandardScaler()
        self.graph = None
        self.threshold = None
        
        self.metrics.interpretability_score = 0.6
        self.metrics.maintenance_complexity = 0.5
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Build k-NN graph
        self.graph = kneighbors_graph(X_scaled, self.k, mode='distance', include_self=False)
        
        # Compute centrality scores for training data
        G = nx.from_scipy_sparse_array(self.graph)
        
        try:
            centrality = nx.degree_centrality(G)
            scores = np.array([centrality.get(i, 0) for i in range(len(X_train))])
        except:
            scores = np.array(self.graph.sum(axis=1)).flatten()
        
        # Low centrality = anomaly
        self.threshold = np.percentile(scores, self.threshold_percentile)
        self._train_data = X_scaled
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores > 0.5).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        
        # For each test point, compute distance to k-nearest training points
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=self.k)
        nn.fit(self._train_data)
        
        distances, _ = nn.kneighbors(X_scaled)
        mean_distances = distances.mean(axis=1)
        
        # High distance = low connectivity = anomaly
        scores = mean_distances / (np.median(mean_distances) * 2)
        return np.clip(scores, 0, 1)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Model data as a graph; anomalies are poorly connected nodes',
            'inductive_bias': 'Normal points are well-connected in the data manifold',
            'strengths': 'Captures complex relationships, topology-aware',
            'weaknesses': 'Computationally expensive, graph construction choices matter',
            'best_for': 'Relational data, network intrusion detection'
        }
    
    def get_model_size(self) -> float:
        if self.graph is None:
            return 0.0
        return (self.graph.data.nbytes + self._train_data.nbytes) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        wrong = y_test != y_pred
        for idx in np.where(wrong)[0][:n_cases]:
            failures.append({
                'index': int(idx),
                'type': 'false_positive' if y_pred[idx] == 1 else 'false_negative'
            })
        return failures