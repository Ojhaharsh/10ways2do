"""
Approach 9: Hybrid Recommendation Systems

Philosophy: Combine multiple approaches.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from ..core.base_model import BaseApproach
from .approach_01_popularity import PopularityRecommender
from .approach_02_collaborative_filtering import ItemBasedCF
from .approach_04_matrix_factorization import SVDRecommender


class HybridRecommender(BaseApproach):
    """Combine multiple recommendation strategies."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Hybrid Recommender", config)
        
        self.popularity = PopularityRecommender()
        self.cf = ItemBasedCF({'k_neighbors': 30})
        self.mf = SVDRecommender({'n_factors': 30})
        
        self.weights = config.get('weights', [0.2, 0.4, 0.4]) if config else [0.2, 0.4, 0.4]
        self.n_items = 0
        self.train_matrix = None
        
        self.metrics.interpretability_score = 0.5
        self.metrics.maintenance_complexity = 0.7
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None) -> None:
        
        self.train_matrix = train_matrix
        self.n_items = train_matrix.shape[1]
        
        self.popularity.train(train_matrix, train_interactions)
        self.cf.train(train_matrix, train_interactions)
        self.mf.train(train_matrix, train_interactions)
        
        self.is_trained = True
    
    def predict(self, user_ids: List[int], k: int = 10) -> List[List[int]]:
        return [self.recommend(u, k) for u in user_ids]
    
    def recommend(self, user_id: int, k: int = 10,
                  exclude_items: List[int] = None) -> List[int]:
        
        # Get recommendations from each model
        pop_recs = self.popularity.recommend(user_id, k * 2, exclude_items)
        cf_recs = self.cf.recommend(user_id, k * 2, exclude_items)
        mf_recs = self.mf.recommend(user_id, k * 2, exclude_items)
        
        # Score-based combination
        scores = np.zeros(self.n_items)
        
        for rank, item in enumerate(pop_recs):
            scores[item] += self.weights[0] * (1.0 / (rank + 1))
        
        for rank, item in enumerate(cf_recs):
            scores[item] += self.weights[1] * (1.0 / (rank + 1))
        
        for rank, item in enumerate(mf_recs):
            scores[item] += self.weights[2] * (1.0 / (rank + 1))
        
        # Exclude already rated
        already_rated = self.train_matrix[user_id] > 0
        scores[already_rated] = -np.inf
        
        if exclude_items:
            for item in exclude_items:
                scores[item] = -np.inf
        
        return np.argsort(-scores)[:k].tolist()
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Combine multiple recommendation strategies for robustness',
            'inductive_bias': 'Different methods capture different aspects of preference',
            'strengths': 'More robust, handles diverse user types, graceful degradation',
            'weaknesses': 'Complex tuning, slower, potential conflicts',
            'best_for': 'Production systems requiring reliability'
        }
    
    def get_model_size(self) -> float:
        return (self.popularity.get_model_size() + 
                self.cf.get_model_size() + 
                self.mf.get_model_size())
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return []