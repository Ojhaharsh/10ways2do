"""
Approach 1: Popularity-Based Recommendation

Philosophy: Recommend what's popular.
- No personalization
- Strong baseline
- Cold start friendly
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter

from ..core.base_model import BaseApproach


class PopularityRecommender(BaseApproach):
    """Recommend most popular items."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Popularity-Based", config)
        
        self.item_popularity = None
        self.top_items = None
        self.n_items = 0
        
        self.metrics.interpretability_score = 1.0
        self.metrics.maintenance_complexity = 0.1
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None) -> None:
        
        self.n_items = train_matrix.shape[1]
        
        # Count interactions per item
        if train_interactions:
            item_counts = Counter(i for _, i, _ in train_interactions)
        else:
            item_counts = Counter()
            for i in range(self.n_items):
                item_counts[i] = np.sum(train_matrix[:, i] > 0)
        
        self.item_popularity = np.zeros(self.n_items)
        for item, count in item_counts.items():
            self.item_popularity[item] = count
        
        self.top_items = np.argsort(-self.item_popularity)
        self.is_trained = True
    
    def predict(self, user_ids: List[int], k: int = 10) -> List[List[int]]:
        """Return top-k popular items for each user."""
        return [self.top_items[:k].tolist() for _ in user_ids]
    
    def recommend(self, user_id: int, k: int = 10, 
                  exclude_items: List[int] = None) -> List[int]:
        """Recommend items for a single user."""
        recommendations = []
        exclude_set = set(exclude_items) if exclude_items else set()
        
        for item in self.top_items:
            if item not in exclude_set:
                recommendations.append(item)
                if len(recommendations) >= k:
                    break
        
        return recommendations
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Popular items are likely relevant to everyone',
            'inductive_bias': 'Assumes homogeneous preferences across users',
            'strengths': 'Simple, fast, no cold start for items, strong baseline',
            'weaknesses': 'No personalization, popularity bias, ignores user preferences',
            'best_for': 'Cold start, baseline comparison, when personalization not critical'
        }
    
    def get_model_size(self) -> float:
        if self.item_popularity is None:
            return 0.0
        return self.item_popularity.nbytes / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return [{'reason': 'Popularity baseline has no personalization'}]