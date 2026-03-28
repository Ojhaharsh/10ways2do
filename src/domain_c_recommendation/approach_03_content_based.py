"""
Approach 3: Content-Based Filtering

Philosophy: Recommend items similar to what user liked.
- Uses item features
- User profile from preferences
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from ..core.base_model import BaseApproach


class ContentBasedRecommender(BaseApproach):
    """Content-based recommendation using item features."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Content-Based", config)
        
        self.item_features = None
        self.user_profiles = None
        self.train_matrix = None
        
        self.metrics.interpretability_score = 0.9
        self.metrics.maintenance_complexity = 0.3
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None,
              item_features: np.ndarray = None) -> None:
        
        self.train_matrix = train_matrix
        n_users, n_items = train_matrix.shape
        
        # Use provided features or generate random
        if item_features is not None:
            self.item_features = normalize(item_features)
        else:
            # Generate synthetic features
            n_features = 20
            self.item_features = normalize(np.random.randn(n_items, n_features))
        
        # Build user profiles as weighted average of liked item features
        self.user_profiles = np.zeros((n_users, self.item_features.shape[1]))
        
        for u in range(n_users):
            rated_mask = train_matrix[u] > 0
            if rated_mask.sum() > 0:
                ratings = train_matrix[u, rated_mask]
                features = self.item_features[rated_mask]
                # Weight by rating
                weights = ratings - ratings.mean() + 1
                self.user_profiles[u] = np.average(features, axis=0, weights=weights)
        
        self.user_profiles = normalize(self.user_profiles)
        self.is_trained = True
    
    def predict(self, user_ids: List[int], k: int = 10) -> List[List[int]]:
        return [self.recommend(u, k) for u in user_ids]
    
    def recommend(self, user_id: int, k: int = 10,
                  exclude_items: List[int] = None) -> List[int]:
        
        user_profile = self.user_profiles[user_id].reshape(1, -1)
        
        # Compute similarity to all items
        scores = cosine_similarity(user_profile, self.item_features)[0]
        
        # Exclude already rated
        already_rated = self.train_matrix[user_id] > 0
        scores[already_rated] = -np.inf
        
        if exclude_items:
            for item in exclude_items:
                scores[item] = -np.inf
        
        return np.argsort(-scores)[:k].tolist()
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Users like items similar to what they liked before',
            'inductive_bias': 'Item features capture what makes items appealing',
            'strengths': 'No cold start for new items, transparent, domain knowledge',
            'weaknesses': 'Limited novelty, requires good features, cold start for users',
            'best_for': 'When item features are meaningful, content-rich domains'
        }
    
    def get_model_size(self) -> float:
        if self.item_features is None:
            return 0.0
        return (self.item_features.nbytes + self.user_profiles.nbytes) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return []