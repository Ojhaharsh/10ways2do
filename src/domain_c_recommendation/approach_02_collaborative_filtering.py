"""
Approach 2: Collaborative Filtering

Philosophy: Similar users like similar items.
- User-based and Item-based CF
- Memory-based approach
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from ..core.base_model import BaseApproach


class UserBasedCF(BaseApproach):
    """User-based collaborative filtering."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("User-Based CF", config)
        
        self.k_neighbors = config.get('k_neighbors', 50) if config else 50
        self.user_similarity = None
        self.train_matrix = None
        
        self.metrics.interpretability_score = 0.8
        self.metrics.maintenance_complexity = 0.4
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None) -> None:
        
        self.train_matrix = train_matrix.copy()
        
        # Compute user-user similarity
        # Normalize by user mean
        user_means = np.true_divide(
            train_matrix.sum(axis=1),
            (train_matrix > 0).sum(axis=1) + 1e-8
        )
        
        normalized = train_matrix.copy()
        for u in range(len(train_matrix)):
            mask = train_matrix[u] > 0
            normalized[u, mask] -= user_means[u]
        
        self.user_similarity = cosine_similarity(normalized)
        np.fill_diagonal(self.user_similarity, 0)
        
        self.user_means = user_means
        self.is_trained = True
    
    def predict(self, user_ids: List[int], k: int = 10) -> List[List[int]]:
        return [self.recommend(u, k) for u in user_ids]
    
    def recommend(self, user_id: int, k: int = 10,
                  exclude_items: List[int] = None) -> List[int]:
        
        # Get k most similar users
        similarities = self.user_similarity[user_id]
        top_users = np.argsort(-similarities)[:self.k_neighbors]
        
        # Predict scores for all items
        scores = np.zeros(self.train_matrix.shape[1])
        sim_sum = np.zeros(self.train_matrix.shape[1])
        
        for neighbor in top_users:
            sim = similarities[neighbor]
            if sim <= 0:
                continue
            
            rated_items = self.train_matrix[neighbor] > 0
            scores[rated_items] += sim * (
                self.train_matrix[neighbor, rated_items] - self.user_means[neighbor]
            )
            sim_sum[rated_items] += sim
        
        # Normalize
        valid = sim_sum > 0
        scores[valid] = self.user_means[user_id] + scores[valid] / sim_sum[valid]
        
        # Exclude already rated items
        already_rated = self.train_matrix[user_id] > 0
        scores[already_rated] = -np.inf
        
        if exclude_items:
            for item in exclude_items:
                scores[item] = -np.inf
        
        return np.argsort(-scores)[:k].tolist()
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Users with similar rating patterns will like similar items',
            'inductive_bias': 'User behavior is the best predictor of preferences',
            'strengths': 'Interpretable, captures user taste, serendipitous recommendations',
            'weaknesses': 'Sparsity issues, scalability, cold start for new users',
            'best_for': 'When user similarity is meaningful, smaller user bases'
        }
    
    def get_model_size(self) -> float:
        if self.user_similarity is None:
            return 0.0
        return (self.user_similarity.nbytes + self.train_matrix.nbytes) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return []


class ItemBasedCF(BaseApproach):
    """Item-based collaborative filtering."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Item-Based CF", config)
        
        self.k_neighbors = config.get('k_neighbors', 50) if config else 50
        self.item_similarity = None
        self.train_matrix = None
        
        self.metrics.interpretability_score = 0.8
        self.metrics.maintenance_complexity = 0.4
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None) -> None:
        
        self.train_matrix = train_matrix.copy()
        
        # Compute item-item similarity
        self.item_similarity = cosine_similarity(train_matrix.T)
        np.fill_diagonal(self.item_similarity, 0)
        
        self.is_trained = True
    
    def predict(self, user_ids: List[int], k: int = 10) -> List[List[int]]:
        return [self.recommend(u, k) for u in user_ids]
    
    def recommend(self, user_id: int, k: int = 10,
                  exclude_items: List[int] = None) -> List[int]:
        
        user_ratings = self.train_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            # Cold start: return random items
            return list(range(k))
        
        # Score each item based on similarity to rated items
        scores = np.zeros(self.train_matrix.shape[1])
        
        for item in range(len(scores)):
            if user_ratings[item] > 0:
                scores[item] = -np.inf
                continue
            
            # Get similarities to user's rated items
            sims = self.item_similarity[item, rated_items]
            top_k = np.argsort(-sims)[:self.k_neighbors]
            
            valid_sims = sims[top_k]
            valid_ratings = user_ratings[rated_items[top_k]]
            
            if valid_sims.sum() > 0:
                scores[item] = np.dot(valid_sims, valid_ratings) / (valid_sims.sum() + 1e-8)
        
        if exclude_items:
            for item in exclude_items:
                scores[item] = -np.inf
        
        return np.argsort(-scores)[:k].tolist()
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Items rated similarly by users are similar',
            'inductive_bias': 'Item relationships are stable and meaningful',
            'strengths': 'More stable than user-based, precomputable, explainable',
            'weaknesses': 'Less serendipitous, cold start for new items',
            'best_for': 'Larger catalogs, when item similarity is meaningful'
        }
    
    def get_model_size(self) -> float:
        if self.item_similarity is None:
            return 0.0
        return self.item_similarity.nbytes / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return []