"""
Synthetic data generator for Recommendation domain.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RecommendationData:
    """Container for recommendation data."""
    user_item_matrix: np.ndarray
    user_features: Optional[np.ndarray]
    item_features: Optional[np.ndarray]
    interactions: List[Tuple[int, int, float]]
    n_users: int
    n_items: int


class RecommendationGenerator:
    """Generate synthetic recommendation data."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate_matrix_factorization_data(
        self,
        n_users: int = 1000,
        n_items: int = 500,
        n_factors: int = 10,
        sparsity: float = 0.98,
        noise_std: float = 0.5
    ) -> RecommendationData:
        """Generate data based on latent factor model."""
        
        user_factors = self.rng.randn(n_users, n_factors)
        item_factors = self.rng.randn(n_items, n_factors)
        
        true_ratings = user_factors @ item_factors.T
        
        user_bias = self.rng.randn(n_users) * 0.5
        item_bias = self.rng.randn(n_items) * 0.5
        global_bias = 3.0
        
        true_ratings = true_ratings + user_bias[:, np.newaxis] + item_bias + global_bias
        true_ratings += self.rng.randn(n_users, n_items) * noise_std
        true_ratings = np.clip(true_ratings, 1, 5)
        
        mask = self.rng.random((n_users, n_items)) > sparsity
        observed_ratings = np.where(mask, true_ratings, 0)
        
        interactions = []
        for u in range(n_users):
            for i in range(n_items):
                if mask[u, i]:
                    interactions.append((u, i, observed_ratings[u, i]))
        
        return RecommendationData(
            user_item_matrix=observed_ratings,
            user_features=user_factors,
            item_features=item_factors,
            interactions=interactions,
            n_users=n_users,
            n_items=n_items
        )
    
    def generate_implicit_data(
        self,
        n_users: int = 1000,
        n_items: int = 500,
        n_factors: int = 10,
        avg_interactions: int = 50
    ) -> Dict:
        """Generate implicit feedback data (clicks, views)."""
        
        user_factors = self.rng.randn(n_users, n_factors)
        item_factors = self.rng.randn(n_items, n_factors)
        
        affinities = user_factors @ item_factors.T
        probs = 1 / (1 + np.exp(-affinities))
        
        interactions = []
        for u in range(n_users):
            n_interact = self.rng.poisson(avg_interactions)
            item_probs = probs[u] / probs[u].sum()
            items = self.rng.choice(n_items, size=min(n_interact, n_items), 
                                   replace=False, p=item_probs)
            for i in items:
                interactions.append((u, int(i), 1.0))
        
        return {
            'interactions': interactions,
            'n_users': n_users,
            'n_items': n_items,
            'user_factors': user_factors,
            'item_factors': item_factors
        }


def create_recommendation_dataset(
    n_users: int = 1000,
    n_items: int = 500,
    sparsity: float = 0.95,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Dict:
    """Create train/test split for recommendation."""
    
    generator = RecommendationGenerator(seed)
    data = generator.generate_matrix_factorization_data(
        n_users=n_users,
        n_items=n_items,
        sparsity=sparsity
    )
    
    rng = np.random.RandomState(seed)
    interactions = data.interactions.copy()
    rng.shuffle(interactions)
    
    split_idx = int(len(interactions) * (1 - test_ratio))
    train_interactions = interactions[:split_idx]
    test_interactions = interactions[split_idx:]
    
    train_matrix = np.zeros((n_users, n_items))
    for u, i, r in train_interactions:
        train_matrix[u, i] = r
    
    test_items = {}
    for u, i, r in test_interactions:
        if u not in test_items:
            test_items[u] = []
        if r >= 4:
            test_items[u].append(i)
    
    return {
        'train_matrix': train_matrix,
        'train_interactions': train_interactions,
        'test_interactions': test_interactions,
        'test_items': test_items,
        'n_users': n_users,
        'n_items': n_items,
        'user_features': data.user_features,
        'item_features': data.item_features
    }


if __name__ == "__main__":
    dataset = create_recommendation_dataset(n_users=500, n_items=200)
    print(f"Train interactions: {len(dataset['train_interactions'])}")
    print(f"Test users with relevant items: {len(dataset['test_items'])}")