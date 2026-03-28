"""
Tests for Domain C: Recommendation
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain_c_recommendation.data_generator import (
    RecommendationGenerator, create_recommendation_dataset
)
from src.domain_c_recommendation.approach_01_popularity import PopularityRecommender
from src.domain_c_recommendation.approach_04_matrix_factorization import SVDRecommender
from src.core.metrics import compute_ranking_metrics


class TestDataGenerator:
    """Test recommendation data generation."""
    
    def test_matrix_generation(self):
        generator = RecommendationGenerator(seed=42)
        data = generator.generate_matrix_factorization_data(
            n_users=100, n_items=50, sparsity=0.9
        )
        
        assert data.user_item_matrix.shape == (100, 50)
        assert len(data.interactions) > 0
    
    def test_dataset_creation(self):
        dataset = create_recommendation_dataset(n_users=100, n_items=50)
        
        assert 'train_matrix' in dataset
        assert 'test_items' in dataset
        assert dataset['train_matrix'].shape == (100, 50)


class TestPopularityRecommender:
    """Test popularity-based recommendations."""
    
    def test_training(self):
        dataset = create_recommendation_dataset(n_users=100, n_items=50)
        
        model = PopularityRecommender()
        model.train(dataset['train_matrix'], dataset['train_interactions'])
        
        assert model.is_trained
        assert model.top_items is not None
    
    def test_recommendation(self):
        dataset = create_recommendation_dataset(n_users=100, n_items=50)
        
        model = PopularityRecommender()
        model.train(dataset['train_matrix'], dataset['train_interactions'])
        
        recs = model.recommend(0, k=10)
        
        assert len(recs) == 10
        assert len(set(recs)) == 10  # No duplicates


class TestSVDRecommender:
    """Test SVD-based recommendations."""
    
    def test_training(self):
        dataset = create_recommendation_dataset(n_users=100, n_items=50)
        
        model = SVDRecommender({'n_factors': 10})
        model.train(dataset['train_matrix'])
        
        assert model.is_trained
        assert model.user_factors is not None
    
    def test_recommendation(self):
        dataset = create_recommendation_dataset(n_users=100, n_items=50)
        
        model = SVDRecommender({'n_factors': 10})
        model.train(dataset['train_matrix'])
        
        recs = model.recommend(0, k=10)
        
        assert len(recs) == 10


class TestRankingMetrics:
    """Test ranking metrics."""
    
    def test_perfect_ranking(self):
        y_true = [[1, 2, 3]]
        y_pred = [[1, 2, 3, 4, 5]]
        
        metrics = compute_ranking_metrics(y_true, y_pred, k_values=[5])
        
        assert metrics['recall@5'] == 1.0
        assert metrics['precision@5'] == 0.6  # 3 relevant out of 5
    
    def test_ndcg(self):
        y_true = [[1, 2]]
        y_pred = [[1, 3, 2]]  # 1 is first, 2 is third
        
        metrics = compute_ranking_metrics(y_true, y_pred, k_values=[5])
        
        assert 0 < metrics['ndcg@5'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])