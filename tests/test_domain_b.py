"""
Tests for Domain B: Anomaly Detection
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain_b_anomaly_detection.data_generator import (
    TimeSeriesGenerator, create_anomaly_dataset
)
from src.domain_b_anomaly_detection.approach_01_statistical import StatisticalAnomalyDetector
from src.domain_b_anomaly_detection.approach_03_tree_based import IsolationForestDetector
from src.core.metrics import compute_anomaly_metrics


class TestDataGenerator:
    """Test anomaly data generation."""
    
    def test_base_series_generation(self):
        generator = TimeSeriesGenerator(seed=42)
        series = generator.generate_base_series(n_samples=100, n_features=3)
        
        assert series.shape == (100, 3)
        assert not np.isnan(series).any()
    
    def test_anomaly_injection(self):
        generator = TimeSeriesGenerator(seed=42)
        X = generator.generate_base_series(n_samples=1000, n_features=2)
        X_anom, y, specs = generator.inject_anomalies(X, anomaly_ratio=0.05)
        
        assert X_anom.shape == X.shape
        assert len(y) == len(X)
        assert y.sum() > 0  # Some anomalies exist
    
    def test_dataset_creation(self):
        dataset = create_anomaly_dataset(n_train=500, n_val=100, n_test=100)
        
        assert 'train' in dataset
        assert dataset['train']['X'].shape[0] == 500
        assert dataset['test']['y'].shape[0] == 100


class TestStatisticalDetector:
    """Test statistical anomaly detection."""
    
    def test_zscore_detection(self):
        detector = StatisticalAnomalyDetector({'method': 'zscore', 'threshold': 3.0})
        
        # Normal data
        X_normal = np.random.randn(100, 3)
        y_normal = np.zeros(100)
        
        detector.train(X_normal, y_normal)
        
        # Test with outlier
        X_test = np.random.randn(10, 3)
        X_test[0] = [10, 10, 10]  # Clear outlier
        
        predictions = detector.predict(X_test)
        
        assert predictions[0] == 1  # Should detect outlier
    
    def test_scoring(self):
        detector = StatisticalAnomalyDetector()
        X = np.random.randn(100, 3)
        
        detector.train(X, np.zeros(100))
        scores = detector.score(X)
        
        assert len(scores) == 100
        assert all(0 <= s <= 1 for s in scores)


class TestIsolationForest:
    """Test Isolation Forest detector."""
    
    def test_training_and_prediction(self):
        detector = IsolationForestDetector({'n_estimators': 50})
        
        X_train = np.random.randn(200, 5)
        detector.train(X_train, np.zeros(200))
        
        X_test = np.random.randn(20, 5)
        predictions = detector.predict(X_test)
        
        assert len(predictions) == 20
        assert all(p in [0, 1] for p in predictions)


class TestAnomalyMetrics:
    """Test anomaly metrics."""
    
    def test_perfect_detection(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1])
        
        metrics = compute_anomaly_metrics(y_true, y_pred)
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_with_scores(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        
        metrics = compute_anomaly_metrics(y_true, y_pred, y_scores)
        
        assert 'roc_auc' in metrics
        assert 'avg_precision' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])