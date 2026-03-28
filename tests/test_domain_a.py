"""
Tests for Domain A: Information Extraction
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain_a_information_extraction.data_generator import (
    ResumeGenerator, create_ie_dataset, get_ie_fields
)
from src.domain_a_information_extraction.approach_01_rule_based import RuleBasedIE
from src.domain_a_information_extraction.approach_02_classical_ml import ClassicalMLIE
from src.core.metrics import compute_ie_metrics


class TestDataGenerator:
    """Test data generation."""
    
    def test_resume_generation(self):
        generator = ResumeGenerator(seed=42)
        text, data = generator.generate_resume()
        
        assert text is not None
        assert len(text) > 0
        assert data.name is not None
        assert data.email is not None
    
    def test_dataset_creation(self):
        dataset = create_ie_dataset(n_train=100, n_val=20, n_test=20)
        
        assert 'train' in dataset
        assert 'val' in dataset
        assert 'test' in dataset
        assert len(dataset['train']['X']) == 100
        assert len(dataset['test']['X']) == 20
    
    def test_noise_injection(self):
        generator = ResumeGenerator(seed=42)
        text, _ = generator.generate_resume()
        
        noisy = generator.add_noise_to_text(text, 0.1)
        assert noisy != text
        
        clean = generator.add_noise_to_text(text, 0.0)
        assert clean == text


class TestRuleBasedIE:
    """Test rule-based extraction."""
    
    def test_email_extraction(self):
        model = RuleBasedIE()
        model.train([], [])
        
        texts = ["Contact me at john@example.com for details."]
        results = model.predict(texts)
        
        assert results[0]['email'] == 'john@example.com'
    
    def test_phone_extraction(self):
        model = RuleBasedIE()
        model.train([], [])
        
        texts = ["Call me at 555-123-4567"]
        results = model.predict(texts)
        
        assert results[0]['phone'] is not None
        assert '555' in results[0]['phone']
    
    def test_philosophy(self):
        model = RuleBasedIE()
        philosophy = model.get_philosophy()
        
        assert 'mental_model' in philosophy
        assert 'strengths' in philosophy
        assert 'weaknesses' in philosophy


class TestClassicalMLIE:
    """Test classical ML extraction."""
    
    def test_training(self):
        dataset = create_ie_dataset(n_train=50, n_val=10, n_test=10)
        
        model = ClassicalMLIE()
        model.train(
            dataset['train']['X'],
            dataset['train']['y']
        )
        
        assert model.is_trained
    
    def test_prediction(self):
        dataset = create_ie_dataset(n_train=50, n_val=10, n_test=10)
        
        model = ClassicalMLIE()
        model.train(dataset['train']['X'], dataset['train']['y'])
        
        predictions = model.predict(dataset['test']['X'])
        
        assert len(predictions) == len(dataset['test']['X'])
        assert isinstance(predictions[0], dict)


class TestMetrics:
    """Test metrics computation."""
    
    def test_ie_metrics(self):
        y_true = [{'name': 'John Doe', 'email': 'john@test.com'}]
        y_pred = [{'name': 'John Doe', 'email': 'john@test.com'}]
        fields = ['name', 'email']
        
        metrics = compute_ie_metrics(y_true, y_pred, fields)
        
        assert metrics['overall_exact_match'] == 1.0
    
    def test_partial_match(self):
        y_true = [{'name': 'John Doe'}]
        y_pred = [{'name': 'John Smith'}]
        fields = ['name']
        
        metrics = compute_ie_metrics(y_true, y_pred, fields)
        
        assert metrics['name_exact_match'] == 0.0
        assert metrics['name_partial_match'] > 0  # "John" matches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])