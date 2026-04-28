"""Approach 1: Basic CNN for image classification."""

from src.core.base_model import BaseApproach
import numpy as np


class BasicCNN(BaseApproach):
    """Basic convolutional neural network (simulated)."""

    def __init__(self, seed: int = 42):
        super().__init__(name="approach_01_basic_cnn", seed=seed)
        self.rng = np.random.RandomState(seed)

    def fit(self, X_train, y_train, **kwargs):
        """Fit basic CNN model."""
        self.classes_ = np.unique(y_train)
        self.mean_ = X_train.mean(axis=0)
        return {"training_samples": len(X_train), "classes": len(self.classes_)}

    def predict(self, X_test):
        """Predict using CNN."""
        # Simulate CNN prediction with simple pattern matching
        scores = np.random.randn(len(X_test), len(self.classes_))
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X_test):
        """Predict probabilities."""
        scores = np.random.randn(len(X_test), len(self.classes_))
        proba = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
        return proba
