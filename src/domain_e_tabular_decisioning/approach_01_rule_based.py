"""Approach 1: Rule-based decisioning."""

from typing import Dict, Optional

import numpy as np

from ..core.base_model import BaseApproach


class RuleBasedDecisioner(BaseApproach):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Rule-Based Thresholding", config)
        self.thresholds = None
        self.metrics.interpretability_score = 0.98
        self.metrics.maintenance_complexity = 0.35

    def train(self, X_train, y_train, X_val=None, y_val=None):
        pos = X_train[y_train == 1]
        neg = X_train[y_train == 0]
        pos_mean = np.mean(pos, axis=0)
        neg_mean = np.mean(neg, axis=0)
        direction = np.sign(pos_mean - neg_mean)
        threshold = (pos_mean + neg_mean) / 2.0
        self.thresholds = {"direction": direction, "threshold": threshold}
        self.is_trained = True

    def predict(self, X):
        scores = self.score(X)
        return (scores >= 0.5).astype(int)

    def score(self, X):
        if not self.is_trained or self.thresholds is None:
            raise ValueError("Model not trained")
        direction = self.thresholds["direction"]
        threshold = self.thresholds["threshold"]
        margin = np.mean((X - threshold) * direction, axis=1)
        return 1.0 / (1.0 + np.exp(-margin))

    def get_philosophy(self):
        return {
            "mental_model": "Risk is explicit thresholds over key signals.",
            "inductive_bias": "Linear threshold rules with domain-intuitive logic.",
            "strengths": "Transparent, auditable, and stable.",
            "weaknesses": "Limited interaction modeling.",
            "best_for": "Regulated environments requiring explainability.",
        }
