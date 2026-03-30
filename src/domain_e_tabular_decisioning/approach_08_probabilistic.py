"""Approach 8: Probabilistic decisioning."""

from typing import Dict, Optional

from sklearn.naive_bayes import GaussianNB

from ..core.base_model import BaseApproach


class ProbabilisticDecisioner(BaseApproach):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Probabilistic (Naive Bayes)", config)
        self.model = GaussianNB(var_smoothing=float(self.config.get("var_smoothing", 1e-9)))
        self.metrics.interpretability_score = 0.72
        self.metrics.maintenance_complexity = 0.15

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_philosophy(self):
        return {
            "mental_model": "Estimate class likelihoods and apply Bayes rule.",
            "inductive_bias": "Conditional independence across features.",
            "strengths": "Fast and robust under sparse information.",
            "weaknesses": "Independence assumption can be unrealistic.",
            "best_for": "Lightweight probabilistic baseline.",
        }
