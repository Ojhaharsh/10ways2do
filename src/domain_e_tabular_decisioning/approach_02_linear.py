"""Approach 2: Linear models."""

from typing import Dict, Optional

from sklearn.linear_model import LogisticRegression

from ..core.base_model import BaseApproach


class LinearDecisioner(BaseApproach):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Linear (Logistic Regression)", config)
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=float(self.config.get("C", 1.0)),
            solver="lbfgs",
        )
        self.metrics.interpretability_score = 0.9
        self.metrics.maintenance_complexity = 0.2

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_philosophy(self):
        return {
            "mental_model": "Risk is a weighted sum of evidence.",
            "inductive_bias": "Approximately linear log-odds relationship.",
            "strengths": "Simple, calibrated, robust baseline.",
            "weaknesses": "Misses non-linear interactions.",
            "best_for": "Strong baseline and transparent scorecards.",
        }
