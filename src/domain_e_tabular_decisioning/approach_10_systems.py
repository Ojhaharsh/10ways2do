"""Approach 10: Systems-oriented decisioning."""

from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ..core.base_model import BaseApproach


class SystemsDecisioner(BaseApproach):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Systems (Calibrated Thresholding)", config)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.threshold = 0.5
        self.metrics.interpretability_score = 0.82
        self.metrics.maintenance_complexity = 0.3

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        if X_val is not None and y_val is not None:
            scores = self.model.predict_proba(X_val)[:, 1]
            candidate_thresholds = np.linspace(0.2, 0.8, 31)
            best_t = 0.5
            best_f1 = -1.0
            for t in candidate_thresholds:
                f1 = f1_score(y_val, (scores >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)
            self.threshold = best_t
        self.is_trained = True

    def predict(self, X):
        return (self.score(X) >= self.threshold).astype(int)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_philosophy(self):
        return {
            "mental_model": "Optimize decisions for operational objectives, not raw probabilities.",
            "inductive_bias": "Simple calibrated model with policy-tuned threshold.",
            "strengths": "Operationally aligned and explainable.",
            "weaknesses": "Depends on validation-policy assumptions.",
            "best_for": "Real systems where precision-recall policy matters.",
        }
