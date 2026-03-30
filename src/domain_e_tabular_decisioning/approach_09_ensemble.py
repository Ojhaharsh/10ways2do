"""Approach 9: Ensemble decisioning."""

from typing import Dict, Optional

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from ..core.base_model import BaseApproach


class EnsembleDecisioner(BaseApproach):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Ensemble (Voting)", config)
        lr = LogisticRegression(max_iter=500, class_weight="balanced")
        rf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
        nb = GaussianNB()
        self.model = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("nb", nb)],
            voting="soft",
        )
        self.metrics.interpretability_score = 0.5
        self.metrics.maintenance_complexity = 0.5

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_philosophy(self):
        return {
            "mental_model": "Average complementary model biases for reliability.",
            "inductive_bias": "Diverse weak assumptions improve robustness.",
            "strengths": "Stable performance and reduced variance.",
            "weaknesses": "Harder maintenance and diagnostics.",
            "best_for": "Production reliability under mixed conditions.",
        }
