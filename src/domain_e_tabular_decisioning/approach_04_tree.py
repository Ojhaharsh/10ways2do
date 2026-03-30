"""Approach 4: Tree-based decisioning."""

from typing import Dict, Optional

from sklearn.ensemble import RandomForestClassifier

from ..core.base_model import BaseApproach


class TreeDecisioner(BaseApproach):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Tree-Based (Random Forest)", config)
        self.model = RandomForestClassifier(
            n_estimators=int(self.config.get("n_estimators", 200)),
            max_depth=self.config.get("max_depth", None),
            min_samples_leaf=int(self.config.get("min_samples_leaf", 2)),
            random_state=42,
            n_jobs=-1,
        )
        self.metrics.interpretability_score = 0.6
        self.metrics.maintenance_complexity = 0.35

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_philosophy(self):
        return {
            "mental_model": "Risk comes from many decision paths averaged together.",
            "inductive_bias": "Hierarchical threshold interactions.",
            "strengths": "Strong tabular performance and robustness.",
            "weaknesses": "Can be less calibrated than linear models.",
            "best_for": "General-purpose tabular decisioning.",
        }
