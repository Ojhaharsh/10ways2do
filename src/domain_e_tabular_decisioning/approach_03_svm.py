"""Approach 3: SVM decisioning."""

from typing import Dict, Optional

from sklearn.svm import SVC

from ..core.base_model import BaseApproach


class SVMDecisioner(BaseApproach):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("SVM (RBF)", config)
        self.model = SVC(
            C=float(self.config.get("C", 2.0)),
            gamma=self.config.get("gamma", "scale"),
            probability=True,
        )
        self.metrics.interpretability_score = 0.45
        self.metrics.maintenance_complexity = 0.4

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_philosophy(self):
        return {
            "mental_model": "Find a max-margin boundary in transformed feature space.",
            "inductive_bias": "Kernelized geometric separation.",
            "strengths": "Strong non-linear separation on medium-sized data.",
            "weaknesses": "Less scalable and less interpretable.",
            "best_for": "Medium tabular tasks with complex boundaries.",
        }
