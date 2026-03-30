"""Approach 6: Neural MLP decisioning."""

from typing import Dict, Optional

from sklearn.neural_network import MLPClassifier

from ..core.base_model import BaseApproach


class NeuralDecisioner(BaseApproach):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Neural (MLP)", config)
        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(self.config.get("hidden_layer_sizes", (64, 32))),
            alpha=float(self.config.get("alpha", 1e-4)),
            learning_rate_init=float(self.config.get("learning_rate_init", 1e-3)),
            max_iter=int(self.config.get("max_iter", 200)),
            random_state=42,
        )
        self.metrics.interpretability_score = 0.25
        self.metrics.maintenance_complexity = 0.55

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_philosophy(self):
        return {
            "mental_model": "Learn distributed non-linear interactions end-to-end.",
            "inductive_bias": "Flexible function approximation with dense layers.",
            "strengths": "Captures non-linear patterns without hand features.",
            "weaknesses": "Less interpretable and can be unstable.",
            "best_for": "Complex tabular interactions with enough data.",
        }
