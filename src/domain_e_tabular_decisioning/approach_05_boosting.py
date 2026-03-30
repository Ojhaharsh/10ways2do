"""Approach 5: Boosting decisioning."""

from typing import Dict, Optional

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from sklearn.ensemble import GradientBoostingClassifier

from ..core.base_model import BaseApproach


class BoostingDecisioner(BaseApproach):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Boosting (XGBoost)", config)
        if HAS_XGB:
            self.model = XGBClassifier(
                n_estimators=int(self.config.get("n_estimators", 200)),
                max_depth=int(self.config.get("max_depth", 4)),
                learning_rate=float(self.config.get("learning_rate", 0.05)),
                subsample=float(self.config.get("subsample", 0.9)),
                colsample_bytree=float(self.config.get("colsample_bytree", 0.9)),
                eval_metric="logloss",
                random_state=42,
            )
        else:
            self.model = GradientBoostingClassifier(random_state=42)
        self.metrics.interpretability_score = 0.55
        self.metrics.maintenance_complexity = 0.45

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.decision_function(X)

    def get_philosophy(self):
        return {
            "mental_model": "Sequentially fix residual errors with strong weak-learners.",
            "inductive_bias": "Additive non-linear trees optimized for hard examples.",
            "strengths": "Often top tabular accuracy.",
            "weaknesses": "Hyperparameter-sensitive and less transparent.",
            "best_for": "High-performance tabular risk prediction.",
        }
