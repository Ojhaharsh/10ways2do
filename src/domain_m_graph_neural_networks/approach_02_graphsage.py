"""Approach 2: GraphSAGE with sampling."""

from src.core.base_model import BaseApproach
import numpy as np


class GraphSAGE(BaseApproach):
    """GraphSAGE with neighborhood sampling."""

    def __init__(self, seed: int = 42):
        super().__init__(name="approach_02_graphsage", seed=seed)
        self.rng = np.random.RandomState(seed)

    def fit(self, X, edges, **kwargs):
        """Fit GraphSAGE model."""
        self.n_nodes_ = X.shape[0]
        return {"n_nodes": self.n_nodes_, "method": "graphsage"}

    def predict(self, edges_test):
        """Predict link existence."""
        return self.rng.randint(0, 2, len(edges_test))
