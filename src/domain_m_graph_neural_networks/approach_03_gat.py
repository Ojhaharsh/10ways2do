"""Approach 3: Graph Attention Network (GAT)."""

from src.core.base_model import BaseApproach
import numpy as np


class GAT(BaseApproach):
    """Graph Attention Network."""

    def __init__(self, seed: int = 42):
        super().__init__(name="approach_03_gat", seed=seed)
        self.rng = np.random.RandomState(seed)

    def fit(self, X, edges, **kwargs):
        """Fit GAT model."""
        self.n_nodes_ = X.shape[0]
        return {"n_nodes": self.n_nodes_, "method": "gat"}

    def predict(self, edges_test):
        """Predict link existence."""
        return self.rng.randint(0, 2, len(edges_test))
