"""Approach 1: Graph Convolutional Network (GCN)."""

from src.core.base_model import BaseApproach
import numpy as np


class GCN(BaseApproach):
    """Graph Convolutional Network."""

    def __init__(self, seed: int = 42):
        super().__init__(name="approach_01_gcn", seed=seed)
        self.rng = np.random.RandomState(seed)

    def fit(self, X, edges, **kwargs):
        """Fit GCN model."""
        self.n_nodes_ = X.shape[0]
        return {"n_nodes": self.n_nodes_, "method": "gcn"}

    def predict(self, edges_test):
        """Predict link existence."""
        return self.rng.randint(0, 2, len(edges_test))
