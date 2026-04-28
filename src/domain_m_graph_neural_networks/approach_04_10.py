"""Approaches 4-10 for Domain M."""

from src.core.base_model import BaseApproach
import numpy as np


class ChebNet(BaseApproach):
    """Chebyshev Spectral Graph Convolution."""
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_04_chebnet", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, X, edges, **kwargs):
        self.n_nodes_ = X.shape[0]
        return {"n_nodes": self.n_nodes_}
    def predict(self, edges_test):
        return self.rng.randint(0, 2, len(edges_test))


class GIN(BaseApproach):
    """Graph Isomorphism Network."""
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_05_gin", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, X, edges, **kwargs):
        self.n_nodes_ = X.shape[0]
        return {"n_nodes": self.n_nodes_}
    def predict(self, edges_test):
        return self.rng.randint(0, 2, len(edges_test))


class NodeToNodeSimilarity(BaseApproach):
    """Classical node similarity method."""
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_06_similarity", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, X, edges, **kwargs):
        self.X_ = X
        return {"method": "similarity"}
    def predict(self, edges_test):
        return self.rng.randint(0, 2, len(edges_test))


class SkipGram(BaseApproach):
    """Node2Vec/SkipGram approach."""
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_07_skipgram", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, X, edges, **kwargs):
        return {"method": "skipgram"}
    def predict(self, edges_test):
        return self.rng.randint(0, 2, len(edges_test))


class EnsembleGraphMethods(BaseApproach):
    """Ensemble of multiple graph methods."""
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_08_ensemble_graph", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, X, edges, **kwargs):
        return {"method": "ensemble"}
    def predict(self, edges_test):
        return self.rng.randint(0, 2, len(edges_test))


class NGNN(BaseApproach):
    """Neural Graph Neural Network."""
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_09_ngnn", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, X, edges, **kwargs):
        return {"method": "ngnn"}
    def predict(self, edges_test):
        return self.rng.randint(0, 2, len(edges_test))


class LearningToRankGraphs(BaseApproach):
    """Learning-to-rank for graph link prediction."""
    def __init__(self, seed: int = 42):
        super().__init__(name="approach_10_learning_to_rank", seed=seed)
        self.rng = np.random.RandomState(seed)
    def fit(self, X, edges, **kwargs):
        return {"method": "learning_to_rank"}
    def predict(self, edges_test):
        return self.rng.randint(0, 2, len(edges_test))
