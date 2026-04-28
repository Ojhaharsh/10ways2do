"""Few-shot learning dataset generator."""
import numpy as np
from typing import Dict, Any

def generate_few_shot_dataset(n_ways: int = 5, n_shots: int = 5, n_queries: int = 15, n_features: int = 64, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n_support = n_ways * n_shots
    n_query = n_ways * n_queries
    X_support = rng.randn(n_support, n_features).astype(np.float32)
    y_support = np.repeat(np.arange(n_ways), n_shots)
    X_query = rng.randn(n_query, n_features).astype(np.float32)
    y_query = np.repeat(np.arange(n_ways), n_queries)
    return {"X_support": X_support, "y_support": y_support, "X_query": X_query, "y_query": y_query}
