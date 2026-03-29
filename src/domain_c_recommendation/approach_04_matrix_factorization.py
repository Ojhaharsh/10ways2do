"""
Approach 4: Matrix Factorization

Philosophy: Learn latent factors for users and items.
- SVD, ALS, NMF
- Captures latent preferences
"""

from typing import Dict, List, Any, Optional
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF

from ..core.base_model import BaseApproach


def _robust_linear_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax=b with numerical safeguards for ill-conditioned systems."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        pass

    # Retry with small diagonal jitter before falling back to least squares.
    n = a.shape[0]
    eye = np.eye(n, dtype=a.dtype)
    for jitter in (1e-8, 1e-6, 1e-4):
        try:
            return np.linalg.solve(a + jitter * eye, b)
        except np.linalg.LinAlgError:
            continue

    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    return x


class SVDRecommender(BaseApproach):
    """SVD-based matrix factorization."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Matrix Factorization (SVD)", config)
        
        self.n_factors = config.get('n_factors', 50) if config else 50
        
        self.user_factors = None
        self.item_factors = None
        self.user_means = None
        self.global_mean = 0
        self.train_matrix = None
        
        self.metrics.interpretability_score = 0.5
        self.metrics.maintenance_complexity = 0.4
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None) -> None:
        
        self.train_matrix = train_matrix.copy()
        
        # Center the matrix
        self.global_mean = train_matrix[train_matrix > 0].mean()
        
        centered = train_matrix.copy()
        mask = centered > 0
        centered[mask] -= self.global_mean
        centered[~mask] = 0
        
        # SVD
        n_factors = min(self.n_factors, min(train_matrix.shape) - 1)
        U, sigma, Vt = svds(centered, k=n_factors)
        
        # Reorder (svds returns in ascending order)
        idx = np.argsort(-sigma)
        sigma = sigma[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]
        
        self.user_factors = U * np.sqrt(sigma)
        self.item_factors = Vt.T * np.sqrt(sigma)
        
        self.is_trained = True
    
    def predict(self, user_ids: List[int], k: int = 10) -> List[List[int]]:
        return [self.recommend(u, k) for u in user_ids]
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair."""
        score = self.global_mean + np.dot(
            self.user_factors[user_id], 
            self.item_factors[item_id]
        )
        return np.clip(score, 1, 5)
    
    def recommend(self, user_id: int, k: int = 10,
                  exclude_items: List[int] = None) -> List[int]:
        
        scores = self.global_mean + self.user_factors[user_id] @ self.item_factors.T
        
        # Exclude already rated
        already_rated = self.train_matrix[user_id] > 0
        scores[already_rated] = -np.inf
        
        if exclude_items:
            for item in exclude_items:
                scores[item] = -np.inf
        
        return np.argsort(-scores)[:k].tolist()
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Users and items exist in a shared latent space',
            'inductive_bias': 'Preferences can be explained by low-rank structure',
            'strengths': 'Captures complex patterns, handles sparsity, scalable',
            'weaknesses': 'Cold start problem, implicit feedback handling',
            'best_for': 'Large sparse matrices, when latent factors are meaningful'
        }
    
    def get_model_size(self) -> float:
        if self.user_factors is None:
            return 0.0
        return (self.user_factors.nbytes + self.item_factors.nbytes) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return []


class ALSRecommender(BaseApproach):
    """Alternating Least Squares for implicit feedback."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Matrix Factorization (ALS)", config)
        
        self.n_factors = config.get('n_factors', 50) if config else 50
        self.regularization = config.get('regularization', 0.01) if config else 0.01
        self.iterations = config.get('iterations', 15) if config else 15
        
        self.user_factors = None
        self.item_factors = None
        self.train_matrix = None
        
        self.metrics.interpretability_score = 0.5
        self.metrics.maintenance_complexity = 0.5
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None) -> None:
        
        self.train_matrix = train_matrix.copy()
        n_users, n_items = train_matrix.shape
        
        # Convert to confidence matrix
        confidence = 1 + 40 * train_matrix  # c_ui = 1 + alpha * r_ui
        preference = (train_matrix > 0).astype(float)
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        reg = self.regularization * np.eye(self.n_factors)
        
        for iteration in range(self.iterations):
            # Fix items, solve for users
            for u in range(n_users):
                Cu = np.diag(confidence[u])
                A = self.item_factors.T @ Cu @ self.item_factors + reg
                b = self.item_factors.T @ Cu @ preference[u]
                self.user_factors[u] = _robust_linear_solve(A, b)
            
            # Fix users, solve for items
            for i in range(n_items):
                Ci = np.diag(confidence[:, i])
                A = self.user_factors.T @ Ci @ self.user_factors + reg
                b = self.user_factors.T @ Ci @ preference[:, i]
                self.item_factors[i] = _robust_linear_solve(A, b)
        
        self.is_trained = True
    
    def predict(self, user_ids: List[int], k: int = 10) -> List[List[int]]:
        return [self.recommend(u, k) for u in user_ids]
    
    def recommend(self, user_id: int, k: int = 10,
                  exclude_items: List[int] = None) -> List[int]:
        
        scores = self.user_factors[user_id] @ self.item_factors.T
        
        already_rated = self.train_matrix[user_id] > 0
        scores[already_rated] = -np.inf
        
        if exclude_items:
            for item in exclude_items:
                scores[item] = -np.inf
        
        return np.argsort(-scores)[:k].tolist()
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Minimize weighted squared error with confidence',
            'inductive_bias': 'Implicit feedback indicates varying confidence',
            'strengths': 'Handles implicit feedback, parallelizable, proven at scale',
            'weaknesses': 'Memory intensive, needs tuning',
            'best_for': 'Implicit feedback data (clicks, views, purchases)'
        }
    
    def get_model_size(self) -> float:
        if self.user_factors is None:
            return 0.0
        return (self.user_factors.nbytes + self.item_factors.nbytes) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return []