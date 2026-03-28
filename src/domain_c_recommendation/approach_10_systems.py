"""
Approach 10: Systems Perspective for Recommendations

Philosophy: Production considerations.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import time
from collections import defaultdict

from ..core.base_model import BaseApproach


class CachedRecommender(BaseApproach):
    """Production recommender with caching and monitoring."""
    
    def __init__(self, base_recommender: BaseApproach, config: Optional[Dict] = None):
        super().__init__(f"Cached({base_recommender.name})", config)
        
        self.base = base_recommender
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.latencies = []
        
        self.cache_ttl = config.get('cache_ttl', 3600) if config else 3600
        self.max_cache_size = config.get('max_cache_size', 10000) if config else 10000
        
        self.metrics.interpretability_score = self.base.metrics.interpretability_score
        self.metrics.maintenance_complexity = self.base.metrics.maintenance_complexity + 0.2
    
    def train(self, train_matrix: np.ndarray, train_interactions: List = None,
              X_val: Any = None, y_val: Any = None) -> None:
        self.base.train(train_matrix, train_interactions, X_val, y_val)
        self.cache.clear()
        self.is_trained = True
    
    def predict(self, user_ids: List[int], k: int = 10) -> List[List[int]]:
        return [self.recommend(u, k) for u in user_ids]
    
    def recommend(self, user_id: int, k: int = 10,
                  exclude_items: List[int] = None) -> List[int]:
        
        start = time.time()
        
        cache_key = (user_id, k, tuple(exclude_items) if exclude_items else None)
        
        if cache_key in self.cache:
            self.cache_hits += 1
            result = self.cache[cache_key]
        else:
            self.cache_misses += 1
            result = self.base.recommend(user_id, k, exclude_items)
            
            if len(self.cache) < self.max_cache_size:
                self.cache[cache_key] = result
        
        self.latencies.append(time.time() - start)
        
        return result
    
    def get_stats(self) -> Dict[str, float]:
        total_requests = self.cache_hits + self.cache_misses
        return {
            'cache_hit_rate': self.cache_hits / max(1, total_requests),
            'cache_size': len(self.cache),
            'avg_latency_ms': np.mean(self.latencies) * 1000 if self.latencies else 0,
            'p95_latency_ms': np.percentile(self.latencies, 95) * 1000 if len(self.latencies) > 10 else 0
        }
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Add production features: caching, monitoring, fallbacks',
            'inductive_bias': 'Same as base model',
            'strengths': 'Production-ready, low latency with cache, observable',
            'weaknesses': 'Cache staleness, memory overhead',
            'best_for': 'Production deployment'
        }
    
    def get_model_size(self) -> float:
        return self.base.get_model_size()
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        return self.base.collect_failure_cases(X_test, y_test, y_pred, n_cases)