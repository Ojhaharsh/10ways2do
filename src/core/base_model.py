"""
Base model class and interfaces for all approaches
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import time
import traceback
import psutil
import numpy as np
from memory_profiler import memory_usage


@dataclass
class ModelMetrics:
    """Container for all model metrics"""
    # Performance metrics (domain-specific)
    primary_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Data efficiency
    data_efficiency: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Robustness
    robustness: Dict[float, Dict[str, float]] = field(default_factory=dict)
    
    # Systems metrics
    training_time: float = 0.0
    inference_latency_mean: float = 0.0
    inference_latency_std: float = 0.0
    inference_latency_p50: float = 0.0
    inference_latency_p95: float = 0.0
    inference_latency_p99: float = 0.0
    memory_training_mb: float = 0.0
    memory_inference_mb: float = 0.0
    model_size_mb: float = 0.0
    throughput_qps: float = 0.0
    
    # Interpretability (0-1 score)
    interpretability_score: float = 0.0
    
    # Maintenance complexity (0-1 score, lower is better)
    maintenance_complexity: float = 0.0
    
    # Failure cases
    failure_cases: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'primary_metrics': self.primary_metrics,
            'data_efficiency': self.data_efficiency,
            'robustness': self.robustness,
            'training_time': self.training_time,
            'inference_latency_mean': self.inference_latency_mean,
            'inference_latency_std': self.inference_latency_std,
            'inference_latency_p50': self.inference_latency_p50,
            'inference_latency_p95': self.inference_latency_p95,
            'inference_latency_p99': self.inference_latency_p99,
            'memory_training_mb': self.memory_training_mb,
            'memory_inference_mb': self.memory_inference_mb,
            'model_size_mb': self.model_size_mb,
            'throughput_qps': self.throughput_qps,
            'interpretability_score': self.interpretability_score,
            'maintenance_complexity': self.maintenance_complexity,
            'failure_cases': self.failure_cases,
            'metadata': self.metadata
        }


class BaseApproach(ABC):
    """
    Base class for all modeling approaches.
    
    Each approach must implement:
    - train(): Training logic
    - predict(): Inference logic
    - get_philosophy(): Description of the approach's mental model
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.is_trained = False
        self.metrics = ModelMetrics()
        self._model = None
    
    @abstractmethod
    def train(self, X_train: Any, y_train: Any, 
              X_val: Optional[Any] = None, y_val: Optional[Any] = None) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_philosophy(self) -> Dict[str, str]:
        """
        Return a description of this approach's philosophy.
        
        Returns:
            Dict with keys:
            - 'mental_model': Core idea
            - 'inductive_bias': What assumptions does it make?
            - 'strengths': Where does it excel?
            - 'weaknesses': Where does it fail?
            - 'best_for': Ideal use cases
        """
        pass
    
    def train_with_timing(self, X_train: Any, y_train: Any,
                          X_val: Optional[Any] = None, 
                          y_val: Optional[Any] = None) -> float:
        """Train with timing measurement"""
        start_time = time.time()
        
        # Measure memory during training
        mem_usage = memory_usage(
            (self.train, (X_train, y_train, X_val, y_val)),
            max_usage=True
        )
        
        training_time = time.time() - start_time
        
        self.metrics.training_time = training_time
        self.metrics.memory_training_mb = mem_usage
        self.is_trained = True
        
        return training_time
    
    def measure_latency(self, X_single: Any, n_iterations: int = 100,
                        warmup: int = 10) -> Dict[str, float]:
        """Measure inference latency"""
        if not self.is_trained:
            raise ValueError("Model must be trained before measuring latency")
        
        # Warmup
        for _ in range(warmup):
            self.predict(X_single)
        
        # Measure
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.predict(X_single)
            latencies.append(time.perf_counter() - start)
        
        latencies = np.array(latencies) * 1000  # Convert to ms
        
        self.metrics.inference_latency_mean = float(np.mean(latencies))
        self.metrics.inference_latency_std = float(np.std(latencies))
        self.metrics.inference_latency_p50 = float(np.percentile(latencies, 50))
        self.metrics.inference_latency_p95 = float(np.percentile(latencies, 95))
        self.metrics.inference_latency_p99 = float(np.percentile(latencies, 99))
        
        return {
            'mean': self.metrics.inference_latency_mean,
            'std': self.metrics.inference_latency_std,
            'p50': self.metrics.inference_latency_p50,
            'p95': self.metrics.inference_latency_p95,
            'p99': self.metrics.inference_latency_p99
        }
    
    def measure_memory_inference(self, X: Any) -> float:
        """Measure memory usage during inference"""
        mem_usage = memory_usage(
            (self.predict, (X,)),
            max_usage=True
        )
        self.metrics.memory_inference_mb = mem_usage
        return mem_usage
    
    def evaluate_data_efficiency(self, X_train: Any, y_train: Any,
                                  X_test: Any, y_test: Any,
                                  sample_sizes: List[int],
                                  evaluation_fn: callable) -> Dict[int, Dict[str, float]]:
        """
        Evaluate model performance with different training set sizes.
        
        Args:
            X_train: Full training features
            y_train: Full training labels
            X_test: Test features
            y_test: Test labels
            sample_sizes: List of training sizes to test
            evaluation_fn: Function(y_true, y_pred) -> Dict[str, float]
        """
        results = {}
        
        for size in sample_sizes:
            if size > len(X_train):
                continue
                
            # Sample training data
            indices = np.random.choice(len(X_train), size, replace=False)
            
            if isinstance(X_train, np.ndarray):
                X_subset = X_train[indices]
                y_subset = y_train[indices]
            else:
                X_subset = [X_train[i] for i in indices]
                y_subset = [y_train[i] for i in indices]
            
            # Train fresh model
            self._reset_model()
            self.train(X_subset, y_subset)
            
            # Evaluate
            y_pred = self.predict(X_test)
            metrics = evaluation_fn(y_test, y_pred)
            results[size] = metrics
        
        self.metrics.data_efficiency = results
        return results
    
    def evaluate_robustness(self, X_test: Any, y_test: Any,
                            noise_levels: List[float],
                            add_noise_fn: callable,
                            evaluation_fn: callable) -> Dict[float, Dict[str, float]]:
        """
        Evaluate model robustness to noise.
        
        Args:
            X_test: Test features
            y_test: Test labels
            noise_levels: List of noise levels to test
            add_noise_fn: Function(X, noise_level) -> X_noisy
            evaluation_fn: Function(y_true, y_pred) -> Dict[str, float]
        """
        results = {}
        
        for noise in noise_levels:
            X_noisy = add_noise_fn(X_test, noise)
            y_pred = self.predict(X_noisy)
            metrics = evaluation_fn(y_test, y_pred)
            results[noise] = metrics
        
        self.metrics.robustness = results
        return results
    
    def collect_failure_cases(self, X_test: Any, y_test: Any,
                               y_pred: Any, n_cases: int = 10) -> List[Dict]:
        """Collect and analyze failure cases"""
        # To be overridden by specific implementations
        return []
    
    def _reset_model(self) -> None:
        """Reset model for fresh training"""
        self._model = None
        self.is_trained = False
    
    def get_model_size(self) -> float:
        """Get model size in MB"""
        # To be overridden by specific implementations
        return 0.0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', trained={self.is_trained})"