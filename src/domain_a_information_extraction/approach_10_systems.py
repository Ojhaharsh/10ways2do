"""
Approach 10: Systems Perspective for Information Extraction

Philosophy: Evaluate production readiness beyond accuracy.
- Latency requirements
- Memory constraints  
- Scalability
- Cost analysis
- Monitoring and observability
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import threading
import queue
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import json

from ..core.base_model import BaseApproach


@dataclass
class SystemsMetrics:
    """Comprehensive systems metrics"""
    # Latency
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_max_ms: float = 0.0
    
    # Throughput
    throughput_qps: float = 0.0
    throughput_batch: float = 0.0
    
    # Memory
    memory_idle_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_per_request_mb: float = 0.0
    
    # Scalability
    scaling_efficiency: Dict[int, float] = field(default_factory=dict)
    
    # Cost (estimated)
    cost_per_1k_requests: float = 0.0
    
    # Reliability
    error_rate: float = 0.0
    timeout_rate: float = 0.0


class SystemsEvaluator:
    """
    Evaluates models from a systems perspective.
    """
    
    def __init__(self, model: BaseApproach):
        self.model = model
        self.metrics = SystemsMetrics()
    
    def evaluate_latency(self, X_test: List[str], 
                         n_iterations: int = 100,
                         warmup: int = 10) -> Dict[str, float]:
        """Measure inference latency distribution"""
        
        # Warmup
        for i in range(min(warmup, len(X_test))):
            self.model.predict([X_test[i]])
        
        # Measure single-item latency
        latencies = []
        for i in range(n_iterations):
            idx = i % len(X_test)
            start = time.perf_counter()
            self.model.predict([X_test[idx]])
            latencies.append((time.perf_counter() - start) * 1000)
        
        latencies = np.array(latencies)
        
        self.metrics.latency_p50_ms = float(np.percentile(latencies, 50))
        self.metrics.latency_p95_ms = float(np.percentile(latencies, 95))
        self.metrics.latency_p99_ms = float(np.percentile(latencies, 99))
        self.metrics.latency_max_ms = float(np.max(latencies))
        
        return {
            'p50': self.metrics.latency_p50_ms,
            'p95': self.metrics.latency_p95_ms,
            'p99': self.metrics.latency_p99_ms,
            'max': self.metrics.latency_max_ms
        }
    
    def evaluate_throughput(self, X_test: List[str],
                            duration_seconds: float = 10.0) -> Dict[str, float]:
        """Measure throughput (queries per second)"""
        
        # Single-threaded throughput
        start = time.time()
        count = 0
        while time.time() - start < duration_seconds:
            idx = count % len(X_test)
            self.model.predict([X_test[idx]])
            count += 1
        
        self.metrics.throughput_qps = count / duration_seconds
        
        # Batch throughput
        batch_sizes = [1, 8, 16, 32]
        batch_throughputs = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(X_test):
                continue
            
            start = time.time()
            count = 0
            while time.time() - start < duration_seconds / 4:
                batch = X_test[:batch_size]
                self.model.predict(batch)
                count += batch_size
            
            batch_throughputs[batch_size] = count / (time.time() - start)
        
        self.metrics.throughput_batch = max(batch_throughputs.values()) if batch_throughputs else 0
        
        return {
            'single_qps': self.metrics.throughput_qps,
            'batch_throughputs': batch_throughputs,
            'max_throughput': self.metrics.throughput_batch
        }
    
    def evaluate_memory(self, X_test: List[str]) -> Dict[str, float]:
        """Measure memory usage"""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Idle memory
        process = psutil.Process()
        self.metrics.memory_idle_mb = process.memory_info().rss / (1024 * 1024)
        
        # Peak memory during batch processing
        peak_memory = self.metrics.memory_idle_mb
        
        for i in range(min(100, len(X_test))):
            self.model.predict([X_test[i]])
            current = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current)
        
        self.metrics.memory_peak_mb = peak_memory
        self.metrics.memory_per_request_mb = (peak_memory - self.metrics.memory_idle_mb) / 100
        
        return {
            'idle_mb': self.metrics.memory_idle_mb,
            'peak_mb': self.metrics.memory_peak_mb,
            'per_request_mb': self.metrics.memory_per_request_mb
        }
    
    def evaluate_scalability(self, X_test: List[str],
                             worker_counts: List[int] = [1, 2, 4, 8]) -> Dict[int, float]:
        """Evaluate scaling with concurrent workers"""
        
        base_throughput = None
        scaling_results = {}
        
        for n_workers in worker_counts:
            # Use threading for I/O bound, processing for CPU bound
            start = time.time()
            count = 0
            duration = 5.0
            
            def worker_task(items):
                return self.model.predict(items)
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                while time.time() - start < duration:
                    idx = count % len(X_test)
                    futures.append(executor.submit(worker_task, [X_test[idx]]))
                    count += 1
                    
                    # Limit pending futures
                    if len(futures) > n_workers * 10:
                        futures = [f for f in futures if not f.done()]
            
            throughput = count / duration
            
            if base_throughput is None:
                base_throughput = throughput
            
            efficiency = throughput / (base_throughput * n_workers) if base_throughput > 0 else 0
            scaling_results[n_workers] = efficiency
            self.metrics.scaling_efficiency[n_workers] = efficiency
        
        return scaling_results
    
    def evaluate_reliability(self, X_test: List[str],
                             n_iterations: int = 1000,
                             timeout_ms: float = 5000) -> Dict[str, float]:
        """Evaluate error and timeout rates"""
        
        errors = 0
        timeouts = 0
        
        for i in range(min(n_iterations, len(X_test) * 10)):
            idx = i % len(X_test)
            
            try:
                start = time.perf_counter()
                result = self.model.predict([X_test[idx]])
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                if elapsed_ms > timeout_ms:
                    timeouts += 1
                
                # Check for empty results
                if not result or not result[0]:
                    errors += 1
                    
            except Exception as e:
                errors += 1
        
        self.metrics.error_rate = errors / n_iterations
        self.metrics.timeout_rate = timeouts / n_iterations
        
        return {
            'error_rate': self.metrics.error_rate,
            'timeout_rate': self.metrics.timeout_rate,
            'success_rate': 1 - self.metrics.error_rate
        }
    
    def estimate_cost(self, requests_per_day: int = 10000) -> Dict[str, float]:
        """Estimate operational costs"""
        
        # Compute cost based on resources
        # Assumptions for cloud pricing
        cpu_cost_per_hour = 0.05  # $/hour per vCPU
        memory_cost_per_gb_hour = 0.01  # $/hour per GB
        gpu_cost_per_hour = 1.0  # $/hour per GPU (if used)
        
        # Estimate resources needed
        time_per_request_hours = self.metrics.latency_p50_ms / 1000 / 3600
        memory_gb = self.metrics.memory_peak_mb / 1024
        
        # Daily compute time
        daily_compute_hours = requests_per_day * time_per_request_hours * 2  # 2x for overhead
        
        # Estimate cost
        cpu_cost = daily_compute_hours * cpu_cost_per_hour
        memory_cost = memory_gb * 24 * memory_cost_per_gb_hour  # 24h for always-on
        
        daily_cost = cpu_cost + memory_cost
        
        self.metrics.cost_per_1k_requests = (daily_cost / requests_per_day) * 1000
        
        return {
            'daily_cost': daily_cost,
            'monthly_cost': daily_cost * 30,
            'cost_per_1k_requests': self.metrics.cost_per_1k_requests
        }
    
    def full_evaluation(self, X_test: List[str]) -> Dict[str, Any]:
        """Run full systems evaluation"""
        
        results = {
            'latency': self.evaluate_latency(X_test),
            'throughput': self.evaluate_throughput(X_test, duration_seconds=5.0),
            'memory': self.evaluate_memory(X_test),
            'reliability': self.evaluate_reliability(X_test, n_iterations=100),
            'cost': self.estimate_cost()
        }
        
        # Add scalability if possible
        try:
            results['scalability'] = self.evaluate_scalability(X_test, [1, 2, 4])
        except:
            results['scalability'] = {}
        
        return results


class SystemsWrapper(BaseApproach):
    """
    Wrapper that adds systems monitoring to any model.
    """
    
    def __init__(self, base_model: BaseApproach, config: Optional[Dict] = None):
        super().__init__(f"Systems({base_model.name})", config)
        self.base_model = base_model
        self.evaluator = None
        
        # Metrics collection
        self.request_latencies = []
        self.request_count = 0
        self.error_count = 0
        
        self.metrics.interpretability_score = base_model.metrics.interpretability_score
        self.metrics.maintenance_complexity = base_model.metrics.maintenance_complexity + 0.1
    
    def train(self, X_train: List[str], y_train: List[Dict],
              X_val: Optional[List[str]] = None,
              y_val: Optional[List[Dict]] = None) -> None:
        """Train base model"""
        self.base_model.train(X_train, y_train, X_val, y_val)
        self.evaluator = SystemsEvaluator(self.base_model)
        self.is_trained = True
    
    def predict(self, X: List[str]) -> List[Dict[str, Any]]:
        """Predict with monitoring"""
        start = time.perf_counter()
        
        try:
            result = self.base_model.predict(X)
            self.request_count += len(X)
        except Exception as e:
            self.error_count += len(X)
            raise
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.request_latencies.append(elapsed / len(X))
        
        return result
    
    def get_live_metrics(self) -> Dict[str, float]:
        """Get live operational metrics"""
        if not self.request_latencies:
            return {}
        
        latencies = np.array(self.request_latencies[-1000:])  # Last 1000
        
        return {
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.request_count),
            'latency_p50': float(np.percentile(latencies, 50)),
            'latency_p95': float(np.percentile(latencies, 95)),
            'latency_mean': float(np.mean(latencies))
        }
    
    def run_systems_evaluation(self, X_test: List[str]) -> Dict[str, Any]:
        """Run comprehensive systems evaluation"""
        if self.evaluator is None:
            self.evaluator = SystemsEvaluator(self.base_model)
        return self.evaluator.full_evaluation(X_test)
    
    def get_philosophy(self) -> Dict[str, str]:
        base_philosophy = self.base_model.get_philosophy()
        return {
            'mental_model': f"Production wrapper around: {base_philosophy['mental_model']}",
            'inductive_bias': base_philosophy['inductive_bias'],
            'strengths': 'Adds monitoring, metrics, and operational insights',
            'weaknesses': 'Slight overhead from monitoring',
            'best_for': 'Production deployment, capacity planning, SLA management'
        }
    
    def get_model_size(self) -> float:
        return self.base_model.get_model_size()
    
    def collect_failure_cases(self, X_test: List[str], y_test: List[Dict],
                               y_pred: List[Dict], n_cases: int = 10) -> List[Dict]:
        return self.base_model.collect_failure_cases(X_test, y_test, y_pred, n_cases)