"""Distributed and auto-scaling benchmark runner."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .benchmark_utils import BENCHMARK_PROTOCOL_VERSION


@dataclass
class BenchmarkJob:
    """Single benchmark job."""
    domain: str
    seed: int
    job_id: str
    priority: int = 1


@dataclass
class BenchmarkResult:
    """Result of a benchmark job."""
    job_id: str
    domain: str
    seed: int
    status: str  # "success", "failed", "timeout"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    runtime_seconds: float = 0.0


class DistributedBenchmarkRunner:
    """Run benchmarks across multiple processes/machines."""
    
    def __init__(self, max_workers: Optional[int] = None, timeout_seconds: int = 3600):
        self.max_workers = max_workers or mp.cpu_count()
        self.timeout_seconds = timeout_seconds
        self.results: List[BenchmarkResult] = []
        self.job_queue: List[BenchmarkJob] = []
    
    def add_job(self, domain: str, seed: int, job_id: str, priority: int = 1):
        """Add job to queue."""
        job = BenchmarkJob(domain=domain, seed=seed, job_id=job_id, priority=priority)
        self.job_queue.append(job)
    
    def add_domain_jobs(self, domain: str, seeds: List[int], base_priority: int = 1):
        """Add multiple jobs for a domain."""
        for i, seed in enumerate(seeds):
            job_id = f"{domain}_seed_{seed}_run_{i}"
            self.add_job(domain, seed, job_id, priority=base_priority)
    
    def _execute_job(self, job: BenchmarkJob, run_fn: Callable) -> BenchmarkResult:
        """Execute single job."""
        start_time = time.time()
        
        try:
            result = run_fn(domain=job.domain, seed=job.seed, timeout=self.timeout_seconds)
            return BenchmarkResult(
                job_id=job.job_id,
                domain=job.domain,
                seed=job.seed,
                status="success",
                result=result,
                runtime_seconds=time.time() - start_time,
            )
        except asyncio.TimeoutError:
            return BenchmarkResult(
                job_id=job.job_id,
                domain=job.domain,
                seed=job.seed,
                status="timeout",
                error="Benchmark exceeded timeout",
                runtime_seconds=time.time() - start_time,
            )
        except Exception as e:
            return BenchmarkResult(
                job_id=job.job_id,
                domain=job.domain,
                seed=job.seed,
                status="failed",
                error=str(e),
                runtime_seconds=time.time() - start_time,
            )
    
    def run_parallel(self, run_fn: Callable, max_retries: int = 2) -> List[BenchmarkResult]:
        """Run all jobs in parallel."""
        # Sort by priority (higher priority first)
        jobs = sorted(self.job_queue, key=lambda j: j.priority, reverse=True)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._execute_job, job, run_fn): job for job in jobs}
            
            for future in futures:
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    self.results.append(result)
                except Exception as e:
                    job = futures[future]
                    self.results.append(
                        BenchmarkResult(
                            job_id=job.job_id,
                            domain=job.domain,
                            seed=job.seed,
                            status="failed",
                            error=f"Executor error: {str(e)}",
                        )
                    )
        
        return self.results
    
    def save_results(self, output_dir: str):
        """Save results to disk."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results_data = {
            "generated_at": str(Path(output_dir).stem),
            "protocol_version": BENCHMARK_PROTOCOL_VERSION,
            "total_jobs": len(self.job_queue),
            "successful_jobs": sum(1 for r in self.results if r.status == "success"),
            "failed_jobs": sum(1 for r in self.results if r.status == "failed"),
            "timeout_jobs": sum(1 for r in self.results if r.status == "timeout"),
            "total_runtime_seconds": sum(r.runtime_seconds for r in self.results),
            "results": [
                {
                    "job_id": r.job_id,
                    "domain": r.domain,
                    "seed": r.seed,
                    "status": r.status,
                    "runtime_seconds": r.runtime_seconds,
                    "error": r.error,
                }
                for r in self.results
            ],
        }
        
        with open(Path(output_dir) / "distributed_results.json", "w") as f:
            json.dump(results_data, f, indent=2)


class AutoScalingRunner:
    """Auto-scale benchmark resources based on load."""
    
    def __init__(self, base_workers: int = 4, max_workers: int = 32):
        self.base_workers = base_workers
        self.max_workers = max_workers
        self.current_workers = base_workers
        self.queue_length_history: List[int] = []
        self.execution_times: List[float] = []
    
    def estimate_required_workers(self, queue_length: int, avg_execution_time: float) -> int:
        """Estimate required workers based on queue and execution time."""
        self.queue_length_history.append(queue_length)
        
        if avg_execution_time > 0:
            estimated_workers = int(queue_length / (avg_execution_time + 1))
            estimated_workers = max(self.base_workers, min(estimated_workers, self.max_workers))
            return estimated_workers
        
        return self.current_workers
    
    def update_worker_count(self, queue_length: int, recent_execution_times: List[float]):
        """Update worker count based on current load."""
        if recent_execution_times:
            avg_time = sum(recent_execution_times) / len(recent_execution_times)
        else:
            avg_time = 0
        
        new_worker_count = self.estimate_required_workers(queue_length, avg_time)
        
        if new_worker_count != self.current_workers:
            print(f"Auto-scaling: {self.current_workers} -> {new_worker_count} workers")
            self.current_workers = new_worker_count
    
    def get_current_worker_count(self) -> int:
        """Get current worker count."""
        return self.current_workers


def create_domain_sweep_jobs(
    domains: List[str],
    seeds_per_domain: int = 5,
    base_seed: int = 42,
) -> List[BenchmarkJob]:
    """Create jobs for sweeping across all domains."""
    jobs = []
    job_counter = 0
    
    for domain in domains:
        for i in range(seeds_per_domain):
            seed = base_seed + i * 1000
            job_id = f"domain_{domain}_seed_{seed}"
            job = BenchmarkJob(
                domain=domain,
                seed=seed,
                job_id=job_id,
                priority=1,
            )
            jobs.append(job)
            job_counter += 1
    
    return jobs
