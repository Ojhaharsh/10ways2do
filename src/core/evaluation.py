"""
Comprehensive evaluation framework
"""

from typing import Dict, List, Any, Optional, Tuple, Type
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .base_model import BaseApproach, ModelMetrics
from .metrics import MetricsComputer


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    approach_name: str
    philosophy: Dict[str, str]
    metrics: ModelMetrics
    insights: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'approach_name': self.approach_name,
            'philosophy': self.philosophy,
            'metrics': self.metrics.to_dict(),
            'insights': self.insights
        }


class Evaluator:
    """
    Comprehensive evaluator for comparing approaches.
    """
    
    def __init__(self, domain: str, config: Optional[Dict] = None):
        """
        Args:
            domain: One of 'ie', 'anomaly', 'recommendation', 'timeseries'
            config: Evaluation configuration
        """
        self.domain = domain
        self.config = config or {}
        self.results: List[EvaluationResult] = []
    
    def evaluate_approach(
        self,
        approach: BaseApproach,
        X_train: Any, y_train: Any,
        X_val: Any, y_val: Any,
        X_test: Any, y_test: Any,
        add_noise_fn: Optional[callable] = None,
        **metric_kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single approach comprehensively.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {approach.name}")
        print(f"{'='*60}")
        
        # 1. Train with timing
        print("Training...")
        training_time = approach.train_with_timing(X_train, y_train, X_val, y_val)
        print(f"  Training time: {training_time:.2f}s")
        
        # 2. Get predictions
        print("Predicting...")
        y_pred = approach.predict(X_test)
        
        # 3. Compute primary metrics
        print("Computing primary metrics...")
        primary_metrics = MetricsComputer.compute(
            self.domain, y_test, y_pred, **metric_kwargs
        )
        approach.metrics.primary_metrics = primary_metrics
        
        for name, value in list(primary_metrics.items())[:5]:
            print(f"  {name}: {value:.4f}")
        
        # 4. Measure latency
        print("Measuring latency...")
        if hasattr(X_test, '__len__') and len(X_test) > 0:
            if isinstance(X_test, np.ndarray):
                X_single = X_test[:1]
            elif isinstance(X_test, list):
                X_single = [X_test[0]]
            else:
                X_single = X_test[:1]
            
            latency = approach.measure_latency(X_single)
            print(f"  Latency p50: {latency['p50']:.2f}ms, p95: {latency['p95']:.2f}ms")
        
        # 5. Measure memory
        print("Measuring memory...")
        approach.measure_memory_inference(X_test[:100] if len(X_test) > 100 else X_test)
        print(f"  Inference memory: {approach.metrics.memory_inference_mb:.2f}MB")
        
        # 6. Data efficiency
        print("Evaluating data efficiency...")
        sample_sizes = self.config.get('efficiency_samples', [100, 500, 1000, 2000, 5000])
        
        def eval_fn(y_true, y_pred):
            return MetricsComputer.compute(self.domain, y_true, y_pred, **metric_kwargs)
        
        approach.evaluate_data_efficiency(
            X_train, y_train, X_test, y_test,
            sample_sizes, eval_fn
        )
        
        # 7. Robustness
        if add_noise_fn is not None:
            print("Evaluating robustness...")
            noise_levels = self.config.get('noise_levels', [0.0, 0.1, 0.2, 0.3])
            approach.evaluate_robustness(
                X_test, y_test, noise_levels, add_noise_fn, eval_fn
            )
        
        # 8. Collect failure cases
        print("Collecting failure cases...")
        approach.metrics.failure_cases = approach.collect_failure_cases(
            X_test, y_test, y_pred, n_cases=10
        )
        
        # 9. Get model size
        approach.metrics.model_size_mb = approach.get_model_size()
        
        # 10. Generate insights
        insights = self._generate_insights(approach)
        
        result = EvaluationResult(
            approach_name=approach.name,
            philosophy=approach.get_philosophy(),
            metrics=approach.metrics,
            insights=insights
        )
        
        self.results.append(result)
        return result
    
    def evaluate_all(
        self,
        approaches: List[BaseApproach],
        X_train: Any, y_train: Any,
        X_val: Any, y_val: Any,
        X_test: Any, y_test: Any,
        add_noise_fn: Optional[callable] = None,
        **metric_kwargs
    ) -> List[EvaluationResult]:
        """Evaluate all approaches"""
        results = []
        
        for approach in tqdm(approaches, desc="Evaluating approaches"):
            try:
                result = self.evaluate_approach(
                    approach,
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    add_noise_fn,
                    **metric_kwargs
                )
                results.append(result)
            except Exception as e:
                print(f"Error evaluating {approach.name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def _generate_insights(self, approach: BaseApproach) -> Dict[str, Any]:
        """Generate insights from evaluation results"""
        metrics = approach.metrics
        insights = {}
        
        # Data efficiency insight
        if metrics.data_efficiency:
            sizes = sorted(metrics.data_efficiency.keys())
            if len(sizes) >= 2:
                first_metric = list(metrics.data_efficiency[sizes[0]].values())[0]
                last_metric = list(metrics.data_efficiency[sizes[-1]].values())[0]
                
                if first_metric > 0:
                    improvement = (last_metric - first_metric) / first_metric
                    insights['data_efficiency_improvement'] = improvement
                    
                    if improvement < 0.1:
                        insights['data_efficiency_verdict'] = 'Data efficient - works well with small data'
                    elif improvement < 0.3:
                        insights['data_efficiency_verdict'] = 'Moderate data needs'
                    else:
                        insights['data_efficiency_verdict'] = 'Data hungry - needs large datasets'
        
        # Robustness insight
        if metrics.robustness:
            noise_levels = sorted(metrics.robustness.keys())
            if len(noise_levels) >= 2:
                first_metric = list(metrics.robustness[noise_levels[0]].values())[0]
                last_metric = list(metrics.robustness[noise_levels[-1]].values())[0]
                
                if first_metric > 0:
                    degradation = (first_metric - last_metric) / first_metric
                    insights['robustness_degradation'] = degradation
                    
                    if degradation < 0.1:
                        insights['robustness_verdict'] = 'Highly robust to noise'
                    elif degradation < 0.3:
                        insights['robustness_verdict'] = 'Moderately robust'
                    else:
                        insights['robustness_verdict'] = 'Sensitive to noise'
        
        # Latency insight
        if metrics.inference_latency_p95 > 0:
            if metrics.inference_latency_p95 < 10:
                insights['latency_verdict'] = 'Real-time capable (<10ms)'
            elif metrics.inference_latency_p95 < 100:
                insights['latency_verdict'] = 'Interactive speed (<100ms)'
            elif metrics.inference_latency_p95 < 1000:
                insights['latency_verdict'] = 'Batch processing speed (<1s)'
            else:
                insights['latency_verdict'] = 'Slow - needs optimization'
        
        return insights
    
    def compare(self) -> pd.DataFrame:
        """Create comparison table of all evaluated approaches"""
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for result in self.results:
            row = {
                'Approach': result.approach_name,
                'Philosophy': result.philosophy.get('mental_model', ''),
            }
            
            # Primary metrics
            for name, value in result.metrics.primary_metrics.items():
                row[f'Metric_{name}'] = value
            
            # Systems metrics
            row['Training_Time_s'] = result.metrics.training_time
            row['Latency_p95_ms'] = result.metrics.inference_latency_p95
            row['Memory_MB'] = result.metrics.memory_inference_mb
            row['Model_Size_MB'] = result.metrics.model_size_mb
            
            # Interpretability & Maintenance
            row['Interpretability'] = result.metrics.interpretability_score
            row['Maintenance_Complexity'] = result.metrics.maintenance_complexity
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_results(self, path: str) -> None:
        """Save results to JSON"""
        results_dict = [r.to_dict() for r in self.results]
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
    
    def generate_report(self) -> str:
        """Generate markdown report"""
        report = []
        report.append(f"# Evaluation Report: {self.domain.upper()}\n")
        report.append(f"## Summary\n")
        report.append(f"Evaluated {len(self.results)} approaches\n\n")
        
        # Comparison table
        df = self.compare()
        report.append("## Comparison Table\n")
        report.append(df.to_markdown(index=False))
        report.append("\n\n")
        
        # Per-approach details
        report.append("## Detailed Analysis\n")
        for result in self.results:
            report.append(f"### {result.approach_name}\n")
            report.append(f"**Philosophy:** {result.philosophy.get('mental_model', 'N/A')}\n\n")
            report.append(f"**Strengths:** {result.philosophy.get('strengths', 'N/A')}\n\n")
            report.append(f"**Weaknesses:** {result.philosophy.get('weaknesses', 'N/A')}\n\n")
            
            if result.insights:
                report.append("**Insights:**\n")
                for key, value in result.insights.items():
                    report.append(f"- {key}: {value}\n")
            
            report.append("\n")
        
        return "\n".join(report)