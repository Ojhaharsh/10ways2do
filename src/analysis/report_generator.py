"""
Generate comprehensive reports from benchmark results.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd


class ReportGenerator:
    """Generate markdown and HTML reports from results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.domains = {
            'domain_a': 'Information Extraction',
            'domain_b': 'Anomaly Detection', 
            'domain_c': 'Recommendation',
            'domain_d': 'Time Series Forecasting',
            'domain_e': 'Tabular Decisioning'
        }

    def _load_aggregated_rows(self, domain: str) -> List[Dict[str, Any]]:
        """Load successful aggregated rows for a domain when available."""
        aggregated_path = self.results_dir / domain / "results_aggregated.json"
        if not aggregated_path.exists():
            return []

        with open(aggregated_path) as f:
            payload = json.load(f)

        approaches = payload.get('approaches', []) if isinstance(payload, dict) else []
        return [row for row in approaches if isinstance(row, dict) and row.get('success', False)]
    
    def load_domain_results(self, domain: str) -> Optional[List[Dict]]:
        """Load results for a specific domain."""
        results_path = self.results_dir / domain / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                payload = json.load(f)
                if isinstance(payload, list):
                    return payload
                if isinstance(payload, dict) and isinstance(payload.get('approaches'), list):
                    return payload.get('approaches')

        # New format fallback
        aggregated_path = self.results_dir / domain / "results_aggregated.json"
        if aggregated_path.exists():
            with open(aggregated_path) as f:
                payload = json.load(f)

            approaches = payload.get('approaches', []) if isinstance(payload, dict) else []
            normalized = []

            for entry in approaches:
                if not entry.get('success', False):
                    continue

                metrics = entry.get('metrics_summary')
                if metrics is None:
                    metrics = entry.get('primary_metrics_summary', {})

                normalized_metrics = {}
                for metric_name, metric_stats in metrics.items():
                    if isinstance(metric_stats, dict):
                        normalized_metrics[metric_name] = metric_stats.get('mean', 0.0)

                timing_summary = entry.get('timing_summary', {})
                training_time = timing_summary.get('training_time', {}).get('mean', 0.0)

                normalized.append(
                    {
                        'name': entry.get('name', 'unknown'),
                        'metrics': normalized_metrics,
                        'training_time': training_time,
                        'significance_vs_best': entry.get('significance_vs_best'),
                        'success': True,
                    }
                )

            if normalized:
                return normalized

        return None
    
    def generate_domain_report(self, domain: str) -> str:
        """Generate report for a single domain."""
        results = self.load_domain_results(domain)
        if not results:
            return f"# {self.domains.get(domain, domain)}\n\nNo results available.\n"
        
        report = []
        domain_name = self.domains.get(domain, domain)
        
        report.append(f"# {domain_name}: Benchmark Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary table
        report.append("## Summary\n")
        
        summary_data = []
        for r in results:
            if r.get('success', False):
                metrics = r.get('metrics', {})
                primary = metrics.get('primary_metrics', metrics)
                
                row = {
                    'Approach': r['name'],
                    'Training Time (s)': f"{r.get('training_time', 0):.2f}",
                }
                
                # Add primary metrics
                for key, value in list(primary.items())[:3]:
                    if isinstance(value, (int, float)):
                        row[key] = f"{value:.4f}"
                
                summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            try:
                report.append(df.to_markdown(index=False))
            except ImportError:
                report.append(df.to_string(index=False))
        
        # Detailed results per approach
        report.append("\n## Detailed Results\n")
        
        for r in results:
            if not r.get('success', False):
                continue
            
            report.append(f"### {r['name']}\n")
            
            # Philosophy
            philosophy = r.get('philosophy', {})
            if philosophy:
                report.append("**Philosophy:**\n")
                report.append(f"- Mental Model: {philosophy.get('mental_model', 'N/A')}\n")
                report.append(f"- Strengths: {philosophy.get('strengths', 'N/A')}\n")
                report.append(f"- Weaknesses: {philosophy.get('weaknesses', 'N/A')}\n")
                report.append(f"- Best For: {philosophy.get('best_for', 'N/A')}\n")
            
            # Metrics
            metrics = r.get('metrics', {})
            report.append("\n**Metrics:**\n")
            
            primary = metrics.get('primary_metrics', {})
            for key, value in primary.items():
                if isinstance(value, (int, float)):
                    report.append(f"- {key}: {value:.4f}\n")
            
            # Systems metrics
            if metrics.get('inference_latency_p95'):
                report.append(f"- Latency p95: {metrics['inference_latency_p95']:.2f}ms\n")
            if metrics.get('memory_inference_mb'):
                report.append(f"- Memory: {metrics['memory_inference_mb']:.2f}MB\n")
            if metrics.get('model_size_mb'):
                report.append(f"- Model Size: {metrics['model_size_mb']:.2f}MB\n")
            
            # Failure cases
            failures = metrics.get('failure_cases', [])
            if failures:
                report.append("\n**Sample Failure Cases:**\n")
                for failure in failures[:3]:
                    report.append(f"- Index {failure.get('index', '?')}: {failure.get('reason', failure.get('type', 'Unknown'))}\n")
            
            report.append("\n")

        report.append("## Statistical Significance\n")
        report.append(
            "Cohen's d guide: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large.\n"
        )
        significance_lines = []

        def _cohens_d_label(value: Any) -> str:
            if not isinstance(value, (int, float)):
                return "N/A"
            abs_value = abs(float(value))
            if abs_value < 0.2:
                return "negligible"
            if abs_value < 0.5:
                return "small"
            if abs_value < 0.8:
                return "medium"
            return "large"

        for r in results:
            if not r.get('success', False):
                continue

            significance = r.get('significance_vs_best')
            if not isinstance(significance, dict):
                continue

            best_approach = significance.get('best_approach', 'N/A')
            p_value = significance.get('p_value')
            is_best = bool(significance.get('is_best', False))
            mean_diff = significance.get('mean_diff_vs_best', 0.0)
            effect_size = significance.get('cohens_d_vs_best')

            p_value_str = f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "N/A"
            effect_size_str = f"{effect_size:.4f}" if isinstance(effect_size, (int, float)) else "N/A"
            effect_size_label = _cohens_d_label(effect_size)
            if is_best:
                significance_lines.append(
                    f"- {r['name']}: reference best approach (p-value={p_value_str}, "
                    f"Cohen's d={effect_size_str}, effect={effect_size_label})"
                )
            else:
                significance_lines.append(
                    f"- {r['name']}: compared vs {best_approach}, "
                    f"p-value={p_value_str}, mean difference={mean_diff:.4f}, "
                    f"Cohen's d={effect_size_str}, effect={effect_size_label}"
                )

        if significance_lines:
            report.extend([line + "\n" for line in significance_lines])
        else:
            report.append("No significance information available for this domain.\n")
        
        # Key insights
        report.append("## Key Insights\n")
        report.append(self._generate_domain_insights(domain, results))
        
        return "\n".join(report)
    
    def _generate_domain_insights(self, domain: str, results: List[Dict]) -> str:
        """Generate domain-specific insights."""
        insights = []
        
        successful = [r for r in results if r.get('success', False)]
        if not successful:
            return "No successful results to analyze.\n"
        
        # Find best performers
        metrics_key = {
            'domain_a': ('primary_metrics', 'overall_exact_match'),
            'domain_b': ('', 'f1'),
            'domain_c': ('', 'ndcg@10'),
            'domain_d': ('', 'rmse'),
            'domain_e': ('', 'f1')
        }
        
        container, metric = metrics_key.get(domain, ('', 'accuracy'))
        
        best_perf = None
        best_name = None
        fastest = None
        fastest_name = None
        
        for r in successful:
            metrics = r.get('metrics', {})
            if container:
                value = metrics.get(container, {}).get(metric, 0)
            else:
                value = metrics.get(metric, 0)
            
            if value is not None:
                if domain == 'domain_d':  # Lower is better for RMSE
                    if best_perf is None or value < best_perf:
                        best_perf = value
                        best_name = r['name']
                else:
                    if best_perf is None or value > best_perf:
                        best_perf = value
                        best_name = r['name']
            
            train_time = r.get('training_time', float('inf'))
            if fastest is None or train_time < fastest:
                fastest = train_time
                fastest_name = r['name']
        
        if best_name:
            insights.append(f"1. **Best Performance**: {best_name} achieved the best {metric} of {best_perf:.4f}\n")
        
        if fastest_name:
            insights.append(f"2. **Fastest Training**: {fastest_name} trained in {fastest:.2f} seconds\n")
        
        # Philosophy insights
        insights.append("\n### Approach Categories:\n")
        
        rule_based = [r['name'] for r in successful if 'rule' in r['name'].lower() or 'statistical' in r['name'].lower()]
        ml_based = [r['name'] for r in successful if 'ml' in r['name'].lower() or 'tree' in r['name'].lower()]
        dl_based = [r['name'] for r in successful if any(x in r['name'].lower() for x in ['lstm', 'cnn', 'transformer', 'neural', 'deep'])]
        hybrid = [r['name'] for r in successful if 'hybrid' in r['name'].lower() or 'ensemble' in r['name'].lower()]
        
        if rule_based:
            insights.append(f"- **Rule/Statistical**: {', '.join(rule_based)} - High interpretability, low latency\n")
        if ml_based:
            insights.append(f"- **Classical ML**: {', '.join(ml_based)} - Good balance of performance and speed\n")
        if dl_based:
            insights.append(f"- **Deep Learning**: {', '.join(dl_based)} - Best accuracy, higher compute\n")
        if hybrid:
            insights.append(f"- **Hybrid**: {', '.join(hybrid)} - Production-ready combinations\n")
        
        return "".join(insights)
    
    def generate_full_report(self) -> str:
        """Generate complete cross-domain report."""
        report = []
        
        report.append("# ML Philosophy Benchmark: Complete Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## Overview\n")
        report.append("""
This benchmark evaluates 10 fundamentally different ML approaches across 4 real-world domains:

1. **Information Extraction** - Extracting structured data from text
2. **Anomaly Detection** - Identifying unusual patterns in data
3. **Recommendation** - Suggesting relevant items to users
4. **Time Series Forecasting** - Predicting future values

Each approach represents a different **mental model** for solving ML problems, with distinct 
trade-offs in terms of accuracy, speed, interpretability, data efficiency, and robustness.
""")
        
        # Per-domain reports
        for domain, name in self.domains.items():
            report.append(f"\n---\n")
            report.append(self.generate_domain_report(domain))
        
        # Cross-domain insights
        report.append("\n---\n")
        report.append("# Cross-Domain Insights\n")
        report.append(self._generate_cross_domain_insights())
        
        # Recommendations
        report.append("\n## Recommendations by Use Case\n")
        report.append(self._generate_recommendations())
        
        return "\n".join(report)
    
    def _generate_cross_domain_insights(self) -> str:
        """Generate insights comparing across domains."""
        dynamic_lines: List[str] = []

        dynamic_lines.append("## Cross-Domain Statistical Summary")
        winners = []

        for domain, domain_name in self.domains.items():
            rows = self._load_aggregated_rows(domain)
            if not rows:
                dynamic_lines.append(f"- {domain_name}: No aggregated results available.")
                continue

            best_row = next(
                (r for r in rows if isinstance(r.get('significance_vs_best'), dict) and r['significance_vs_best'].get('is_best')),
                None,
            )

            if best_row is None:
                dynamic_lines.append(f"- {domain_name}: No significance annotations available.")
                continue

            sig = best_row['significance_vs_best']
            best_name = best_row.get('name', sig.get('best_approach', 'N/A'))
            best_mean = sig.get('best_mean')
            direction = "higher-is-better" if sig.get('higher_is_better', True) else "lower-is-better"
            best_mean_str = f"{best_mean:.4f}" if isinstance(best_mean, (int, float)) else "N/A"

            competitors_significant = 0
            competitors_total = 0
            for row in rows:
                if row.get('name') == best_name:
                    continue
                row_sig = row.get('significance_vs_best')
                if not isinstance(row_sig, dict):
                    continue
                p_value = row_sig.get('p_value')
                if isinstance(p_value, (int, float)):
                    competitors_total += 1
                    if p_value < float(row_sig.get('alpha', 0.05)):
                        competitors_significant += 1

            dynamic_lines.append(
                f"- {domain_name}: best={best_name}, best_mean={best_mean_str}, "
                f"criterion={direction}, significant_vs_best={competitors_significant}/{competitors_total}."
            )
            winners.append(best_name)

        if winners:
            winner_counts: Dict[str, int] = {}
            for winner in winners:
                winner_counts[winner] = winner_counts.get(winner, 0) + 1
            top_winner = max(winner_counts, key=winner_counts.get)
            dynamic_lines.append(
                f"- Most frequent domain winner: {top_winner} ({winner_counts[top_winner]} domain(s))."
            )

        dynamic_block = "\n".join(dynamic_lines)

        return dynamic_block + "\n\n" + """
## Universal Patterns

### 1. Rule-Based Methods
- **Across all domains**: Highest interpretability, lowest latency
- **Trade-off**: Limited coverage, requires manual maintenance
- **Best when**: Patterns are well-defined and stable

### 2. Classical ML (Trees, Linear Models)
- **Across all domains**: Good performance with moderate data
- **Trade-off**: Requires feature engineering
- **Best when**: Interpretability matters, moderate scale

### 3. Deep Learning (RNN, CNN, Transformers)
- **Across all domains**: Best raw performance with sufficient data
- **Trade-off**: Higher compute, less interpretable
- **Best when**: Accuracy is paramount, data is abundant

### 4. Pretrained Models / Transfer Learning
- **Across all domains**: Reduces data requirements significantly
- **Trade-off**: Model size, domain mismatch possible
- **Best when**: Limited labeled data, similar pretrain domain

### 5. Hybrid Approaches
- **Across all domains**: Often best production choice
- **Trade-off**: More complex to maintain
- **Best when**: Reliability is critical, diverse inputs

### 6. Systems Considerations
- **Across all domains**: Often the deciding factor
- **Key metrics**: Latency, throughput, memory, cost
- **Best practice**: Always evaluate with production constraints
"""
    
    def _generate_recommendations(self) -> str:
        """Generate use-case based recommendations."""
        return """
### By Priority:

**Maximum Accuracy:**
- Use pretrained transformers (BERT, etc.)
- Ensemble multiple approaches
- Accept higher latency and cost

**Low Latency (<10ms):**
- Rule-based or simple ML models
- Heavy caching for repeated queries
- Model distillation from larger models

**Limited Training Data (<1000 samples):**
- Transfer learning / pretrained models
- Rule-based augmentation
- Few-shot learning with LLMs

**High Interpretability:**
- Rule-based systems
- Linear models with feature importance
- Decision trees / SHAP explanations

**Production Reliability:**
- Hybrid approaches with fallbacks
- Ensemble for robustness
- Comprehensive monitoring

**Minimal Infrastructure:**
- Rule-based or classical ML
- Avoid GPU requirements
- Consider LLM APIs for complex tasks
"""
    
    def save_report(self, output_path: str = "results/REPORT.md") -> None:
        """Generate and save the full report."""
        report = self.generate_full_report()
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {output_path}")