"""
Cross-domain analysis comparing approaches across all domains.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np


class CrossDomainAnalyzer:
    """Analyze and compare results across all domains."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.domains = ['domain_a', 'domain_b', 'domain_c', 'domain_d']
        self.domain_names = {
            'domain_a': 'Information Extraction',
            'domain_b': 'Anomaly Detection',
            'domain_c': 'Recommendation',
            'domain_d': 'Time Series'
        }
        self.results = {}
    
    def load_results(self) -> None:
        """Load results from all domains."""
        for domain in self.domains:
            results_path = self.results_dir / domain / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    self.results[domain] = json.load(f)
    
    def get_approach_categories(self) -> Dict[str, List[str]]:
        """Categorize approaches across domains."""
        return {
            'rule_based': ['Rule-Based', 'Statistical', 'Popularity'],
            'classical_ml': ['Classical ML', 'Distance-Based', 'Collaborative Filtering', 'Content-Based'],
            'tree_based': ['Tree-Based', 'Isolation Forest', 'Matrix Factorization'],
            'deep_learning': ['RNN/LSTM', 'CNN', 'Autoencoder', 'Deep Learning'],
            'transformer': ['Transformer', 'Pretrained Transformer'],
            'llm': ['Prompt-Based LLM'],
            'hybrid': ['Hybrid', 'Ensemble'],
            'systems': ['Systems', 'Streaming', 'Cached']
        }
    
    def compare_philosophies(self) -> pd.DataFrame:
        """Compare approach philosophies across domains."""
        rows = []
        
        for domain, results in self.results.items():
            for result in results:
                if result.get('success', False):
                    philosophy = result.get('philosophy', {})
                    rows.append({
                        'Domain': self.domain_names.get(domain, domain),
                        'Approach': result['name'],
                        'Mental Model': philosophy.get('mental_model', 'N/A'),
                        'Strengths': philosophy.get('strengths', 'N/A'),
                        'Weaknesses': philosophy.get('weaknesses', 'N/A'),
                        'Best For': philosophy.get('best_for', 'N/A')
                    })
        
        return pd.DataFrame(rows)
    
    def analyze_trade_offs(self) -> Dict[str, Any]:
        """Analyze trade-offs across approaches."""
        analysis = {
            'accuracy_vs_speed': [],
            'interpretability_vs_performance': [],
            'data_efficiency': [],
            'robustness': []
        }
        
        for domain, results in self.results.items():
            for result in results:
                if not result.get('success', False):
                    continue
                
                metrics = result.get('metrics', {})
                
                # Extract relevant metrics based on domain
                if domain == 'domain_a':
                    accuracy = metrics.get('primary_metrics', {}).get('overall_exact_match', 0)
                elif domain == 'domain_b':
                    accuracy = metrics.get('f1', 0)
                elif domain == 'domain_c':
                    accuracy = metrics.get('ndcg@10', 0)
                else:
                    accuracy = 1.0 / (metrics.get('rmse', 1) + 0.001)
                
                training_time = result.get('training_time', 0)
                
                analysis['accuracy_vs_speed'].append({
                    'domain': domain,
                    'approach': result['name'],
                    'accuracy': accuracy,
                    'training_time': training_time
                })
        
        return analysis
    
    def generate_insights(self) -> List[str]:
        """Generate cross-domain insights."""
        insights = []
        
        insights.append("## Cross-Domain Insights\n")
        
        insights.append("### 1. Rule-Based Methods")
        insights.append("- Consistently high precision across all domains")
        insights.append("- Zero training time, fully interpretable")
        insights.append("- Limited by coverage and maintenance burden\n")
        
        insights.append("### 2. Classical ML")
        insights.append("- Good balance of performance and interpretability")
        insights.append("- Requires careful feature engineering")
        insights.append("- Works well with moderate data sizes\n")
        
        insights.append("### 3. Deep Learning")
        insights.append("- Best raw performance with sufficient data")
        insights.append("- Higher computational cost and complexity")
        insights.append("- Less interpretable, harder to debug\n")
        
        insights.append("### 4. Transformers")
        insights.append("- State-of-the-art on many tasks")
        insights.append("- Require more data and compute")
        insights.append("- Pre-training significantly helps\n")
        
        insights.append("### 5. Hybrid Approaches")
        insights.append("- Often best production choice")
        insights.append("- Combine strengths of multiple methods")
        insights.append("- More complex to maintain\n")
        
        insights.append("### 6. Systems Considerations")
        insights.append("- Latency requirements often dictate choices")
        insights.append("- Caching and batching essential for scale")
        insights.append("- Monitoring and fallbacks critical\n")
        
        return insights
    
    def create_summary_report(self) -> str:
        """Create comprehensive summary report."""
        self.load_results()
        
        report = []
        report.append("# ML Philosophy Benchmark: Cross-Domain Analysis\n")
        
        # Philosophy comparison
        report.append("## Approach Philosophies\n")
        df = self.compare_philosophies()
        if not df.empty:
            report.append(df.to_markdown(index=False))
        
        # Trade-offs
        report.append("\n## Trade-off Analysis\n")
        trade_offs = self.analyze_trade_offs()
        
        if trade_offs['accuracy_vs_speed']:
            report.append("### Accuracy vs Training Time\n")
            for item in trade_offs['accuracy_vs_speed'][:10]:
                report.append(f"- {item['approach']} ({item['domain']}): "
                            f"Accuracy={item['accuracy']:.3f}, Time={item['training_time']:.2f}s")
        
        # Insights
        report.append("\n")
        report.extend(self.generate_insights())
        
        return "\n".join(report)