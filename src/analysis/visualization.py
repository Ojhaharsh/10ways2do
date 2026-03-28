"""
Visualization utilities for results.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class Visualizer:
    """Create visualizations for benchmark results."""
    
    def __init__(self, style: str = 'seaborn'):
        if HAS_PLOTTING:
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except:
                plt.style.use('ggplot')
            self.colors = sns.color_palette("husl", 12)
    
    def plot_comparison(self, results: List[Dict], metric: str,
                        title: str = None, save_path: str = None) -> None:
        """Plot comparison bar chart."""
        if not HAS_PLOTTING:
            print("Matplotlib not available for plotting")
            return
        
        names = []
        values = []
        
        for r in results:
            if r.get('success', False):
                names.append(r['name'][:20])  # Truncate long names
                metrics = r.get('metrics', {})
                if isinstance(metrics, dict) and 'primary_metrics' in metrics:
                    val = metrics['primary_metrics'].get(metric, 0)
                else:
                    val = metrics.get(metric, 0)
                values.append(float(val) if val else 0)
        
        if not names:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(names, values, color=self.colors[:len(names)])
        
        ax.set_xlabel(metric)
        ax.set_title(title or f'Comparison: {metric}')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_trade_off(self, results: List[Dict], 
                       x_metric: str, y_metric: str,
                       title: str = None, save_path: str = None) -> None:
        """Plot trade-off scatter plot."""
        if not HAS_PLOTTING:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, r in enumerate(results):
            if not r.get('success', False):
                continue
            
            metrics = r.get('metrics', {})
            x_val = metrics.get(x_metric, r.get(x_metric, 0))
            y_val = metrics.get(y_metric, r.get(y_metric, 0))
            
            if x_val is None:
                x_val = 0
            if y_val is None:
                y_val = 0
            
            ax.scatter(float(x_val), float(y_val), s=150, 
                      c=[self.colors[i % len(self.colors)]], 
                      label=r['name'][:25], alpha=0.7, edgecolors='black')
        
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(title or f'{y_metric} vs {x_metric}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_radar(self, results: List[Dict], metrics: List[str],
                   title: str = None, save_path: str = None) -> None:
        """Plot radar chart comparing approaches."""
        if not HAS_PLOTTING:
            return
        
        approaches = []
        all_values = []
        
        for r in results:
            if not r.get('success', False):
                continue
            approaches.append(r['name'][:15])
            
            row = []
            m = r.get('metrics', {})
            for metric in metrics:
                if isinstance(m, dict) and 'primary_metrics' in m:
                    val = m['primary_metrics'].get(metric, 0)
                else:
                    val = m.get(metric, 0)
                row.append(float(val) if val else 0)
            all_values.append(row)
        
        if not approaches or not all_values:
            return
        
        # Normalize values to 0-1
        all_values = np.array(all_values)
        mins = all_values.min(axis=0)
        maxs = all_values.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        normalized = (all_values - mins) / ranges
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, (approach, values) in enumerate(zip(approaches, normalized)):
            values = values.tolist() + [values[0]]
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=approach, color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.1, color=self.colors[i % len(self.colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        ax.set_title(title or 'Multi-Metric Comparison', size=14, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_data_efficiency(self, results: List[Dict], 
                             save_path: str = None) -> None:
        """Plot data efficiency curves."""
        if not HAS_PLOTTING:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, r in enumerate(results):
            if not r.get('success', False):
                continue
            
            metrics = r.get('metrics', {})
            efficiency = metrics.get('data_efficiency', {})
            
            if not efficiency:
                continue
            
            sizes = sorted([int(k) for k in efficiency.keys()])
            values = []
            for size in sizes:
                size_metrics = efficiency.get(str(size), efficiency.get(size, {}))
                if size_metrics:
                    val = list(size_metrics.values())[0] if size_metrics else 0
                    values.append(float(val))
            
            if sizes and values:
                ax.plot(sizes, values, 'o-', label=r['name'][:20],
                       color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Performance')
        ax.set_title('Data Efficiency Comparison')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_robustness(self, results: List[Dict], 
                        save_path: str = None) -> None:
        """Plot robustness to noise."""
        if not HAS_PLOTTING:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, r in enumerate(results):
            if not r.get('success', False):
                continue
            
            metrics = r.get('metrics', {})
            robustness = metrics.get('robustness', {})
            
            if not robustness:
                continue
            
            noise_levels = sorted([float(k) for k in robustness.keys()])
            values = []
            for noise in noise_levels:
                noise_metrics = robustness.get(str(noise), robustness.get(noise, {}))
                if noise_metrics:
                    val = list(noise_metrics.values())[0] if noise_metrics else 0
                    values.append(float(val))
            
            if noise_levels and values:
                ax.plot(noise_levels, values, 'o-', label=r['name'][:20],
                       color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Performance')
        ax.set_title('Robustness to Noise')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_all_plots(self, results: List[Dict], output_dir: str,
                         domain: str = "unknown") -> None:
        """Create all standard plots for a domain."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main comparison
        primary_metrics = {
            'domain_a': 'overall_exact_match',
            'domain_b': 'f1',
            'domain_c': 'ndcg@10',
            'domain_d': 'rmse'
        }
        
        metric = primary_metrics.get(domain, 'accuracy')
        self.plot_comparison(results, metric, 
                            f'{domain}: {metric}',
                            str(output_path / f'comparison_{metric}.png'))
        
        # Trade-off: accuracy vs training time
        self.plot_trade_off(results, 'training_time', metric,
                           f'{domain}: Performance vs Training Time',
                           str(output_path / 'tradeoff_time.png'))
        
        # Data efficiency
        self.plot_data_efficiency(results, str(output_path / 'data_efficiency.png'))
        
        # Robustness
        self.plot_robustness(results, str(output_path / 'robustness.png'))