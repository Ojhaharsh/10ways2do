"""Leaderboard and dashboard generation for benchmark results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class LeaderboardEntry:
    """Single leaderboard entry."""
    
    def __init__(
        self,
        domain: str,
        approach: str,
        accuracy: float,
        speed_rank: Optional[int] = None,
        cost_rank: Optional[int] = None,
        fairness_score: Optional[float] = None,
        robustness_score: Optional[float] = None,
        timestamp: Optional[str] = None,
    ):
        self.domain = domain
        self.approach = approach
        self.accuracy = accuracy
        self.speed_rank = speed_rank
        self.cost_rank = cost_rank
        self.fairness_score = fairness_score
        self.robustness_score = robustness_score
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.composite_score = self._compute_composite_score()
    
    def _compute_composite_score(self) -> float:
        """Compute weighted composite score."""
        score = self.accuracy
        if self.robustness_score is not None:
            score = 0.7 * score + 0.3 * self.robustness_score
        if self.fairness_score is not None:
            score = 0.8 * score + 0.2 * self.fairness_score
        return score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain,
            "approach": self.approach,
            "accuracy": self.accuracy,
            "composite_score": self.composite_score,
            "speed_rank": self.speed_rank,
            "cost_rank": self.cost_rank,
            "fairness_score": self.fairness_score,
            "robustness_score": self.robustness_score,
            "timestamp": self.timestamp,
        }


class LeaderboardGenerator:
    """Generate leaderboards from benchmark results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.entries: List[LeaderboardEntry] = []
    
    def add_entry(self, entry: LeaderboardEntry):
        """Add entry to leaderboard."""
        self.entries.append(entry)
    
    def generate_domain_leaderboard(self, domain: str) -> pd.DataFrame:
        """Generate per-domain leaderboard."""
        domain_entries = [e for e in self.entries if e.domain == domain]
        if not domain_entries:
            return pd.DataFrame()
        
        df = pd.DataFrame([e.to_dict() for e in domain_entries])
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    
    def generate_overall_leaderboard(self) -> pd.DataFrame:
        """Generate overall cross-domain leaderboard."""
        if not self.entries:
            return pd.DataFrame()
        
        # Group by approach and compute average scores
        approach_scores = {}
        for entry in self.entries:
            if entry.approach not in approach_scores:
                approach_scores[entry.approach] = []
            approach_scores[entry.approach].append(entry.composite_score)
        
        leaderboard_data = []
        for approach, scores in approach_scores.items():
            leaderboard_data.append({
                "approach": approach,
                "avg_composite_score": sum(scores) / len(scores),
                "n_domains": len(scores),
                "std_dev": (sum((s - (sum(scores) / len(scores))) ** 2 for s in scores) / len(scores)) ** 0.5,
            })
        
        df = pd.DataFrame(leaderboard_data)
        return df.sort_values("avg_composite_score", ascending=False).reset_index(drop=True)
    
    def save_leaderboards(self, output_dir: str = "results"):
        """Save all leaderboards."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save overall
        overall_lb = self.generate_overall_leaderboard()
        overall_lb.to_csv(Path(output_dir) / "leaderboard_overall.csv", index=False)
        
        # Save per-domain
        domains = set(e.domain for e in self.entries)
        for domain in sorted(domains):
            domain_lb = self.generate_domain_leaderboard(domain)
            domain_lb.to_csv(Path(output_dir) / f"leaderboard_{domain}.csv", index=False)
        
        # Save JSON version
        leaderboard_json = {
            "generated_at": datetime.utcnow().isoformat(),
            "overall": self.generate_overall_leaderboard().to_dict("records"),
            "by_domain": {domain: self.generate_domain_leaderboard(domain).to_dict("records") for domain in domains},
        }
        
        with open(Path(output_dir) / "leaderboard.json", "w") as f:
            json.dump(leaderboard_json, f, indent=2)


class DashboardGenerator:
    """Generate dashboard HTML and metrics visualizations."""
    
    @staticmethod
    def generate_dashboard_html(
        leaderboard_dir: str = "results",
        output_file: str = "dashboard.html",
    ) -> None:
        """Generate dashboard HTML."""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>ML Philosophy Benchmark Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-card h3 {{ margin-top: 0; color: #0066cc; }}
        .stat-card .value {{ font-size: 32px; font-weight: bold; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #0066cc; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 8px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; }}
        .timestamp {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ML Philosophy Benchmark Dashboard</h1>
        <p>16 Domains × 10 Approaches = 160 Method Comparisons</p>
        <p class="timestamp">Generated: {datetime.utcnow().isoformat()}</p>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Domains</h3>
                <div class="value">16</div>
                <p>A-P coverage including vision, graphs, RL, and multimodal</p>
            </div>
            <div class="stat-card">
                <h3>Approaches per Domain</h3>
                <div class="value">10</div>
                <p>From rule-based to modern neural architectures</p>
            </div>
            <div class="stat-card">
                <h3>Total Comparisons</h3>
                <div class="value">160</div>
                <p>Complete approach matrix across all domains</p>
            </div>
            <div class="stat-card">
                <h3>Advanced Metrics</h3>
                <div class="value">8+</div>
                <p>Cost, fairness, robustness, and leaderboards</p>
            </div>
        </div>
        
        <h2>Domains Overview</h2>
        <table>
            <tr>
                <th>Domain</th>
                <th>Category</th>
                <th>Key Metrics</th>
                <th>Status</th>
            </tr>
            <tr><td>A-K</td><td>Original Domains</td><td>Accuracy, F1, AUC</td><td>✓ Complete</td></tr>
            <tr><td>L</td><td>Computer Vision</td><td>ImageNet-style classification</td><td>✓ New</td></tr>
            <tr><td>M</td><td>Graph Neural Networks</td><td>Link prediction, scalability</td><td>✓ New</td></tr>
            <tr><td>N</td><td>Few-Shot Learning</td><td>N-way K-shot accuracy</td><td>✓ New</td></tr>
            <tr><td>O</td><td>Reinforcement Learning</td><td>Regret, cumulative reward</td><td>✓ New</td></tr>
            <tr><td>P</td><td>Multimodal Learning</td><td>Cross-modal alignment, accuracy</td><td>✓ New</td></tr>
        </table>
        
        <h2>Key Features</h2>
        <ul>
            <li><strong>16 Domains</strong>: Comprehensive coverage across ML subfields</li>
            <li><strong>Advanced Metrics</strong>: Cost tracking, fairness assessment, robustness testing</li>
            <li><strong>Reproducibility</strong>: Deterministic seeding, multi-run validation</li>
            <li><strong>Leaderboards</strong>: Per-domain and cross-domain rankings</li>
            <li><strong>CI/CD Integration</strong>: Automated benchmarking and release gates</li>
            <li><strong>Executive Summaries</strong>: Benchmark cards for governance and reporting</li>
        </ul>
    </div>
</body>
</html>
"""
        
        with open(output_file, "w") as f:
            f.write(html_content)
        
        print(f"Dashboard generated: {output_file}")
