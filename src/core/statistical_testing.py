"""Advanced statistical testing for benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


class StatisticalTest:
    """Base class for statistical tests."""
    
    @staticmethod
    def welch_ttest(
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Perform Welch's t-test (does not assume equal variances)."""
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        
        mean_diff = group1.mean() - group2.mean()
        ci_95 = 1.96 * np.sqrt(group1.var() / len(group1) + group2.var() / len(group2))
        
        return {
            "test": "Welch's t-test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_alpha": float(alpha) if p_value < alpha else None,
            "mean_difference": float(mean_diff),
            "ci_95": float(ci_95),
            "effect_size": float(mean_diff / (np.sqrt((group1.var() + group2.var()) / 2))),
        }
    
    @staticmethod
    def mann_whitney_u_test(
        group1: np.ndarray,
        group2: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Perform Mann-Whitney U test (non-parametric)."""
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        
        return {
            "test": "Mann-Whitney U test",
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "significant_at_alpha": float(alpha) if p_value < alpha else None,
        }
    
    @staticmethod
    def kruskal_wallis_test(
        *groups: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Perform Kruskal-Wallis test (non-parametric ANOVA)."""
        h_stat, p_value = stats.kruskal(*groups)
        
        return {
            "test": "Kruskal-Wallis test",
            "h_statistic": float(h_stat),
            "p_value": float(p_value),
            "n_groups": len(groups),
            "significant_at_alpha": float(alpha) if p_value < alpha else None,
        }
    
    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Compute bootstrap confidence interval."""
        rng = np.random.RandomState(seed)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = rng.choice(data, size=len(data), replace=True)
            bootstrap_means.append(bootstrap_sample.mean())
        
        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            "test": "Bootstrap CI",
            "point_estimate": float(data.mean()),
            "ci_lower": float(np.percentile(bootstrap_means, lower_percentile)),
            "ci_upper": float(np.percentile(bootstrap_means, upper_percentile)),
            "confidence_level": confidence,
            "n_bootstrap": n_bootstrap,
        }
    
    @staticmethod
    def effect_size_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return float((group1.mean() - group2.mean()) / (pooled_std + 1e-8))


class ComparisonAnalyzer:
    """Analyze comparisons between approaches."""
    
    @staticmethod
    def pairwise_comparisons(
        results: Dict[str, List[float]],
        test_type: str = "welch",
    ) -> Dict[str, Any]:
        """Perform pairwise comparisons between all approaches."""
        from itertools import combinations
        
        comparisons = {}
        
        for (approach1, scores1), (approach2, scores2) in combinations(results.items(), 2):
            pair_name = f"{approach1}_vs_{approach2}"
            scores1_arr = np.array(scores1)
            scores2_arr = np.array(scores2)
            
            if test_type == "welch":
                comparison = StatisticalTest.welch_ttest(scores1_arr, scores2_arr)
            elif test_type == "mann-whitney":
                comparison = StatisticalTest.mann_whitney_u_test(scores1_arr, scores2_arr)
            else:
                comparison = {}
            
            comparison["mean_1"] = float(scores1_arr.mean())
            comparison["mean_2"] = float(scores2_arr.mean())
            comparison["winner"] = approach1 if scores1_arr.mean() > scores2_arr.mean() else approach2
            
            comparisons[pair_name] = comparison
        
        return comparisons
    
    @staticmethod
    def statistical_ranking(
        results: Dict[str, List[float]],
        confidence: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """Create statistically-grounded ranking."""
        rankings = []
        
        for approach, scores in results.items():
            scores_arr = np.array(scores)
            ci = StatisticalTest.bootstrap_confidence_interval(scores_arr, confidence=confidence)
            
            rankings.append({
                "approach": approach,
                "mean_score": float(scores_arr.mean()),
                "std_dev": float(scores_arr.std()),
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "n_runs": len(scores),
            })
        
        # Sort by mean score
        rankings.sort(key=lambda x: x["mean_score"], reverse=True)
        
        # Add rank and mark overlapping CIs
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1
            ranking["ci_overlaps"] = []
            
            for j, other in enumerate(rankings):
                if i != j:
                    # Check if confidence intervals overlap
                    if not (ranking["ci_upper"] < other["ci_lower"] or ranking["ci_lower"] > other["ci_upper"]):
                        ranking["ci_overlaps"].append(other["approach"])
        
        return rankings


class SignificanceReport:
    """Generate significance report for benchmark results."""
    
    @staticmethod
    def generate_report(
        results_by_approach: Dict[str, List[float]],
        output_dir: str = "results",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Generate comprehensive significance report."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Pairwise comparisons
        pairwise = ComparisonAnalyzer.pairwise_comparisons(results_by_approach)
        
        # Statistical ranking
        ranking = ComparisonAnalyzer.statistical_ranking(results_by_approach)
        
        # Omnibus test (Kruskal-Wallis)
        groups = [np.array(scores) for scores in results_by_approach.values()]
        omnibus = StatisticalTest.kruskal_wallis_test(*groups, alpha=alpha)
        
        report = {
            "generated_at": str(Path(output_dir).stem),
            "alpha": alpha,
            "omnibus_test": omnibus,
            "ranking": ranking,
            "pairwise_comparisons": pairwise,
            "n_approaches": len(results_by_approach),
            "n_comparisons": len(pairwise),
        }
        
        # Save report
        with open(Path(output_dir) / "significance_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report
