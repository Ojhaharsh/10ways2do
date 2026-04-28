"""Advanced metrics: cost, latency, fairness, and robustness analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PerformanceMetrics:
    """Captures performance metrics beyond accuracy."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class CostMetrics:
    """Captures computational cost metrics."""
    training_time_seconds: float
    inference_time_ms_per_sample: float
    peak_memory_mb: float
    model_size_mb: float
    flops_billions: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class RobustnessMetrics:
    """Captures robustness to distribution shift and adversarial perturbations."""
    accuracy_clean: float
    accuracy_noisy: float
    accuracy_adversarial: float
    robustness_score: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class FairnessMetrics:
    """Captures fairness across demographic groups."""
    overall_accuracy: float
    group_min_accuracy: float
    group_max_accuracy: float
    disparity_ratio: float
    equalized_odds_diff: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class AdvancedMetricsCalculator:
    """Calculate advanced metrics for model evaluation."""
    
    @staticmethod
    def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> PerformanceMetrics:
        """Compute classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        auc_roc = None
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                from sklearn.metrics import roc_auc_score
                auc_roc = roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                pass
        
        return PerformanceMetrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1, auc_roc=auc_roc)
    
    @staticmethod
    def compute_robustness_metrics(
        model: Any,
        X_clean: np.ndarray,
        y_true: np.ndarray,
        noise_level: float = 0.1,
        seed: int = 42,
    ) -> RobustnessMetrics:
        """Compute robustness metrics under distribution shift."""
        from sklearn.metrics import accuracy_score
        
        # Clean accuracy
        try:
            y_pred_clean = model.predict(X_clean)
            accuracy_clean = accuracy_score(y_true, y_pred_clean)
        except Exception:
            accuracy_clean = 0.0
        
        # Noisy accuracy
        rng = np.random.RandomState(seed)
        X_noisy = X_clean + rng.normal(0, noise_level, X_clean.shape)
        try:
            y_pred_noisy = model.predict(X_noisy)
            accuracy_noisy = accuracy_score(y_true, y_pred_noisy)
        except Exception:
            accuracy_noisy = 0.0
        
        # Adversarial accuracy (simple FGSM-like perturbation)
        X_adversarial = X_clean + np.sign(rng.randn(*X_clean.shape)) * (noise_level * 2)
        try:
            y_pred_adversarial = model.predict(X_adversarial)
            accuracy_adversarial = accuracy_score(y_true, y_pred_adversarial)
        except Exception:
            accuracy_adversarial = 0.0
        
        robustness_score = (accuracy_clean + accuracy_noisy + accuracy_adversarial) / 3.0
        
        return RobustnessMetrics(
            accuracy_clean=accuracy_clean,
            accuracy_noisy=accuracy_noisy,
            accuracy_adversarial=accuracy_adversarial,
            robustness_score=robustness_score,
        )
    
    @staticmethod
    def compute_fairness_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
    ) -> FairnessMetrics:
        """Compute fairness metrics across demographic groups."""
        from sklearn.metrics import accuracy_score
        
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        group_accuracies = []
        for group_id in np.unique(groups):
            mask = groups == group_id
            if mask.sum() > 0:
                group_acc = accuracy_score(y_true[mask], y_pred[mask])
                group_accuracies.append(group_acc)
        
        if not group_accuracies:
            return FairnessMetrics(
                overall_accuracy=overall_accuracy,
                group_min_accuracy=0.0,
                group_max_accuracy=0.0,
                disparity_ratio=1.0,
                equalized_odds_diff=0.0,
            )
        
        group_min_accuracy = min(group_accuracies)
        group_max_accuracy = max(group_accuracies)
        disparity_ratio = group_max_accuracy / (group_min_accuracy + 1e-8)
        equalized_odds_diff = group_max_accuracy - group_min_accuracy
        
        return FairnessMetrics(
            overall_accuracy=overall_accuracy,
            group_min_accuracy=group_min_accuracy,
            group_max_accuracy=group_max_accuracy,
            disparity_ratio=disparity_ratio,
            equalized_odds_diff=equalized_odds_diff,
        )


class CostTracker:
    """Track computational cost during model training and inference."""
    
    def __init__(self):
        self.training_start_time: Optional[float] = None
        self.inference_times: List[float] = []
        self.peak_memory_mb: float = 0.0
        self.model_size_mb: float = 0.0
    
    def start_training(self):
        """Mark start of training."""
        self.training_start_time = time.time()
    
    def end_training(self) -> float:
        """Mark end of training and return elapsed time."""
        if self.training_start_time is None:
            return 0.0
        elapsed = time.time() - self.training_start_time
        self.training_start_time = None
        return elapsed
    
    def record_inference_time(self, time_ms: float):
        """Record inference time for a sample."""
        self.inference_times.append(time_ms)
    
    def get_cost_metrics(self) -> CostMetrics:
        """Get aggregated cost metrics."""
        training_time = 0.0
        if self.training_start_time is not None:
            training_time = time.time() - self.training_start_time
        
        inference_per_sample = np.mean(self.inference_times) if self.inference_times else 0.0
        
        return CostMetrics(
            training_time_seconds=training_time,
            inference_time_ms_per_sample=inference_per_sample,
            peak_memory_mb=self.peak_memory_mb,
            model_size_mb=self.model_size_mb,
            flops_billions=None,
        )


def save_advanced_metrics(
    metrics: Dict[str, Any],
    output_dir: str,
    metric_type: str = "advanced",
) -> None:
    """Save advanced metrics to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filepath = Path(output_dir) / f"metrics_{metric_type}.json"
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def load_advanced_metrics(
    output_dir: str,
    metric_type: str = "advanced",
) -> Dict[str, Any]:
    """Load advanced metrics from JSON."""
    filepath = Path(output_dir) / f"metrics_{metric_type}.json"
    if not filepath.exists():
        return {}
    
    with open(filepath, "r") as f:
        return json.load(f)
