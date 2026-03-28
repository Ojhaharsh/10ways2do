"""
Comprehensive metrics for all domains
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
)
from collections import defaultdict


# ==================== INFORMATION EXTRACTION METRICS ====================

def compute_ie_metrics(y_true: List[Dict], y_pred: List[Dict],
                        fields: List[str]) -> Dict[str, float]:
    """
    Compute information extraction metrics.
    
    Args:
        y_true: List of ground truth dictionaries
        y_pred: List of predicted dictionaries
        fields: List of field names to evaluate
    
    Returns:
        Dictionary of metrics
    """
    exact_matches = defaultdict(list)
    partial_matches = defaultdict(list)
    
    for true, pred in zip(y_true, y_pred):
        for field in fields:
            true_val = true.get(field, "")
            pred_val = pred.get(field, "")
            
            # Exact match
            exact_matches[field].append(true_val == pred_val)
            
            # Partial match (Jaccard similarity)
            if true_val and pred_val:
                true_tokens = set(str(true_val).lower().split())
                pred_tokens = set(str(pred_val).lower().split())
                if true_tokens or pred_tokens:
                    jaccard = len(true_tokens & pred_tokens) / len(true_tokens | pred_tokens)
                    partial_matches[field].append(jaccard)
                else:
                    partial_matches[field].append(1.0)
            elif not true_val and not pred_val:
                partial_matches[field].append(1.0)
            else:
                partial_matches[field].append(0.0)
    
    # Aggregate metrics
    metrics = {}
    
    for field in fields:
        metrics[f'{field}_exact_match'] = np.mean(exact_matches[field])
        metrics[f'{field}_partial_match'] = np.mean(partial_matches[field])
    
    # Overall metrics
    all_exact = [v for vals in exact_matches.values() for v in vals]
    all_partial = [v for vals in partial_matches.values() for v in vals]
    
    metrics['overall_exact_match'] = np.mean(all_exact)
    metrics['overall_partial_match'] = np.mean(all_partial)
    metrics['overall_f1'] = 2 * metrics['overall_exact_match'] * metrics['overall_partial_match'] / \
                           (metrics['overall_exact_match'] + metrics['overall_partial_match'] + 1e-8)
    
    return metrics


def compute_ner_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, float]:
    """
    Compute NER-style metrics for token-level extraction.
    
    Args:
        y_true: List of true label sequences
        y_pred: List of predicted label sequences
    
    Returns:
        Dictionary of metrics
    """
    # Flatten
    true_flat = [label for seq in y_true for label in seq]
    pred_flat = [label for seq in y_pred for label in seq]
    
    # Get unique labels (excluding 'O')
    labels = list(set(true_flat + pred_flat) - {'O'})
    
    metrics = {
        'accuracy': accuracy_score(true_flat, pred_flat),
        'macro_f1': f1_score(true_flat, pred_flat, labels=labels, average='macro', zero_division=0),
        'micro_f1': f1_score(true_flat, pred_flat, labels=labels, average='micro', zero_division=0),
        'weighted_f1': f1_score(true_flat, pred_flat, labels=labels, average='weighted', zero_division=0)
    }
    
    # Per-label metrics
    for label in labels:
        binary_true = [1 if l == label else 0 for l in true_flat]
        binary_pred = [1 if l == label else 0 for l in pred_flat]
        
        metrics[f'{label}_precision'] = precision_score(binary_true, binary_pred, zero_division=0)
        metrics[f'{label}_recall'] = recall_score(binary_true, binary_pred, zero_division=0)
        metrics[f'{label}_f1'] = f1_score(binary_true, binary_pred, zero_division=0)
    
    return metrics


# ==================== ANOMALY DETECTION METRICS ====================

def compute_anomaly_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute anomaly detection metrics.
    
    Args:
        y_true: Binary ground truth (1 = anomaly)
        y_pred: Binary predictions (1 = anomaly)
        y_scores: Anomaly scores (higher = more anomalous)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # False positive rate (critical for anomaly detection)
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # If we have scores, compute ranking metrics
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            metrics['avg_precision'] = average_precision_score(y_true, y_scores)
        except ValueError:
            metrics['roc_auc'] = 0.0
            metrics['avg_precision'] = 0.0
        
        # Precision at K
        for k in [10, 50, 100]:
            if len(y_scores) >= k:
                top_k_indices = np.argsort(y_scores)[-k:]
                metrics[f'precision_at_{k}'] = np.sum(y_true[top_k_indices]) / k
    
    return metrics


# ==================== RECOMMENDATION METRICS ====================

def compute_ranking_metrics(y_true: List[List[int]], y_pred: List[List[int]],
                            k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute ranking/recommendation metrics.
    
    Args:
        y_true: List of relevant item lists per user
        y_pred: List of recommended item lists per user (ordered by relevance)
        k_values: K values for precision/recall@K
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Precision, Recall, F1 at K
    for k in k_values:
        precisions = []
        recalls = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            true_set = set(true_items)
            pred_set = set(pred_items[:k])
            
            if len(pred_set) > 0:
                precision = len(true_set & pred_set) / len(pred_set)
            else:
                precision = 0.0
            
            if len(true_set) > 0:
                recall = len(true_set & pred_set) / len(true_set)
            else:
                recall = 1.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        metrics[f'precision@{k}'] = np.mean(precisions)
        metrics[f'recall@{k}'] = np.mean(recalls)
        
        p, r = metrics[f'precision@{k}'], metrics[f'recall@{k}']
        metrics[f'f1@{k}'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    # NDCG
    for k in k_values:
        ndcgs = []
        for true_items, pred_items in zip(y_true, y_pred):
            ndcg = compute_ndcg(true_items, pred_items[:k])
            ndcgs.append(ndcg)
        metrics[f'ndcg@{k}'] = np.mean(ndcgs)
    
    # Mean Reciprocal Rank
    mrrs = []
    for true_items, pred_items in zip(y_true, y_pred):
        true_set = set(true_items)
        for i, item in enumerate(pred_items):
            if item in true_set:
                mrrs.append(1.0 / (i + 1))
                break
        else:
            mrrs.append(0.0)
    metrics['mrr'] = np.mean(mrrs)
    
    # Hit Rate
    for k in k_values:
        hits = []
        for true_items, pred_items in zip(y_true, y_pred):
            true_set = set(true_items)
            pred_set = set(pred_items[:k])
            hits.append(1.0 if len(true_set & pred_set) > 0 else 0.0)
        metrics[f'hit_rate@{k}'] = np.mean(hits)
    
    # Coverage (unique items recommended)
    all_recommended = set()
    for pred_items in y_pred:
        all_recommended.update(pred_items[:max(k_values)])
    
    all_true = set()
    for true_items in y_true:
        all_true.update(true_items)
    
    metrics['coverage'] = len(all_recommended) / len(all_true) if all_true else 0
    
    return metrics


def compute_ndcg(true_items: List[int], pred_items: List[int]) -> float:
    """Compute NDCG for a single user"""
    true_set = set(true_items)
    
    # DCG
    dcg = 0.0
    for i, item in enumerate(pred_items):
        if item in true_set:
            dcg += 1.0 / np.log2(i + 2)
    
    # Ideal DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), len(pred_items))))
    
    return dcg / idcg if idcg > 0 else 0.0


# ==================== TIME SERIES METRICS ====================

def compute_timeseries_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                naive_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute time series forecasting metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        naive_pred: Naive baseline predictions (for relative metrics)
    
    Returns:
        Dictionary of metrics
    """
    # Remove NaN/Inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
    }
    
    # MAPE (handle zeros)
    mask_nonzero = y_true != 0
    if mask_nonzero.sum() > 0:
        metrics['mape'] = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
    else:
        metrics['mape'] = float('inf')
    
    # Symmetric MAPE
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask_nonzero = denominator != 0
    if mask_nonzero.sum() > 0:
        metrics['smape'] = np.mean(2 * np.abs(y_true[mask_nonzero] - y_pred[mask_nonzero]) / denominator[mask_nonzero]) * 100
    else:
        metrics['smape'] = float('inf')
    
    # MASE (if naive prediction provided)
    if naive_pred is not None:
        naive_pred = naive_pred[mask]
        naive_mae = mean_absolute_error(y_true, naive_pred)
        if naive_mae > 0:
            metrics['mase'] = metrics['mae'] / naive_mae
        else:
            metrics['mase'] = float('inf')
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Direction accuracy (for financial applications)
    if len(y_true) > 1:
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        metrics['direction_accuracy'] = np.mean(true_direction == pred_direction)
    
    return metrics


# ==================== UNIFIED EVALUATION ====================

class MetricsComputer:
    """Unified metrics computation across domains"""
    
    @staticmethod
    def compute(domain: str, y_true: Any, y_pred: Any, 
                **kwargs) -> Dict[str, float]:
        """
        Compute metrics for a given domain.
        
        Args:
            domain: One of 'ie', 'anomaly', 'recommendation', 'timeseries'
            y_true: Ground truth
            y_pred: Predictions
            **kwargs: Domain-specific arguments
        """
        if domain == 'ie':
            return compute_ie_metrics(y_true, y_pred, kwargs.get('fields', []))
        elif domain == 'ner':
            return compute_ner_metrics(y_true, y_pred)
        elif domain == 'anomaly':
            return compute_anomaly_metrics(y_true, y_pred, kwargs.get('y_scores'))
        elif domain == 'recommendation':
            return compute_ranking_metrics(y_true, y_pred, kwargs.get('k_values', [5, 10, 20]))
        elif domain == 'timeseries':
            return compute_timeseries_metrics(y_true, y_pred, kwargs.get('naive_pred'))
        else:
            raise ValueError(f"Unknown domain: {domain}")