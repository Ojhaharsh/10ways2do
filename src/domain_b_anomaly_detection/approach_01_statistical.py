"""
Approach 1: Statistical Methods for Anomaly Detection

Philosophy: Use statistical properties to identify outliers.
- Z-score, IQR, Grubbs test
- Assumes data follows known distributions
- Simple, interpretable, fast
"""

from typing import Dict, List, Any, Optional
import numpy as np
from scipy import stats

from ..core.base_model import BaseApproach


class StatisticalAnomalyDetector(BaseApproach):
    """
    Statistical anomaly detection using Z-score and IQR.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Statistical (Z-score + IQR)", config)
        
        self.method = config.get('method', 'zscore') if config else 'zscore'
        self.threshold = config.get('threshold', 3.0) if config else 3.0
        self.iqr_multiplier = config.get('iqr_multiplier', 1.5) if config else 1.5
        
        self.means = None
        self.stds = None
        self.q1 = None
        self.q3 = None
        self.iqr = None
        
        self.metrics.interpretability_score = 1.0
        self.metrics.maintenance_complexity = 0.1
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        """Learn statistics from training data (typically normal samples)"""
        
        # Use only normal samples if labels provided
        if y_train is not None:
            normal_mask = y_train == 0
            X_normal = X_train[normal_mask]
        else:
            X_normal = X_train
        
        # Z-score parameters
        self.means = np.mean(X_normal, axis=0)
        self.stds = np.std(X_normal, axis=0) + 1e-8
        
        # IQR parameters
        self.q1 = np.percentile(X_normal, 25, axis=0)
        self.q3 = np.percentile(X_normal, 75, axis=0)
        self.iqr = self.q3 - self.q1 + 1e-8
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        scores = self.score(X)
        return (scores > 0.5).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (0-1, higher = more anomalous)"""
        
        if self.method == 'zscore':
            z_scores = np.abs((X - self.means) / self.stds)
            max_z = np.max(z_scores, axis=1)
            # Convert to 0-1 score
            scores = 1 - np.exp(-max_z / self.threshold)
            
        elif self.method == 'iqr':
            lower = self.q1 - self.iqr_multiplier * self.iqr
            upper = self.q3 + self.iqr_multiplier * self.iqr
            
            below = (X < lower).any(axis=1)
            above = (X > upper).any(axis=1)
            scores = (below | above).astype(float)
            
        elif self.method == 'combined':
            # Combine Z-score and IQR
            z_scores = np.abs((X - self.means) / self.stds)
            z_anomaly = np.max(z_scores, axis=1) > self.threshold
            
            lower = self.q1 - self.iqr_multiplier * self.iqr
            upper = self.q3 + self.iqr_multiplier * self.iqr
            iqr_anomaly = ((X < lower) | (X > upper)).any(axis=1)
            
            scores = (z_anomaly | iqr_anomaly).astype(float)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return scores
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Anomalies are statistically unlikely under assumed distribution',
            'inductive_bias': 'Data is normally distributed (or at least unimodal)',
            'strengths': 'Simple, fast, interpretable, no training needed',
            'weaknesses': 'Assumes normality, ignores temporal patterns, univariate focus',
            'best_for': 'Quick baseline, interpretable results, Gaussian-like data'
        }
    
    def get_model_size(self) -> float:
        if self.means is None:
            return 0.0
        return (self.means.nbytes + self.stds.nbytes + 
                self.q1.nbytes + self.q3.nbytes) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        
        # False negatives (missed anomalies)
        fn_mask = (y_test == 1) & (y_pred == 0)
        fn_indices = np.where(fn_mask)[0][:n_cases // 2]
        
        for idx in fn_indices:
            z_scores = np.abs((X_test[idx] - self.means) / self.stds)
            failures.append({
                'index': int(idx),
                'type': 'false_negative',
                'z_scores': z_scores.tolist(),
                'max_z': float(np.max(z_scores)),
                'reason': f'Max Z-score {np.max(z_scores):.2f} below threshold {self.threshold}'
            })
        
        # False positives
        fp_mask = (y_test == 0) & (y_pred == 1)
        fp_indices = np.where(fp_mask)[0][:n_cases // 2]
        
        for idx in fp_indices:
            z_scores = np.abs((X_test[idx] - self.means) / self.stds)
            failures.append({
                'index': int(idx),
                'type': 'false_positive',
                'z_scores': z_scores.tolist(),
                'max_z': float(np.max(z_scores)),
                'reason': f'Normal point with high Z-score {np.max(z_scores):.2f}'
            })
        
        return failures


class MADDetector(BaseApproach):
    """
    Median Absolute Deviation based detector.
    More robust to outliers than Z-score.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Statistical (MAD)", config)
        
        self.threshold = config.get('threshold', 3.5) if config else 3.5
        self.medians = None
        self.mads = None
        
        self.metrics.interpretability_score = 1.0
        self.metrics.maintenance_complexity = 0.1
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        
        if y_train is not None:
            X_normal = X_train[y_train == 0]
        else:
            X_normal = X_train
        
        self.medians = np.median(X_normal, axis=0)
        self.mads = np.median(np.abs(X_normal - self.medians), axis=0) + 1e-8
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores > 0.5).astype(int)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        # Modified Z-score using MAD
        modified_z = 0.6745 * np.abs(X - self.medians) / self.mads
        max_z = np.max(modified_z, axis=1)
        return (max_z > self.threshold).astype(float)
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Use robust statistics (median, MAD) resistant to outliers',
            'inductive_bias': 'Outliers should not influence the baseline',
            'strengths': 'Robust to existing outliers in training, simple',
            'weaknesses': 'Still ignores temporal patterns, assumes symmetry',
            'best_for': 'Contaminated training data, heavy-tailed distributions'
        }
    
    def get_model_size(self) -> float:
        if self.medians is None:
            return 0.0
        return (self.medians.nbytes + self.mads.nbytes) / (1024 * 1024)
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        failures = []
        wrong = y_test != y_pred
        for idx in np.where(wrong)[0][:n_cases]:
            failures.append({
                'index': int(idx),
                'type': 'false_positive' if y_pred[idx] == 1 else 'false_negative',
                'values': X_test[idx].tolist()
            })
        return failures