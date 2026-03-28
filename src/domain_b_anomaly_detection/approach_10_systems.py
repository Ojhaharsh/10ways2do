"""
Approach 10: Systems Perspective for Anomaly Detection

Philosophy: Production-ready monitoring and alerting.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import time
from collections import deque
from dataclasses import dataclass

from ..core.base_model import BaseApproach


@dataclass
class AlertConfig:
    """Configuration for alerting."""
    threshold: float = 0.5
    min_consecutive: int = 3
    cooldown_seconds: float = 60.0
    severity_levels: Dict[float, str] = None
    
    def __post_init__(self):
        if self.severity_levels is None:
            self.severity_levels = {
                0.5: 'warning',
                0.7: 'critical',
                0.9: 'emergency'
            }


class StreamingAnomalyDetector(BaseApproach):
    """
    Production-ready streaming anomaly detector with:
    - Online updates
    - Alerting
    - Metrics collection
    """
    
    def __init__(self, base_detector: BaseApproach, config: Optional[Dict] = None):
        super().__init__(f"Streaming({base_detector.name})", config)
        
        self.detector = base_detector
        self.alert_config = AlertConfig()
        
        # Streaming state
        self.window_size = config.get('window_size', 100) if config else 100
        self.recent_scores = deque(maxlen=self.window_size)
        self.recent_predictions = deque(maxlen=self.window_size)
        
        # Alert state
        self.consecutive_anomalies = 0
        self.last_alert_time = 0
        self.alert_history = []
        
        # Statistics
        self.total_processed = 0
        self.total_anomalies = 0
        self.processing_times = deque(maxlen=1000)
        
        self.metrics.interpretability_score = self.detector.metrics.interpretability_score
        self.metrics.maintenance_complexity = self.detector.metrics.maintenance_complexity + 0.2
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        self.detector.train(X_train, y_train, X_val, y_val)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.detector.predict(X)
    
    def process_stream(self, x: np.ndarray) -> Dict[str, Any]:
        """Process a single data point from a stream."""
        start_time = time.time()
        
        # Get prediction and score
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        score = float(self.detector.score(x)[0])
        prediction = int(score > self.alert_config.threshold)
        
        # Update state
        self.recent_scores.append(score)
        self.recent_predictions.append(prediction)
        self.total_processed += 1
        
        # Check for alert
        alert = None
        if prediction == 1:
            self.total_anomalies += 1
            self.consecutive_anomalies += 1
            
            if self.consecutive_anomalies >= self.alert_config.min_consecutive:
                current_time = time.time()
                if current_time - self.last_alert_time > self.alert_config.cooldown_seconds:
                    alert = self._create_alert(score)
                    self.last_alert_time = current_time
        else:
            self.consecutive_anomalies = 0
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return {
            'score': score,
            'prediction': prediction,
            'alert': alert,
            'processing_time_ms': processing_time * 1000,
            'stats': self.get_stats()
        }
    
    def _create_alert(self, score: float) -> Dict[str, Any]:
        """Create an alert."""
        severity = 'info'
        for threshold, sev in sorted(self.alert_config.severity_levels.items()):
            if score >= threshold:
                severity = sev
        
        alert = {
            'timestamp': time.time(),
            'severity': severity,
            'score': score,
            'consecutive_count': self.consecutive_anomalies,
            'recent_anomaly_rate': sum(self.recent_predictions) / len(self.recent_predictions)
        }
        
        self.alert_history.append(alert)
        return alert
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        return {
            'total_processed': self.total_processed,
            'total_anomalies': self.total_anomalies,
            'anomaly_rate': self.total_anomalies / max(1, self.total_processed),
            'recent_anomaly_rate': sum(self.recent_predictions) / max(1, len(self.recent_predictions)),
            'avg_score': np.mean(self.recent_scores) if self.recent_scores else 0,
            'avg_processing_time_ms': np.mean(self.processing_times) * 1000 if self.processing_times else 0,
            'p95_processing_time_ms': np.percentile(self.processing_times, 95) * 1000 if len(self.processing_times) > 10 else 0
        }
    
    def get_philosophy(self) -> Dict[str, str]:
        return {
            'mental_model': 'Production wrapper with streaming, alerting, and monitoring',
            'inductive_bias': 'Same as underlying detector',
            'strengths': 'Production-ready, alerting, statistics, streaming',
            'weaknesses': 'Added complexity and overhead',
            'best_for': 'Production deployment of anomaly detection'
        }
    
    def get_model_size(self) -> float:
        return self.detector.get_model_size()
    
    def collect_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, n_cases: int = 10) -> List[Dict]:
        return self.detector.collect_failure_cases(X_test, y_test, y_pred, n_cases)