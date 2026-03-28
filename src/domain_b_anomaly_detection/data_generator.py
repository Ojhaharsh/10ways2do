"""
Synthetic data generator for Anomaly Detection domain.
Generates multi-variate time series with various anomaly types.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AnomalyType(Enum):
    POINT = "point"           # Single point outlier
    CONTEXTUAL = "contextual" # Normal value, wrong context
    COLLECTIVE = "collective" # Pattern anomaly
    SEASONAL = "seasonal"     # Seasonal pattern break
    TREND = "trend"          # Sudden trend change
    LEVEL_SHIFT = "level"    # Permanent level change


@dataclass
class AnomalySpec:
    """Specification for an anomaly"""
    type: AnomalyType
    start_idx: int
    end_idx: int
    severity: float  # 1-5 scale
    feature_idx: int = 0


class TimeSeriesGenerator:
    """Generate synthetic time series with anomalies"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate_base_series(self, n_samples: int, n_features: int = 5,
                             seasonality: bool = True,
                             trend: bool = True) -> np.ndarray:
        """Generate base normal time series"""
        
        X = np.zeros((n_samples, n_features))
        t = np.arange(n_samples)
        
        for f in range(n_features):
            # Base signal
            signal = self.rng.randn(n_samples) * 0.5
            
            # Add seasonality
            if seasonality:
                period = self.rng.randint(20, 100)
                amplitude = self.rng.uniform(0.5, 2.0)
                phase = self.rng.uniform(0, 2 * np.pi)
                signal += amplitude * np.sin(2 * np.pi * t / period + phase)
            
            # Add trend
            if trend:
                slope = self.rng.uniform(-0.001, 0.001)
                signal += slope * t
            
            # Add autocorrelation
            for i in range(1, n_samples):
                signal[i] += 0.3 * signal[i-1]
            
            X[:, f] = signal
        
        # Add feature correlations
        correlation_matrix = self._generate_correlation_matrix(n_features)
        X = X @ correlation_matrix
        
        return X
    
    def _generate_correlation_matrix(self, n: int) -> np.ndarray:
        """Generate a valid correlation matrix"""
        A = self.rng.randn(n, n)
        return A @ A.T / n
    
    def inject_anomalies(self, X: np.ndarray, 
                         anomaly_ratio: float = 0.05,
                         anomaly_types: Optional[List[AnomalyType]] = None
                         ) -> Tuple[np.ndarray, np.ndarray, List[AnomalySpec]]:
        """Inject anomalies into the time series"""
        
        n_samples, n_features = X.shape
        X_anomalous = X.copy()
        y = np.zeros(n_samples, dtype=int)  # 0 = normal, 1 = anomaly
        specs = []
        
        if anomaly_types is None:
            anomaly_types = list(AnomalyType)
        
        n_anomalies = int(n_samples * anomaly_ratio)
        
        # Distribute anomalies
        anomalies_per_type = max(1, n_anomalies // len(anomaly_types))
        
        for anomaly_type in anomaly_types:
            for _ in range(anomalies_per_type):
                spec = self._inject_single_anomaly(
                    X_anomalous, y, anomaly_type, n_features
                )
                if spec:
                    specs.append(spec)
        
        return X_anomalous, y, specs
    
    def _inject_single_anomaly(self, X: np.ndarray, y: np.ndarray,
                                anomaly_type: AnomalyType,
                                n_features: int) -> Optional[AnomalySpec]:
        """Inject a single anomaly"""
        
        n_samples = X.shape[0]
        feature_idx = self.rng.randint(0, n_features)
        severity = self.rng.uniform(2, 5)
        
        # Find a non-anomalous region
        attempts = 0
        while attempts < 100:
            if anomaly_type in [AnomalyType.POINT, AnomalyType.CONTEXTUAL]:
                start_idx = self.rng.randint(10, n_samples - 10)
                end_idx = start_idx + 1
            else:
                length = self.rng.randint(5, 20)
                start_idx = self.rng.randint(10, n_samples - length - 10)
                end_idx = start_idx + length
            
            if y[start_idx:end_idx].sum() == 0:
                break
            attempts += 1
        else:
            return None
        
        # Calculate standard deviation for scaling
        std = np.std(X[:, feature_idx])
        
        if anomaly_type == AnomalyType.POINT:
            # Single point outlier
            X[start_idx, feature_idx] += severity * std * self.rng.choice([-1, 1])
            y[start_idx] = 1
            
        elif anomaly_type == AnomalyType.CONTEXTUAL:
            # Value is normal but in wrong context (e.g., high value at night)
            # Swap with a value from different context
            swap_idx = (start_idx + n_samples // 2) % n_samples
            X[start_idx, feature_idx], X[swap_idx, feature_idx] = \
                X[swap_idx, feature_idx], X[start_idx, feature_idx]
            y[start_idx] = 1
            
        elif anomaly_type == AnomalyType.COLLECTIVE:
            # Pattern anomaly - unusual pattern
            pattern = np.sin(np.linspace(0, 4 * np.pi, end_idx - start_idx)) * severity * std
            X[start_idx:end_idx, feature_idx] += pattern
            y[start_idx:end_idx] = 1
            
        elif anomaly_type == AnomalyType.SEASONAL:
            # Break in seasonal pattern
            X[start_idx:end_idx, feature_idx] *= (1 + severity * 0.2)
            y[start_idx:end_idx] = 1
            
        elif anomaly_type == AnomalyType.TREND:
            # Sudden trend change
            trend_change = np.linspace(0, severity * std, end_idx - start_idx)
            X[start_idx:end_idx, feature_idx] += trend_change
            y[start_idx:end_idx] = 1
            
        elif anomaly_type == AnomalyType.LEVEL_SHIFT:
            # Permanent level shift (marks only the transition as anomaly)
            X[start_idx:, feature_idx] += severity * std
            y[start_idx:start_idx + 5] = 1  # Only transition is anomaly
        
        return AnomalySpec(
            type=anomaly_type,
            start_idx=start_idx,
            end_idx=end_idx,
            severity=severity,
            feature_idx=feature_idx
        )
    
    def generate_dataset(self, n_samples: int = 10000,
                         n_features: int = 5,
                         anomaly_ratio: float = 0.05) -> Dict:
        """Generate complete dataset"""
        
        # Generate base series
        X = self.generate_base_series(n_samples, n_features)
        
        # Inject anomalies
        X_anomalous, y, specs = self.inject_anomalies(X, anomaly_ratio)
        
        return {
            'X': X_anomalous,
            'y': y,
            'X_clean': X,
            'anomaly_specs': specs,
            'n_features': n_features,
            'anomaly_ratio': y.mean()
        }


class NetworkLogGenerator:
    """Generate synthetic network log data with anomalies"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate_logs(self, n_samples: int = 10000,
                      anomaly_ratio: float = 0.02) -> Dict:
        """Generate network log data"""
        
        # Features: bytes_in, bytes_out, packets, duration, port, protocol
        X = np.zeros((n_samples, 6))
        
        # Normal traffic patterns
        X[:, 0] = self.rng.exponential(1000, n_samples)  # bytes_in
        X[:, 1] = self.rng.exponential(500, n_samples)   # bytes_out
        X[:, 2] = self.rng.poisson(10, n_samples)        # packets
        X[:, 3] = self.rng.exponential(30, n_samples)    # duration
        X[:, 4] = self.rng.choice([80, 443, 8080, 22, 21], n_samples)  # port
        X[:, 5] = self.rng.choice([0, 1, 2], n_samples)  # protocol (TCP, UDP, ICMP)
        
        y = np.zeros(n_samples, dtype=int)
        
        # Inject anomalies
        n_anomalies = int(n_samples * anomaly_ratio)
        anomaly_indices = self.rng.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = self.rng.choice(['ddos', 'scan', 'exfil', 'unusual'])
            
            if anomaly_type == 'ddos':
                X[idx, 0] *= 100  # Very high bytes in
                X[idx, 2] *= 50  # Many packets
            elif anomaly_type == 'scan':
                X[idx, 3] = 0.1  # Very short duration
                X[idx, 2] = 1    # Single packet
            elif anomaly_type == 'exfil':
                X[idx, 1] *= 100  # Very high bytes out
            elif anomaly_type == 'unusual':
                X[idx, 4] = 31337  # Unusual port
            
            y[idx] = 1
        
        return {
            'X': X,
            'y': y,
            'feature_names': ['bytes_in', 'bytes_out', 'packets', 
                            'duration', 'port', 'protocol'],
            'anomaly_ratio': y.mean()
        }


def create_anomaly_dataset(
    n_train: int = 8000,
    n_val: int = 1000,
    n_test: int = 1000,
    n_features: int = 5,
    anomaly_ratio: float = 0.05,
    data_type: str = 'timeseries',
    seed: int = 42
) -> Dict:
    """Create complete anomaly detection dataset"""
    
    if data_type == 'timeseries':
        generator = TimeSeriesGenerator(seed)
        
        train_data = generator.generate_dataset(n_train, n_features, anomaly_ratio)
        val_data = generator.generate_dataset(n_val, n_features, anomaly_ratio)
        test_data = generator.generate_dataset(n_test, n_features, anomaly_ratio)
        
    elif data_type == 'network':
        generator = NetworkLogGenerator(seed)
        
        train_data = generator.generate_logs(n_train, anomaly_ratio)
        val_data = generator.generate_logs(n_val, anomaly_ratio)
        test_data = generator.generate_logs(n_test, anomaly_ratio)
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return {
        'train': {'X': train_data['X'], 'y': train_data['y']},
        'val': {'X': val_data['X'], 'y': val_data['y']},
        'test': {'X': test_data['X'], 'y': test_data['y']},
        'n_features': train_data.get('n_features', train_data['X'].shape[1]),
        'anomaly_ratio': train_data['anomaly_ratio'],
        'data_type': data_type
    }


def add_noise_to_data(X: np.ndarray, noise_level: float) -> np.ndarray:
    """Add Gaussian noise to data"""
    if noise_level == 0:
        return X
    noise = np.random.randn(*X.shape) * noise_level * np.std(X, axis=0)
    return X + noise


if __name__ == "__main__":
    # Test generation
    dataset = create_anomaly_dataset(n_train=1000, n_test=200)
    print(f"Train shape: {dataset['train']['X'].shape}")
    print(f"Train anomaly ratio: {dataset['train']['y'].mean():.4f}")
    print(f"Test shape: {dataset['test']['X'].shape}")