"""
Data loading utilities for all domains.

Provides unified interface for loading, caching, and preprocessing data
across all benchmark domains.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np


class DataLoader:
    """
    Unified data loader for all domains.
    
    Handles:
    - Loading from various formats (JSON, CSV, pickle, numpy)
    - Caching processed data
    - Train/val/test splitting
    - Data validation
    """
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_json(self, filepath: str) -> Any:
        """Load data from JSON file."""
        path = self.data_dir / filepath
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_json(self, data: Any, filepath: str) -> None:
        """Save data to JSON file."""
        path = self.data_dir / filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_pickle(self, filepath: str) -> Any:
        """Load data from pickle file."""
        path = self.data_dir / filepath
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def save_pickle(self, data: Any, filepath: str) -> None:
        """Save data to pickle file."""
        path = self.data_dir / filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_numpy(self, filepath: str) -> np.ndarray:
        """Load numpy array from file."""
        path = self.data_dir / filepath
        return np.load(path)
    
    def save_numpy(self, data: np.ndarray, filepath: str) -> None:
        """Save numpy array to file."""
        path = self.data_dir / filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data)
    
    def load_csv(self, filepath: str, **kwargs) -> 'pd.DataFrame':
        """Load data from CSV file."""
        import pandas as pd
        path = self.data_dir / filepath
        return pd.read_csv(path, **kwargs)
    
    def save_csv(self, data: 'pd.DataFrame', filepath: str, **kwargs) -> None:
        """Save data to CSV file."""
        path = self.data_dir / filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, **kwargs)
    
    def get_cache_path(self, name: str) -> Path:
        """Get path for cached data."""
        return self.cache_dir / f"{name}.pkl"
    
    def load_cached(self, name: str) -> Optional[Any]:
        """Load cached data if available."""
        cache_path = self.get_cache_path(name)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_cached(self, data: Any, name: str) -> None:
        """Save data to cache."""
        cache_path = self.get_cache_path(name)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def clear_cache(self, name: Optional[str] = None) -> None:
        """Clear cached data."""
        if name:
            cache_path = self.get_cache_path(name)
            if cache_path.exists():
                cache_path.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()


class DataSplitter:
    """
    Utility for splitting data into train/val/test sets.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def split_indices(
        self,
        n_samples: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate train/val/test indices.
        
        Args:
            n_samples: Total number of samples
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        indices = self.rng.permutation(n_samples)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        return train_indices, val_indices, test_indices
    
    def split_array(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, np.ndarray]:
        """
        Split arrays into train/val/test sets.
        
        Args:
            X: Feature array
            y: Label array (optional)
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            Dictionary with train/val/test splits
        """
        train_idx, val_idx, test_idx = self.split_indices(
            len(X), train_ratio, val_ratio, test_ratio
        )
        
        result = {
            'X_train': X[train_idx],
            'X_val': X[val_idx],
            'X_test': X[test_idx],
        }
        
        if y is not None:
            result['y_train'] = y[train_idx]
            result['y_val'] = y[val_idx]
            result['y_test'] = y[test_idx]
        
        return result
    
    def split_list(
        self,
        data: List[Any],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Split a list into train/val/test sets.
        
        Args:
            data: List to split
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            Tuple of (train_list, val_list, test_list)
        """
        train_idx, val_idx, test_idx = self.split_indices(
            len(data), train_ratio, val_ratio, test_ratio
        )
        
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        test_data = [data[i] for i in test_idx]
        
        return train_data, val_data, test_data
    
    def temporal_split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, np.ndarray]:
        """
        Split time series data temporally (no shuffling).
        
        Args:
            X: Feature array (assumed to be in temporal order)
            y: Label array (optional)
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            Dictionary with train/val/test splits
        """
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        result = {
            'X_train': X[:train_end],
            'X_val': X[train_end:val_end],
            'X_test': X[val_end:],
        }
        
        if y is not None:
            result['y_train'] = y[:train_end]
            result['y_val'] = y[train_end:val_end]
            result['y_test'] = y[val_end:]
        
        return result


class DataValidator:
    """
    Utility for validating data quality.
    """
    
    @staticmethod
    def check_missing(data: np.ndarray) -> Dict[str, Any]:
        """Check for missing values."""
        if np.issubdtype(data.dtype, np.floating):
            n_nan = np.isnan(data).sum()
            n_inf = np.isinf(data).sum()
            return {
                'has_missing': n_nan > 0 or n_inf > 0,
                'n_nan': int(n_nan),
                'n_inf': int(n_inf),
                'pct_missing': float((n_nan + n_inf) / data.size * 100)
            }
        return {'has_missing': False, 'n_nan': 0, 'n_inf': 0, 'pct_missing': 0.0}
    
    @staticmethod
    def check_shape(data: np.ndarray, expected_shape: Tuple) -> bool:
        """Check if data has expected shape."""
        if len(expected_shape) != len(data.shape):
            return False
        
        for actual, expected in zip(data.shape, expected_shape):
            if expected is not None and actual != expected:
                return False
        
        return True
    
    @staticmethod
    def check_range(data: np.ndarray, min_val: float = None, 
                    max_val: float = None) -> Dict[str, Any]:
        """Check if data is within expected range."""
        actual_min = float(np.nanmin(data))
        actual_max = float(np.nanmax(data))
        
        in_range = True
        if min_val is not None and actual_min < min_val:
            in_range = False
        if max_val is not None and actual_max > max_val:
            in_range = False
        
        return {
            'in_range': in_range,
            'actual_min': actual_min,
            'actual_max': actual_max,
            'expected_min': min_val,
            'expected_max': max_val
        }
    
    @staticmethod
    def check_class_balance(labels: np.ndarray) -> Dict[str, Any]:
        """Check class balance in labels."""
        unique, counts = np.unique(labels, return_counts=True)
        
        total = len(labels)
        class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        class_pct = {int(u): float(c / total * 100) for u, c in zip(unique, counts)}
        
        # Check imbalance ratio
        if len(counts) > 1:
            imbalance_ratio = float(max(counts) / min(counts))
        else:
            imbalance_ratio = 1.0
        
        return {
            'n_classes': len(unique),
            'class_distribution': class_dist,
            'class_percentages': class_pct,
            'imbalance_ratio': imbalance_ratio,
            'is_balanced': imbalance_ratio < 3.0
        }
    
    @staticmethod
    def validate_dataset(
        X: np.ndarray,
        y: np.ndarray = None,
        expected_features: int = None
    ) -> Dict[str, Any]:
        """
        Comprehensive dataset validation.
        
        Args:
            X: Feature array
            y: Label array (optional)
            expected_features: Expected number of features
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'n_samples': len(X),
            'n_features': X.shape[1] if X.ndim > 1 else 1,
            'dtype': str(X.dtype),
            'shape': X.shape,
        }
        
        # Check missing values
        results['missing'] = DataValidator.check_missing(X)
        
        # Check shape
        if expected_features is not None:
            expected_shape = (None, expected_features)
            results['shape_valid'] = DataValidator.check_shape(X, expected_shape)
        
        # Check range
        results['range'] = DataValidator.check_range(X)
        
        # Check labels if provided
        if y is not None:
            results['labels'] = {
                'n_samples': len(y),
                'dtype': str(y.dtype),
            }
            results['labels']['balance'] = DataValidator.check_class_balance(y)
            results['labels']['missing'] = DataValidator.check_missing(y)
        
        return results


def load_domain_data(
    domain: str,
    data_dir: str = "data",
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Load data for a specific domain.
    
    Args:
        domain: One of 'a', 'b', 'c', 'd'
        data_dir: Base data directory
        use_cache: Whether to use cached data
        
    Returns:
        Dictionary with train/val/test data
    """
    loader = DataLoader(data_dir)
    cache_name = f"domain_{domain}_data"
    
    if use_cache:
        cached = loader.load_cached(cache_name)
        if cached is not None:
            return cached
    
    # Generate synthetic data based on domain
    if domain == 'a':
        from src.domain_a_information_extraction.data_generator import create_ie_dataset
        data = create_ie_dataset()
    elif domain == 'b':
        from src.domain_b_anomaly_detection.data_generator import create_anomaly_dataset
        data = create_anomaly_dataset()
    elif domain == 'c':
        from src.domain_c_recommendation.data_generator import create_recommendation_dataset
        data = create_recommendation_dataset()
    elif domain == 'd':
        from src.domain_d_time_series.data_generator import create_timeseries_dataset
        data = create_timeseries_dataset()
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    if use_cache:
        loader.save_cached(data, cache_name)
    
    return data