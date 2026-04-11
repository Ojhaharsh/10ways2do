"""Synthetic dataset generator for fraud risk assessment."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def create_fraud_risk_dataset(
    n_samples: int = 5000,
    n_features: int = 26,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Create train/val/test splits for binary fraud risk prediction."""
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(4, n_features // 3),
        n_redundant=max(2, n_features // 8),
        n_clusters_per_class=2,
        class_sep=0.9,
        weights=[0.9, 0.1],
        random_state=seed,
    )

    x_temp, x_test, y_temp, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=seed + 1,
        stratify=y_temp,
    )

    return {
        "train": {"X": np.asarray(x_train), "y": np.asarray(y_train)},
        "val": {"X": np.asarray(x_val), "y": np.asarray(y_val)},
        "test": {"X": np.asarray(x_test), "y": np.asarray(y_test)},
    }
