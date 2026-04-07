"""Synthetic dataset generator for operations optimization outcomes."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def create_operations_dataset(
    n_samples: int = 5000,
    n_features: int = 20,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Create train/val/test splits for continuous operations objective prediction."""
    x, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(4, n_features // 2),
        noise=8.0,
        random_state=seed,
    )

    y = y + 0.1 * (x[:, 0] ** 2) - 0.05 * (x[:, 1] * x[:, 2])

    x_temp, x_test, y_temp, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=seed + 1,
    )

    return {
        "train": {"X": np.asarray(x_train), "y": np.asarray(y_train)},
        "val": {"X": np.asarray(x_val), "y": np.asarray(y_val)},
        "test": {"X": np.asarray(x_test), "y": np.asarray(y_test)},
    }
